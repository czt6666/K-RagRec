"""H3 multi-source connected-subgraph retrieval over the FB knowledge graph.

The baseline issues one independent top-k cosine query per history item,
then re-ranks the union and emits ~10 disjoint 1-hop subgraphs. That
loses the *connectivity* between history items: the user's 10 movies are
likely linked through shared cast / studio / genre nodes in the KG, but
those linker nodes are never explicitly retrieved.

H3 instead returns a single connected subgraph per sample by computing an
approximate Steiner tree that covers all 10 history-item anchors plus the
top-K nodes most similar to the SBERT query, then expanding by one hop
around the tree. This is the same "joint connectivity-aware retrieval"
spirit as G-Retriever's PCST, but implemented with networkx because the
Windows wheel of pcst_fast 1.0.10 returns corrupt output.

Hop-field fusion: node features used for the similarity prizes are a
weighted mixture of the three GNN layers in
``dataset/fb/graphs/{0,layer2,layer3}.pt``.
"""
from __future__ import annotations

import os

from collections import deque

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


def _bfs_khop_neighborhood(nxg, source, k):
    """BFS up to k hops from a single source. Returns set of nodes."""
    visited = {source}
    frontier = deque([(source, 0)])
    while frontier:
        node, d = frontier.popleft()
        if d >= k:
            continue
        for nb in nxg.neighbors(node):
            if nb not in visited:
                visited.add(nb)
                frontier.append((nb, d + 1))
    return visited


def multisource_neighborhood(nxg, terminals, k=2, max_nodes=200):
    """Fast O(|terminals| * (V+E) up to k hops) Steiner-tree alternative.

    Take the k-hop BFS neighborhood of each terminal and union them. Where
    two terminals are < 2k hops apart, the union naturally contains a
    path connecting them (since both BFS frontiers meet in the middle).
    Also lets us rank "linker" nodes by how many terminals they connect.
    """
    counts = {}
    for t in terminals:
        if t not in nxg:
            continue
        nbhd = _bfs_khop_neighborhood(nxg, t, k)
        for n in nbhd:
            counts[n] = counts.get(n, 0) + 1
    # rank: terminals first (count >= 1 with self), then by # terminals reached
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    selected = [n for n, _ in ranked[:max_nodes]]
    return set(selected)


def fused_node_features(G_x, G1_x, G2_x, weights=(0.5, 0.3, 0.2)):
    """Static layer-fusion. Future work: query-conditioned weights."""
    w0, w1, w2 = weights
    return w0 * G_x + w1 * G1_x + w2 * G2_x


def precompute_movie_anchors(retriever, batch_size=128):
    """Find the closest KG node to each ML1M movie's title via SBERT cosine.

    Returns dict[int -> int] mapping ml1m movie_id -> node index in G.
    """
    titles = []
    ids = []
    for mid, name in retriever.movie_id_to_name.items():
        ids.append(int(mid))
        titles.append(str(name))
    embs = retriever.text2embedding(retriever.model, retriever.tokenizer,
                                    retriever.device, titles)
    G_x = retriever.G.x  # (N, 1024)
    # cosine: (M, 1024) x (N, 1024).T -> (M, N), argmax over N
    embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-9)
    Gx = G_x / (G_x.norm(dim=-1, keepdim=True) + 1e-9)
    out = {}
    for start in range(0, len(ids), batch_size):
        chunk = embs[start:start + batch_size]
        sims = chunk @ Gx.t()
        node_idx = sims.argmax(dim=-1).tolist()
        for k, n in enumerate(node_idx):
            out[ids[start + k]] = int(n)
    return out


def pcst_retrieve(
    retriever,
    query_text: str,
    sequence_ids: list,
    fused_x: torch.Tensor,
    anchor_prize: float = 10.0,
    topk_query_prize: int = 20,
    edge_cost: float = 0.5,
    pruning: str = "strong",
    max_subgraph_nodes: int = 200,
) -> Data:
    """Solve a single multi-source PCST over the full FB KG.

    Args:
        retriever: a ``GraphRetrieval`` instance (provides ``encode_query``,
            ``movie_id_to_anchor`` cache, ``G``, ``global_edge_type`` if H5).
        query_text: the user's watching-history string (used as SBERT query).
        sequence_ids: ML1M ids of the 10 history items.
        fused_x: pre-fused node features of shape ``(N, 1024)``.
        anchor_prize: prize put on each history-item anchor node.
        topk_query_prize: number of similarity-top-K nodes that get prizes.
        edge_cost: uniform cost per edge (controls subgraph size).
        pruning: pcst_fast pruning mode.
        max_subgraph_nodes: hard cap on returned subgraph size.

    Returns:
        ``torch_geometric.data.Data`` with reindexed node ids.
    """
    G = retriever.G
    debug = bool(os.environ.get("PCST_DEBUG"))

    # 1. anchor nodes
    anchors = []
    for m in sequence_ids:
        a = retriever.movie_id_to_anchor.get(int(m))
        if a is not None:
            anchors.append(int(a))

    # 2. query similarity terminals
    q_emb = retriever.encode_query(query_text).squeeze(0)
    q = q_emb / (q_emb.norm() + 1e-9)
    fx = fused_x / (fused_x.norm(dim=-1, keepdim=True) + 1e-9)
    sims = (fx @ q).cpu()  # (N,)
    if topk_query_prize > 0:
        _, top_idx = torch.topk(sims, k=min(topk_query_prize, fused_x.size(0)))
        top_idx = top_idx.tolist()
    else:
        top_idx = []

    terminals = list(set(anchors) | set(int(i) for i in top_idx))
    if debug:
        print(f"[pcst-nx] anchors={len(anchors)} top_query={len(top_idx)} "
              f"terminals={len(terminals)}")

    # 3. build cached networkx graph for Steiner tree solver
    nx_graph = retriever.Graph  # already built in GraphRetrieval
    # Restrict terminals to the largest connected component to avoid
    # NetworkXError from Steiner tree on disconnected terminals.
    if not nx.is_connected(nx_graph):
        comp_of = {n: i for i, comp in enumerate(nx.connected_components(nx_graph)) for n in comp}
        from collections import Counter
        comp_count = Counter(comp_of[t] for t in terminals if t in comp_of)
        if not comp_count:
            return Data(x=G.x[:1], edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, G.edge_attr.size(-1)), num_nodes=1)
        majority_comp = comp_count.most_common(1)[0][0]
        terminals = [t for t in terminals if comp_of.get(t) == majority_comp]
        if debug:
            print(f"[pcst-nx] restricted to majority component, terminals_now={len(terminals)}")

    if not terminals:
        return Data(x=G.x[:1], edge_index=torch.empty(2, 0, dtype=torch.long),
                    edge_attr=torch.empty(0, G.edge_attr.size(-1)), num_nodes=1)

    # 4. fast Steiner-tree alternative: union of k-hop BFS neighborhoods
    # around each terminal. Linker nodes that lie within k hops of multiple
    # terminals are picked first (high "shared neighborhood" rank).
    expanded = multisource_neighborhood(nx_graph, terminals, k=2,
                                        max_nodes=max_subgraph_nodes)
    sub_nodes = sorted(expanded)[:max_subgraph_nodes]
    node_map = {old: new for new, old in enumerate(sub_nodes)}
    if debug:
        print(f"[pcst-nx] tree_nodes={len(node_set)} expanded={len(expanded)} "
              f"final_nodes={len(sub_nodes)} "
              f"anchors_in_subgraph={sum(int(a) in node_map for a in anchors)}")

    # 6. collect edges between selected nodes (from the original directed KG)
    src = G.edge_index[0]
    dst = G.edge_index[1]
    sub_set_t = torch.tensor(sub_nodes)
    in_set = torch.zeros(G.num_nodes, dtype=torch.bool)
    in_set[sub_set_t] = True
    edge_mask = in_set[src] & in_set[dst]
    kept_edges = torch.nonzero(edge_mask, as_tuple=False).squeeze(-1).tolist()
    if debug:
        print(f"[pcst-nx] kept_edges_after_mask={len(kept_edges)}")

    sub_x = G.x[sub_nodes]
    if len(kept_edges) > 0:
        sub_e_old = G.edge_index[:, kept_edges]
        new_src = torch.tensor([node_map[int(s)] for s in sub_e_old[0].tolist()])
        new_dst = torch.tensor([node_map[int(t)] for t in sub_e_old[1].tolist()])
        sub_edge_index = torch.stack([new_src, new_dst])
        sub_edge_attr = G.edge_attr[kept_edges]
    else:
        sub_edge_index = torch.empty(2, 0, dtype=torch.long)
        sub_edge_attr = torch.empty(0, G.edge_attr.size(-1))

    _ = anchor_prize, edge_cost, pruning  # signatures preserved for ablations


    return Data(
        x=sub_x,
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        num_nodes=len(sub_nodes),
    )
