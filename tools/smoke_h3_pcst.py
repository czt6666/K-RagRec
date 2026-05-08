"""End-to-end H3 PCST smoke test on real ML1M+FB data, no LLM. Run from
repo root with PYTHONPATH=methods/h3_pcst."""
import json
import time

from retrieve import GraphRetrieval


def main():
    print("[1/3] Loading retriever (this also precomputes 3883 movie anchors)...")
    t0 = time.time()
    r = GraphRetrieval(model_name="sbert", path="dataset/fb")
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  fused_x shape : {tuple(r.fused_x.shape)}")
    print(f"  anchor count  : {len(r.movie_id_to_anchor)}")
    print(f"  G2 (layer3) x : {tuple(r.G2.x.shape)}")

    print("\n[2/3] Loading first sample...")
    with open("dataset/ML1M/10000_data_id_20.json") as f:
        sample = json.load(f)[0]
    seq_ids = json.loads(sample["sequence_ids"])
    print(f"  sequence_ids  : {seq_ids}")

    print("\n[3/3] PCST solve...")
    t0 = time.time()
    out = r.pcst_retrieval_topk(
        sample["input"], seq_ids,
        anchor_prize=10.0, topk_query_prize=20,
        edge_cost=0.5, max_subgraph_nodes=200,
    )
    elapsed = time.time() - t0
    sg = out[0]
    print(f"  done in {elapsed:.2f}s")
    print(f"  subgraph nodes: {sg.num_nodes}")
    print(f"  subgraph edges: {sg.edge_index.shape[1]}")
    print(f"  x shape       : {tuple(sg.x.shape)}")
    print(f"  edge_attr     : "
          f"{tuple(sg.edge_attr.shape) if sg.edge_attr is not None else None}")

    # Sanity: connectedness check
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(sg.num_nodes))
    G.add_edges_from(sg.edge_index.t().tolist())
    n_components = nx.number_connected_components(G)
    print(f"  connected components: {n_components}")

    print("\n[OK] H3 PCST smoke test passed.")


if __name__ == "__main__":
    main()
