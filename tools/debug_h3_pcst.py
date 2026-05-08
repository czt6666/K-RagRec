"""Probe the PCST solver under different parameter regimes to find a
configuration that actually returns a connected subgraph."""
import json

import numpy as np
import torch
import pcst_fast

from retrieve import GraphRetrieval


def main():
    r = GraphRetrieval(model_name="sbert", path="dataset/fb")
    with open("dataset/ML1M/10000_data_id_20.json") as f:
        sample = json.load(f)[0]
    seq_ids = json.loads(sample["sequence_ids"])

    # Anchors
    anchors = [r.movie_id_to_anchor[m] for m in seq_ids if m in r.movie_id_to_anchor]
    print(f"anchors ({len(anchors)}): {anchors}")

    # Are the anchors close in the KG?
    import networkx as nx
    g_full = nx.Graph()
    g_full.add_nodes_from(range(r.G.num_nodes))
    g_full.add_edges_from(r.G.edge_index.t().tolist())
    print(f"KG: {g_full.number_of_nodes()} nodes, {g_full.number_of_edges()} edges, "
          f"avg degree {2*g_full.number_of_edges()/g_full.number_of_nodes():.2f}")

    # Distance among anchors (will be expensive; sample)
    a0 = anchors[0]
    for a in anchors[1:5]:
        try:
            d = nx.shortest_path_length(g_full, a0, a)
            print(f"  shortest path {a0} -> {a}: {d}")
        except nx.NetworkXNoPath:
            print(f"  no path {a0} -> {a}")

    # Sweep PCST params
    edge_index_np = r.G.edge_index.cpu().numpy().T.astype(np.int64)
    n = r.G.num_nodes

    for anchor_prize in [10.0, 100.0, 1000.0]:
        for edge_cost in [1.0, 0.1, 0.01]:
            for pruning in ["strong", "simple", "gw"]:
                prizes = np.zeros(n, dtype=np.float32)
                for a in anchors:
                    prizes[a] += anchor_prize
                costs = np.full(edge_index_np.shape[0], float(edge_cost), dtype=np.float32)
                verts, kept = pcst_fast.pcst_fast(edge_index_np, prizes, costs,
                                                  -1, 1, pruning, 0)
                print(f"  anchor={anchor_prize:6.0f} cost={edge_cost:5.2f} pruning={pruning:7s} "
                      f"-> nodes={len(verts):3d} edges={len(kept):3d}")


if __name__ == "__main__":
    main()
