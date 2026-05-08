"""Tiny test: load only the KG graph (no SBERT) and run nx.steiner_tree."""
import time

import networkx as nx
import torch
from networkx.algorithms.approximation import steiner_tree


def main():
    print("Loading KG edge structure...")
    g = torch.load("dataset/fb/graphs/0.pt", weights_only=False)
    print(f"  nodes={g.num_nodes}, edges={g.edge_index.shape[1]}")

    print("Building networkx graph...")
    t0 = time.time()
    nxg = nx.Graph()
    nxg.add_nodes_from(range(g.num_nodes))
    nxg.add_edges_from(g.edge_index.t().tolist())
    print(f"  done in {time.time()-t0:.1f}s, "
          f"connected={nx.is_connected(nxg)}")

    # Use the same anchors from the debug script
    anchors = [3322, 2913, 1219, 1170, 4085, 4142, 3263, 422, 3310, 1571]
    print(f"\nSteiner tree on {len(anchors)} anchors...")
    t0 = time.time()
    st = steiner_tree(nxg, anchors)
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  tree nodes: {st.number_of_nodes()}, edges: {st.number_of_edges()}")
    print(f"  components: {nx.number_connected_components(st)}")
    print(f"  contains all anchors? {all(a in st.nodes for a in anchors)}")


if __name__ == "__main__":
    main()
