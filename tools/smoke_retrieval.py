"""
Smoke test: verify the KG retrieval pipeline works end-to-end without loading
the LLM. Useful for confirming data + SBERT + PCST_fast wiring before
attempting full training (which requires server-grade GPUs).

Run from repo root with the method whose retriever you want to smoke-test on
PYTHONPATH:
    $env:PYTHONPATH = "methods/baseline"
    python tools/smoke_retrieval.py
"""
import json
import time
import sys

from retrieve import GraphRetrieval


def main():
    print("[1/4] Instantiating GraphRetrieval (loads SBERT + 3 KG layers)...")
    t0 = time.time()
    retriever = GraphRetrieval(model_name="sbert", path="dataset/fb")
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  G  (layer 0) : {retriever.G.num_nodes} nodes, "
          f"{retriever.G.edge_index.shape[1]} edges, "
          f"x.shape={tuple(retriever.G.x.shape)}")
    print(f"  G1 (layer 1) : x.shape={tuple(retriever.G1.x.shape)}")
    print(f"  movies known : {len(retriever.movie_id_to_name)}")
    print(f"  popularity sort length: {len(retriever.sorted_item_ids)}")

    print("\n[2/4] Loading first sample from dataset/ML1M/10000_data_id_20.json...")
    with open("dataset/ML1M/10000_data_id_20.json", "r") as f:
        sample = json.load(f)[0]
    seq_ids = json.loads(sample["sequence_ids"])
    print(f"  input          : {sample['input'][:80]}...")
    print(f"  target letter  : {sample['output']}")
    print(f"  sequence_ids   : {seq_ids}")

    print("\n[3/4] Calling whether_retrieval (popularity selective policy)...")
    adaptive_ratio = 5
    k = adaptive_ratio * len(seq_ids)
    retrieve_movies = retriever.whether_retrieval(seq_ids, k)
    print(f"  k = adaptive_ratio*len(seq) = {adaptive_ratio}*{len(seq_ids)} = {k}")
    print(f"  selected items ({len(retrieve_movies)}): {retrieve_movies}")
    print(f"  names: {[retriever.movie_id_to_name.get(m, '?') for m in retrieve_movies[:5]]}...")

    print("\n[4/4] Calling retrieval_topk (sub_graph_numbers=3, reranking_numbers=5)...")
    t0 = time.time()
    graphs = retriever.retrieval_topk(
        sample["input"], retrieve_movies,
        topk_nodes=3, topk_rerank_nodes=5,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.2f}s")
    print(f"  returned {len(graphs)} subgraphs (expected 2*reranking_numbers = 10)")
    for i, g in enumerate(graphs[:3]):
        print(f"    subgraph[{i}]: x={tuple(g.x.shape)}, "
              f"edge_index={tuple(g.edge_index.shape)}, "
              f"edge_attr={tuple(g.edge_attr.shape) if g.edge_attr is not None else None}")

    print("\n[OK] Retrieval pipeline smoke test passed.")


if __name__ == "__main__":
    main()
