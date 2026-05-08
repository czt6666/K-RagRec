"""Step into pcst_retrieve to understand why edges go missing."""
import json

from retrieve import GraphRetrieval
from src.pcst_retrieval import pcst_retrieve


def main():
    r = GraphRetrieval(model_name="sbert", path="dataset/fb")
    with open("dataset/ML1M/10000_data_id_20.json") as f:
        sample = json.load(f)[0]
    seq_ids = json.loads(sample["sequence_ids"])

    print("Calling pcst_retrieve directly...")
    sg = pcst_retrieve(
        r, sample["input"], seq_ids,
        fused_x=r.fused_x,
        anchor_prize=10.0,
        topk_query_prize=0,        # no query prizes
        edge_cost=1.0,             # match debug
        max_subgraph_nodes=200,
    )
    print(f"  no query prizes, cost=1.0   -> nodes={sg.num_nodes} edges={sg.edge_index.shape[1]}")

    sg = pcst_retrieve(
        r, sample["input"], seq_ids,
        fused_x=r.fused_x,
        anchor_prize=10.0,
        topk_query_prize=20,
        edge_cost=0.5,
        max_subgraph_nodes=200,
    )
    print(f"  query top-20, cost=0.5      -> nodes={sg.num_nodes} edges={sg.edge_index.shape[1]}")

    # bigger params
    sg = pcst_retrieve(
        r, sample["input"], seq_ids,
        fused_x=r.fused_x,
        anchor_prize=20.0,
        topk_query_prize=50,
        edge_cost=0.3,
        max_subgraph_nodes=200,
    )
    print(f"  query top-50, cost=0.3      -> nodes={sg.num_nodes} edges={sg.edge_index.shape[1]}")


if __name__ == "__main__":
    main()
