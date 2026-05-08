"""Verify H5 RGCN / CompGCN forward without loading the LLM. Run from repo
root with PYTHONPATH=methods/h5_rel_gnn."""
import torch
from torch_geometric.data import Data, Batch

from src.model.gnn import load_gnn_model

torch.manual_seed(0)
print("Available GNNs:", list(load_gnn_model.keys()))

# Build a tiny fake subgraph batch (mimics what retrieve.py emits)
def make_data(n=8, e=12, in_dim=1024, num_rel=20):
    return Data(
        x=torch.randn(n, in_dim),
        edge_index=torch.randint(0, n, (2, e)),
        edge_attr=torch.randn(e, in_dim),
        edge_type=torch.randint(0, num_rel, (e,)),
        num_nodes=n,
    )

batch = Batch.from_data_list([make_data() for _ in range(3)])
print(f"batch.x={tuple(batch.x.shape)} ei={tuple(batch.edge_index.shape)} "
      f"edge_attr={tuple(batch.edge_attr.shape)} edge_type={tuple(batch.edge_type.shape)}")

for name in ['gcn', 'gat', 'gt', 'graphsage', 'rgcn', 'compgcn']:
    cls = load_gnn_model[name]
    gnn = cls(in_channels=1024, hidden_channels=1024, out_channels=1024,
              num_layers=4, dropout=0.0, num_heads=4)
    kwargs = {}
    if name in ('rgcn',):
        kwargs['edge_type'] = batch.edge_type
    out, _ = gnn(batch.x, batch.edge_index, batch.edge_attr, **kwargs)
    print(f"  OK  {name:10s} -> {tuple(out.shape)}")

print("\n[OK] H5 GNN smoke test passed.")
