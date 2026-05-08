"""Verify H1 GraphQFormer forward without loading the LLM. Run from repo
root with PYTHONPATH=methods/h1_qformer."""
import torch

from src.model.qformer import GraphQFormer

torch.manual_seed(0)
qformer = GraphQFormer(gnn_dim=1024, llm_dim=4096, num_query_tokens=8, num_heads=8)
n_params = sum(p.numel() for p in qformer.parameters())
print(f"GraphQFormer params: {n_params:,}  ({n_params/1e6:.2f} M)")

# Variable subgraph counts per sample
for n in [1, 5, 10, 20]:
    x = torch.randn(n, 1024)
    out = qformer(x)
    assert out.shape == (8, 4096), f"unexpected shape {out.shape} for n={n}"
    print(f"  N={n:2d} -> tokens shape {tuple(out.shape)}")

print("\n[OK] H1 Q-Former smoke test passed.")
