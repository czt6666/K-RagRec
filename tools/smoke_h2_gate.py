"""Verify H2 RetrievalGate forward without loading the LLM. Run from repo
root with PYTHONPATH=methods/h2_gate."""
import torch

from src.model.gate import RetrievalGate

torch.manual_seed(0)
gate = RetrievalGate(emb_dim=1024, hidden=256)
print(f"RetrievalGate params: {sum(p.numel() for p in gate.parameters()):,}")

# Vary subgraph count
for n in [1, 5, 10]:
    subgraph_embeds = torch.randn(n, 1024)
    query = torch.randn(1024)
    weights, logits = gate(subgraph_embeds, query)
    assert weights.shape == (n,) and logits.shape == (n,)
    print(f"  N={n:2d}  weights={[round(v, 3) for v in weights.tolist()]}  "
          f"mean={weights.mean():.3f}")

# Gumbel-Sigmoid: deterministic in eval, stochastic in train
torch.manual_seed(1)
logits = torch.randn(8) * 2
print(f"\nGumbel-Sigmoid hard masks (3 samples):")
for _ in range(3):
    mask = gate.gumbel_sigmoid(logits, tau=1.0, hard=True)
    print(f"  {mask.tolist()}")

# Verify gradient flows through gate
torch.manual_seed(2)
gate.train()
subgraph_embeds = torch.randn(5, 1024, requires_grad=False)
query = torch.randn(1024, requires_grad=False)
weights, _ = gate(subgraph_embeds, query)
loss = (subgraph_embeds * weights.unsqueeze(-1)).sum()
loss.backward()
nz = sum(p.grad.abs().sum().item() > 0 for p in gate.parameters() if p.grad is not None)
print(f"\nGradient flow OK: {nz} of {sum(1 for _ in gate.parameters())} parameters have non-zero gradient.")
print("\n[OK] H2 gate smoke test passed.")
