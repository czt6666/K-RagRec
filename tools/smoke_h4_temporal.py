"""Verify H4 SequentialEncoder forward without loading the LLM. Run from
repo root with PYTHONPATH=methods/h4_temporal."""
import torch

from src.model.temporal import SequentialEncoder

torch.manual_seed(0)
enc = SequentialEncoder(vocab_size=4000, max_seq_len=32, d_model=256,
                        nhead=4, num_layers=2, dropout=0.0, out_dim=1024)
print(f"SequentialEncoder params: {sum(p.numel() for p in enc.parameters()):,}")

# 3-sample batch with variable real lengths via padding (id=0)
ids = torch.tensor([
    [1339, 1366, 17, 159, 509, 3416, 1748, 912, 1203, 3267] + [0]*22,
    [858, 260, 1221, 1287, 1387, 2571, 1214, 2028, 589, 1291] + [0]*22,
    [3897, 1, 1265, 588, 3114, 3255, 2396, 34, 2599, 1923] + [0]*22,
])
out = enc(ids)
print(f"  no time -> {tuple(out.shape)}")
assert out.shape == (3, 1024)

# With time deltas (seconds since now)
ts = torch.tensor([
    [864000, 432000, 86400, 3600, 1800, 600, 60, 30, 10, 1] + [0]*22,
] * 3, dtype=torch.float32)
out_t = enc(ids, time_deltas=ts)
print(f"  with time -> {tuple(out_t.shape)}, max abs diff vs no-time: {(out - out_t).abs().max():.4f}")

print("\n[OK] H4 sequential encoder smoke test passed.")
