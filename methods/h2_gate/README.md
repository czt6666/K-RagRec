# H2 — Learnable retrieval gate

> Forked from `methods/baseline/`. Augments the popularity-threshold rule
> with a small MLP gate that scores each retrieved subgraph against the
> user-history query embedding, producing a soft (sigmoid) or hard
> (Gumbel-Sigmoid straight-through) weight that re-weights subgraph
> contributions before pooling. The popularity selector still controls
> *which* items get sent to retrieval (so retrieval-time speed is
> preserved); the gate then decides *how much each retrieved subgraph
> contributes* in the LLM forward.

## Files changed vs baseline

| file | what changed |
|---|---|
| `src/model/gate.py` | **new** — `RetrievalGate(emb_dim, hidden)`. Concatenates `[g, q, |g-q|, g*q]` (1024×4 → 256 → 1) and outputs sigmoid weights. Includes a `gumbel_sigmoid(logits, tau, hard)` static method for straight-through binary selection. |
| `src/model/graph_llm.py` | Adds `self.gate`. In `encode_graphs`, after GNN scatter-mean, multiplies each subgraph embedding by the gate weight. Optionally enables Gumbel-Sigmoid in training. Adds an L1-style `_last_gate_sparsity_loss` that the `forward()` returns alongside CE so the model is incentivized to keep the gate sparse. |
| `train.py` / `evaluate.py` | Compute SBERT embedding of the watching-history string per sample (`retrieval_model.encode_query(input)`) and put it in `sample['query_emb']`. |
| `src/config.py` | + `--gate_use_gumbel`, `--gate_tau`, `--gate_sparsity_lambda`. |

## Run (PowerShell)

```powershell
$env:PYTHONPATH = "methods/h2_gate"

# Default: soft sigmoid weighting, no sparsity penalty
python methods/h2_gate/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "<llama-2-7b path>" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5

# Hard binary selection via Gumbel-Sigmoid + sparsity penalty
python methods/h2_gate/train.py `
    --model_name graph_llm --llm_model_name 7b `
    --llm_model_path "<llama-2-7b path>" `
    --llm_frozen True --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 `
    --gate_use_gumbel --gate_tau 1.0 --gate_sparsity_lambda 0.01

# Evaluate (any of the above)
python methods/h2_gate/evaluate.py `
    --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m `
    --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5
```

## Smoke test

```powershell
$env:PYTHONPATH = "methods/h2_gate"
python tools/smoke_h2_gate.py
```

Expected: `RetrievalGate params: 1,049,089`, then per-N weight tables, three
binary Gumbel masks, and `Gradient flow OK: 4 of 4 parameters ...`,
ending with `[OK] H2 gate smoke test passed.`

## Ablation grid

| config | what it tests |
|---|---|
| baseline (no gate) | reference number |
| `--gate_sparsity_lambda 0` (soft sigmoid) | continuous re-weighting only |
| `--gate_use_gumbel --gate_sparsity_lambda 0.0` | binary selection, no sparsity push |
| `--gate_use_gumbel --gate_sparsity_lambda 0.01` | binary + sparsity (paper headline) |
| `--gate_use_gumbel --gate_sparsity_lambda 0.05` | aggressive sparsity |

## Things to watch when training

- The gate is **jointly trained with the rest of the model**. Loss = CE +
  `gate_sparsity_lambda * mean(weights)`. Without sparsity, the gate
  defaults to ≈0.5 weights everywhere and offers little benefit.
- `--gate_tau` between 0.5 and 1.5 typically works; lower = sharper
  binary, harder optimization.
- The popularity-based pre-filter in `whether_retrieval` is unchanged, so
  this method preserves retrieval-time efficiency. The gate only
  re-weights what was already retrieved.
