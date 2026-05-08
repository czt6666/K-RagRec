# H1 — Graph-Q-Former multi-token projector

> Forked from `methods/baseline/`. Replaces the baseline's
> `MLP -> mean(dim=0)` graph-to-LLM bridge — which collapses N retrieved
> subgraphs into a single soft-prompt token — with a small cross-attention
> Q-Former that emits `q` query tokens regardless of N. See
> `paper/INNOVATIONS.md` H1 for motivation and `paper/CODE_ANALYSIS.md` §5
> row 1 for the underlying bug.

## Files changed vs baseline

| file | what changed |
|---|---|
| `src/model/qformer.py` | **new** — `GraphQFormer(gnn_dim, llm_dim, num_query_tokens, num_heads)`. Cross-attention from `q` learnable query tokens over the `(N, gnn_dim)` projected subgraph stack, followed by LayerNorm + FFN. Output: `(q, llm_dim)`. |
| `src/model/graph_llm.py` | Drops `self.projector`. Adds `self.qformer = GraphQFormer(...)`. In `forward()` and `inference()`, replaces `mean(dim=0, keepdim=True)` over projected embeddings with a direct `self.qformer(g_embeds)` call so all `q` tokens enter the LLM prefix. |
| `src/config.py` | + `--num_query_tokens` (default 8), `--qformer_num_heads` (default 8), `--qformer_dropout` (default 0.0), `--llm_hidden_dim` (default 4096; use 3584 for Qwen2-7B). |

The retrieval pipeline (`retrieve.py`) and the GNN encoder (`src/model/gnn.py`) are untouched.

## Run (PowerShell)

```powershell
$env:PYTHONPATH = "methods/h1_qformer"

# Training (LLaMA-2-7B, q=8 query tokens)
python methods/h1_qformer/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "<llama-2-7b path>" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 `
    --num_query_tokens 8 --qformer_num_heads 8 `
    --llm_hidden_dim 4096

# For Qwen2-7B: use --llm_hidden_dim 3584

# Evaluation
python methods/h1_qformer/evaluate.py `
    --model_name graph_llm `
    --llm_model_name 7b --llm_frozen True --dataset ml1m `
    --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 `
    --num_query_tokens 8 --qformer_num_heads 8 --llm_hidden_dim 4096
```

## Smoke test

```powershell
$env:PYTHONPATH = "methods/h1_qformer"
python tools/smoke_h1_qformer.py
```

Expected output: `GraphQFormer params: 144,787,456 (144.79 M)` and four
shapes `(8, 4096)`, ending with `[OK] H1 Q-Former smoke test passed.`

## Ablation grid (suggested)

| config | what it tests |
|---|---|
| baseline (mean) | reference number from paper Table 1 |
| `--num_query_tokens 1` | should be ≈ baseline (degenerate Q-Former) |
| `--num_query_tokens 4` | medium |
| `--num_query_tokens 8` | default — paper headline number |
| `--num_query_tokens 16` | upper end; check prompt length |
| concat all N (no Q-Former) | "what if we just concat" baseline; will need a small code variant |

## Parameter cost

The Q-Former adds ~145 M trainable params (cross-attention with
`embed_dim=4096, heads=8`, plus a 2× FFN). That's ~2 % of LLaMA-2-7B.
Worth tuning `qformer_num_heads` and the FFN expansion factor (currently
×2 inside `qformer.py`) if you need to shrink it.
