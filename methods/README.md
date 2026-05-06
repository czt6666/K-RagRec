# `methods/` — Method Index

This directory holds one self-contained subdirectory per method. The original K-RagRec implementation lives in `baseline/` and is treated as **read-only** — it serves as the reproducible reference for the ACL 2025 paper. Every other directory is a *copy* of `baseline/` with targeted modifications, so we can compare against the unmodified original at any time.

## Naming Convention

| Directory | Purpose | Status |
|---|---|---|
| `baseline/` | Original K-RagRec code (the ACL 2025 paper). **Do not modify.** | ✅ committed |
| `h1_qformer/` | H1 — Multi-token Graph-Q-Former projector (replace mean pooling) | ⏳ planned |
| `h2_gate/` | H2 — Learnable adaptive retrieval gate (replace popularity heuristic) | ⏳ planned |
| `h3_pcst/` | H3 — Multi-source PCST + path-aware retrieval | ⏳ planned |
| `h5_rel_gnn/` | H5 — Relation-aware GNN encoder (CompGCN / HGT) | ⏳ planned |
| `k2_ragrec/` | Combined H1 + H2 + H3 + H5 — final method for the paper | ⏳ planned |

H/M/L priority and motivation: see `paper/INNOVATIONS.md`.

## Directory Workflow

When starting a new method (e.g. `h1_qformer`):

```powershell
# 1. Copy baseline into a new method directory
Copy-Item -Recurse methods/baseline methods/h1_qformer

# 2. Add a method-level README documenting:
#    - which baseline commit it forks from
#    - the specific files/lines being changed
#    - how to run training & evaluation
New-Item methods/h1_qformer/README.md
```

## How to Run (PowerShell)

The cwd is always the repo root (`K-ragrec/`). Datasets resolve relative to cwd; `PYTHONPATH` controls where Python finds the `src/` package.

```powershell
$env:PYTHONPATH = "methods/<method_name>"
python methods/<method_name>/train.py --args ...
python methods/<method_name>/evaluate.py --args ...
```

For example, to run the baseline:

```powershell
$env:PYTHONPATH = "methods/baseline"
python methods/baseline/train.py `
    --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m `
    --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5
```

> Do NOT `cd` into a method directory before running. The hardcoded paths inside the scripts (`dataset/ML1M/...`, `dataset/fb`, `output/...`) all resolve relative to the cwd.

## Why Not a Single Tree With Branches?

Method-as-directory is preferred over git branches for two reasons:

1. **Side-by-side diffs.** We can `Compare-Object` (or `git diff --no-index`) any pair of methods to see exactly what changed without juggling branches.
2. **Mixing methods is trivial.** When we combine H1 + H2 + H3 into `k2_ragrec/`, we just copy + cherry-pick file fragments rather than resolving cross-branch merges.

## Reading Order

Before touching code:

1. `paper/CODE_ANALYSIS.md` — line-level audit of the baseline (paper-vs-code gaps, code smells)
2. `paper/RELATED_WORKS.md` — 46-paper survey across 8 categories
3. `paper/INNOVATIONS.md` — prioritized backlog with sprint plan
