# K-RagRec (Reorganized for Innovation Experiments)

This repository started from the official PyTorch implementation of the ACL 2025 paper *"Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation"* and is being reorganized to host follow-up methods that build on top of K-RagRec.

The original code lives untouched in `methods/baseline/` and serves as a reproducible reference. New methods are added as sibling directories under `methods/` (one directory per method, copied from baseline and modified in place). See `methods/README.md` for the per-method index.

## Repository Layout

```
K-ragrec/
├── methods/
│   ├── README.md                # method index + per-method runbooks
│   └── baseline/                # original K-RagRec code (read-only reference)
│       ├── train.py
│       ├── evaluate.py
│       ├── retrieve.py
│       ├── run.sh
│       └── src/
├── dataset/                     # shared datasets (ML1M / ml-20m / book / fb)
├── paper/                       # paper PDF + analysis docs
│   ├── K-RagRec.pdf
│   ├── CODE_ANALYSIS.md         # paper-vs-code audit
│   ├── RELATED_WORKS.md         # 46-paper related-work survey
│   └── INNOVATIONS.md           # H/M/L innovation backlog
├── pyproject.toml, uv.lock
└── README.md
```

## Environment

- Python==3.9
- numpy==1.23.4
- torch==2.4.1
- cuda==11.8.89
- transformers==4.45.2
- networkx==2.8.7
- peft==0.12.0

## Datasets

Three datasets are supported: MovieLens-1M, MovieLens-20M, Amazon Book. Only ML1M is processed in-repo; raw data for the other two is provided. KG comes from Freebase, filtered per dataset.

Processed ML1M + KG: https://drive.google.com/file/d/1MlEPkRj47WrdXECUiz5D6Ie1oMv4hKC9/view

## Running the Baseline

> **Reproducing on a server?** See `tools/server_checklist.md` for the full Linux/bash runbook (uv install, dependency pinning, smoke test, tmux long-train tips, expected paper numbers).
>
> The PowerShell snippets below are for local dev / smoke checks. LLaMA-2-7B does not fit on consumer GPUs in fp16 — full training requires the server.

All commands assume the repo root (`K-ragrec/`) as the working directory. `dataset/` paths inside the code are resolved relative to cwd, so do NOT `cd` into `methods/baseline/`.

`PYTHONPATH` is set to `methods/baseline` so that `from src.model import ...` resolves correctly.

### Retrieval-only smoke test (no LLM, runs anywhere)

```powershell
$env:PYTHONPATH = "methods/baseline"
python tools/smoke_retrieval.py
```

Expected last line: `[OK] Retrieval pipeline smoke test passed.`

### Train

```powershell
$env:PYTHONPATH = "methods/baseline"
python methods/baseline/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "D:\hf_cache\hub\models--meta-llama--Llama-2-7b-hf\snapshots\01c7f73d771dfac7d292323805ebc428287df4f9" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt `
    --gnn_num_layers 4 `
    --sub_graph_numbers 3 `
    --reranking_numbers 5 `
    --adaptive_ratio 5
```

### Evaluate

```powershell
$env:PYTHONPATH = "methods/baseline"
python methods/baseline/evaluate.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "D:\hf_cache\hub\models--meta-llama--Llama-2-7b-hf\snapshots\01c7f73d771dfac7d292323805ebc428287df4f9" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt `
    --gnn_num_layers 4 `
    --sub_graph_numbers 3 `
    --reranking_numbers 5 `
    --adaptive_ratio 5
```

### Multi-GPU

```powershell
$env:CUDA_VISIBLE_DEVICES = "0,1"
$env:PYTHONPATH = "methods/baseline"
python methods/baseline/train.py ...
```

## Running Other Methods

Each new method directory follows the same pattern. Replace `methods/baseline` with `methods/<method_name>` in both `PYTHONPATH` and the script path. See `methods/README.md` for the per-method runbooks.

## Hardware

The original paper runs on 2× NVIDIA A6000-48GB. For single-GPU setups, switch to `graph_llm_for_one_GPU` (if/when present in a method's `src/model/`).

## Citation

```
@article{wang2025knowledge,
    title={Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation},
    author={Wang, Shijie and Fan, Wenqi and Feng, Yue and Ma, Xinyu and Wang, Shuaiqiang and Yin, Dawei},
    journal={arXiv preprint arXiv:2501.02226},
    year={2025}
}
```
