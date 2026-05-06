# Server Setup & Run Checklist (baseline)

> Local dev box (Windows + 8 GB GPU) is too small for LLaMA-2-7B. Real training
> happens on a Linux server. This file is the end-to-end runbook for that box.
> Commands assume **bash on Linux**.

---

## 0. Hardware checklist

| Item | Min | Comfortable | Why |
|---|---|---|---|
| GPU | 1× 24 GB (e.g., A10, RTX 3090, A5000) | 2× 48 GB (A6000, paper config) | LLaMA-2-7B fp16 ≈ 14 GB + GNN + projector + activations |
| CUDA driver | 11.8+ | 12.x | torch 2.4.1 default ships cu118 |
| RAM | 32 GB | 64 GB | model load peak ≈ 28 GB before sharding |
| Disk | 60 GB | 100 GB | data 1 GB + KG 1 GB + LLaMA 13 GB + ckpts |
| Network | reachable to HuggingFace | — | for model download |

If the server has multiple GPUs the code uses them automatically via
`device_map="auto"` (`graph_llm.py:30-44`).

---

## 1. Pull code + data + LLM weights

```bash
# Code
git clone <your-repo-url> K-ragrec
cd K-ragrec

# Datasets — ML1M + processed Freebase KG
#   download from: https://drive.google.com/file/d/1MlEPkRj47WrdXECUiz5D6Ie1oMv4hKC9/view
#   place under dataset/ so the tree looks like:
#     dataset/ML1M/{10000_data_id_20.json, ratings_45.txt, movies_id_name.txt, ml1m_raw/}
#     dataset/fb/graphs/{0.pt, layer2_embeddings_W.pt, layer3_embeddings_W.pt}
#     dataset/fb/{filtered_full_fb.txt, fb_entity_names.tsv, ...}

# Verify after extraction
ls dataset/ML1M/10000_data_id_20.json dataset/fb/graphs/0.pt

# LLaMA-2-7B (gated; needs HF account with Llama-2 access approval)
pip install -U "huggingface_hub[cli]"
huggingface-cli login          # paste token
huggingface-cli download meta-llama/Llama-2-7b-hf \
    --local-dir ~/hf_cache/Llama-2-7b-hf \
    --local-dir-use-symlinks False
# remember this path; we'll pass it via --llm_model_path below
```

---

## 2. Python + dependencies

Two equivalent paths — pick one. The repo ships `pyproject.toml` + `uv.lock` so `uv` is the most reproducible.

### A. uv (recommended, mirrors uv.lock exactly)

```bash
# Install uv (one-liner)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env  # or restart shell

# Pin Python 3.9 in the project venv
uv python install 3.9
uv venv --python 3.9
source .venv/bin/activate

# Install everything from pyproject + uv.lock
uv sync
```

### B. Plain pip + venv

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Torch 2.4.1 + CUDA 11.8 (matches paper's environment)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# PyG stack — these wheels must match torch 2.4.1 + cu118
pip install torch-geometric==2.6.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu118.html

# Rest of the stack
pip install transformers==4.45.2 peft==0.12.0 accelerate
pip install networkx==2.8.7 pandas scikit-learn sentencepiece gensim wandb
pip install pcst-fast
```

> If `torch-scatter` build fails: ensure CUDA toolkit matches torch's bundled
> CUDA, or fall back to the prebuilt wheel index above.

---

## 3. Verify the install

```bash
export PYTHONPATH=methods/baseline

python - <<'PY'
import importlib
for m in ["torch","torch_geometric","torch_scatter","transformers","peft",
         "networkx","pandas","sklearn","pcst_fast","gensim","sentencepiece"]:
    mod = importlib.import_module(m)
    print(f"OK  {m:25s} {getattr(mod, '__version__', '?')}")

import torch
print("CUDA available:", torch.cuda.is_available(),
      "device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory/1024**3:.1f} GiB)")
PY
```

Expected: every module prints `OK`; `CUDA available: True`; at least one GPU listed.

---

## 4. Retrieval-only smoke test (no LLM)

Validates SBERT + KG layers + PCST without burning the GPU on a 7B load.

```bash
PYTHONPATH=methods/baseline python tools/smoke_retrieval.py
```

Expected last line: `[OK] Retrieval pipeline smoke test passed.`

If this fails, the issue is data/SBERT/PyG wiring — fix before moving on.

---

## 5. Train baseline

```bash
export PYTHONPATH=methods/baseline
export CUDA_VISIBLE_DEVICES=0,1   # adjust for available GPUs

# Run inside a tmux session so SSH disconnect doesn't kill it
tmux new -s krag-train

python methods/baseline/train.py \
    --model_name graph_llm \
    --llm_model_name 7b \
    --llm_model_path "$HOME/hf_cache/Llama-2-7b-hf" \
    --llm_frozen True \
    --dataset ml1m \
    --batch_size 5 \
    --gnn_model_name gt \
    --gnn_num_layers 4 \
    --sub_graph_numbers 3 \
    --reranking_numbers 5 \
    --adaptive_ratio 5 \
    2>&1 | tee output/train_baseline.log

# detach: Ctrl+B then D; reattach: tmux attach -t krag-train
```

**What gets written:**
- Checkpoint: `output/ml1m/model_name_graph_llm_..._checkpoint_best.pth` (overwritten every epoch — `train.py:90`).
- No automatic validation split. Best == latest.

**Expected runtime:**
- ~2 s per sample for retrieval (uncached) + LLM forward
- 9 000 train samples × 3 epochs ≈ 15-20 hours on 2× A6000

**Common gotchas:**
- `cwd` must be repo root, not `methods/baseline/` — `train.py:31` reads `dataset/ML1M/...` relative to cwd.
- If you see `OOM` immediately on load: try `CUDA_VISIBLE_DEVICES=0` (single GPU) and let `device_map="auto"` shard across CPU/GPU; or shrink `--batch_size 2`.
- `train.py:43` maps letter A-T to indices via a 19-deep ternary chain — works but slow; ignore for now.

---

## 6. Evaluate against paper Table 1

```bash
export PYTHONPATH=methods/baseline
python methods/baseline/evaluate.py \
    --model_name graph_llm \
    --llm_model_name 7b \
    --llm_model_path "$HOME/hf_cache/Llama-2-7b-hf" \
    --llm_frozen True \
    --dataset ml1m \
    --batch_size 5 \
    --gnn_model_name gt \
    --gnn_num_layers 4 \
    --sub_graph_numbers 3 \
    --reranking_numbers 5 \
    --adaptive_ratio 5 \
    2>&1 | tee output/eval_baseline.log
```

**Target (paper Table 1, ML1M, LLaMA-2 Frozen LLM w/ PT):**
- ACC ≈ 0.435
- Recall@3 ≈ 0.725
- Recall@5 ≈ 0.831

`evaluate.py:107` only logs `Recall@1/3/5/10`. **Treat Recall@1 as ACC proxy** (top-1 hit rate). If you want literal ACC, parse the printed `pred` list and compare to the gold A-T letter — see `paper/CODE_ANALYSIS.md` §5 row 8 for the gap.

---

## 7. Known deviations (don't "fix" these for the baseline run)

| Where | Issue | Action |
|---|---|---|
| `evaluate.py:64` | `whether_retrieval(adaptive_ratio*sequence_id, 5)` — `int*list` is list repetition; differs from `train.py:62` | Leave as-is; matches paper numbers |
| `graph_llm.py:168/227` | N retrieved subgraphs are `mean`-pooled into 1 soft token (paper Eq. 8 says concat) | Leave as-is for baseline; H1 will fix |
| `train.py:90` | Best ckpt = last epoch, no validation | Leave as-is; matches paper numbers |
| `train.py` | `adjust_learning_rate` defined but never called | Leave as-is; matches paper numbers |
| `retrieve.py` | `layer3_embeddings_W.pt` produced by `index_KG.py` but never loaded | Leave as-is; H3 will use it |

A full audit lives in `paper/CODE_ANALYSIS.md`.

---

## 8. After the baseline reproduces

1. Save the eval log + checkpoint with a date stamp:
   ```bash
   mv output/ml1m/model_name_graph_llm_..._checkpoint_best.pth \
      output/ml1m/baseline_$(date +%Y%m%d).pth
   cp output/eval_baseline.log output/eval_baseline_$(date +%Y%m%d).log
   ```
2. Lock the numbers into `paper/CODE_ANALYSIS.md` §6 so future innovations have a reference point.
3. Move on to H1 (`methods/h1_qformer/`) — see `paper/INNOVATIONS.md`.
