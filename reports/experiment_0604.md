# Experiment Report: 0604 Run — K-RagRec (ML1M)

## Overview

Full training and evaluation of the GraphLLM baseline on the ML1M dataset using the updated knowledge graph (`dataset/fb/graphs/0.pt`).

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | GraphLLM (LLaMA-7B frozen + Graph Transformer) |
| Dataset | ML1M (9000 train / 1000 test) |
| GNN | Graph Transformer, 4 layers, hidden_dim=1024 |
| Epochs | 10 |
| Batch size | 2 |
| Learning rate | 1e-5 |
| Warmup epochs | 1 |
| adaptive_ratio | 5 |
| sub_graph_numbers | 3 |
| reranking_numbers | 5 |
| GPU | NVIDIA RTX 3090 (24GB), single card (CUDA_VISIBLE_DEVICES=2) |

## Training

### Loss per Epoch

| Epoch | First Loss | Last Loss | Avg Loss | Steps |
|-------|-----------|-----------|----------|-------|
| 0 | 7.934 | 7.836 | 7.966 | 4500 |
| 1 | 7.934 | 1.201 | 1.160 | 4500 |
| 2 | 2.539 | 1.004 | 0.928 | 4500 |
| 3 | 2.243 | 0.960 | 0.900 | 4500 |
| 4 | 2.137 | 0.985 | 0.876 | 4500 |
| 5 | 2.163 | 0.961 | 0.858 | 4500 |
| 6 | 1.795 | 0.954 | 0.841 | 4500 |
| 7 | 1.509 | 0.801 | 0.828 | 4500 |
| 8 | 1.443 | 0.641 | 0.815 | 4500 |
| 9 | 1.438 | 0.520 | 0.804 | 4500 |

- **Total steps**: 45,000
- **Total training time**: ~8.75 hours (31,487 seconds)
- **Overall avg loss**: 1.597 → **min loss**: 7.18e-5

Loss dropped sharply from epoch 0 (~8.0) to epoch 1 (~1.2) and continued converging to ~0.5–0.8 by epoch 9, indicating stable learning.

## Evaluation Results

Evaluated on the 1000 held-out test samples (indices 9000–10000).

| Metric | Score |
|--------|-------|
| **Recall@1** | **5.4%** |
| **Recall@3** | **14.8%** |
| **Recall@5** | **25.5%** |
| **Recall@10** | **51.9%** |
| Accuracy (top-1) | 5.4% |

## Comparison with Previous Run (0601)

The 0601 run used the old `0.pt` graph data, crashed mid-training (GPU instability), and only completed 7/10 epochs in ~44 hours.

| Metric | 0601 (old 0.pt, epoch 6) | 0604 (new 0.pt, epoch 9) | Delta |
|--------|--------------------------|--------------------------|-------|
| Recall@1 | 4.5% | **5.4%** | +0.9% |
| Recall@3 | 15.5% | **14.8%** | -0.7% |
| Recall@5 | 25.2% | **25.5%** | +0.3% |
| Recall@10 | 50.2% | **51.9%** | +1.7% |
| Avg Loss | 2.51 | **1.60** | -0.91 |
| Training Time | ~44h (7 epochs) | **~8.75h (10 epochs)** | -83% |

The new 0.pt graph data improves Recall@1 and Recall@10. Training was also significantly faster due to engineering fixes.

## Engineering Notes

### Issues Resolved

1. **GPU utilization fluctuation**: Graph retrieval (SBERT + NetworkX) was called inline during training, causing CPU-GPU sync stalls. Fixed by pre-caching all 9000 training graphs before the training loop.

2. **OOM on training step 1**: After step 0, Adam optimizer allocates momentum tensors (~160 MB). Combined with LLM (~14 GiB) + forward pass activations, this exceeded 24 GiB on step 1. Fixed by:
   - Reducing `batch_size` from 5 → 2 (reduces peak activation memory ~60%)
   - `optimizer.zero_grad(set_to_none=True)` (frees gradient tensors rather than zeroing)
   - `torch.cuda.empty_cache()` after each optimizer step

3. **Pre-computation disk cache**: With 9000 samples at ~2.4s each, pre-computation takes ~6 hours. Added disk caching to `dataset/ML1M/cached_graphs_train_r5_sg3_rr5.pt` (11 GB). Subsequent runs (or crash restarts) load from cache instantly.

4. **CUDA device mismatch**: Tokenizer returned CPU tensors passed directly to `word_embedding` on CUDA. Fixed by `.to(_embed_device)` on both BOS and PAD token tensors in `graph_llm.py`.
