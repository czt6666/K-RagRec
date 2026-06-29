#!/usr/bin/env bash
# Run from repo root: bash methods/baseline_llama/run_all.sh
# Sequential: (1) Qwen no-RAG, (2) LLaMA-3.1-8B no-RAG, (3) LLaMA-3.1-8B + N-token injection train+eval
# All on GPU 2,3 only.

set -e

echo "=== [1/4] Qwen2.5-7B no-RAG ==="
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/test_no_rag.py \
  --llm_model_name qwen2.5_7b_chat \
  --dataset ml1m \
  --output_dir output_qwen_norag
echo "Qwen no-RAG done."

echo "=== [2/4] LLaMA-3.1-8B-Instruct no-RAG ==="
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/test_no_rag.py \
  --llm_model_name llama3_8b_chat \
  --dataset ml1m \
  --output_dir output_llama3_norag
echo "LLaMA-3.1 no-RAG done."

echo "=== [3/4] LLaMA-3.1-8B-Instruct + N-token injection: training ==="
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/train.py \
  --model_name graph_llm \
  --llm_model_name llama3_8b_chat \
  --llm_frozen True \
  --dataset ml1m \
  --batch_size 2 \
  --num_epochs 10 \
  --gnn_model_name gt \
  --gnn_num_layers 4 \
  --adaptive_ratio 5 \
  --sub_graph_numbers 3 \
  --reranking_numbers 5 \
  --output_dir output_llama3_injection
echo "LLaMA-3.1 training done."

echo "=== [4/4] LLaMA-3.1-8B-Instruct + N-token injection: evaluation ==="
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/evaluate.py \
  --model_name graph_llm \
  --llm_model_name llama3_8b_chat \
  --llm_frozen True \
  --dataset ml1m \
  --batch_size 2 \
  --num_epochs 10 \
  --gnn_model_name gt \
  --gnn_num_layers 4 \
  --adaptive_ratio 5 \
  --sub_graph_numbers 3 \
  --reranking_numbers 5 \
  --output_dir output_llama3_injection
echo "LLaMA-3.1 evaluation done."

echo "=== All experiments complete ==="
