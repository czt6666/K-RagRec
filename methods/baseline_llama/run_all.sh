#!/usr/bin/env bash
# Run from repo root: bash methods/baseline_llama/run_all.sh
# Sequential on GPU 2,3:
#   1. Qwen2.5-7B no-RAG  (skipped if results already exist)
#   2. Mistral-7B-Instruct no-RAG
#   3. LLaMA-3.1-8B-Instruct no-RAG
#   4. LLaMA-3.1-8B-Instruct + N-token injection (train + eval)

set -e

echo "=== [1/5] Qwen2.5-7B no-RAG ==="
if [ -f "output/result/qwen_norag/ml1m/norag_results.json" ]; then
  echo "Already done, skipping."
else
  CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/test_no_rag.py \
    --llm_model_name qwen2.5_7b_chat \
    --dataset ml1m \
    --output_dir output/result/qwen_norag
fi
echo "Qwen no-RAG done."

echo "=== [2/5] Mistral-7B-Instruct-v0.3 no-RAG ==="
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/test_no_rag.py \
  --llm_model_name mistral_7b_chat \
  --dataset ml1m \
  --output_dir output/result/mistral_norag
echo "Mistral no-RAG done."

echo "=== [3/5] LLaMA-3.1-8B-Instruct no-RAG ==="
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/test_no_rag.py \
  --llm_model_name llama3_8b_chat \
  --dataset ml1m \
  --output_dir output/result/llama3_norag
echo "LLaMA-3.1 no-RAG done."

echo "=== [4/5] LLaMA-3.1-8B-Instruct + N-token injection: training ==="
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
  --output_dir output/result/llama3_injection
echo "LLaMA-3.1 training done."

echo "=== [5/5] LLaMA-3.1-8B-Instruct + N-token injection: evaluation ==="
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
  --output_dir output/result/llama3_injection
echo "LLaMA-3.1 evaluation done."

echo "=== All experiments complete ==="
