#!/usr/bin/env bash
# Run all four ML1M experiments on GPU 2,3 with detailed raw logging.
# Sequential execution; skips any experiment whose result JSON already exists.

set -e

echo "=== [1/4] Qwen2.5-7B no-RAG ==="
if [ -f "output/result/qwen_norag/ml1m/norag_results.json" ]; then
  echo "Already done, skipping."
else
  CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/test_no_rag.py \
    --llm_model_name qwen2.5_7b_chat \
    --dataset ml1m \
    --output_dir output/result/qwen_norag
fi
echo "Qwen no-RAG done."

echo "=== [2/4] LLaMA-3.1-8B no-RAG ==="
if [ -f "output/result/llama3_norag/ml1m/norag_results.json" ]; then
  echo "Already done, skipping."
else
  CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/test_no_rag.py \
    --llm_model_name llama3_8b_chat \
    --dataset ml1m \
    --output_dir output/result/llama3_norag
fi
echo "LLaMA-3.1 no-RAG done."

echo "=== [3/4] Qwen2.5-7B + N-token injection (train + eval) ==="
if [ -f "output/result/qwen_injection/ml1m/baseline_results.json" ]; then
  echo "Already done, skipping."
else
  bash methods/baseline_qwen/run_qwen_0623.sh
fi
echo "Qwen injection done."

echo "=== [4/4] LLaMA-3.1-8B + N-token injection (train + eval) ==="
if [ -f "output/result/llama3_injection/ml1m/baseline_results.json" ]; then
  echo "Already done, skipping."
else
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
fi
echo "LLaMA-3.1 injection done."

echo "=== All four experiments complete ==="
