#!/usr/bin/env bash
# Run four additional experiments on GPU 2,3
# 1. llama2 + rag (frozen)
# 2. llama2 + rag + LoRA
# 3. llama3 + rag + LoRA
# 4. qwen2 + rag + LoRA

set -e

echo "=== [1/4] LLaMA-2-7B + N-token injection (frozen) ==="
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/train.py \
  --model_name graph_llm \
  --llm_model_name 7b \
  --llm_frozen True \
  --dataset ml1m \
  --batch_size 2 \
  --num_epochs 10 \
  --gnn_model_name gt \
  --gnn_num_layers 4 \
  --adaptive_ratio 5 \
  --sub_graph_numbers 3 \
  --reranking_numbers 5 \
  --output_dir output/result/llama2_injection

echo "LLaMA-2 training done. Starting evaluation..."
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/evaluate.py \
  --model_name graph_llm \
  --llm_model_name 7b \
  --llm_frozen True \
  --dataset ml1m \
  --batch_size 2 \
  --num_epochs 10 \
  --gnn_model_name gt \
  --gnn_num_layers 4 \
  --adaptive_ratio 5 \
  --sub_graph_numbers 3 \
  --reranking_numbers 5 \
  --output_dir output/result/llama2_injection
echo "LLaMA-2 frozen done."

echo "=== [2/4] LLaMA-2-7B + N-token injection + LoRA ==="
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/train.py \
  --model_name graph_llm \
  --llm_model_name 7b \
  --llm_frozen False \
  --dataset ml1m \
  --batch_size 2 \
  --num_epochs 10 \
  --gnn_model_name gt \
  --gnn_num_layers 4 \
  --adaptive_ratio 5 \
  --sub_graph_numbers 3 \
  --reranking_numbers 5 \
  --output_dir output/result/llama2_injection_lora

echo "LLaMA-2 LoRA training done. Starting evaluation..."
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/evaluate.py \
  --model_name graph_llm \
  --llm_model_name 7b \
  --llm_frozen False \
  --dataset ml1m \
  --batch_size 2 \
  --num_epochs 10 \
  --gnn_model_name gt \
  --gnn_num_layers 4 \
  --adaptive_ratio 5 \
  --sub_graph_numbers 3 \
  --reranking_numbers 5 \
  --output_dir output/result/llama2_injection_lora
echo "LLaMA-2 LoRA done."

echo "=== [3/4] LLaMA-3.1-8B + N-token injection + LoRA ==="
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/train.py \
  --model_name graph_llm \
  --llm_model_name llama3_8b_chat \
  --llm_frozen False \
  --dataset ml1m \
  --batch_size 2 \
  --num_epochs 10 \
  --gnn_model_name gt \
  --gnn_num_layers 4 \
  --adaptive_ratio 5 \
  --sub_graph_numbers 3 \
  --reranking_numbers 5 \
  --output_dir output/result/llama3_injection_lora

echo "LLaMA-3 LoRA training done. Starting evaluation..."
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_llama/evaluate.py \
  --model_name graph_llm \
  --llm_model_name llama3_8b_chat \
  --llm_frozen False \
  --dataset ml1m \
  --batch_size 2 \
  --num_epochs 10 \
  --gnn_model_name gt \
  --gnn_num_layers 4 \
  --adaptive_ratio 5 \
  --sub_graph_numbers 3 \
  --reranking_numbers 5 \
  --output_dir output/result/llama3_injection_lora
echo "LLaMA-3 LoRA done."

echo "=== [4/4] Qwen2.5-7B + N-token injection + LoRA ==="
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_qwen/train.py \
  --model_name graph_llm \
  --llm_model_name qwen2.5_7b_chat \
  --llm_frozen False \
  --dataset ml1m \
  --batch_size 2 \
  --num_epochs 10 \
  --gnn_model_name gt \
  --gnn_num_layers 4 \
  --adaptive_ratio 5 \
  --sub_graph_numbers 3 \
  --reranking_numbers 5 \
  --output_dir output/result/qwen_injection_lora

echo "Qwen LoRA training done. Starting evaluation..."
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_qwen/evaluate.py \
  --model_name graph_llm \
  --llm_model_name qwen2.5_7b_chat \
  --llm_frozen False \
  --dataset ml1m \
  --batch_size 2 \
  --num_epochs 10 \
  --gnn_model_name gt \
  --gnn_num_layers 4 \
  --adaptive_ratio 5 \
  --sub_graph_numbers 3 \
  --reranking_numbers 5 \
  --output_dir output/result/qwen_injection_lora
echo "Qwen LoRA done."

echo "=== All four additional experiments complete ==="
