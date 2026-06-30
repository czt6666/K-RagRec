# Run from repo root (paths like dataset/ML1M/... and dataset/fb are repo-root-relative):
#   bash methods/baseline_qwen/run_qwen_0623.sh

echo "Starting training..."
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_qwen/train.py --model_name graph_llm --llm_model_name qwen2.5_7b_chat --llm_frozen True --dataset ml1m --batch_size 2 --num_epochs 10 --gnn_model_name gt --gnn_num_layers 4 --adaptive_ratio 5 --sub_graph_numbers 3 --reranking_numbers 5 --output_dir output/result/qwen_injection
echo "Training completed."

echo "Starting evaluation..."
CUDA_VISIBLE_DEVICES=2,3 uv run python methods/baseline_qwen/evaluate.py --model_name graph_llm --llm_model_name qwen2.5_7b_chat --llm_frozen True --dataset ml1m --batch_size 2 --num_epochs 10 --gnn_model_name gt --gnn_num_layers 4 --adaptive_ratio 5 --sub_graph_numbers 3 --reranking_numbers 5 --output_dir output/result/qwen_injection
echo "Evaluation completed."
