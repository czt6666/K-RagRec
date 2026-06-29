# Baseline

> ⚠️ **必须指定不同的 `--output_dir`**。baseline 与 H1-H5 的 checkpoint 文件名不包含方法名，默认都写到 `output/ml1m/` 下，会互相覆盖。训 baseline 用 `--output_dir output_baseline`，训 H1 用 `--output_dir output_h1`，以此类推。

## 运行（Linux / Bash）

```bash
export PYTHONPATH="methods/baseline"

# 训练
uv run python methods/baseline/train.py \
    --model_name graph_llm \
    --llm_model_name 7b \
    --llm_model_path "/data/hf_cache/hub/models--meta-llama--Llama-2-7b-hf" \
    --llm_frozen True \
    --dataset ml1m \
    --batch_size 5 \
    --gnn_model_name gt --gnn_num_layers 4 \
    --adaptive_ratio 5 --sub_graph_numbers 3 --reranking_numbers 5 \
    --output_dir output_baseline

# 评测
uv run python methods/baseline/evaluate.py \
    --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m \
    --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 \
    --adaptive_ratio 5 --sub_graph_numbers 3 --reranking_numbers 5 \
    --output_dir output_baseline
```
