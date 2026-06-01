#!/bin/bash
cd /root/workspace/python/K-RagRec
export CUDA_VISIBLE_DEVICES=0,1
uv run python methods/baseline/train.py \
    --model_name graph_llm \
    --llm_model_name 7b \
    --llm_frozen True \
    --dataset ml1m \
    --batch_size 5 \
    --gnn_model_name gt \
    --gnn_num_layers 4 \
    --adaptive_ratio 5 \
    --sub_graph_numbers 3 \
    --reranking_numbers 5 \
    --output_dir output_0601 \
    --num_epochs 10 \
    --warmup_epochs 1 \
    > /root/workspace/python/K-RagRec/output_0601_train.log 2>&1
