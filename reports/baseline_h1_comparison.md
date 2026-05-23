# Baseline vs H1 (Graph-Q-Former) 实验对比报告

## 1. 实验设置

| 配置项 | 值 |
|---|---|
| 数据集 | ML1M（训练 9,000 / 测试 1,000） |
| LLM | LLaMA-2-7B (frozen) |
| GNN | GraphTransformer (4 layers, 1024 dim) |
| Batch size | 5 |
| Epochs | 3 |
| Learning rate | 1e-5 |
| Retrieval | SBERT + adaptive ratio 5 + 3 subgraphs + reranking 5 |
| 评估指标 | Recall@1/3/5/10, Accuracy |

> **注意**：baseline 与 H1-H5 的 checkpoint 文件名默认相同，必须使用不同的 `--output_dir`，否则会互相覆盖。详见各方法 README.md。

---

## 2. Baseline 结果

### 2.1 训练概况

- **训练耗时**：约 18.8 小时（4 卡 3090）
- **总步数**：5,400 步（9000 / 5 * 3 epochs）
- **Loss 变化**：8.12 → 1.52（平均），最低 1.30
- **Checkpoint**：`output_baseline/ml1m/...checkpoint_best.pth`（370 MB，仅可训练参数）

```json
{
  "method": "baseline",
  "dataset": "ml1m",
  "num_epochs": 3,
  "batch_size": 5,
  "total_steps": 5400,
  "avg_loss": 1.532,
  "min_loss": 1.298,
  "max_loss": 8.154,
  "elapsed_seconds": 67706.4
}
```

### 2.2 评估结果

| 指标 | 值 |
|---|---|
| **Recall@1** | **4.9%** |
| Recall@3 | 14.2% |
| Recall@5 | 23.7% |
| Recall@10 | 49.9% |

> Recall@1 等价于 Top-1 Accuracy（检查正确答案是否在模型排名第一的预测中）。

---

## 3. 与论文指标的对比

### 3.1 论文报告（表 9，ML1M + LLaMA-2-7B）

| 模型 | 论文 Accuracy |
|---|---|
| GCN | 39.7% |
| GAT | 42.0% |
| **GT (GraphTransformer)** | **42.9%** |
| GraphSAGE | 41.8% |

### 3.2 差距分析

| 对比项 | 论文 | 我们的实现 | 差距 |
|---|---|---|---|
| GT Accuracy | **42.9%** | **4.9%** | **-38.0 pp** |

**差距巨大的可能原因**：

1. **训练/评估检索不一致**：
   - `train.py`：`whether_retrieval(sequence_id, adaptive_ratio*len(sequence_id))`
   - `evaluate.py`：`whether_retrieval(adaptive_ratio*sequence_id, 5)`
   - 两者检索逻辑不同，导致分布偏移。

2. **Prompt 微小差异**：
   - `train.py` 使用 `### Instruction`
   - `evaluate.py` 使用 `## Instruction`
   - 虽然对 LLM 影响有限，但训练-评估不一致总不是好事。

3. **学习率与优化**：
   - 代码中定义了 `adjust_learning_rate`（cosine warmup），但 `train.py` 实际**没有调用**。
   - 恒定的 1e-5 对 GNN + 投影层可能偏低。

4. **epoch 末直接覆盖 best**：
   - `train.py` 每轮结束直接覆盖 `checkpoint_best.pth`，没有验证集挑选最佳模型。

5. **LLM 路径被硬编码覆盖**：
   - `train.py` 中 `args.llm_model_path = llama_model_path[args.llm_model_name]` 会覆盖用户传入的路径。

6. **模型可能未真正收敛**：
   - Loss 从 8.1 降到 1.5，但 1.5 对 20 类分类来说仅比随机（~3.0）好一倍。
   - 模型可能学到了部分模式，但远未达到论文水平。

---

## 4. H1 —— Graph-Q-Former

### 4.1 设计思路

H1 用 **Graph-Q-Former** 替换了 baseline 的 `MLP-Projector + Mean-Pool`：

- **Baseline**：N 个子图 → GNN → mean pooling → 投影 → **1 个 soft prompt token**
- **H1**：N 个子图 → GNN → Q-Former (cross-attention) → **q 个 learnable query tokens**（默认 q=8）

Q-Former 包含约 **145M 可训练参数**（约为 LLaMA-2-7B 的 2%），通过 cross-attention 让固定数量的 query token 自适应地聚合子图信息，避免 mean pooling 的信息损失。

### 4.2 文件改动

| 文件 | 改动 |
|---|---|
| `src/model/qformer.py` | 新增 `GraphQFormer` |
| `src/model/graph_llm.py` | 去掉 projector，接入 Q-Former |
| `src/config.py` | 新增 `--num_query_tokens`, `--qformer_num_heads`, `--llm_hidden_dim` |

---

## 5. H1 运行命令

### 训练

```bash
uv run python methods/h1_qformer/train.py \
    --model_name graph_llm \
    --llm_model_name 7b \
    --llm_model_path "/data/hf_cache/hub/models--meta-llama--Llama-2-7b-hf" \
    --llm_frozen True \
    --dataset ml1m \
    --batch_size 5 \
    --gnn_model_name gt --gnn_num_layers 4 \
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 \
    --num_query_tokens 8 --qformer_num_heads 8 \
    --llm_hidden_dim 4096 \
    --output_dir output_h1
```

### 评估

```bash
uv run python methods/h1_qformer/evaluate.py \
    --model_name graph_llm \
    --llm_model_name 7b \
    --llm_frozen True \
    --dataset ml1m \
    --batch_size 5 \
    --gnn_model_name gt --gnn_num_layers 4 \
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 \
    --num_query_tokens 8 --qformer_num_heads 8 \
    --llm_hidden_dim 4096 \
    --output_dir output_h1
```

> 预计训练时间：~18 小时（4 卡 3090，batch_size=5）

---

## 6. 预期对比维度

跑完 H1 后，建议从以下几个维度对比：

| 维度 | Baseline | H1 |
|---|---|---|
| **Recall@1 / Accuracy** | 4.9% | ? |
| Recall@3 | 14.2% | ? |
| Recall@5 | 23.7% | ? |
| Recall@10 | 49.9% | ? |
| **可训练参数量** | ~370 MB (GNN+Projector) | ~370 MB + 145 MB (Q-Former) |
| 训练时间 | ~18.8 h | ? |

---

## 7. 附录：结果文件位置

| 方法 | 训练日志 | 训练摘要 | 评估结果 |
|---|---|---|---|
| Baseline | `output_baseline/ml1m/baseline_train_log.json` | `output_baseline/ml1m/baseline_train_summary.json` | `output_baseline/ml1m/baseline_results.json` |
| H1 | `output_h1/ml1m/h1_qformer_train_log.json` | `output_h1/ml1m/h1_qformer_train_summary.json` | `output_h1/ml1m/h1_qformer_results.json` |
