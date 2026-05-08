# H1 —— Graph-Q-Former 多 token 投影器

> 从 `methods/baseline/` 分叉。把 baseline 那种 `MLP -> mean(dim=0)` 的「图到 LLM」桥接（它把 N 个检索到的子图压成一个 soft prompt token）替换为一个小型 cross-attention Q-Former，无论 N 多大都输出固定 q 个 query token。动机和上下文见 `paper/INNOVATIONS.md` 的 H1 节，以及 `paper/CODE_ANALYSIS.md` 第 5 节第 1 行（baseline 的 mean 池化 bug）。

## 相对 baseline 的文件改动

| 文件 | 改了什么 |
|---|---|
| `src/model/qformer.py` | **新增** —— `GraphQFormer(gnn_dim, llm_dim, num_query_tokens, num_heads)`。`q` 个可学习 query token 通过 cross-attention 在 `(N, gnn_dim)` 投影后的子图序列上聚合，外接 LayerNorm + FFN。输出 `(q, llm_dim)`。 |
| `src/model/graph_llm.py` | 去掉 `self.projector`。新增 `self.qformer = GraphQFormer(...)`。`forward()` 和 `inference()` 中，把对投影 embedding 的 `mean(dim=0, keepdim=True)` 换成直接 `self.qformer(g_embeds)`，让全部 `q` 个 token 都进 LLM 的 prefix。 |
| `src/config.py` | 新增 `--num_query_tokens`（默认 8）、`--qformer_num_heads`（默认 8）、`--qformer_dropout`（默认 0.0）、`--llm_hidden_dim`（默认 4096，Qwen2-7B 用 3584）。 |

检索 pipeline（`retrieve.py`）和 GNN 编码器（`src/model/gnn.py`）保持不变。

## 运行（PowerShell）

```powershell
$env:PYTHONPATH = "methods/h1_qformer"

# 训练（LLaMA-2-7B，q=8）
python methods/h1_qformer/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "<llama-2-7b 路径>" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 `
    --num_query_tokens 8 --qformer_num_heads 8 `
    --llm_hidden_dim 4096

# Qwen2-7B 用 --llm_hidden_dim 3584

# 评测
python methods/h1_qformer/evaluate.py `
    --model_name graph_llm `
    --llm_model_name 7b --llm_frozen True --dataset ml1m `
    --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 `
    --num_query_tokens 8 --qformer_num_heads 8 --llm_hidden_dim 4096
```

## 冒烟测试

```powershell
$env:PYTHONPATH = "methods/h1_qformer"
python tools/smoke_h1_qformer.py
```

期望输出：`GraphQFormer params: 144,787,456 (144.79 M)`，4 种 N 值都返回 `(8, 4096)`，最后 `[OK] H1 Q-Former smoke test passed.`

## 建议的消融

| 配置 | 检验什么 |
|---|---|
| baseline (mean) | 论文表 1 的参考数字 |
| `--num_query_tokens 1` | 应该接近 baseline（退化的 Q-Former） |
| `--num_query_tokens 4` | 中等容量 |
| `--num_query_tokens 8` | 默认 —— 论文主结果 |
| `--num_query_tokens 16` | 更大容量；注意 prompt 长度 |
| concat 全部 N（不用 Q-Former） | 「直接拼接」对照基线；需要小改一版代码 |

## 参数量

Q-Former 大约 145M 可训参数（cross-attention `embed_dim=4096, heads=8`，FFN 2× 扩展）。约为 LLaMA-2-7B 的 2%。如果想瘦身，可调 `qformer_num_heads` 和 `qformer.py` 里的 FFN 扩展系数（当前 ×2）。
