# H2 —— 可学习检索门控

> 从 `methods/baseline/` 分叉。在 baseline 的「按流行度阈值筛 item」策略之外，再叠一层 MLP 门控：它对每个被检索到的子图，结合用户历史 query 嵌入，输出一个软（sigmoid）或硬（Gumbel-Sigmoid 直通梯度）的权重，在池化前就对子图贡献做重加权。流行度筛选仍然控制**哪些 item 进检索**（保留检索时的速度优势），门控控制**每个检索结果在 LLM 前向中权重多大**。

## 相对 baseline 的文件改动

| 文件 | 改了什么 |
|---|---|
| `src/model/gate.py` | **新增** —— `RetrievalGate(emb_dim, hidden)`。把 `[g, q, |g-q|, g*q]`（1024×4 → 256 → 1）拼起来过 MLP，输出 sigmoid 权重。还提供静态方法 `gumbel_sigmoid(logits, tau, hard)` 用于直通的二值选择。 |
| `src/model/graph_llm.py` | 新增 `self.gate`。`encode_graphs` 在 GNN scatter-mean 之后，把每个子图嵌入乘以门控权重。训练时可启用 Gumbel-Sigmoid。新增 L1 风格的 `_last_gate_sparsity_loss`，由 `forward()` 与 CE 一起返回，鼓励门控变得稀疏。 |
| `train.py` / `evaluate.py` | 每条样本都对 watching-history 字符串做一次 `retrieval_model.encode_query(input)`（SBERT 嵌入），存进 `sample['query_emb']`。 |
| `src/config.py` | 新增 `--gate_use_gumbel`、`--gate_tau`、`--gate_sparsity_lambda`。 |

> ⚠️ **必须指定不同的 `--output_dir`**。baseline 与 H1-H5 的 checkpoint 文件名不包含方法名，默认都写到 `output/ml1m/` 下，会互相覆盖。训 baseline 用 `--output_dir output_baseline`，训 H2 用 `--output_dir output_h2`，以此类推。

## 运行（PowerShell）

```powershell
$env:PYTHONPATH = "methods/h2_gate"

# 默认：软 sigmoid 加权，无稀疏惩罚
python methods/h2_gate/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "<llama-2-7b 路径>" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 `
    --output_dir output_h2

# 硬二值选择 + 稀疏惩罚
python methods/h2_gate/train.py `
    --model_name graph_llm --llm_model_name 7b `
    --llm_model_path "<llama-2-7b 路径>" `
    --llm_frozen True --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 `
    --gate_use_gumbel --gate_tau 1.0 --gate_sparsity_lambda 0.01 `
    --output_dir output_h2

# 评测
python methods/h2_gate/evaluate.py `
    --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m `
    --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 `
    --output_dir output_h2
```

## 冒烟测试

```powershell
$env:PYTHONPATH = "methods/h2_gate"
python tools/smoke_h2_gate.py
```

期望：`RetrievalGate params: 1,049,089`，每个 N 都打出权重表，3 个 Gumbel 二值掩码，`Gradient flow OK: 4 of 4 parameters ...`，最后 `[OK] H2 gate smoke test passed.`

## 消融网格

| 配置 | 检验什么 |
|---|---|
| baseline（无门控） | 参考数字 |
| `--gate_sparsity_lambda 0`（软 sigmoid） | 仅连续重加权 |
| `--gate_use_gumbel --gate_sparsity_lambda 0.0` | 二值选择，不施稀疏压力 |
| `--gate_use_gumbel --gate_sparsity_lambda 0.01` | 二值 + 稀疏（论文主结果） |
| `--gate_use_gumbel --gate_sparsity_lambda 0.05` | 强稀疏 |

## 训练时需要注意的

- 门控**与模型其它部分联合训练**。Loss = CE + `gate_sparsity_lambda * mean(weights)`。不加稀疏的话门控大概率收敛到 ≈0.5，没什么用。
- `--gate_tau` 取 0.5–1.5 一般可用；越低越接近硬二值，但优化更难。
- `whether_retrieval` 里那套基于流行度的预筛选没动，所以这版仍保留 baseline 的检索速度优势。门控只对**已检索到**的子图做重加权。
