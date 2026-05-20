# H5 —— 关系感知 GNN 编码器

> 从 `methods/baseline/` 分叉。新增两个关系感知 GNN 编码器（RGCN、CompGCN），让编码器能用上 Freebase KG 中的 ~177 种关系类型，而不是把 `edge_attr` 当成无差别的 1024 维向量。动机详见 `paper/INNOVATIONS.md` 的 H5 节。

## 相对 baseline 的文件改动

| 文件 | 改了什么 |
|---|---|
| `src/model/gnn.py` | 新增 `RGCN`（用 PyG 的 `RGCNConv` 作用在离散 `edge_type` id 上）和 `CompGCN`（自定义 `MessagePassing`，乘性组合算子 `phi(h,r)=h*r`）。已注册到 `load_gnn_model` 字典，键为 `rgcn` 和 `compgcn`。 |
| `retrieve.py` | `__init__` 中通过 `torch.unique(G.edge_attr, dim=0, return_inverse=True)` 派生出全图 `edge_type` 张量（177 个 unique 关系嵌入 → 177 个 id）。`get_first_order_subgraph` 里把 `edge_type` 切片附加到每个 `Data` 上。 |
| `src/model/graph_llm.py` | `encode_graphs` 检查 GNN 的 forward 签名；如果它接受 `edge_type`，就把 `graphs.edge_type` 透传过去（`Batch.from_data_list` 会自动 batch 这个字段）。其它 GNN 不受影响。 |

> ⚠️ **必须指定不同的 `--output_dir`**。baseline 与 H1-H5 的 checkpoint 文件名不包含方法名，默认都写到 `output/ml1m/` 下，会互相覆盖。训 baseline 用 `--output_dir output_baseline`，训 H5 用 `--output_dir output_h5`，以此类推。

## 运行（PowerShell）

```powershell
$env:PYTHONPATH = "methods/h5_rel_gnn"

# RGCN 编码器
python methods/h5_rel_gnn/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "<llama-2-7b 路径>" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name rgcn `
    --gnn_num_layers 4 `
    --sub_graph_numbers 3 `
    --reranking_numbers 5 `
    --adaptive_ratio 5 `
    --output_dir output_h5

# CompGCN 编码器
python methods/h5_rel_gnn/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "<llama-2-7b 路径>" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name compgcn `
    --gnn_num_layers 4 `
    --sub_graph_numbers 3 `
    --reranking_numbers 5 `
    --adaptive_ratio 5 `
    --output_dir output_h5

# 评测（--gnn_model_name 与训练时一致）
python methods/h5_rel_gnn/evaluate.py `
    --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m `
    --batch_size 5 --gnn_model_name rgcn --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 `
    --output_dir output_h5
```

## 冒烟测试

```powershell
$env:PYTHONPATH = "methods/h5_rel_gnn"
python tools/smoke_h5_gnn.py     # 在合成 batch 上测全部 GNN 的 forward
python tools/smoke_retrieval.py  # 验证带 edge_type 的检索仍然正常
```

期望分别打印 `[OK] H5 GNN smoke test passed.` 和检索冒烟通过。

## 论文要报告的消融矩阵

| `--gnn_model_name` | 来源 |
|---|---|
| `gcn` | baseline |
| `gat` | baseline |
| `gt` | baseline（论文默认） |
| `graphsage` | baseline |
| `rgcn` | **H5 新增** |
| `compgcn` | **H5 新增** |

在 ML1M / ML20M / Book × LLaMA-2-7B 上对比。预期 `rgcn` 和 `compgcn` 比 baseline 那 4 个 GNN 高 1–3% ACC，因为 Freebase 上 ~177 种关系类型的语义在 baseline 那种「edge 视为 1024 维向量」的写法里几乎损失殆尽。
