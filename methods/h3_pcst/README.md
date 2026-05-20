# H3 —— 路径感知多源连通子图检索

> 从 `methods/baseline/` 分叉。把 baseline 「每个历史 item 各自做 cosine top-k → 得到 ~10 个互不相连的 1-hop 子图」的做法（这种做法把跨 item 的链接节点完全丢掉了）换成「每条样本一个连通子图」：以全部 10 个历史 item 为锚点，再加上对 user query SBERT 余弦相似度最高的 K 个节点，最终输出一个连通的子图。这个子图自然能覆盖到电影之间「同演员 / 同制片厂 / 同类型」这类把它们串起来的中间节点。
>
> 实现说明：文件名仍叫 `pcst_retrieval.py`，是因为最初的设计是用 G-Retriever 那一套 Prize-Collecting Steiner Tree。但 `pcst_fast 1.0.10` 的 Windows wheel 经检验输出错乱（`verts` 全是 0）；`nx.steiner_tree` 在这个 14k 节点的 KG 上一次要跑 51 分钟。最终落地的算法是**多源 2-hop BFS 邻域并集**：在多个锚点 2-hop 范围内同时出现的节点排名最高，从而以 O(|T|·(V+E)) 的复杂度（每条样本 ~0.1s）恢复 Steiner tree 那种「链接节点」的语义。诊断过程留在 `tools/test_steiner.py` 和 `tools/debug_h3_pcst.py` 里。

## 相对 baseline 的文件改动

| 文件 | 改了什么 |
|---|---|
| `src/pcst_retrieval.py` | **新增** —— `multisource_neighborhood(nxg, terminals, k, max_nodes)`（快速 Steiner 替代）；`pcst_retrieve(retriever, query_text, sequence_ids, fused_x, ...)`（完整入口，包含 SBERT 相似度奖励 + hop-field 融合特征）；`precompute_movie_anchors()` 在 retriever 启动时把 ML1M-movie-id → KG-node-id 映射全表算好。 |
| `retrieve.py` | `__init__` 加载之前未使用的 `layer3_embeddings_W.pt`，把三层 KG 表征按 0.5 / 0.3 / 0.2 加权融合存进 `self.fused_x`，并在 retriever 启动时一次性预算好 `self.movie_id_to_anchor`（3883 条；一次 batch SBERT 调用）。新增 `pcst_retrieval_topk()` 方法，返回长度为 1 的列表（一个大的连通子图）。 |
| `train.py` / `evaluate.py` | 把原来的 `whether_retrieval` + `retrieval_topk` 调用替换成单次 `pcst_retrieval_topk(input, sequence_id, ...)`。 |
| `src/config.py` | 新增 `--pcst_anchor_prize`、`--pcst_topk_query_prize`、`--pcst_edge_cost`、`--pcst_max_nodes`。（其中 `_anchor_prize` 和 `_edge_cost` 是早期 PCST 时代留下的，目前未真正使用；常调的是 `--pcst_topk_query_prize` 和 `--pcst_max_nodes`。） |

> ⚠️ **必须指定不同的 `--output_dir`**。baseline 与 H1-H5 的 checkpoint 文件名不包含方法名，默认都写到 `output/ml1m/` 下，会互相覆盖。训 baseline 用 `--output_dir output_baseline`，训 H3 用 `--output_dir output_h3`，以此类推。

## 运行（PowerShell）

```powershell
$env:PYTHONPATH = "methods/h3_pcst"

python methods/h3_pcst/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "<llama-2-7b 路径>" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt --gnn_num_layers 4 `
    --pcst_anchor_prize 10.0 `
    --pcst_topk_query_prize 20 `
    --pcst_max_nodes 200 `
    --output_dir output_h3

python methods/h3_pcst/evaluate.py `
    --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m `
    --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 `
    --pcst_anchor_prize 10.0 --pcst_topk_query_prize 20 --pcst_max_nodes 200 `
    --output_dir output_h3
```

## 冒烟测试

```powershell
$env:PYTHONPATH = "methods/h3_pcst"
python tools/smoke_h3_pcst.py
```

期望最后一段：
```
subgraph nodes: 200
subgraph edges: 587
connected components: 1
[OK] H3 PCST smoke test passed.
```

retriever 加载约 14s（绝大部分用于一次性 SBERT 编码 3883 个电影标题）。每条样本的 PCST 求解约 0.09s。

## 消融网格

| 配置 | 检验什么 |
|---|---|
| `--pcst_max_nodes 50` | 紧凑子图、上下文少 |
| `--pcst_max_nodes 200` | 默认 |
| `--pcst_max_nodes 500` | 上下文更丰富，LLM prefix 更长 |
| `--pcst_topk_query_prize 0` | 只用锚点，不加 query 相似度奖励 |
| `--pcst_topk_query_prize 50` | 强 query 扩展 |
| baseline 检索（`methods/baseline/`） | 2 N 个互不相连的 top-k 子图（对照） |

## 为什么是 2-hop BFS 而不是真正的 PCST

1. **pcst_fast（G-Retriever 用的那个库）**：Windows 预编译 wheel 输出乱码（`verts: [0,0,0,...]`），用 4 节点小图就能复现。短期没时间去 debug 那个 C++ 扩展。
2. **nx.steiner_tree（KMB 近似）**：在 14669 节点的图上每条样本要 51 分钟；算法复杂度 O(|T|·SP)，SP 即「从每个 terminal 跑一次 dijkstra_predecessors」。9000 条训练样本的训练循环根本不可用。
3. **多源 2-hop BFS**：O(|T|·(V+E)) ≈ 0.1s 一条样本。「链接节点」的直觉得到保留：任何在两个锚点 2 hop 内都出现的节点，必然落在它们之间一条 ≤4 hop 的路径上；我们按「与几个锚点共享邻域」给节点排名。本 KG 直径约 6，锚点之间通常 2 hop 可达，所以这种近似已经能抓到大部分自然的 Steiner 结构。实证结果：返回的子图是 1 个连通分量。

如果以后跑到一台能正确工作的 `pcst_fast`（比如 Linux），可以替换掉 multi-source-BFS 那条路径，回到基于 prize/cost 的 PCST —— 下游的 Q-Former 和门控并不在意检索器是怎么产出子图的。
