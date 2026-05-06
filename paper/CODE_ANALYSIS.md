# K-RagRec 代码实现深度分析报告

> 目的：精确到文件 + 行号，梳理 K-RagRec 论文的实现细节，把"论文说的"与"代码做的"逐条对照，定位差异、潜在缺陷与可改造的接口。后续创新点设计将基于本报告。

---

## 0. 项目骨架

```
K-ragrec/
├── train.py                     # 训练入口：load 数据 → 检索 → 前向 → 反传 → 存 ckpt
├── evaluate.py                  # 评测入口：load ckpt → 检索 → 推理 → Recall@k
├── retrieve.py                  # 知识图谱子图检索 + 重排
├── run.sh                       # 一键脚本（CUDA_VISIBLE_DEVICES=0,1）
├── src/
│   ├── config.py                # argparse 超参
│   ├── model/
│   │   ├── __init__.py          # 模型注册 + LLM 路径表
│   │   ├── graph_llm.py         # 主模型：GNN + Projector + frozen LLM
│   │   └── gnn.py               # GCN / GAT / GraphTransformer / SAGE
│   ├── processing_kg/
│   │   ├── index_KG.py          # KG 三元组 → embedding → 多层 GCN 嵌入
│   │   └── lm_modeling.py       # SBERT / Contriever / Word2Vec 语义编码
│   └── utils/
│       ├── ckpt.py              # 保存 / 加载断点
│       ├── lm_modeling.py       # 同 processing_kg/lm_modeling.py（重复文件）
│       ├── lr_schedule.py       # cosine warmup
│       └── seed.py
└── dataset/
    ├── ML1M / ml-20m / book     # 三个数据集
    └── fb/                      # Freebase KG（节点/边 csv + 多层 .pt 嵌入）
```

---

## 1. 数据流 & 训练 pipeline

### 1.1 入口 `train.py:26-91`

```
seed_everything → load_model[graph_llm](args)            # 实例化 GraphLLM
                → GraphRetrieval('sbert', 'dataset/fb')  # 检索器
                → 读 dataset/ML1M/10000_data_id_20.json[:9000]  # 训练集
                → 逐 batch:
                    · whether_retrieval(sequence_id, adaptive_ratio*len)   # 选哪些电影需要检索
                    · retrieval_topk(input, list, K=3, N=5)                # 取子图 + 重排
                    · model.forward(sample)
                    · loss.backward()
                每个 epoch 末 _save_checkpoint(...is_best=True)            # ⚠ 直接覆盖为 best，未做验证
```

**关键发现**:
- `train.py:31` 训练数据硬编码为 `dataset/ML1M/10000_data_id_20.json`，dataset 参数其实没起作用。
- `train.py:43` 用 20 个三元 if 把 A-T 映射到 0-19；冗长且写在 list-comp 里。
- `train.py:90` 每个 epoch 都覆写 best ckpt，没有 validation／early stopping。`patience` 参数（config:14）形同虚设。
- 没有学习率调度调用，`adjust_learning_rate` 定义了但 train.py 没用。

### 1.2 入口 `evaluate.py:22-110`

- `evaluate.py:35` 同样硬编码 `dataset/ML1M/10000_data_id_20.json[9000:10000]` 作为测试集。
- `evaluate.py:64` **bug**：`retrieval_model.whether_retrieval(args.adaptive_ratio*sequence_id, 5)` —— 这里 `sequence_id` 是 list；`int * list = list 重复`，把交互序列重复 N 次再传入；第二个参数固定为 5（不再受 adaptive_ratio 控制）。这与 `train.py:62`（`whether_retrieval(sequence_id, adaptive_ratio*len(sequence_id))`）行为不一致 → **训练/推理分布不一致**。
- `evaluate.py:38-50` 自定义 recall_at_k：对 `predicted` 截前 10，找到 actual 时按 index ≤1/3/5/10 返回 (1/0,1,1,1) 等。逻辑可读性差，但功能正确。
- `evaluate.py:73` 调用 `model.inference(sample)` 取 sorted_indices（A-T 的 logits 排序）。

---

## 2. 模型核心 `src/model/graph_llm.py`

### 2.1 GraphLLM 类结构（line 47-321）

| 组件 | 行号 | 说明 |
|---|---|---|
| Tokenizer + LLM | 62-72 | `AutoModelForCausalLM` 加载 LLaMA-2/3 / QWEN，FP16 |
| LLM 冻结 / LoRA | 74-96 | `--llm_frozen=True` 时全冻结；否则 LoRA r=8/α=16, target=q,v_proj |
| `graph_encoder` | 101-108 | GNN_Encoding，`load_gnn_model[args.gnn_model_name](...)` 默认 `gt`(GraphTransformer) |
| `projector` | 110-114 | `Linear(1024→2048) → Sigmoid → Linear(2048→4096)`, hard-code 4096 维（LLaMA 维度），qwen 注释为 3584 但代码没切换 |
| `word_embedding` | 116 | LLM 输入嵌入层引用 |

### 2.2 子图编码 `encode_graphs()` 131-142

```python
for graphs in graphs_list:                                    # 每个样本一个 list
    graphs = Batch.from_data_list(graphs).to(device)          # PyG batch
    n_embeds, _ = self.graph_encoder(x, edge_index, edge_attr)
    g_embeds = scatter(n_embeds, batch, dim=0, reduce='mean')  # 子图级 mean pool
graph_embeds_list.append(g_embeds)                             # 形状 [N_subgraphs, hidden]
```

### 2.3 ⚠ 关键差异点：forward / inference 168-169 与 226-228

```python
sample_graph_embeds = torch.cat([proj.unsqueeze(0) for proj in projected_graph_embeds_list[i]], dim=0)
sample_graph_embeds = sample_graph_embeds.mean(dim=0, keepdim=True)  # ← 全部子图再做一次 mean
inputs_embeds = torch.cat([bos_embeds, sample_graph_embeds, inputs_embeds], dim=0)
```

- 论文（公式 8）：`h_G^ = MLP([h_g1; ...; h_gN])` 是**逐子图保留 N 个 soft token**。
- 代码：把 N 个子图投影向量做了一次 **mean** → 仅 1 个 soft token 拼到 BOS 后面。
- 影响：丢掉了"每个子图独立的语义槽"，LLM 无法分辨不同 hop 不同实体的信息；与论文消融"-Encoding 掉点 37%/45.9%"对应的方法实现不一致，是潜在创新切入点。

### 2.4 inference 评分 256-305

- 用 `option_indices` 把 A-T 映射到 LLaMA-2 tokenizer 的 ID（319/350/.../323），qwen/llama3 用 32-51。
- 取 `generation.scores[0]`（生成的第一个 token 的 logits）→ 仅在 A-T 候选间 softmax → argsort 返回排序索引。
- 即评估实际上是单 token 选择题，**不是真正的生成式推荐**。

### 2.5 GNN 实现 `src/model/gnn.py`

- 4 种 conv: GCN(L17)、GraphTransformer(L35-61)、GAT(L63-89)、SAGE(L92-118)。
- 只有 `GraphTransformer` 用了 `edge_attr`(作为 edge_dim 输入)；GCN/SAGE 直接忽略边特征；GAT 形参接收但未使用。**意味着对边类型的语义信息利用极不充分，是可改造点。**
- BatchNorm + ReLU + Dropout 标配，最后一层 conv 不做 norm/激活。

---

## 3. 检索模块 `retrieve.py`

### 3.1 类初始化 `GraphRetrieval.__init__` 17-29

| 资源 | 加载来源 | 含义 |
|---|---|---|
| `self.G` | `dataset/fb/graphs/0.pt` | KG 节点 = SBERT 嵌入；边 = SBERT 嵌入；**未经过 GNN（0层）** |
| `self.G1` | `dataset/fb/graphs/layer2_embeddings_W.pt` | 经过 1 次 GCN 卷积的节点嵌入（"2-hop"） |
| `self.Graph` | networkx Graph 从 `self.G.edge_index` 构建 | 用于查邻居 |
| `sorted_item_ids` | 按 `ratings_45.txt` 中交互次数升序排序 | 流行度热度反向排（low→high） |
| `movie_id_to_name` | `movies_id_name.txt` | id ↔ 标题 |

> 注：`index_KG.py:107-117` 实际还生成了 `layer3_embeddings_W.pt`，但 retrieve.py 只加载到第二层；论文说 hop-field 应该跨 l-hop，**未在代码中真正使用 ≥3 层嵌入**。

### 3.2 `whether_retrieval` 64-69（流行度选择策略）

```python
retrieve_list = sorted(watching_list, key=lambda x: self.sorted_item_ids.index(x))  # 按热度升序
return retrieve_list[:int(k)]
```

- 论文：阈值 p=50%，对低于阈值（冷门）的 item 检索。
- 代码：取热度最低的前 `k=adaptive_ratio*len(sequence_id)` 个 item 去检索；`adaptive_ratio` 默认 5，整数从 1-10 表示 0.1-1（config:44 注释）。
- ⚠ `self.sorted_item_ids.index(x)` 是 O(N) 线性查表，对每条样本要排 10 次 → 总体 O(N·10·M) 显著偏慢；可换 dict。
- 排序使用绝对交互数，**冷启动 / 长尾物品被无差别拉到前面**；缺少 personalization 维度（不同用户对"热度"的容忍不一样）。

### 3.3 `retrieval_topk`（核心检索） 31-56

```
对每个需检索的 movie:
    item_name → SBERT q_emb
    一阶检索: retrieval_topk(self.G, q_emb, K=3)   # 用 0 层（PLM 直接嵌入）
    二阶检索: retrieval_topk(self.G1, q_emb, K=3)  # 用 1 层 GCN 嵌入
全局重排:
    global_q_emb = SBERT(input)                    # ⚠ 仅历史观影列表，不含任务描述！
    一阶 re_rank(global_q_emb, all_first_order_nodes, N=5)
    二阶 re_rank(global_q_emb, all_second_order_nodes, N=5)
对重排后的节点，调用 get_first_order_subgraph() 截取局部子图
最终返回 list[Data]，长度 = 2N（一阶 N + 二阶 N）
```

**关键观察**:
- 检索是**节点级**的 top-k（226-232 行 `retrieval_topk(graph, q_emb, ...)` 直接对 `graph.x` 做 cosine），并不是"子图级"。返回的 selected_nodes 之后再现取邻接得到子图。
- "一阶"用 `self.G.x`（PLM 0 层），"二阶"用 `self.G1.x`（GCN 1 层）；**这与论文中 "GNN_Indexing 通过聚合 l-hop 邻居生成嵌入" 的语义吻合**，但论文说应该有多层，代码事实只两层。
- 查询编码 `global_q_emb` 仅基于 `input`（观影序列字符串），**没拼任务指令**（"Given user's watching history, select..."），与论文 3.6 节"用 recommendation prompt 做 query"描述不一致。

### 3.4 `get_first_order_subgraph` 107-128

- 选定 node + 它在 networkx 图中的所有邻居 → 取局部子图。
- 节点特征用 `self.G.x`（PLM 嵌入）；边特征用 `self.G.edge_attr`（PLM 边嵌入）。
- ⚠ `node_to_subgraph_index = {old: new for new, old in enumerate(subgraph_nodes)}`：subgraph_nodes 是 set，**遍历顺序不确定**，导致每次重映射都不一样；不影响 GNN 输出的均值（图本身一样），但破坏了"节点 0 = 中心节点"这种局部含义。

### 3.5 `re_ranking` 195-211

- 拿一阶/二阶检索得到的 node 列表，再用 `q_emb`（即 `input` 的 SBERT 嵌入）与每个节点的特征做 cosine，取 top-N。
- **没有学习参数**，重排完全是固定相似度；与论文里"re-ranking ensure most relevant at top"描述一致但能力有限。

---

## 4. 数据规模 & 训练超参

| 项 | 值 | 来源 |
|---|---|---|
| 训练集大小 | 9 000（取自 ML1M 的前 9000） | train.py:40 |
| 测试集大小 | 1 000（[9000:10000]） | evaluate.py:86 |
| Batch size | 5 (train) / 5 (eval) | run.sh; config.py 默认 3 |
| Epochs | 3 | config.py:21 |
| LR | 1e-5；warmup_epochs=1（**未生效**） | config.py:12, 22 |
| Weight decay | 0.05 | config.py:13 |
| GNN | GraphTransformer 4 层；hidden=1024；heads=4 | run.sh; config.py:36-41 |
| Sub_graph K | 3（每 item 取 3 子图） | run.sh |
| Re-rank N | 5（最终保留 5×2=10 子图） | run.sh |
| adaptive_ratio | 5（控制 whether_retrieval 选几个 item） | config.py:44 |
| 候选数 M | 20（从 A-T 选） | dataset 文件名 + 提示模板 |
| 最大新 token | 64（评测只用第一 token） | config.py:33 |

KG 规模（论文表 3，对应 dataset/fb）：
- ML1M：3 498 items in KG / 250 631 entities / 264 relations / 348 979 triples
- ML20M：20 139 / 1 278 544 / 436 / 1 827 361
- Book：91 700 / 186 954 / 16 / 259 861

---

## 5. 论文 vs 代码 差异对照表（高优先级 bug & 不一致）

| # | 论文描述 | 代码事实 | 影响 |
|---|---|---|---|
| 1 | 公式(8) 子图嵌入逐个拼接为 N 个 soft token | graph_llm.py:168 `mean(dim=0)` 把 N 子图压成 1 token | 严重信息损失，是创新切入点 |
| 2 | hop-field 索引跨 l-hop（多层） | retrieve.py 只用 G(0层) + G1(1层) | 多层 hop 没真用上，layer3 .pt 闲置 |
| 3 | re-ranking query = task desc + history | retrieve.py:44 仅用 history string | 重排信号弱，不含任务上下文 |
| 4 | 候选 M=20，含 1 正 19 负 | 数据文件已是 20 项（A-T） | 一致 |
| 5 | 流行度阈值 p ∈ [0,1] | 代码用 `adaptive_ratio*len(seq)` 的整数；evaluate.py:64 与 train.py:62 不一致（list 重复 vs 取子集） | **训练-推理分布不一致** |
| 6 | popularity 由人工设置 | 用绝对交互计数，未做归一化 / 用户级 | 长尾物品被过度选 |
| 7 | GNN_Encoding 与 GNN_Indexing 不同参数 | 代码两者用同一类型 GraphTransformer，但参数独立 | OK，但缺消融对比 |
| 8 | 评测 Accuracy / Recall@3 / Recall@5 | evaluate.py 仅打印 R@1/3/5/10，没算 ACC | 无大碍，需补 ACC |
| 9 | 边特征与边类型应被利用 | 仅 GraphTransformer 用 edge_attr；GAT/GCN/SAGE 忽略 | 边类型语义浪费 |
| 10 | LLM 冻结 + LoRA 双模式 | 代码内部 hard-code lora_r=8（与论文表 4 lora_r=4 不同） | 与论文超参冲突 |
| 11 | LR cosine warmup | 代码定义 `adjust_learning_rate` 但 train.py 未调用 | 实际是恒定 LR |
| 12 | epoch 末做 best 选择 | train.py:90 直接覆盖 best | 没有验证集挑 best |

---

## 6. 论文消融数据回顾（用于设计创新点对比基线）

LLaMA-2-7B / ML1M：
- K-RagRec: ACC=0.435, R@3=0.725, R@5=0.831
- 去 -Indexing：ACC ↓
- 去 -Popularity：ACC ↓
- 去 -Re-ranking：ACC ↓
- 去 -Encoding：**掉 37%**（最关键）→ ML1M ACC 约 0.274
- 在 Book 上去 -Encoding：**掉 45.9%**

GNN 类型对比（表 9 ML1M LLaMA-2）：
- GCN 0.397 / GAT 0.420 / GT 0.429 / SAGE 0.418
- ACC 差距 ≤ 3%，鲁棒但说明 GNN 选型本身收益有限。

GNN 层数（表 10 Book）：
- 3 层 0.496 / 4 层 0.506 / 5 层 0.498 → 4 层最佳。

---

## 7. 可改造的接口点（→ 创新点的着力位置）

按"代码改造成本 × 论文增量"打分（H/M/L）：

| 接口 | 文件:行 | 改造空间 | 成本 |
|---|---|---|---|
| 子图嵌入 mean → 多 soft token / cross-attention | graph_llm.py:168 / 226 | **H** 信息保真 | L |
| `whether_retrieval` 的策略 | retrieve.py:64-69 | **H** 个性化/学习式选择 | M |
| 重排 query 不含任务描述 | retrieve.py:44 | **H** 检索质量 | L |
| 重排器无参数 | retrieve.py:195-211 | **H** 可学习 reranker (Cross-encoder / GNN) | M |
| GNN 不利用边类型 | gnn.py 全部 | **M** 异质 / R-GCN / CompGCN | M |
| 索引仅 2 层 | retrieve.py:24-25 + index_KG.py | **M** 真正的 hop-field（多层 fusion） | M |
| Projector Sigmoid+Linear 简单 | graph_llm.py:110-114 | **L** Q-Former 风格 / Perceiver | M |
| LLM 冻结 + 仅 LoRA | graph_llm.py:74-96 | **M** 引入 RecLM-emb / TokenRec 思路 | H |
| 时序信号缺失 | 整套未建模时间 | **H** 时序 GNN / Bi-Mamba / SASRec 蒸馏 | H |
| 评测仅 single-token | graph_llm.py:256-305 | **L** 添加 listwise / 生成式评测 | L |
| 没有 hard-negative | dataset 已固定 19 负 | **M** 难负挖掘 + curriculum | M |
| 没有图结构上的对比/对齐损失 | graph_llm.py 仅 CE | **H** Graph-Text alignment, InfoNCE | M |
| Popularity = 全局静态 | retrieve.py:71-83 | **H** Dynamic / user-aware popularity | L |
| 数据集硬编码 ML1M | train.py:31 / evaluate.py:35 | 工程修复 | L |
| 训练 / 推理 retrieve 不一致 | evaluate.py:64 vs train.py:62 | bug fix | L |
| 缺 validation set & best ckpt | train.py:79-91 | 工程修复 | L |

---

## 8. 一句话总结

> **K-RagRec 的代码实现是"论文 + 工程妥协"的产物**：核心 RAG 流（流行度筛选 → 跨层节点检索 → 子图重排 → GNN 编码 → projector → frozen LLM）齐全，但论文里最关键的"逐子图 soft prompt"被简化成"全部 mean 池化"，hop-field 仅用了 2 层，重排器没有学习参数，时序/异质 KG 信号几乎未利用。
> 这些"实现差距"恰好是后续创新最容易拿到收益的位置。
