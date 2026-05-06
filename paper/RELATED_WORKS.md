# 近期相关论文综述（2024-2026）

> 围绕 K-RagRec 论文的核心范式（Knowledge Graph + Retrieval Augmented Generation + LLM-based Recommendation），整理 30 篇近期相关工作的核心思想、与 K-RagRec 的差异、可借鉴点。
> 分类：① GraphRAG 通用方法；② KG-RAG 推荐；③ LLM 推荐增强；④ 图表征 / 图 Transformer；⑤ 时序 / 序列推荐；⑥ 自适应检索 / 反思检索；⑦ 多跳推理 / 路径检索；⑧ 工具组件（reranker / soft prompt / MoE）。
>
> 标记：⭐= 与 K-RagRec 直接互补，可考虑融入创新点。

---

## ① GraphRAG 通用方法

### 1. ⭐ G-Retriever (NeurIPS 2024) – arXiv:2402.07630
- **核心**：把子图检索建模为 Prize-Collecting Steiner Tree (PCST) 优化。同时为节点和边赋"奖金"，求带权连通子图。
- **优势**：相比 K-RagRec 的"先 top-k 节点 → 取邻接子图"的两步法，PCST 是端到端联合优化，能保留连通的逻辑链。可解释；在 WebQSP 上能裁掉 99% 节点，speed-up ~3×。
- **可借鉴**：把 K-RagRec 的 cosine top-k 检索升级为 PCST，能保证子图连通；适合多跳推荐场景。

### 2. ⭐ NodeRAG (arXiv:2504.11544, 2025)
- **核心**：异质节点结构（实体节点、文本块节点、社区节点、属性节点等），每种节点不同的检索策略 + 个性化 PageRank 做检索。
- **结果**：在 MuSiQue 上 46.29% > GraphRAG 41.71% > LightRAG 36.0%；token 用量更少。
- **可借鉴**：把"item 节点 + KG 实体节点 + 用户节点 + 类型节点"统一为异质 graph，让"用户兴趣社区"也参与检索。

### 3. LightRAG (EMNLP 2025) – HKU
- **核心**：双层检索（low-level 实体 + high-level 概念），向量+图混合索引；查询延迟降 30%。
- **可借鉴**：低层=单 item 一阶子图，高层=用户兴趣聚类社区，给 K-RagRec 加"全局视图"。

### 4. Microsoft GraphRAG
- **核心**：先在文档语料上构 KG，做 community summarization，再用 community report 增强生成。
- **借鉴价值低**（更适合 doc QA，但 community 思想可用于推荐域聚类）。

### 5. ⭐ Paths-over-Graph (PoG, WWW 2025) – arXiv:2410.14211
- **核心**：动态多跳路径探索 + 图结构剪枝；对多实体问题首次提出 "deep path detection"。
- **结果**：在 KGQA 5 数据集上比 ToG 平均 +18.9%。
- **可借鉴**：把"用户多个历史 item"看成"多实体 query"，通过 PoG 风格的多源路径联合搜索得到子图。这是 K-RagRec 现在没有的能力。

### 6. ⭐ Graph-Constrained Reasoning (GCR, ICLR 2025)
- **核心**：用 KG-Trie 把 KG 路径编码成前缀树，约束 LLM 解码必须沿合法路径走，保证 100% 可信推理。
- **可借鉴**：可以在生成"为什么推荐 X"的解释时，约束 LLM 沿 KG 真实路径生成。

### 7. ⭐ Think-on-Graph 2.0 (ICLR 2025)
- **核心**：迭代检索 + 推理协同，结构化 KG 与非结构化文本紧耦合。
- **可借鉴**：item 不仅有 KG，还有 review/description 长文本，二者迭代检索能补全冷启动信息。

### 8. RAP-RAG (MDPI 2025)
- **核心**：异质加权图索引 + 自适应 planner，根据查询特征动态选择检索策略。
- **可借鉴**：不同用户/不同 query 复杂度走不同的检索路径。

---

## ② KG-RAG 推荐 / 解释推荐

### 9. ⭐ G-Refer (WWW 2025)
- **核心**：基于 GraphRAG 的可解释推荐；混合图检索（结构 + 语义），知识剪枝后转人类可读文本。
- **与 K-RagRec 差异**：G-Refer 主打"解释"，K-RagRec 主打"准确"。但 G-Refer 的混合检索/剪枝模块直接可借用。

### 10. ⭐ KG-RAG for Cold-Start Recommendation (arXiv:2505.20773, 2025)
- **核心**：动态知识图构造 + 自适应候选检索；专为冷启动设计。
- **可借鉴**：K-RagRec 的"流行度阈值"是 hard rule，可改为"动态构造冷启动子图"。

### 11. ⭐ IKGR – Intent-Based KG Recommender (2025)
- **核心**：用 LLM 提取 user intent → 构造 intent-augmented KG → embedding translation 层；针对稀疏连通性。
- **可借鉴**：把 LLM 抽到的"用户意图实体"显式插入 KG，再做 RAG。

### 12. AgentCF (WWW 2024)
- **核心**：用户/item 都用 LLM agent 模拟，agent 之间通过模拟交互学习。
- **借鉴价值低**，但 agent 化的"用户 agent + KG agent"可作为长尾增量。

### 13. RecLM: Recommendation Instruction Tuning (ACL 2025)
- **核心**：为推荐设计专门的 instruction-tuning 数据集，多种任务统一格式。
- **可借鉴**：K-RagRec 的 prompt 模板单一（"select from A-T"），可换成多任务 instruction。

### 14. CoRA (2024) / CoLLM (2024)
- **核心**：把协同过滤 embedding 作为 LLM 的 weight 调制 / 直接拼接。CoRA 通过 hyper-network 调整 LLM 权重。
- **可借鉴**：K-RagRec 没有显式 collaborative signal，可以加上 LightGCN 或 MF embedding 作为 graph token 之一。

### 15. ⭐ Comprehending KG with LLM for RecSys (CoLaKG, WSDM 2025)
- **核心**：LLM 为 KG 子图生成自然语言摘要，作为软提示注入；同时建模局部 + 全局兴趣。
- **可借鉴**：K-RagRec 仅用 GNN 嵌入做 soft token，可补充 LLM-summarized 文本，形成双 token 互补。

---

## ③ LLM 推荐增强（生成式 / 微调）

### 16. ⭐ TallRec (RecSys 2023) – arXiv:2305.00447
- 基线类工作：用 LoRA 把 LLM 微调到推荐任务上，是 K-RagRec 的对比 baseline 之一。

### 17. LLaRA (SIGIR 2024)
- **核心**：用 SASRec 等 CF 模型的 item embedding 替换 LLM 中的 item token，缓解 ID 表征稀疏。
- **可借鉴**：把 K-RagRec 的"图 soft token"扩展为"图 + 协同 + 文本"三 token 融合。

### 18. ⭐ TokenRec (KDD 2025)
- **核心**：用 vector quantization 把用户/item 编码成离散 token，注入 LLM 词表。
- **可借鉴**：把 K-RagRec 的连续 graph embedding 量化为离散 token，便于 LLM 复用且节省序列长度。

### 19. ⭐ Semantic IDs for Generative Search & Recommendation (Spotify 2025)
- **核心**：RQ-KMeans / RQ-VAE 构造 multi-task semantic ID；search 与 rec 联合训练帕累托最优。
- **可借鉴**：把 K-RagRec 的子图也编为 semantic ID，让 LLM 可生成式地预测下一个 item。

### 20. SASRecLLM (arXiv:2507.05733, 2025)
- **核心**：SASRec 作为 collaborative encoder + LoRA LLM；mapping layer 对齐维度。
- **可借鉴**：把 SASRec 的"序列建模"能力拼到 K-RagRec 的"图建模"上，弥补时序信息缺失。

### 21. Lost in Sequence (arXiv:2502.13909, 2025)
- **核心**：实证发现 LLM4Rec 在捕捉序列变化上不如 SASRec；提出 SCaLRec 校准方案。
- **借鉴**：警示——只用 LLM/Graph 不建模序列演化是次优。

---

## ④ 图表征 / 图 Transformer

### 22. ⭐ HGT (Heterogeneous Graph Transformer)
- **核心**：节点/边/meta-relation specific 参数 + 隐式 meta-path 学习。
- **可借鉴**：把 K-RagRec 的 GraphTransformer (代码 src/model/gnn.py:35-61) 升级为 HGT，处理 Freebase 中 264 种关系。

### 23. ⭐ HMT – Heterogeneous Memory Transformer (Sci. Rep. 2025)
- **核心**：图 memory module + Transformer 处理异质图 + 长程上下文。
- **可借鉴**：在 K-RagRec 的 GNN 编码后加一个 memory bank 跨样本共享。

### 24. EHG (2025)
- **核心**：参数减 25%，训练快 20%，在异质图节点分类上 SOTA。
- **可借鉴**：减小 GNN 计算量；推荐场景在线推理对延迟敏感。

### 25. CompGCN (ICLR 2020) / R-GCN
- **核心**：composition operation (sub/mult/circular) 联合编码节点 + 关系，特别适合多关系 KG。
- **可借鉴**：K-RagRec 的 GCN/GAT/SAGE 完全忽略 edge_attr；换成 CompGCN 是低成本高收益。

### 26. CKGE / KGformer
- **核心**：上下文化的 KG 嵌入 Transformer。
- **可借鉴**：用 KG 路径作为序列 → 走 Transformer 自注意。

---

## ⑤ 时序 / 序列推荐

### 27. ⭐ TGODE – Time-Guided Graph Neural ODE (arXiv:2511.18347, 2025)
- **核心**：时间引导的扩散生成器 + 神经 ODE 建模偏好演化；5 数据集提升 10–46%。
- **可借鉴**：把 K-RagRec 数据中的时间戳显式建模（论文与代码完全没用时间戳）。

### 28. Hypergraph Repeated Intent (WWW 2025)
- **核心**：超图建模重复 intent 的时序模式。
- **可借鉴**：用户重复看某类电影/书 → 超图加权重。

### 29. PTGCN (2021)
- **核心**：position + time aware GCN for sequential rec。

---

## ⑥ 自适应检索 / 反思检索

### 30. ⭐ Self-RAG (ICLR 2024) – arXiv:2310.11511
- **核心**：训练 LLM 输出 reflection token：决定何时检索、检索内容是否相关、是否使用。
- **与 K-RagRec 差异**：K-RagRec 用静态 popularity 决定何时检索；Self-RAG 是模型学习决定。
- **可借鉴 (高价值)**：把 popularity rule 替换成可学习的 retrieval gate。

### 31. CRAG – Corrective RAG (ICLR 2024)
- **核心**：用轻量评估器判定检索结果 {Correct/Incorrect/Ambiguous}，相应执行不同补救策略（web search / decompose）。
- **可借鉴**：在子图检索后加一个 critic，判断子图是否对推荐有信息量。

### 32. SCMRAG (AAMAS 2025)
- **核心**：自我修正多跳 RAG；信息不足时主动补检索。
- **可借鉴**：动态决定要不要展开二跳邻域。

### 33. AIR-RAG (2025)
- **核心**：自适应迭代检索，强化学习优化对齐。

### 34. RL-Driven Dynamic RAG (2025)
- **核心**：MDP + RL 把 one-shot 检索变成动态决策。

### 35. ARAG: Agentic RAG for Personalized Recommendation (arXiv:2506.21931, 2025)
- **核心**：多 agent（user-modeling agent / item-retriever agent / planner）协同推荐。
- **可借鉴**：K-RagRec 仍是单线检索；ARAG 的"agent 协同"是新范式。

---

## ⑦ 多跳推理 / 路径检索

### 36. ⭐ Reasoning on Graphs (RoG, ICLR 2024)
- **核心**：planning-retrieval-reasoning 三阶段；先生成 relation path 计划，再去 KG 找路径，最后基于路径推理。
- **可借鉴**：把 K-RagRec 的"子图节点"升级为"关系路径"，更结构化。

### 37. StepChain GraphRAG (arXiv:2510.02827, 2025)
- **核心**：把复杂问题分解为子问题，BFS 检索 + 解释。

### 38. GNN-RAG (arXiv:2405.20139, 2024)
- **核心**：GNN 直接在 KG 上推理出答案 candidate，再用 LLM verbalise。
- **可借鉴**：用 GNN 排候选 item，可作为 K-RagRec 的"协同重排器"。

---

## ⑧ 工具组件

### 39. ⭐ GNP – Graph Neural Prompting (AAAI 2024)
- **核心**：plug-and-play graph neural prompt；含 cross-modality pooling + 自监督 link prediction objective。
- **可借鉴**：K-RagRec 的 projector 是简单 MLP；换成 GNP 的 cross-modality 池化更精准；自监督 link prediction loss 可以加进训练。

### 40. CGP-Tuning (arXiv:2501.04510, 2025)
- **核心**：图-文本 cross-modal 对齐 + 类型感知 embedding；线性计算复杂度。
- **可借鉴**：K-RagRec 的 graph→LLM 投影没有类型感知。

### 41. MolCA / Q-Former 系列
- **核心**：Q-Former 桥接图 / 视觉到 frozen LLM。
- **可借鉴**：用 Q-Former 替换 K-RagRec 的 MLP projector，让"多个子图"通过 cross-attention 选择性进入 LLM。

### 42. Voyage rerank-2.5 / Jina Reranker v2 / Cohere v3.5
- **2025 reranker SOTA**。可用作 K-RagRec 的"子图重排器"，替换无参数 cosine。

### 43. ⭐ Hierarchical Time-Aware MoE (WWW 2025)
- **核心**：两层 MoE，第二层为 Temporal MoE，按时间戳路由 expert。
- **可借鉴**：把 K-RagRec 的 GNN 编码器拆成 multiple experts（不同关系类型 / 不同时间段）。

### 44. ⭐ Facet-Aware Multi-Head MoE (WSDM 2025)
- **核心**：MoE 替换 self-attention 的 query；不同 expert 抓不同偏好维度。
- **可借鉴**：用户兴趣多面性 → 多个 graph token，每个对应一种偏好。

### 45. FLR – Factorized Latent Reasoning (2025)
- **核心**：用多个 disentangled preference factor 表达 user intent；正交/多样/稀疏约束。

### 46. HyMiRec (2025)
- **核心**：disentangled multi-interest learning + 对比学习。

---

## 关键 takeaways（提炼为创新方向）

| 方向 | 提供 idea 的论文 | K-RagRec 当前缺位 |
|---|---|---|
| 子图连通性优化 | G-Retriever (PCST), NodeRAG | 仅 cosine top-k，子图离散 |
| 多跳/路径推理 | RoG, PoG, ToG-2 | 只有 1-hop / 2-hop 节点级 |
| 自适应检索（学习式 gate） | Self-RAG, CRAG, ARAG | hard popularity 阈值 |
| 学习式重排 | Voyage / Jina Reranker | 无参数 cosine |
| 时序信号 | TGODE, SASRec/SASRecLLM | 完全忽略时间戳 |
| 异质关系编码 | CompGCN, HGT, R-GCN | 4 种 GNN 都只把 edge_attr 当向量 |
| 多 soft token / Q-Former | GNP, CGP-Tuning, MolCA | 多子图被 mean 池化为单 token |
| 解耦兴趣 / MoE | Facet MoE, HyMiRec, FLR | 单一图嵌入表征 |
| Graph-Text 对齐损失 | CoLaKG, GNP self-link prediction | 仅 CE loss |
| 协同过滤信号 | LLaRA, CoLLM, TokenRec | 完全缺失 |
| 解释 / 生成 | G-Refer, GCR | 仅 ABCD 选择 |
| 冷启动专用 | KG-RAG cold-start, IKGR | 流行度阈值反向选冷门，但策略粗 |

每个方向都对应一个或多个 K-RagRec 代码改造接口（详见 `CODE_ANALYSIS.md` 第 7 节）。

---

## 论文链接列表（备查）

- K-RagRec: https://aclanthology.org/2025.acl-long.1317.pdf
- G-Retriever: https://arxiv.org/abs/2402.07630
- NodeRAG: https://arxiv.org/abs/2504.11544
- LightRAG: https://aclanthology.org/2025.findings-emnlp.568.pdf
- GraphRAG Survey: https://arxiv.org/abs/2408.08921
- Paths-over-Graph: https://arxiv.org/abs/2410.14211
- Graph-Constrained Reasoning: https://openreview.net/forum?id=6embY8aclt
- Think-on-Graph 2.0: https://arxiv.org/abs/2407.10805
- G-Refer: https://dl.acm.org/doi/10.1145/3696410.3714727
- KG-RAG Cold-Start: https://arxiv.org/abs/2505.20773
- AgentCF: https://dl.acm.org/doi/10.1145/3613904.3642041
- RecLM: https://aclanthology.org/2025.acl-long.751.pdf
- CoLaKG / CoLLM: https://arxiv.org/abs/2310.19488
- TallRec: https://arxiv.org/abs/2305.00447
- LLaRA: SIGIR 2024
- TokenRec: https://arxiv.org/abs/2406.10450
- Semantic IDs Spotify: https://web3.arxiv.org/pdf/2508.10478
- SASRecLLM: https://arxiv.org/abs/2507.05733
- Lost in Sequence: https://arxiv.org/abs/2502.13909
- HGT: https://arxiv.org/abs/2003.01332
- HMT (Sci Reports 2025): https://www.nature.com/articles/s41598-025-28266-1
- CompGCN: https://openreview.net/forum?id=BylA_C4tPr
- TGODE: https://arxiv.org/abs/2511.18347
- Hypergraph Repeated Intent (WWW 2025): https://dl.acm.org/doi/10.1145/3696410.3714896
- Self-RAG: https://arxiv.org/abs/2310.11511
- CRAG: https://arxiv.org/abs/2401.15884
- SCMRAG: https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p50.pdf
- ARAG: https://arxiv.org/abs/2506.21931
- Reasoning on Graphs: https://arxiv.org/abs/2310.01061
- StepChain GraphRAG: https://arxiv.org/abs/2510.02827
- GNN-RAG: https://arxiv.org/abs/2405.20139
- GNP: https://ojs.aaai.org/index.php/AAAI/article/view/29875
- CGP-Tuning: https://arxiv.org/abs/2501.04510
- Hierarchical Time-Aware MoE (WWW 2025): atailab.cn/seminar2025Spring/
- Facet-Aware Multi-Head MoE (WSDM 2025): atailab.cn/seminar2025Spring/
- FLR (2025): arXiv 2604.26760
- HyMiRec (2025): https://arxiv.org/abs/2510.13738
