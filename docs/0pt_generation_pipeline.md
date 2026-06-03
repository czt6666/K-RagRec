# 0.pt 是怎么生成的？谁映射谁？

## 一句话总结

`0.pt` 是知识图谱（KG）里每个节点的**向量指纹**。
- 如果指纹是"名称"的（如"Tom Hanks"）→ 检索有效 ✅
- 如果指纹是"机器ID"的（如"/m/02rdyx7"）→ 检索无效 ❌

作者发布的 `0.pt` 是**名称指纹**（所以检索有效），但他们没告诉你"名称从哪来"。

---

## 一、Freebase 原始数据长什么样？

Freebase 是一个知识图谱，所有实体都用**机器ID（MID）**标识：

```
/m/0dyb1    →  Toy Story（电影）
/m/09w353   →  Jumanji（电影）
/m/03l72gz  →  某个演员
/m/0cgp4pw  →  某个日期
```

三元组文件 `filtered_full_fb.txt` 里全是这种 MID：

```
/m/05z5lc    /film/film/starring    /m/03l72gz
  ↑                ↑                    ↑
电影《教父》      关系：主演            演员马龙·白兰度
（但写成了MID）   （关系也是MID格式）    （也是MID）
```

**问题**：SBERT（语义编码器）认识 "Tom Hanks" 这几个字，但不认识 `/m/03l72gz` 这串乱码。

---

## 二、生成 0.pt 需要什么步骤？

### 步骤 1：MID → 名称映射（关键！缺了这步就完蛋）

需要一张"对照表"：

| Freebase MID | 实体名称（人类可读） |
|---|---|
| /m/0dyb1 | Toy Story |
| /m/03l72gz | Marlon Brando |
| /m/0cgp4pw | 1994-09-23 |

**谁来提供这张表？**
- 论文作者用了某种方式生成（可能是 Freebase API、Wikidata 映射、或自己爬的）
- 但他们**没有发布这张表**
- 本地发布的 `fb_entity_names.tsv` 只有 **3.7% 覆盖率**，不够用

**有这张表 vs 没这张表的区别**：

```
有映射表（作者的做法）：
  /m/0dyb1 → "Toy Story" → SBERT编码 → 向量A
  /m/03l72gz → "Marlon Brando" → SBERT编码 → 向量B

没映射表（本地现状）：
  /m/0dyb1 → "/m/0dyb1" → SBERT编码 → 向量X（乱码的向量）
  /m/03l72gz → "/m/03l72gz" → SBERT编码 → 向量Y（乱码的向量）
```

### 步骤 2：用映射后的名称生成 mapped_filtered_fb.txt

把三元组里的所有 MID 替换成名称：

```
原始（filtered_full_fb.txt）：
  /m/05z5lc    /film/film/starring    /m/03l72gz

映射后（mapped_filtered_fb.txt，作者有，本地没有）：
  The Godfather    starring    Marlon Brando
```

### 步骤 3：index_KG.py 生成 all_nodes.csv 和 0.pt

```python
# index_KG.py 的逻辑

# 1. 读取 mapped_filtered_fb.txt
# 2. 提取所有实体名称 → all_nodes.csv
#    node_id | node_attr
#    0       | The Godfather
#    1       | Marlon Brando
#    ...

# 3. 用 SBERT 编码每个 node_attr
x = SBERT(["The Godfather", "Marlon Brando", ...])
#    → shape [14669, 1024]

# 4. 保存为 0.pt
```

---

## 三、本地发生了什么？

### 场景 A：下载的数据集（正确状态）

```
dataset/fb/
  ├── filtered_full_fb.txt      ← MID 三元组（下载的）
  ├── graphs/
  │     └── 0.pt                ← 名称向量（作者预计算的 ✅）
  └── nodes/
        └── all_nodes.csv       ← 不存在！
```

**注意**：发布的数据集中**没有** `all_nodes.csv`！`DATA_FORMAT.md` 明确写了：
> "They are absent in the distributed snapshot"

### 场景 B：某人运行了 index_KG.py（错误状态）

因为本地没有 `mapped_filtered_fb.txt`，`index_KG.py`  fallback 到了 `filtered_full_fb.txt`（MID 版），生成了：

```
dataset/fb/
  └── nodes/
        └── all_nodes.csv       ← 74.5% 是 MID（❌）
```

但 `0.pt`**没有被重新生成**（或者说，如果重新生成了，就会是错误的 MID 版本）。

---

## 四、核心问题：0.pt 到底是什么？

### 两个可能性

| 可能性 | 0.pt 是什么 | 检索是否有效 | 论文能否达到43% |
|---|---|---|---|
| **A（作者的）** | SBERT("Toy Story") 等**名称向量** | ✅ 有效 | ✅ 能 |
| **B（本地错误）** | SBERT("/m/0dyb1") 等**MID向量** | ❌ 无效 | ❌ 不能 |

**关键证据支持 A**：
1. `DATA_FORMAT.md` 明确说 `0.pt` 是 "text embedding of **each entity name**"
2. 如果 0.pt 是 MID 向量，论文不可能达到 43%（检索完全随机）
3. 作者不会发布一个自己都知道是垃圾的数据

**但我之前覆盖了 0.pt！**

我运行的 `fix_kg_node_names.py` 重新生成了 `0.pt`，但因为没有正确的映射表，它实际上是用 **MID** 重新生成的！

这意味着：**本地的 `0.pt` 现在已经是错误版本了！**

---

## 五、"谁映射谁"的完整链条

```
Freebase Dump（原始）
    │
    │  作者的操作（未开源）
    │  ┌──────────────────────────────────────┐
    │  │ 1. 用 Freebase API / Wikidata / 其他 │
    │  │    生成 MID → 名称 映射表            │
    │  │                                      │
    │  │ 2. 把 filtered_full_fb.txt           │
    │  │    转成 mapped_filtered_fb.txt       │
    │  │    （所有MID替换成名称）              │
    │  │                                      │
    │  │ 3. 运行 index_KG.py                 │
    │  │    生成正确的 all_nodes.csv + 0.pt   │
    │  └──────────────────────────────────────┘
    │
    ▼
Google Drive 发布的数据集
    │
    ├── 0.pt              ← ✅ 名称向量（作者预计算的）
    ├── filtered_full_fb.txt ← MID 三元组（原始）
    ├── fb_entity_names.tsv  ← 部分映射（覆盖率3.7%）
    └── 没有 mapped_filtered_fb.txt
        没有 all_nodes.csv
        没有 edges/all_edges.csv
    │
    ▼
本地运行（错误操作）
    │
    ├── 某人运行 index_KG.py
    │   （找不到 mapped_filtered_fb.txt，
    │    只能读 filtered_full_fb.txt）
    │
    └── 生成了错误的 all_nodes.csv（74.5% MID）
        如果同时重新生成 0.pt → 也是错误的！
```

---

## 六、你现在该怎么办？

### 第一步：确认 0.pt 是否被覆盖了

检查 `dataset/fb/graphs/0.pt` 的修改时间：
- 如果是 **2026/4/1** → 是原始的（正确）✅
- 如果是 **2026/6/3** 或更晚 → 被我覆盖了（错误）❌

### 第二步：如果被覆盖了，恢复它

从服务器上下载原始的 `0.pt`：

```powershell
# PowerShell
scp -P 31500 root@39.106.77.104:/root/workspace/python/K-RagRec/dataset/fb/graphs/0.pt `
    D:\Python\K-ragrec\dataset\fb\graphs\0.pt
```

如果服务器上的也被覆盖了（你重新运行过 index_KG.py），那就需要从 Google Drive 重新下载整个数据集：
https://drive.google.com/file/d/1MlEPkRj47WrdXECUiz5D6Ie1oMv4hKC9/view

### 第三步：验证 0.pt 是否正确

运行这个验证脚本，看检索是否能找到语义相关的节点：

```python
# 验证脚本：输入一部电影，看检索回来的子图中心节点是什么
from retrieve import GraphRetrieval
retrieval = GraphRetrieval('sbert', 'dataset/fb')

# 检索《Toy Story》相关的子图
graphs = retrieval.retrieval_topk(
    "Toy Story",          # query
    [1],                   # retrieve_movies_list（这里随便给一个ID）
    topk_nodes=3,
    topk_rerank_nodes=1
)

# 打印子图中心节点的ID
for g in graphs:
    center_id = g.node_ids[0].item()
    print(f"中心节点ID: {center_id}")
```

如果 `0.pt` 是正确的，检索到的节点应该和 "Toy Story" 语义相关（如其他动画电影、皮克斯公司等）。
如果 `0.pt` 是错误的，检索结果看起来就是随机的。

---

## 七、为什么作者不发布 mapped_filtered_fb.txt？

最可能的原因：

1. **文件太大**：完整的 Freebase MID→名称映射表可能有几十 MB 甚至几百 MB
2. **版权问题**：Freebase 数据有特定许可证，作者可能不想处理合规问题
3. **他们认为这是标准预处理**：就像 CV 论文不会发布"怎么把 PNG 转成张量"的脚本一样，他们认为"MID 映射成名称"是常识性操作

**但这对复现者来说是个坑**：没有映射表，就无法重新生成 `0.pt`，只能依赖作者预计算好的版本。

---

## 八、如果你必须重新生成 0.pt（比如换了 SBERT 模型）

没有 `mapped_filtered_fb.txt` 的情况下，你有两个选择：

### 方案 A：用 Wikidata 补映射（推荐，但需要编程）

Wikidata 有 Freebase MID → Wikidata QID → 多语言标签的映射：
1. 下载 Wikidata 的 `freebase_id` 属性 dump
2. 把 `filtered_full_fb.txt` 中的 MID 映射到 Wikidata QID
3. 再映射到英文/中文标签
4. 生成 `mapped_filtered_fb.txt`

### 方案 B：直接用原始数据集

如果 Google Drive 上的 `0.pt` 是正确的，**永远不要重新生成它**。作者用 `all-roberta-large-v1` 预计算好了，你只需要用它。

---

## 总结

| 问题 | 答案 |
|---|---|
| 谁映射谁？ | Freebase MID（如 `/m/0dyb1`）→ 实体名称（如 "Toy Story"） |
| 谁来映射？ | 论文作者用了某种外部资源（未发布），本地只有不完整的 `fb_entity_names.tsv` |
| 0.pt 是什么？ | 作者用**名称**预计算的 SBERT 向量，不是 MID 向量 |
| 本地 all_nodes.csv 为什么全是 MID？ | 因为没有 `mapped_filtered_fb.txt`，某人用原始 MID 文件生成了它 |
| 这影响训练和推理吗？ | **不影响**，因为核心路径读 `0.pt` 不读 `all_nodes.csv` |
| 但如果 0.pt 被覆盖了？ | **就影响了！** 需要恢复原始的 `0.pt` |