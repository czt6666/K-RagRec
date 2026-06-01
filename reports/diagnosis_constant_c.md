# Baseline 模型恒预测 C 根因诊断报告

> 排查时间：2026-06-01
> 排查对象：`methods/baseline` 下的 GraphLLM（加载 `original_verify/output/...checkpoint_best.pth`）
> 核心问题：加载 checkpoint 后，模型对所有测试样本均预测为 **C**，准确率 ~5%

---

## 一、`fb_entity_names.tsv` 是什么？覆盖率为什么低？

### 1.1 文件来源

`dataset/fb/fb_entity_names.tsv` 是 **Freebase 知识图谱的实体名称映射文件**，来自 Google 的 Freebase 项目（已下线）。格式为：

```
<MID>\t<实体名称>
```

例如：
```
m.010bfczz\t私のベレット
m.010hztxj\t마이크
```

### 1.2 覆盖率低的直接原因

我们将 `filtered_full_fb.txt` 中所有 `/m/XXXXX` 格式的 MID 转换为 `m.XXXXX` 后，去 `fb_entity_names.tsv` 中查找名称，统计结果如下：

| 范围 | 实体总数 | 有名称的实体 | 覆盖率 |
|---|---|---|---|
| `filtered_full_fb.txt` 全部 | 250,631 | 9,288 | **3.7%** |
| 前 14,669 个（`0.pt` 对应子图） | 14,669 | 3,746 | **25.5%** |

**为什么覆盖率这么低？**

Freebase 中有大量 **匿名/辅助节点**，例如：
- 日期实体（`/m/010g913j` → release date）
- 数值实体（runtime、rating 等）
- 中间聚合节点
- 未被命名的 type/instance 节点

这些节点在 `fb_entity_names.tsv` 中没有人类可读名称。因此，即使 `fb_entity_names.tsv` 本身有 52 万条记录，`filtered_full_fb.txt` 中仍有大量 MID 查不到名称。

**对模型的影响：** 模型输入不依赖 `fb_entity_names.tsv`。`0.pt` 中的向量是预计算好的 SBERT 语义向量，只要节点顺序正确，模型就能正常工作。`fb_entity_names.tsv` 只影响 **调试时的可视化输出**。

---

## 二、训练出的问题 vs 预测代码的问题

### 2.1 排查手段：对比 `forward()` 与 `inference()`

我们对同一个测试样本，分别用 **训练模式**（`forward()`）和 **推理模式**（`inference()`）运行，对比关键指标。

#### 2.1.1 训练集 Label 分布

```python
Label distribution in train (9000 samples):
{'R': 459, 'M': 418, 'A': 408, 'E': 462, 'B': 405, 'O': 451, 'L': 473,
 'N': 439, 'J': 409, 'D': 469, 'P': 454, 'S': 458, 'K': 462, 'T': 451,
 'Q': 435, 'H': 459, 'F': 481, 'G': 434, 'C': 488, 'I': 485}
Most common label: ('C', 488)
```

- `C` 是最多的，但只比其他字母多 3%~10%，**不存在极端类别不平衡**。
- 如果模型因为类别不平衡而偏向 C，偏差应该很小，不足以导致 100% 预测 C。

#### 2.1.2 `forward()` 训练模式 Loss 检查

对同一条样本（真实答案是 `Q`），传入不同 label 计算 `forward()` 的 loss：

| 传入的 label | Loss 值 | 说明 |
|---|---|---|
| **真实答案 Q** | **1.6650** | 模型应该把这个压到最低 |
| 错误答案 C | **1.3365** | ⚠️ **比真实答案还低！** |
| 错误答案 A | 1.4767 | 也比真实答案低 |

**关键结论：**
- 模型认为 `C` 比真实答案 `Q` 更"合理"。
- 这说明 **训练没有收敛**，模型没有学会根据输入区分 20 个选项，而是退化成对某个固定选项（C）的系统性偏好。
- 这是 **训练出的问题**，不是预测代码的问题。

#### 2.1.3 `inference()` 实际生成了什么？

我们绕过 `graph_llm.py:inference()` 的排序逻辑，直接调用 `model.generate()` 并解码：

```
生成的 token IDs: [379, 2]
生成的文本: 'H'
第一个生成 token 的 A-T 概率分布 (Top 5):
  C: 0.094246
  J: 0.080613
  E: 0.074555
  G: 0.070771
  H: 0.065453
```

**惊人发现：**
- `model.generate()` 实际生成的第一个 token 是 `379`（`▁H`），不是 `C`！
- `scores[0]` 中 A-T 的最高概率只有 **0.094**（C），**说明模型在第一个生成位置更倾向于生成 A-T 以外的 token**。
- 但 `graph_llm.py:inference()` 的代码逻辑是：**不读取实际生成的文本，而是直接对 `scores[0]` 中 A-T 的 20 个概率做排序，取排名第一的**。

这导致：
- 模型实际生成了 `H`
- 但 `inference()` 报告排名第一的是 `C`
- `evaluate.py` 使用 `inference()` 的结果计算准确率，记录为 `C`

#### 2.1.4 `forward()` vs `inference()` 输入一致性

手动构造两者的 `inputs_embeds`，对比最大差异：

```
forward() 和 inference() 手动构造的 inputs_embeds 最大差异: 0.01953125
```

差异很小（< 0.02），在 float16 精度范围内，**不构成主要问题**。

#### 2.1.5 Graph Embed 是否退化为常量？

对比两个完全不同样本的 `graph_embed`（经过 projector + mean pool 后）：

```
两样本 graph_embed 的 L2 距离: 0.9577
样本1 graph_embed 均值: -0.0012, 标准差: 1.1848
样本2 graph_embed 均值: -0.0012, 标准差: 1.1791
```

- L2 距离 0.96，说明两个样本的 graph_embed **有明显差异**。
- 不存在 GNN/Projector 输出退化为常量的问题。

---

## 三、根因总结

### 根因 1：训练失败（主要）

- `forward(label='C')` 的 loss（1.3365）**低于**真实答案的 loss（1.6650）。
- 模型没有学会根据 graph + text 区分 20 个选项，而是学到了对 `C` 的系统性偏好。
- 可能原因：
  1. **学习率固定 1e-5 偏低**，`adjust_learning_rate`（cosine warmup）定义了但 `train.py` 没有调用。
  2. **3 个 epoch 不足**，loss 从 8.1 降到 1.5，但 1.5 对 20 类仅比随机（~3.0）好一倍，远未收敛。
  3. **mean pool 信息损失**：10 个子图被压缩成 1 个 token，graph 信息严重损失。
  4. **没有 validation set 选 best**：每 epoch 末直接覆盖 best ckpt，评估的可能是欠拟合或过拟合的模型。

### 根因 2：`inference()` 解码逻辑缺陷（次要，但严重）

`graph_llm.py:inference()` 的代码：

```python
scores = generation_output.scores[0].softmax(dim=-1)
specific_logits = torch.tensor(scores[:, option_indices], dtype=torch.float32).softmax(dim=-1)
sorted_indices = specific_logits.argsort(dim=-1, descending=True)
return sorted_indices
```

**问题：**
- 它假设第一个生成 token **一定是 A-T 之一**。
- 但实际上 `model.generate()` 生成的第一个 token 可能是 `▁H`、` `、`\n` 等（A-T 之外）。
- `scores[0]` 中 A-T 的 softmax 概率之和远小于 1，最高仅 ~0.09。
- 即使模型训练好了，如果第一个 token 不是 A-T（例如生成了一个空格），`inference()` 也会错误地报告 A-T 中概率最高的那个。

**正确的做法：**
- 解码 `generation_output.sequences`，提取生成的第一个字符。
- 如果第一个字符是 A-T，直接返回。
- 如果不是，再做 fallback（例如取 `scores[0]` 中 A-T 的 argmax）。

---

## 四、数据流逐层解析

### 4.1 训练数据流（`train.py` → `graph_llm.py:forward()`）

```
10000_data_id_20.json
  │
  ├──► input: "Last Action Hero", "Terminal Velocity", ... (10部电影标题)
  ├──► questions: "A: Big Lebowski, The, B: Abbott and Costello..." (20个选项)
  ├──► output: "Q" (正确答案字母)
  └──► sequence_ids: [485, 548, 2402, ...] (10部电影的ML1M ID)
       │
       ▼ retrieve.py
           whether_retrieval(seq_ids, adaptive_ratio * len(seq_ids))
           ──► retrieve_movies_list: [3440, 548, 2720, ...] (按冷门排序后的电影ID)
           │
           retrieval_topk(input_text, retrieve_movies_list, topk_nodes=3, topk_rerank_nodes=5)
           ──► graphs: list[Data] (10个子图，每个子图 ~20 nodes, 20 edges)
       │
       ▼ sample dict
           {'id': [...], 'graph': [graphs], 'question': [prompt], 'label': ['Q']}
       │
       ▼ graph_llm.py:forward()
           encode_graphs() ──► graph_embeds [10, 1024]
           projector()     ──► [10, 4096]
           mean(dim=0)     ──► [1, 4096]  ← 信息损失点
           cat([BOS, graph_embed, question, EOS_USER, label, EOS])
           ──► inputs_embeds [1, ~430, 4096]
           model(inputs_embeds=..., labels=...) ──► loss
```

**每一步的数据类型：**

| 步骤 | 数据 | 类型/形状 | 说明 |
|---|---|---|---|
| `sequence_ids` | 电影 ID 列表 | `list[int]` | 如 `[485, 548, 2402, ...]` |
| `retrieve_movies_list` | 检索用电影 ID | `list[int]` | 冷门优先排序后的子集 |
| `graphs` | 子图列表 | `list[Data]` | 每个 `Data` 有 `x`, `edge_index`, `edge_attr` |
| `graph_embeds` | GNN 输出 | `torch.Tensor [N_subgraphs, 1024]` | 子图内节点 mean pool |
| `projected` | Projector 输出 | `torch.Tensor [N_subgraphs, 4096]` | MLP 映射到 LLM dim |
| `sample_graph_embeds` | 最终 graph token | `torch.Tensor [1, 4096]` | **10 个子图再 mean pool → 1 个 token** |
| `inputs_embeds` | LLM 输入 | `torch.Tensor [1, seq_len, 4096]` | BOS + graph + question + label |
| `labels` | 训练目标 | `torch.Tensor [1, seq_len]` | 只有 label 位置非 -100 |

### 4.2 推理数据流（`evaluate.py` / `debug_pipeline.py` → `graph_llm.py:inference()`）

```
sample dict (label='')
  │
  ▼ graph_llm.py:inference()
      encode_graphs() ──► 同训练
      projector()     ──► 同训练
      mean(dim=0)     ──► 同训练
      cat([BOS, graph_embed, question, EOS_USER])
      ──► inputs_embeds [1, ~428, 4096]  ← 注意：没有 label！
      model.generate(inputs_embeds=..., max_new_tokens=64)
      ──► generation_output.sequences [1, seq_len]
      ──► generation_output.scores[0] [1, vocab_size]
      scores[0][:, option_indices].argsort(descending=True)
      ──► sorted_indices [1, 20]
```

**关键差异：**

| 对比项 | `forward()` 训练 | `inference()` 推理 |
|---|---|---|
| 输入序列结尾 | `... + EOS_USER + label + EOS` | `... + EOS_USER` |
| 期望下一个 token | `label`（如 `▁Q`） | `label`（如 `▁Q`） |
| 获取预测的方式 | `labels` 参数直接监督 | `generate()` + `scores[0]` 排序 |
| `scores[0]` 中 A-T 最高概率 | ~0.09 | ~0.09 |
| `generate()` 实际生成的 token | N/A（训练时不生成） | `▁H`（非 A-T！） |

---

## 五、结论与建议

### 5.1 为什么恒预测 C？

**训练失败是根本原因。**

- 模型没有收敛，`forward(label='C')` 的 loss 甚至低于真实答案。
- GNN + Projector 的 31.5M 参数在 3 epoch / 固定 1e-5 学习率下，没有学会把 graph 信息映射到正确的选项。
- `mean pool` 把 10 个子图压成 1 个 token，信息严重损失，导致 LLM 几乎看不到有效的 graph 信号。

**`inference()` 的解码逻辑放大了问题。**

- 模型实际生成的第一个 token 往往不是 A-T（如 `▁H`）。
- 但 `inference()` 不读实际生成文本，而是直接对 `scores[0]` 中 A-T 的概率排序。
- 由于模型训练失败，`scores[0]` 中 A-T 的排序总是 C 排第一，导致 `evaluate.py` 记录为 100% C。

### 5.2 修复优先级

| 优先级 | 问题 | 修复方案 |
|---|---|---|
| **P0** | 训练未收敛 | 调用 `adjust_learning_rate`（warmup + cosine decay）；增加 epoch 到 10+；添加 validation set 选 best ckpt |
| **P1** | `inference()` 解码逻辑 | 解码 `sequences` 取第一个字符；只有非 A-T 时才 fallback 到 `scores[0]` 排序 |
| **P2** | mean pool 信息损失 | 参考论文公式 8，把 10 个子图 token 拼接到 prompt 中，而不是 mean pool 成 1 个 |
| **P3** | train/eval prompt 不一致 | 统一 `### Instruction` 和 `## Instruction` |

### 5.3 关于 "正常成绩"

当前 baseline 的真实成绩确实是 **~5%**（Recall@1），这是未收敛模型的表现。论文中 GT 的 42.9% 需要在修复上述训练问题后才能达到。

如果需要在 `debug_pipeline.py` 中**快速拿到非恒定的、有意义的输出**，最立竿见影的修复是：
1. 修复 `inference()` 的解码逻辑（读实际生成文本）。
2. 但即使如此，准确率仍会在 ~5% 左右徘徊，因为模型本身没有学好。
