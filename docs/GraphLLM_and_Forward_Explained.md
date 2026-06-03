# GraphLLM 与 Forward 流程深度讲解

> 目标：零基础理解 GraphLLM 的作用、forward() 内部每一步在做什么、以及为什么模型会 collapse。

---

## 一、GraphLLM 是做什么的？

### 1.1 类比：一个"推荐考试"场景

想象你要做一个**电影推荐系统**。用户告诉你他看过 10 部电影，让你从 20 个选项里选出他最可能喜欢的那一部。

**传统做法**：看"哪些用户和他口味相似"（协同过滤）。

**K-RagRec 的做法**：
> "我不光看他和其他用户的相似度，我还要查**知识图谱（KG）**，看看这些电影和哪些演员、导演、类型有关系，把这些**知识**也告诉大模型，让它做更聪明的判断。"

**GraphLLM 就是干这件事的"总装车间"**。它把 4 个零件组装在一起，让 LLM 能"看到"知识图谱的信息。

---

### 1.2 四个零件

```
┌─────────────────────────────────────────────────────────────┐
│                    GraphLLM 组装车间                         │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐                        │
│  │ 零件1: GNN  │───→│ 零件2: Proj │   把子图变成 LLM 能   │
│  │ (读知识图谱)│    │ (维度翻译)  │   理解的向量          │
│  └─────────────┘    └─────────────┘                        │
│         ↓                                                   │
│  ┌─────────────┐    ┌─────────────┐                        │
│  │ 零件3: Tok  │───→│ 零件4: LLM  │   预测答案字母 A-T    │
│  │ (文本转数字)│    │ (67亿参数)  │                       │
│  └─────────────┘    └─────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 零件 1：graph_encoder（GNN）——"知识图谱阅读器"

```python
self.graph_encoder = load_gnn_model['gt'](...)
```

**作用**：把"子图"（一堆节点和边）压缩成向量。

**什么是子图？**
从知识图谱里检索出来的局部结构。比如：
```
《教父》 ──导演──→ 科波拉
   │
   └──主演──→ 马龙·白兰度
```
这是一个小网络，有 3 个节点和 2 条边。

**GNN 做什么？**
```
子图（节点+边） → GNN → 每个节点变成 1024 维向量
                  → 把所有节点取平均 → 子图级向量 [1, 1024]
```

**为什么取平均？**
因为 LLM 只能接收固定数量的输入。一个子图可能有 37 个节点，不能直接塞 37 个向量给 LLM。所以先用 GNN 把节点信息聚合到子图级别。

> ⚠️ **论文原意**：公式 (8) 说应该把 N 个子图向量**逐个拼接**到 Prompt 前面，让 LLM 看到 N 个独立的 "soft token"。但代码里做了 `mean()`，把 N 个子图压成了 1 个向量。这是已知的实现差异。

---

#### 零件 2：Projector（投影器）——"维度翻译器"

```python
self.projector = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.Sigmoid(),
    nn.Linear(2048, 4096),
)
```

**作用**：把 GNN 的 1024 维向量，翻译成 LLM 能理解的 4096 维向量。

**为什么需要它？**
- GNN 说"日语"（1024 维）
- LLM 说"英语"（4096 维）
- Projector 是"日英翻译器"

类比：你不能直接把一张 1024×1024 的图片塞给一个只接受 4096×4096 的打印机，需要先放大/转换。

---

#### 零件 3：Tokenizer（分词器）——"数字翻译官"

```python
self.tokenizer = AutoTokenizer.from_pretrained(...)
```

**作用**：把人类文字变成 LLM 认识的数字。

**例子**：
| 文本 | Token ID |
|---|---|
| `"R"` | `[390]` |
| `"Below is an instruction..."` | `[13866, 338, 385, ...]`（约 390 个数字） |
| `"<s>[INST]"` | `[1, 29961, 25580, 29962]` |

**为什么需要它？** LLM 只认识数字，不认识字母。

> ⚠️ **关键 Bug**：代码里用了 `BOS = '<s>[INST]'`，但加载的是 **BASE 模型**（`Llama-2-7b-hf`），不是 Chat 模型。BASE 模型不认识 `[INST]`，把它拆成了 3 个普通字符 token。这导致 LLM 完全无法理解指令格式，是 **collapse 的根因之一**。

---

#### 零件 4：LLM（大语言模型）——"大脑"

```python
self.model = AutoModelForCausalLM.from_pretrained(...)
```

**作用**：一个已经预训练好的、有 **67 亿参数**的"语言专家"。它看过互联网上的海量文本，懂得语言规律。

**关键设定**：`llm_frozen = True`
```python
for name, param in model.named_parameters():
    param.requires_grad = False   # 冻结！不训练！
```

**为什么冻结？**
1. 67 亿参数太大了，训练不起（也没有必要）
2. LLM 已经懂语言了，我们只需要教它"如何理解知识图谱的向量"
3. 这个"教学"任务由 GNN + Projector 完成

---

### 1.3 可训练参数分布

| 组件 | 参数量 | 是否训练 |
|---|---|---|
| LLM | 6,738M | ❌ 冻结 |
| GNN | ~31M | ✅ 训练 |
| Projector | ~0.5M | ✅ 训练 |
| **总计** | **~6,770M** | 只有 **31.5M** 在更新 |

形象比喻：
> LLM 是一个固定不动的"大学教授"。GNN + Projector 是两个"学生助理"，负责把 raw 的 KG 信息整理好，用教授能听懂的语言汇报。训练过程就是不断调整这两个助理的汇报方式，让教授更容易答对题。

---

## 二、sample dict 是做什么的？

`sample dict` 就是一个**快递包裹**，`train.py` 把数据打包好，送给 `model.forward()` 处理。

```python
sample = {
    'id': ['query1', 'query2'],           # 样本编号（没啥实际用）
    'graph': [                             # 每个样本检索到的子图列表
        [子图0_Data, 子图1_Data, ..., 子图9_Data],   # 样本 0
        [子图0_Data, 子图1_Data, ..., 子图9_Data],   # 样本 1
    ],
    'question': [                          # 完整的 Prompt 文本
        "Below is an instruction...选项A...选项T...",  # 样本 0
        "Below is an instruction...选项A...选项T...",  # 样本 1
    ],
    'label': ['R', 'M'],                   # 正确答案字母
}
```

你可以把它理解成**函数的输入参数**。`model.forward(sample)` 收到这个包裹后，拆开做以下事情：
- 用 `sample['question']` 构建 Prompt
- 用 `sample['graph']` 做 GNN 编码
- 用 `sample['label']` 计算 loss（看预测和真实答案差多远）

---

## 三、forward() 内部详解

### 3.1 输入：不是文本，是向量矩阵！

这是最难理解的一点。**forward() 拼接的不是文字，是高维向量。**

#### 第一步：把所有东西变成向量

```python
# 1. Prompt 文本 → token IDs → 向量矩阵
input_ids = [13866, 338, 385, ...]   # 390 个整数（token IDs）
inputs_embeds = word_embedding(input_ids)   # shape: [390, 4096]
# 每个 token ID 变成一个 4096 维的向量
# 所以是 390 行 × 4096 列的矩阵

# 2. 答案字母 "R" → 向量
label_input_ids = [390, 2]   # "R" 的 token ID + EOS 结束符
# 也变成向量

# 3. 子图 → GNN → Projector → 向量
sample_graph_embeds = [1, 4096]   # 1 行 × 4096 列
```

**word_embedding 是什么？**
它是 LLM 内部的一个查找表（Lookup Table）：
```
token ID 0   →  [0.01, -0.02, 0.05, ...]  (4096 维)
token ID 1   →  [-0.03, 0.08, 0.01, ...] (4096 维)
token ID 390 →  [0.12, -0.15, 0.03, ...] (4096 维)
```
每个整数都对应一个预训练好的高维向量。

---

### 3.2 拼接过程（cat）

```python
inputs_embeds = torch.cat(
    [bos_embeds, sample_graph_embeds, inputs_embeds],
    dim=0   # 按第 0 维（行）拼接
)
```

**按行拼接**，得到一个更大的矩阵：

```
┌──────────────────────────────────────────────────────────────────┐
│ 位置 0-3   │  BOS token 向量      │  shape: [4, 4096]          │
├──────────────────────────────────────────────────────────────────┤
│ 位置 4     │  Graph 向量          │  shape: [1, 4096]          │
│            │  （子图信息的综合表示）│  ← 这是知识图谱的输入！     │
├──────────────────────────────────────────────────────────────────┤
│ 位置 5-394 │  Prompt 文本向量     │  shape: [390, 4096]        │
│            │  （问题和选项）       │  ← "用户看过...选项A是..."  │
├──────────────────────────────────────────────────────────────────┤
│ 位置 395   │  [/INST] 向量       │  shape: [1, 4096]          │
├──────────────────────────────────────────────────────────────────┤
│ 位置 396   │  "R" 向量           │  shape: [1, 4096]          │
│            │  （正确答案）         │  ← 训练时希望模型输出这个   │
├──────────────────────────────────────────────────────────────────┤
│ 位置 397   │  EOS 向量           │  shape: [1, 4096]          │
│            │  （结束标记）         │                            │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                    总共 [402, 4096] 的矩阵
```

**这个 [402, 4096] 的矩阵就是真正输入 LLM 的东西。**

LLM 看到这个矩阵后，会尝试"预测下一个 token"。训练时，我们让它预测 `"R"` 这个位置，然后和真实的 `"R"` 比较。

---

### 3.3 为什么把子图压缩成 1 个向量也可以？

这是一个**设计选择**，不是必然的。论文和代码在这里有差异。

#### 方案 A（论文公式 8，理论上更好）

```
输入 = BOS(4) + [sub0] + [sub1] + ... + [sub9] + Prompt(390)
长度 = 4 + 10 + 390 = 404 个 token
```
保留 10 个子图向量，让 LLM 分别处理每个子图的信息。

#### 方案 B（代码实际做的）

```python
sample_graph_embeds = mean([sub0, sub1, ..., sub9])  # [1, 4096]
输入 = BOS(4) + [mean] + Prompt(390)
长度 = 4 + 1 + 390 = 395 个 token
```

**为什么取平均也能工作？**
- LLM 的强大之处在于它能从上下文推断信息
- 即使只给 1 个"综合向量"，LLM 也能从中提取有用的模式
- 但这确实**损失了细节**（比如哪个子图更重要），所以论文说拼接更好

**类比**：10 张照片压缩成 1 张平均图
- 你能看出大概是什么场景，但看不清每张照片的细节
- 有信息损失，但不至于完全失效

---

### 3.4 Loss 是怎么计算的？

```python
outputs = self.model(inputs_embeds=full_embeds, labels=label_input_ids)
loss = outputs.loss
```

#### 关键：`labels` 参数的设计

`label_input_ids` 不是完整的答案序列，而是长这样：

```python
[IGNORE_INDEX, IGNORE_INDEX, ..., IGNORE_INDEX, 390, 2]
  ↑ 前 397 个位置            ↑ 最后 2 个位置
    （不计算 loss）             （计算 loss）
```

**解释**：
- `IGNORE_INDEX = -100`，表示"这个位置不用管"
- 只有最后 2 个位置（`"R"` 和 `EOS`）会计算 loss
- 前面的 BOS、Graph、Prompt、[/INST] 都不算 loss

**为什么？**
- 训练目标是让模型学会"看到问题和选项后，输出正确答案"
- 我们不关心模型能不能复述 Prompt，只关心它能不能答对题

#### Loss 的数学本质

```
Loss = -log(模型预测 "R" 的概率)
```

- 如果模型很确定答案是 R（概率 99%），loss ≈ 0.01
- 如果模型随机猜（概率 1/20 = 5%），loss ≈ 3.0
- 如果模型完全跑偏（概率 0.01%），loss ≈ 9.2

**随机初始化模型的 loss ≈ 7.27**，说明模型比随机猜还糟，它严重偏向某个错误答案（collapse）。

---

### 3.5 反向传播更新谁的参数？

```python
loss.backward()   # 计算梯度：loss 对每个参数的导数
optimizer.step()  # 更新参数：沿梯度反方向走一小步
```

**更新谁？**

| 参数组 | 是否更新 | 原因 |
|---|---|---|
| **GNN (graph_encoder)** | ✅ 更新 | 需要学会从子图提取有用的特征 |
| **Projector** | ✅ 更新 | 需要学会把 GNN 输出翻译成 LLM 能懂的语言 |
| **LLM (67亿参数)** | ❌ 不更新 | `requires_grad = False`，冻结了 |

**所以每次反向传播，只更新 GNN + Projector 的 31.5M 参数。**

**形象比喻**：
- LLM 是一个固定不动的"大学教授"
- GNN + Projector 是两个"学生助理"
- 训练就是不断调整助理的汇报方式，让教授更容易答对题
- 教授本身不被改变（他已经是语言专家了）

---

## 四、query_text（Prompt）完整示例

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction: Given the user's watching history, selecting a film that is most likely to interest the user from the options. ### Watching history: "Bram Stoker's Dracula", "Crucible, The", "Sense and Sensibility", "Clockers", "Piano, The", "Trial, The", "Dark City", "Casablanca", "12 Angry Men", "Mariachi, El". ###Options: A: Star Trek III: The Search for Spock, B: Chairman of the Board, C: Up at the Villa, D: Great Race, The, E: Brazil, F: Glengarry Glen Ross, G: Brain That Wouldn't Die, The, H: View to a Kill, A, I: Paths of Glory, J: Mr. Nice Guy, K: Rocky, L: Zero Kelvin, M: Body Heat, N: Doom Generation, The, O: Rock, The, P: When Harry Met Sally..., Q: Absent Minded Professor, The, R: Kids, S: Bowfinger, T: When Night Is Falling. Select a movie from options A to T that the user is most likely to be interested in. Please answer "A" or "B" or "C" or "D" or "E" or "F" or "G" or "H" or "I" or "J" or "K" or "L" or "M" or "N" or "O" or "P" or "Q" or "R" or "S" or "T" only".
```

**长度**：约 1149 个字符，被 Tokenizer 压缩成约 390 个 token IDs。

---

## 五、目前已知的问题清单

| 优先级 | 问题 | 影响 | 修复方式 |
|---|---|---|---|
| **P0** | BASE 模型 + Chat 模板不匹配（`[INST]`） | LLM 看不懂指令，直接 collapse | 换 `Llama-2-7b-chat-hf` 或改模板 |
| **P1** | KG 节点 74.5% 是 Freebase MID，检索语义不匹配 | 检索子图完全无效 | 需要 `mapped_filtered_fb.txt` 重新生成 `0.pt` |
| **P2** | Mean pool 信息损失（论文说应拼接 N 个） | 性能上限被压低 | 改 `graph_llm.py:168` 的 `mean()` 为 `cat()` |

**建议修复顺序**：先 P0（模板），再 P1（KG 节点），最后 P2（mean pool）。
