# 任务总结：Baseline Collapse 根因诊断（2026-06-04）

## 任务目标

诊断 K-RagRec baseline 在服务器上训练后始终输出同一个答案（Recall@1 ~5%，接近随机）的根因。

---

## 关键发现

### P0 - 已修复：Train/Eval 检索分布不一致

- **位置**：`methods/baseline/evaluate.py:64`
- **问题**：`whether_retrieval(args.adaptive_ratio*sequence_id, 5)` 中 `sequence_id` 是 list，`int*list` 导致列表重复，与 `train.py:62` 行为完全不同
- **修复**：`tools/fix_baseline_bugs.py` 已修复
- **状态**：用户已在服务器上修复

### P1 - 核心根因：BASE 模型 + Chat 模板不匹配

- **位置**：`methods/baseline/src/model/graph_llm.py:19-21`
- **问题**：代码加载 `Llama-2-7b-hf`（BASE 模型），但硬编码 `BOS='<s>[INST]'` 和 `EOS_USER='[/INST]'`。BASE 模型的 tokenizer 把 `[INST]` 拆成 3 个普通 token（`[`, `INST`, `]`），模型完全无法理解指令格式
- **影响**：无论输入什么，模型都 collapse 到固定字母
- **修复**：`tools/fix_chat_template.py` 提供方案 B（改模板为纯文本）；更推荐方案 A（换 `Llama-2-7b-chat-hf`）
- **状态**：用户已在服务器上修复

### P2 - 数据问题：KG 节点 74.5% 为 Freebase MID

- **发现**：`dataset/fb/nodes/all_nodes.csv` 中 10,923 / 14,669 节点（74.5%）的 `node_attr` 是 `/m/xxxxx` 格式的 Freebase MID，而非人类可读名称
- **根因**：`index_KG.py` 需要 `mapped_filtered_fb.txt`（MID→名称的三元组文件），但发布的数据集中未包含。本地 fallback 到 `filtered_full_fb.txt`（MID 版本）生成了错误的 `all_nodes.csv`
- **关键验证**：`0.pt` 的余弦相似度测试显示 `0.pt` **不是**基于 MID 的（`0.pt` vs SBERT(MID) 相似度仅 0.05-0.15），证明原始的 `0.pt` 是作者预计算好的**名称 embedding**
- **影响训练/推理？**：**不影响**。核心路径 `retrieve.py` 读 `self.G.x`（来自 `0.pt`），不读 `all_nodes.csv`。`all_nodes.csv` 只影响可视化
- **状态**：`0.pt` 已恢复为正确版本，已上传到服务器

### P3 - 代码 vs 论文差异：Mean Pool 信息损失

- **位置**：`methods/baseline/src/model/graph_llm.py:168`
- **问题**：论文公式 8 要求拼接 N 个子图 soft token，代码做了 `mean(dim=0)` 压成 1 个
- **影响**：性能上限被压低，但不导致 collapse

---

## 已创建的文档

| 文档 | 内容 |
|---|---|
| `docs/GraphLLM_and_Forward_Explained.md` | GraphLLM 零基础讲解 + forward() 内部逐层拆解 |
| `docs/0pt_generation_pipeline.md` | `0.pt` 生成流程、MID→名称映射链、谁映射谁 |
| `docs/Freebase_Link_Explanation.md` | Freebase 开发者页面说明 |
| `docs/TASK_SUMMARY_20250604.md` | 本文件 |

---

## 已修复/保留的工具

| 工具 | 用途 |
|---|---|
| `tools/debug_pipeline.py` | 端到端调试脚本（8 阶段检查） |
| `tools/fix_baseline_bugs.py` | 一键修复 P0 + 学习率 + best ckpt |
| `tools/fix_chat_template.py` | 修复 BASE 模型 Chat 模板不匹配 |
| `tools/diff_official.py` | 对比本地代码与官方 GitHub 版本 |

---

## 下一步行动

1. **服务器上确认 `0.pt` 已正确上传**
2. **重新训练 baseline**，观察：
   - loss 是否从 ~7 降到 ~1 以下
   - evaluate 时输出是否开始多样化（不再固定到同一个字母）
3. **如果训练正常**，warm-start H1（Graph-Q-Former）从 baseline ckpt
4. **如果仍 collapse**，继续排查：学习率、GNN 输出、Projector 初始化

---

## 验证命令速查

```bash
# 服务器上验证 0.pt 没有被覆盖
ls -lh dataset/fb/graphs/
md5sum dataset/fb/graphs/0.pt

# 本地验证检索是否有效
$env:PYTHONPATH = "methods/baseline"
python tools/smoke_retrieval.py
```
