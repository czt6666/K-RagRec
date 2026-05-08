# `methods/` —— 方法索引

本目录每个子目录对应一个方法，且彼此自包含。原始 K-RagRec 实现放在 `baseline/`，按**只读参考**对待——它是 ACL 2025 论文的可复现基线。其它每个目录都是 `baseline/` 的副本，只在需要的地方做了定向修改，这样我们随时可以与未改动的原版做对照。

## 命名约定

| 目录 | 用途 | 状态 |
|---|---|---|
| `baseline/` | 原始 K-RagRec 代码（ACL 2025 论文）。**禁止修改**。 | ✅ 已提交 |
| `h1_qformer/` | H1 —— 多 token Graph-Q-Former 投影器（替换 mean 池化） | ✅ 已实现 |
| `h2_gate/` | H2 —— 可学习自适应检索门控（替换流行度启发式） | ✅ 已实现 |
| `h3_pcst/` | H3 —— 多源 PCST + 路径感知检索 | ✅ 已实现 |
| `h4_temporal/` | H4 —— 时序 / 序列 Transformer 分支 | ✅ 已实现 |
| `h5_rel_gnn/` | H5 —— 关系类型感知 GNN 编码器（CompGCN / RGCN） | ✅ 已实现 |
| `k2_ragrec/` | 组合 H1 + H2 + H3 + H5 —— 论文最终方法 | ⏳ 计划中 |

各方案的优先级和动机详见 `paper/INNOVATIONS.md`。

## 新建方法的工作流

启动一个新方法（例如 `h1_qformer`）的步骤：

```powershell
# 1. 把 baseline 复制到一个新方法目录
Copy-Item -Recurse methods/baseline methods/h1_qformer

# 2. 在该方法目录下加一个 README，记录：
#    - 是从 baseline 哪个 commit 分叉的
#    - 改了哪些文件 / 哪些行
#    - 训练 + 评测怎么跑
New-Item methods/h1_qformer/README.md
```

## 怎么运行（PowerShell）

工作目录始终是仓库根目录（`K-ragrec/`）。数据集路径相对 cwd 解析；`PYTHONPATH` 控制 Python 在哪里找 `src/` 包。

```powershell
$env:PYTHONPATH = "methods/<方法名>"
python methods/<方法名>/train.py --args ...
python methods/<方法名>/evaluate.py --args ...
```

例如，跑 baseline：

```powershell
$env:PYTHONPATH = "methods/baseline"
python methods/baseline/train.py `
    --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m `
    --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5
```

> **不要**先 `cd` 进方法目录再运行。脚本里硬编码的 `dataset/ML1M/...`、`dataset/fb`、`output/...` 都是相对 cwd 解析的。

## 为什么用「方法即目录」而不是 git 分支

方法即目录的优势有二：

1. **能并排 diff**。任意两个方法之间可以 `Compare-Object`（或 `git diff --no-index`）一眼看到差异，不用切来切去。
2. **组合方法很轻松**。把 H1 + H2 + H3 合到 `k2_ragrec/` 时，直接复制 + 拼片段就行，不用解决跨分支 merge 冲突。

## 阅读顺序

动手改代码之前先读：

1. `paper/CODE_ANALYSIS.md` —— baseline 逐行审计（论文与代码差异、代码异味）
2. `paper/RELATED_WORKS.md` —— 46 篇相关论文调研，按 8 大类分组
3. `paper/INNOVATIONS.md` —— 排好优先级的创新清单 + sprint 计划
4. `paper/DATA_FORMAT.md` —— 数据集 + KG 格式逐文件审计（实现新方法前的契约文档）
