# K-RagRec（为创新实验重组的版本）

本仓库源自 ACL 2025 论文 *"Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation"* 的官方 PyTorch 实现，目前已被重新组织，用于承载在 K-RagRec 之上的后续创新方法。

原始代码原封不动地保留在 `methods/baseline/` 中，作为可复现的参考基线。每个新方法以同级子目录的形式加入 `methods/`（一个目录对应一个方法，从 baseline 复制后就地修改）。每个方法的入口和说明见 `methods/README.md`。

## 仓库结构

```
K-ragrec/
├── methods/
│   ├── README.md                # 方法索引 + 每个方法的运行手册
│   └── baseline/                # 原始 K-RagRec 代码（只读参考）
│       ├── train.py
│       ├── evaluate.py
│       ├── retrieve.py
│       ├── run.sh
│       └── src/
├── dataset/                     # 共享数据集（ML1M / ml-20m / book / fb）
├── paper/                       # 论文 PDF + 分析文档
│   ├── K-RagRec.pdf
│   ├── CODE_ANALYSIS.md         # 论文 vs 代码逐行对照
│   ├── RELATED_WORKS.md         # 46 篇相关论文调研
│   └── INNOVATIONS.md           # 高/中/低优先级创新点清单
├── pyproject.toml, uv.lock
└── README.md
```

## 环境

- Python==3.9
- numpy==1.23.4
- torch==2.4.1
- cuda==11.8.89
- transformers==4.45.2
- networkx==2.8.7
- peft==0.12.0

## 数据集

支持三个数据集：MovieLens-1M、MovieLens-20M、Amazon Book。仓库内仅 ML1M 已经处理好，其它两个仅提供原始数据。知识图谱用 Freebase 按数据集过滤而来。

ML1M + KG 处理后的下载链接：https://drive.google.com/file/d/1MlEPkRj47WrdXECUiz5D6Ie1oMv4hKC9/view

## 运行 baseline

> **要在服务器上复现？** 请看 `tools/server_checklist.md`，里面有完整的 Linux/bash 启动手册（uv 安装、依赖锁定、冒烟测试、tmux 长跑技巧、对照论文的目标数字）。
>
> 下面的 PowerShell 片段是给本地开发 / 冒烟检查用的。LLaMA-2-7B 的 fp16 在消费级 GPU 上塞不下，正式训练必须上服务器。

所有命令都在仓库根目录（`K-ragrec/`）下执行。代码里的 `dataset/` 路径是相对当前工作目录解析的，**不要 `cd` 进 `methods/baseline/`**。

`PYTHONPATH` 设为 `methods/baseline`，这样 `from src.model import ...` 才能正常解析。

### 仅检索冒烟测试（不加载 LLM，任何机器都能跑）

```powershell
$env:PYTHONPATH = "methods/baseline"
python tools/smoke_retrieval.py
```

期望最后一行：`[OK] Retrieval pipeline smoke test passed.`

### 训练

```powershell
$env:PYTHONPATH = "methods/baseline"
python methods/baseline/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "D:\hf_cache\hub\models--meta-llama--Llama-2-7b-hf\snapshots\01c7f73d771dfac7d292323805ebc428287df4f9" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt `
    --gnn_num_layers 4 `
    --sub_graph_numbers 3 `
    --reranking_numbers 5 `
    --adaptive_ratio 5
```

### 评测

```powershell
$env:PYTHONPATH = "methods/baseline"
python methods/baseline/evaluate.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "D:\hf_cache\hub\models--meta-llama--Llama-2-7b-hf\snapshots\01c7f73d771dfac7d292323805ebc428287df4f9" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt `
    --gnn_num_layers 4 `
    --sub_graph_numbers 3 `
    --reranking_numbers 5 `
    --adaptive_ratio 5
```

### 多 GPU

```powershell
$env:CUDA_VISIBLE_DEVICES = "0,1"
$env:PYTHONPATH = "methods/baseline"
python methods/baseline/train.py ...
```

## 运行其它方法

每个新方法目录的命令模式相同——把 `methods/baseline` 替换成 `methods/<方法名>` 即可（`PYTHONPATH` 和脚本路径都要换）。每个方法目录下的 README 有针对该方法的具体启动命令。

## 硬件

原论文用 2× NVIDIA A6000-48GB。如果只有单卡 GPU，方法目录下的 `src/model/` 里如果存在 `graph_llm_for_one_GPU` 就改用它。

## 引用

```
@article{wang2025knowledge,
    title={Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation},
    author={Wang, Shijie and Fan, Wenqi and Feng, Yue and Ma, Xinyu and Wang, Shuaiqiang and Yin, Dawei},
    journal={arXiv preprint arXiv:2501.02226},
    year={2025}
}
```
