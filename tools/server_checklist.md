# 服务器部署 + 跑通 baseline 清单

> 本地开发机（Windows + 8GB GPU）跑不动 LLaMA-2-7B，正式训练得放服务器上。
> 本文是上服务器之后的端到端 runbook。命令默认 **Linux + bash**。

---

## 0. 硬件清单

| 项 | 最低 | 推荐 | 原因 |
|---|---|---|---|
| GPU | 1× 24 GB（A10 / RTX 3090 / A5000） | 2× 48 GB（A6000，论文配置） | LLaMA-2-7B fp16 ≈ 14 GB + GNN + projector + 激活 |
| CUDA 驱动 | 11.8+ | 12.x | torch 2.4.1 默认带 cu118 |
| RAM | 32 GB | 64 GB | 模型加载峰值约 28 GB（分片之前） |
| 磁盘 | 60 GB | 100 GB | 数据 1 GB + KG 1 GB + LLaMA 13 GB + ckpt |
| 网络 | 能访问 HuggingFace | — | 拉模型权重用 |

如果服务器有多张 GPU，代码里 `device_map="auto"` 会自动分片（`graph_llm.py:30-44`）。

---

## 1. 拉代码 + 数据 + LLM 权重

```bash
# 代码
git clone <你的仓库 URL> K-ragrec
cd K-ragrec

# 数据集 —— ML1M + 处理过的 Freebase KG
#   下载链接: https://drive.google.com/file/d/1MlEPkRj47WrdXECUiz5D6Ie1oMv4hKC9/view
#   解压到 dataset/ ，最后目录长这样：
#     dataset/ML1M/{10000_data_id_20.json, ratings_45.txt, movies_id_name.txt, ml1m_raw/}
#     dataset/fb/graphs/{0.pt, layer2_embeddings_W.pt, layer3_embeddings_W.pt}
#     dataset/fb/{filtered_full_fb.txt, fb_entity_names.tsv, ...}

# 解压后验证一下
ls dataset/ML1M/10000_data_id_20.json dataset/fb/graphs/0.pt

# LLaMA-2-7B 权重（gated；需要 HF 帐号有 Llama-2 访问权限）
pip install -U "huggingface_hub[cli]"
huggingface-cli login          # 粘贴 token
huggingface-cli download meta-llama/Llama-2-7b-hf \
    --local-dir ~/hf_cache/Llama-2-7b-hf \
    --local-dir-use-symlinks False
# 记住这个路径，下面 --llm_model_path 会用到
```

---

## 2. Python + 依赖

仓库 `pyproject.toml` 锁了 torch 2.5.1 + cu121（与作者机器上预装的版本一致）。下面两条路二选一。

### A. uv（推荐，与 uv.lock 对齐）

```bash
# 装 uv（一行）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env  # 或重启 shell

# 进项目目录后，uv 会按 pyproject 自动建 .venv（默认 Python 3.10+）
uv sync

# torch-scatter 单独装（无 PyPI wheel，需 torch 已就绪，所以放 uv sync 之后）
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

source .venv/bin/activate
```

### B. 纯 pip（系统 Python 已经装好 torch 时）

如果你已经像下面这样把 torch 装在系统 Python 里：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

那就不用 uv，直接把剩下的依赖装进同一个解释器：

```bash
pip install torch-geometric==2.6.1 transformers==4.45.2 peft==0.12.0 \
    accelerate networkx==2.8.7 sentencepiece huggingface-hub \
    pcst-fast gensim pandas scikit-learn wandb
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

> 如果 `torch-scatter` 编译失败：确认 torch 已经能 `import torch` 跑通，CUDA 主版本对得上 (cu121)，再从上面的预编译 wheel index 装。

---

## 3. 验证依赖

```bash
export PYTHONPATH=methods/baseline

python - <<'PY'
import importlib
for m in ["torch","torch_geometric","torch_scatter","transformers","peft",
         "networkx","pandas","sklearn","pcst_fast","gensim","sentencepiece"]:
    mod = importlib.import_module(m)
    print(f"OK  {m:25s} {getattr(mod, '__version__', '?')}")

import torch
print("CUDA available:", torch.cuda.is_available(),
      "device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory/1024**3:.1f} GiB)")
PY
```

期望：每个模块都打印 `OK`；`CUDA available: True`；至少列出一张 GPU。

---

## 4. 仅检索冒烟测试（不加载 LLM）

验证 SBERT + KG 各层 + PCST 链路，避免在 7B 模型上烧 GPU 才发现链路有问题。

```bash
PYTHONPATH=methods/baseline python tools/smoke_retrieval.py
```

期望最后一行：`[OK] Retrieval pipeline smoke test passed.`

如果失败，问题就在数据 / SBERT / PyG 的接线上，先修这块再继续。

---

## 5. 训练 baseline

```bash
export PYTHONPATH=methods/baseline
export CUDA_VISIBLE_DEVICES=0,1   # 按可用 GPU 调整

# 在 tmux 里跑，断开 SSH 也不会被杀
tmux new -s krag-train

python methods/baseline/train.py \
    --model_name graph_llm \
    --llm_model_name 7b \
    --llm_model_path "$HOME/hf_cache/Llama-2-7b-hf" \
    --llm_frozen True \
    --dataset ml1m \
    --batch_size 5 \
    --gnn_model_name gt \
    --gnn_num_layers 4 \
    --sub_graph_numbers 3 \
    --reranking_numbers 5 \
    --adaptive_ratio 5 \
    2>&1 | tee output/train_baseline.log

# 离开会话: Ctrl+B 然后 D；回来再接: tmux attach -t krag-train
```

**会写出什么：**
- 检查点：`output/ml1m/model_name_graph_llm_..._checkpoint_best.pth`（每个 epoch 都覆写一次—— `train.py:90`）。
- 没有自动验证集划分。Best == 最新一次。

**预期耗时：**
- 每条样本 ~2s 检索（无缓存）+ LLM forward
- 9000 训练样本 × 3 epoch ≈ 15-20 小时（2× A6000）

**常见坑：**
- cwd 必须是仓库根，不能是 `methods/baseline/` —— `train.py:31` 是相对 cwd 读 `dataset/ML1M/...` 的。
- 加载就 OOM：试试 `CUDA_VISIBLE_DEVICES=0`（单卡），让 `device_map="auto"` 在 CPU/GPU 之间分片；或者 `--batch_size 2`。
- `train.py:43` 用 19 层三元嵌套把 A-T 映射成 0-19 —— 能跑但慢，先别管。

---

## 6. 评测对照论文表 1

```bash
export PYTHONPATH=methods/baseline
python methods/baseline/evaluate.py \
    --model_name graph_llm \
    --llm_model_name 7b \
    --llm_model_path "$HOME/hf_cache/Llama-2-7b-hf" \
    --llm_frozen True \
    --dataset ml1m \
    --batch_size 5 \
    --gnn_model_name gt \
    --gnn_num_layers 4 \
    --sub_graph_numbers 3 \
    --reranking_numbers 5 \
    --adaptive_ratio 5 \
    2>&1 | tee output/eval_baseline.log
```

**目标（论文表 1，ML1M，LLaMA-2 Frozen LLM w/ PT 那一行）：**
- ACC ≈ 0.435
- Recall@3 ≈ 0.725
- Recall@5 ≈ 0.831

`evaluate.py:107` 只打了 `Recall@1/3/5/10`。**先把 Recall@1 当 ACC 看**（top-1 命中率）。要严格对应 ACC，得自己解析打印的 `pred` 列表与黄金标签 letter 对照——具体差距在 `paper/CODE_ANALYSIS.md` §5 第 8 行有说明。

---

## 7. 已知偏差（**复现 baseline 时不要修**）

| 位置 | 现象 | 处理 |
|---|---|---|
| `evaluate.py:64` | `whether_retrieval(adaptive_ratio*sequence_id, 5)` —— `int*list` 是 list 重复，与 `train.py:62` 行为不一致 | 不动；这就是论文报告时的状态 |
| `graph_llm.py:168/227` | N 个检索子图被 `mean` 池化为 1 个 soft token（论文公式 8 是拼接） | 不动；H1 会修这个 |
| `train.py:90` | best ckpt = 最后一个 epoch，没有验证集 | 不动；论文如此 |
| `train.py` | `adjust_learning_rate` 定义了但从来没被调 | 不动；论文如此 |
| `retrieve.py` | `index_KG.py` 生成的 `layer3_embeddings_W.pt` 从未被加载 | 不动；H3 会用 |

完整审计在 `paper/CODE_ANALYSIS.md`。

---

## 8. baseline 跑通之后

1. 把评测日志和 ckpt 加日期戳归档：
   ```bash
   mv output/ml1m/model_name_graph_llm_..._checkpoint_best.pth \
      output/ml1m/baseline_$(date +%Y%m%d).pth
   cp output/eval_baseline.log output/eval_baseline_$(date +%Y%m%d).log
   ```
2. 把数字写进 `paper/CODE_ANALYSIS.md` §6，作为后续创新的 reference point。
3. 进入 H1（`methods/h1_qformer/`）—— 见 `paper/INNOVATIONS.md`。
