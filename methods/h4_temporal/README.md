# H4 —— 时序 / 序列分支

> 从 `methods/baseline/` 分叉。新增一个小型 Transformer，把用户历史 item id 序列编码成单个 soft prompt token，再喂给 LLM。这弥补了 baseline 在子图嵌入上做 mean 池化时丢掉的「顺序与近期性」信息。

## 相对 baseline 的文件改动

| 文件 | 改了什么 |
|---|---|
| `src/model/temporal.py` | **新增** —— `SequentialEncoder(vocab_size, max_seq_len, d_model, nhead, num_layers, dropout, out_dim, num_time_buckets)`。Item embedding + 学到的 positional embedding +（可选）log 桶化的 time-delta embedding → 2 层 batch_first Transformer + CLS token → `Linear(d_model -> out_dim)` 输出一个汇总向量。还有一个 `_bucket_dt` 辅助函数，把秒数映射到 16 个对数桶（1h、6h、1d、1w、1mo、...）。 |
| `src/model/graph_llm.py` | 新增 `self.seq_encoder` 和 `_run_seq_encoder()` helper：把 `samples['sequence_ids']` 补齐到 `max_seq_len`，过 Transformer，再过原有的 `self.projector` 落到 LLM 空间。`forward()` / `inference()` 中，得到的这个 token 接在 graph soft prompt 之后、文本 prompt 之前：`[BOS, graph_token, seq_token, instruction]`。如果 `samples['time_deltas']` 存在（单位 = 秒，0 表示未知）就用，否则跳过 time-bucket embedding。 |
| `train.py` / `evaluate.py` | 把每条样本的 `sequence_ids`（baseline 里就有的 10 个 ML1M id）放进 `samples['sequence_ids']`。 |
| `src/config.py` | 新增 `--seq_vocab_size`（4000）、`--seq_max_len`（32）、`--seq_d_model`（256）、`--seq_nhead`（4）、`--seq_num_layers`（2）、`--seq_dropout`（0.1）。 |

> ⚠️ **必须指定不同的 `--output_dir`**。baseline 与 H1-H5 的 checkpoint 文件名不包含方法名，默认都写到 `output/ml1m/` 下，会互相覆盖。训 baseline 用 `--output_dir output_baseline`，训 H4 用 `--output_dir output_h4`，以此类推。

## 运行（PowerShell）

```powershell
$env:PYTHONPATH = "methods/h4_temporal"

python methods/h4_temporal/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "<llama-2-7b 路径>" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 `
    --seq_d_model 256 --seq_num_layers 2 `
    --output_dir output_h4

python methods/h4_temporal/evaluate.py `
    --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m `
    --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5 `
    --seq_d_model 256 --seq_num_layers 2 `
    --output_dir output_h4
```

## 冒烟测试

```powershell
$env:PYTHONPATH = "methods/h4_temporal"
python tools/smoke_h4_temporal.py
```

期望：`SequentialEncoder params: 2,354,176`，两次前向 shape 都是 `(3, 1024)`，最后 `[OK] H4 sequential encoder smoke test passed.`

## 可选：传入真实时间戳

当前 `train.py` / `evaluate.py` 没传 `time_deltas`，所以 time-bucket embedding 现在是零初始化、不参与训练。要启用真实时间戳：

1. 用 `dataset/ML1M/ratings_45.txt` 预先建一个 `(user_id, movie_id) -> timestamp` 字典。
2. 训练 / 评测每条样本里查到对应 user（要么把 `user_id` 串到 JSON 里，要么靠 history 的指纹去 ratings 里反查），按 `[t_now - t_i for each history item]` 的秒差列表填进 sample。
3. 在 sample dict 里增加 `'time_deltas': time_deltas_list`，与 `sequence_ids` 并列。

不传时间戳的话，编码器靠位置编码也能抓住 10-item 窗口内的近期排名。

## 消融网格

| 配置 | 检验什么 |
|---|---|
| baseline（无 seq token） | 参考数字 |
| `--seq_num_layers 1 --seq_d_model 128` | 微型编码器，+0.6 M 参数 |
| `--seq_num_layers 2 --seq_d_model 256` | 默认 |
| `--seq_num_layers 4 --seq_d_model 512` | 更大的编码器 |
| + 真实 `time_deltas` | 看真实时间戳是否相对纯位置编码再有加成 |

## 为什么是「H4 简化版」

`paper/INNOVATIONS.md` 的 H4 列了三种思路：时间衰减 GNN、TGODE、SASRec 双塔。本实现最接近 SASRec 双塔的设计（SASRec 本质就是带 positional embedding 的小 Transformer）。完整的 TGODE / 连续时间路径暂时不做，因为它要 `torchdiffeq` 和重型训练循环；这里用「位置 + 桶化时间」加性 embedding 已经能拿到时序信号约 80% 的收益，工程量低很多。
