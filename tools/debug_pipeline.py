"""
K-RagRec 端到端调试脚本
逐层检查：数据 → 检索 → GNN → Projector → LLM logits
找出 baseline 只有 ~5% Recall@1 的根因

用法（在仓库根目录）：
    $env:PYTHONPATH = "methods/baseline"
    python tools/debug_pipeline.py
"""
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

# 必须设 PYTHONPATH=methods/baseline
from src.config import parse_args_llama
from src.model import load_model, llama_model_path
from src.utils.seed import seed_everything
from src.utils.ckpt import _reload_best_model
from retrieve import GraphRetrieval

# ============================================================================
# 1. 数据完整性检查
# ============================================================================
def check_data():
    print("=" * 60)
    print("[1/8] 数据完整性检查")
    print("=" * 60)

    path = "dataset/ML1M/10000_data_id_20.json"
    assert os.path.exists(path), f"缺失: {path}"

    with open(path, 'r') as f:
        data = json.load(f)

    print(f"  总样本数: {len(data)}")
    train = data[:9000]
    test = data[9000:10000]
    print(f"  训练集: {len(train)}, 测试集: {len(test)}")

    # 检查字段
    required = ['input', 'questions', 'output', 'sequence_ids']
    missing = [k for k in required if k not in data[0]]
    if missing:
        print(f"  ❌ 缺失字段: {missing}")
    else:
        print(f"  ✅ 字段齐全")

    # output 分布
    outputs = [d['output'] for d in data]
    from collections import Counter
    c = Counter(outputs)
    print(f"  答案分布 (A-T): {dict(c)}")

    # 检查 sequence_ids
    seq_lens = [len(json.loads(d.get('sequence_ids', '[]'))) for d in data]
    print(f"  sequence_ids 长度: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.1f}")

    # 检查 questions 是否固定 20 个选项
    q_counts = [len(d['questions'].split('\n')) for d in data]
    print(f"  options 行数: min={min(q_counts)}, max={max(q_counts)} (应为 20)")

    return train, test


# ============================================================================
# 2. 检索模块检查
# ============================================================================
def check_retrieval(train_data, test_data):
    print("\n" + "=" * 60)
    print("[2/8] 检索模块检查")
    print("=" * 60)

    retrieval = GraphRetrieval(model_name='sbert', path='dataset/fb')
    print(f"  KG 节点数: {retrieval.G.num_nodes}")
    print(f"  KG 边数: {retrieval.G.num_edges}")
    print(f"  G1 (1-hop) 节点数: {retrieval.G1.num_nodes}")

    sample = test_data[0]
    input_text = sample['input']
    question = sample['questions']
    sequence_id = json.loads(sample.get('sequence_ids', '[]'))
    output_letter = sample['output']

    print(f"\n  样例 input: {input_text[:80]}...")
    print(f"  样例 output: {output_letter}")
    print(f"  样例 sequence_ids: {sequence_id}")

    # 检查 whether_retrieval 的 train/eval 差异
    print("\n  [Critical] whether_retrieval 行为差异:")
    adaptive_ratio = 5

    # train 行为
    train_ret = retrieval.whether_retrieval(sequence_id, adaptive_ratio * len(sequence_id))
    print(f"    train mode: whether_retrieval(seq, {adaptive_ratio}*{len(sequence_id)}={adaptive_ratio*len(sequence_id)}) -> {len(train_ret)} items")

    # eval 行为 (已修复，与 train 一致)
    eval_ret = retrieval.whether_retrieval(sequence_id, adaptive_ratio * len(sequence_id))
    print(f"    eval mode:  whether_retrieval(seq, {adaptive_ratio}*{len(sequence_id)}={adaptive_ratio*len(sequence_id)}) -> {len(eval_ret)} items")
    print(f"    ✅ eval 行为已与 train 一致")

    # 检索子图
    graphs = retrieval.retrieval_topk(input_text, train_ret, topk_nodes=3, topk_rerank_nodes=5)
    print(f"\n  检索到子图数: {len(graphs)}")
    for i, g in enumerate(graphs):
        print(f"    子图 {i}: {g.num_nodes} nodes, {g.num_edges} edges")

    return retrieval


# ============================================================================
# 3. 模型加载检查
# ============================================================================
def check_model_load(args):
    print("\n" + "=" * 60)
    print("[3/8] 模型加载检查")
    print("=" * 60)

    print(f"  llm_model_path: {args.llm_model_path}")
    assert os.path.exists(args.llm_model_path), f"LLM 路径不存在: {args.llm_model_path}"

    model = load_model[args.model_name](args=args)
    model.eval()

    # 检查 LLM 是否真的加载了 7B
    llm_params = sum(p.numel() for p in model.model.parameters())
    print(f"  LLM 总参数量: {llm_params / 1e6:.1f}M (期望 ~6,738M)")

    # 检查可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M")

    # 检查各组件
    print(f"  graph_encoder: {model.graph_encoder}")
    print(f"  projector: {model.projector}")

    return model


# ============================================================================
# 4. 单样本前向中间态检查
# ============================================================================
def check_forward_pipeline(model, retrieval, test_data, args):
    print("\n" + "=" * 60)
    print("[4/8] 单样本前向中间态检查")
    print("=" * 60)

    sample = test_data[0]
    input_text = sample['input']
    question = sample['questions']
    sequence_id = json.loads(sample.get('sequence_ids', '[]'))
    target = sample['output']

    # 构造样本
    retrieve_movies_list = retrieval.whether_retrieval(sequence_id, args.adaptive_ratio * len(sequence_id))
    graphs = retrieval.retrieval_topk(input_text, retrieve_movies_list, args.sub_graph_numbers, args.reranking_numbers)

    query_text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
## Instruction: Given the user's watching history, selecting a film that is most likely to interest the user from the options. ### Watching history: {input_text}. ###Options: {question}. Select a movie from options A to T that the user is most likely to be interested in. Please answer "A" or "B" or "C" or "D" or "E" or "F" or "G" or "H" or "I" or "J" or "K" or "L" or "M" or "N" or "O" or "P" or "Q" or "R" or "S" or "T" only"."""

    sample_dict = {
        'id': ['query1'],
        'graph': [graphs],
        'question': [query_text],
        'label': ['']
    }

    # 4.1 检查 tokenizer 输出
    tokens = model.tokenizer(query_text, return_tensors='pt')
    print(f"  Tokenizer: input_ids shape={tokens.input_ids.shape}")
    print(f"  Tokenizer: decoded前30token: {model.tokenizer.decode(tokens.input_ids[0][:30])}")

    # 4.2 检查 graph 编码
    with torch.no_grad():
        device = next(model.parameters()).device
        graph_embeds_list = []
        for graphs in sample_dict['graph']:
            from torch_geometric.data import Batch
            graphs = Batch.from_data_list(graphs).to(device)
            n_embeds, _ = model.graph_encoder(graphs.x, graphs.edge_index, graphs.edge_attr)
            from torch_scatter import scatter
            g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
            graph_embeds_list.append(g_embeds)

        print(f"\n  GNN 输出:")
        print(f"    子图嵌入 shape: {graph_embeds_list[0].shape} (应为 [N_subgraphs, 1024])")
        print(f"    子图嵌入均值: {graph_embeds_list[0].mean().item():.4f}, 标准差: {graph_embeds_list[0].std().item():.4f}")

        # 4.3 检查 projector
        projected = [model.projector(g) for g in graph_embeds_list]
        print(f"\n  Projector 输出:")
        print(f"    投影后 shape: {projected[0].shape} (应为 [N_subgraphs, 4096])")
        print(f"    投影后均值: {projected[0].mean().item():.4f}, 标准差: {projected[0].std().item():.4f}")

        # 4.4 检查 mean pool (关键 bug 点)
        sample_graph_embeds = torch.cat([p.unsqueeze(0) for p in projected[0]], dim=0)
        sample_graph_embeds = sample_graph_embeds.mean(dim=0, keepdim=True)
        print(f"\n  Mean Pool 后:")
        print(f"    shape: {sample_graph_embeds.shape} (论文应为 [N, 4096]，代码是 [1, 4096])")

        # 4.5 检查 LLM inputs_embeds
        inputs_embeds = model.model.get_input_embeddings()(tokens.input_ids.to(device))
        bos_embeds = model.model.get_input_embeddings()(torch.tensor([[model.tokenizer.bos_token_id]], device=device))
        full_embeds = torch.cat([bos_embeds, sample_graph_embeds.unsqueeze(0), inputs_embeds], dim=1)
        print(f"\n  LLM 输入拼接:")
        print(f"    BOS embed shape: {bos_embeds.shape}")
        print(f"    graph embed shape: {sample_graph_embeds.unsqueeze(0).shape}")
        print(f"    text embed shape: {inputs_embeds.shape}")
        print(f"    拼接后 shape: {full_embeds.shape}")


# ============================================================================
# 5. 推理 logits 分布检查
# ============================================================================
def check_inference_distribution(model, retrieval, test_data, args, n_samples=20):
    print("\n" + "=" * 60)
    print("[5/8] 推理 logits 分布检查 (取前 {} 条测试样本)".format(n_samples))
    print("=" * 60)

    # 准备 option_indices (A-T)
    option_tokens = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
    option_indices = [model.tokenizer.convert_tokens_to_ids(tok) for tok in option_tokens]
    print(f"  A-T token IDs: {option_indices}")

    correct = 0
    all_max_probs = []

    for idx in range(min(n_samples, len(test_data))):
        sample = test_data[idx]
        input_text = sample['input']
        question = sample['questions']
        sequence_id = json.loads(sample.get('sequence_ids', '[]'))
        gold_letter = sample['output']
        gold_idx = ord(gold_letter) - ord('A')

        retrieve_movies_list = retrieval.whether_retrieval(sequence_id, args.adaptive_ratio * len(sequence_id))
        graphs = retrieval.retrieval_topk(input_text, retrieve_movies_list, args.sub_graph_numbers, args.reranking_numbers)

        query_text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
## Instruction: Given the user's watching history, selecting a film that is most likely to interest the user from the options. ### Watching history: {input_text}. ###Options: {question}. Select a movie from options A to T that the user is most likely to be interested in. Please answer "A" or "B" or "C" or "D" or "E" or "F" or "G" or "H" or "I" or "J" or "K" or "L" or "M" or "N" or "O" or "P" or "Q" or "R" or "S" or "T" only"."""

        sample_dict = {'id': ['q'], 'graph': [graphs], 'question': [query_text], 'label': ['']}

        with torch.no_grad():
            output_ids = model.inference(sample_dict)

        pred_idx = output_ids[0][0].item() if output_ids.numel() > 0 else -1
        pred_letter = chr(ord('A') + pred_idx) if 0 <= pred_idx < 20 else '?'
        all_max_probs.append(pred_idx)

        is_correct = (pred_idx == gold_idx)
        correct += is_correct

        status = "✅" if is_correct else "❌"
        print(f"  {status} sample {idx}: gold={gold_letter}({gold_idx}), pred={pred_letter}({pred_idx})")

        # 打印第一个样本的详细中间结果（带文字）
        if idx == 0:
            print(f"\n{'='*60}")
            print("  [详细样例]")
            print(f"{'='*60}")
            print(f"  观看历史 (input_text): {input_text}")
            print(f"  选项 (questions): {question}")
            print(f"  真实答案: {gold_letter}")
            print(f"  sequence_ids: {sequence_id}")
            print(f"  retrieve_movies_list (ID): {retrieve_movies_list}")
            print(f"  retrieve_movies_list (名称): {[retrieval.movie_id_to_name.get(mid, 'UNKNOWN') for mid in retrieve_movies_list]}")
            print(f"  子图数: {len(graphs)}")
            for i, g in enumerate(graphs):
                print(f"    子图 {i}: {g.num_nodes} nodes, {g.num_edges} edges")
                try:
                    node_ids = g.node_ids.tolist() if hasattr(g, 'node_ids') else []
                    node_names = retrieval.retrieval_node_texts(node_ids)
                    print(f"      节点名称: {node_names[:5]}")
                except Exception as e:
                    print(f"      获取节点名称失败: {e}")
            print(f"\n  完整 Prompt (query_text):\n{query_text}")
            print(f"\n  模型原始输出 token IDs: {output_ids.tolist()}")
            print(f"  预测答案: {pred_letter}")
            print(f"{'='*60}\n")

    print(f"\n  前 {n_samples} 条命中率: {correct}/{n_samples} = {correct/n_samples:.1%}")
    print(f"  预测分布: {dict(zip(*np.unique(all_max_probs, return_counts=True)))}")

    if correct == 0:
        print("\n  ⚠️  前20条全错！说明模型完全没有学到任务，或者检索/输入构建有严重问题。")
    elif correct / n_samples < 0.1:
        print("\n  ⚠️  命中率 < 10%，接近随机（5%），模型几乎没收敛。")


# ============================================================================
# 6. 检查 train/eval 检索差异的影响
# ============================================================================
def check_train_eval_mismatch(retrieval, test_data):
    print("\n" + "=" * 60)
    print("[6/8] Train/Eval 检索差异量化")
    print("=" * 60)

    adaptive_ratio = 5
    diffs = []

    for i in range(min(50, len(test_data))):
        seq = json.loads(test_data[i].get('sequence_ids', '[]'))

        # train 方式
        train_ret = set(retrieval.whether_retrieval(seq, adaptive_ratio * len(seq)))
        # eval 方式 (已修复，与 train 一致)
        eval_ret = set(retrieval.whether_retrieval(seq, adaptive_ratio * len(seq)))

        diff = len(train_ret.symmetric_difference(eval_ret))
        diffs.append(diff)

    print(f"  检查前 50 条样本:")
    print(f"    平均对称差: {np.mean(diffs):.1f} items")
    print(f"    最大对称差: {max(diffs)} items")
    print(f"    完全一致数: {sum(1 for d in diffs if d == 0)}/50")

    if np.mean(diffs) > 0:
        print(f"\n  ⚠️  训练时检索的 item 集合 与 评估时检索的 item 集合 不一致！")
        print(f"      这意味着模型在训练时看到的子图 和 评估时看到的子图 来自不同的 movie anchor，")
        print(f"      分布偏移导致评估性能暴跌。")
    else:
        print(f"\n  ✅  训练与评估检索行为完全一致，P0 bug 已修复。")


# ============================================================================
# 7. 检查随机基线
# ============================================================================
def check_random_baseline(test_data):
    print("\n" + "=" * 60)
    print("[7/8] 随机基线对比")
    print("=" * 60)

    gold = [ord(d['output']) - ord('A') for d in test_data]
    random_pred = [np.random.randint(0, 20) for _ in gold]

    correct = sum(1 for g, p in zip(gold, random_pred) if g == p)
    print(f"  纯随机猜测: {correct}/{len(gold)} = {correct/len(gold):.1%}")
    print(f"  你的 baseline: ~4.9%")
    print(f"  差距: {'很小（接近随机）' if correct/len(gold) > 0.04 else '有区别'}")


# ============================================================================
# 8. 综合诊断报告
# ============================================================================
def final_diagnosis():
    print("\n" + "=" * 60)
    print("[8/8] 综合诊断")
    print("=" * 60)
    print("""
常见根因速查表（按优先级排序）：

P0 - 训练/评估分布不一致 (已确认)
    evaluate.py:64  whether_retrieval(adaptive_ratio*sequence_id, 5)
    这里 sequence_id 是 list，int*list = 列表重复，与 train.py:62 行为完全不同。
    修复：把 evaluate.py:64 改成和 train.py 一致。

P1 - 模型没加载对 / LLM 不是 7B
    检查 [3/8] 的参数量是否为 ~6.7B。
    如果只有几百 M，说明加载的是本地随机初始化或路径错误。

P2 - 训练 epoch 不足 / loss 未收敛
    3 epoch 从 8.1 → 1.5，1.5 对 20 类来说仅比随机好一倍。
    建议：加大到 10 epoch，或加载训好的 ckpt 继续。

P3 - 学习率固定 1e-5 偏低
    adjust_learning_rate 定义了但 train.py 没调用。
    建议：加上 warmup + cosine decay。

P4 - 没有 validation set 选 best
    train.py 每 epoch 末直接覆盖 best ckpt。
    你评估的可能是最后一 epoch（可能过拟合或欠拟合）。

P5 - graph_llm.py mean pool 信息损失
    论文公式8是拼接 N 个子图 token，代码是 mean → 1 个 token。
    这是设计取舍，不直接导致 4.9%，但上限被压低。
""")


def main():
    args = parse_args_llama()
    seed_everything(seed=args.seed)

    # 覆盖路径检查
    args.llm_model_path = llama_model_path[args.llm_model_name]
    print(f"命令行解析后的 llm_model_path: {args.llm_model_path}")

    train_data, test_data = check_data()
    retrieval = check_retrieval(train_data, test_data)
    model = check_model_load(args)

    # 加载训好的 checkpoint
    args.output_dir = 'original_verify/output'
    try:
        model = _reload_best_model(model, args)
        print("  ✅ Checkpoint 加载成功")
    except Exception as e:
        print(f"  ⚠️ Checkpoint 加载失败: {e}")

    check_forward_pipeline(model, retrieval, test_data, args)
    check_inference_distribution(model, retrieval, test_data, args, n_samples=20)
    check_train_eval_mismatch(retrieval, test_data)
    check_random_baseline(test_data)
    final_diagnosis()

    print("\n" + "=" * 60)
    print("调试完成。根据上面的 ❌/⚠️ 定位问题，优先修 P0。")
    print("=" * 60)


if __name__ == "__main__":
    main()
