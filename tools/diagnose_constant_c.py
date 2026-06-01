"""
排查模型为什么恒预测 C。
核心思路：对同一个样本，同时用 forward()（训练模式）和 inference()（推理模式）跑一遍，
对比两者的输入和输出，判断是训练问题还是推理代码问题。
"""
import os
import sys
import json
import torch

sys.path.insert(0, "methods/baseline")
from src.config import parse_args_llama
from src.model import load_model, llama_model_path
from src.utils.seed import seed_everything
from src.utils.ckpt import _reload_best_model
from retrieve import GraphRetrieval


def main():
    args = parse_args_llama()
    seed_everything(seed=args.seed)
    args.llm_model_path = llama_model_path[args.llm_model_name]
    args.output_dir = 'original_verify/output'

    # 加载模型 + checkpoint
    model = load_model[args.model_name](args=args)
    try:
        model = _reload_best_model(model, args)
        print("✅ Checkpoint loaded")
    except Exception as e:
        print(f"⚠️ Checkpoint load failed: {e}")
        return

    model.eval()
    retrieval = GraphRetrieval(model_name='sbert', path='dataset/fb')

    # 取第一个测试样本
    with open("dataset/ML1M/10000_data_id_20.json", 'r') as f:
        data = json.load(f)[9000]

    input_text = data['input']
    question = data['questions']
    sequence_id = json.loads(data.get('sequence_ids', '[]'))
    gold_letter = data['output']

    retrieve_movies_list = retrieval.whether_retrieval(sequence_id, args.adaptive_ratio * len(sequence_id))
    graphs = retrieval.retrieval_topk(input_text, retrieve_movies_list, args.sub_graph_numbers, args.reranking_numbers)

    query_text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
## Instruction: Given the user's watching history, selecting a film that is most likely to interest the user from the options. ### Watching history: {input_text}. ###Options: {question}. Select a movie from options A to T that the user is most likely to be interested in. Please answer "A" or "B" or "C" or "D" or "E" or "F" or "G" or "H" or "I" or "J" or "K" or "L" or "M" or "N" or "O" or "P" or "Q" or "R" or "S" or "T" only"."""

    # ============================================================
    # 1. 检查训练样本的 label 分布（从 JSON 中直接统计）
    # ============================================================
    print("\n" + "=" * 60)
    print("[1/6] 训练集 label 分布检查")
    print("=" * 60)
    with open("dataset/ML1M/10000_data_id_20.json", 'r') as f:
        all_data = json.load(f)[:9000]
    from collections import Counter
    label_dist = Counter([d['output'] for d in all_data])
    print(f"  Label distribution in train (9000 samples): {dict(label_dist)}")
    print(f"  Most common label: {label_dist.most_common(1)[0]}")

    # ============================================================
    # 2. 验证 tokenizer 对 A-T 的编码
    # ============================================================
    print("\n" + "=" * 60)
    print("[2/6] Tokenizer A-T 编码验证")
    print("=" * 60)
    for letter in ['A', 'B', 'C', 'Q']:
        tid = model.tokenizer.convert_tokens_to_ids(letter)
        decoded = model.tokenizer.convert_ids_to_tokens(tid)
        print(f"  Letter '{letter}': token_id={tid}, decoded='{decoded}'")

    # ============================================================
    # 3. 用 inference() 跑同一个样本
    # ============================================================
    print("\n" + "=" * 60)
    print("[3/6] inference() 输出检查")
    print("=" * 60)
    sample_infer = {'id': ['q1'], 'graph': [graphs], 'question': [query_text], 'label': ['']}

    with torch.no_grad():
        output_ids = model.inference(sample_infer)

    print(f"  inference() 返回的 sorted_indices: {output_ids[0].tolist()}")
    pred_idx = output_ids[0][0].item()
    pred_letter = chr(ord('A') + pred_idx)
    print(f"  排名第一的选项索引: {pred_idx} -> {pred_letter}")
    print(f"  真实答案: {gold_letter}")

    # 临时 hack：在 inference() 内部 print scores（需要改 graph_llm.py）
    # 这里先手动调用一次 generate，拿到 raw scores
    print("\n  [手动 generate 检查 scores]")
    with torch.no_grad():
        # 复制 inference() 的前半段逻辑
        questions = model.tokenizer([query_text], add_special_tokens=False)
        eos_user_tokens = model.tokenizer('[/INST]', add_special_tokens=False)
        bos_embeds = model.word_embedding(model.tokenizer('<s>[INST]', add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = model.word_embedding(torch.tensor(model.tokenizer.pad_token_id)).unsqueeze(0)

        graph_embeds_list = model.encode_graphs(sample_infer)
        projected_graph_embeds_list = [model.projector(g) for g in graph_embeds_list]

        input_ids = questions.input_ids[0] + eos_user_tokens.input_ids
        inputs_embeds = model.word_embedding(torch.tensor(input_ids).to(model.model.device))
        sample_graph_embeds = torch.cat([proj.unsqueeze(0) for proj in projected_graph_embeds_list[0]], dim=0).mean(dim=0, keepdim=True)
        inputs_embeds = torch.cat([bos_embeds, sample_graph_embeds, inputs_embeds], dim=0)
        # Cast to model dtype (float16) to avoid dtype mismatch
        inputs_embeds = inputs_embeds.to(model.model.dtype)

        # pad to batch
        max_length = inputs_embeds.shape[0]
        attention_mask = torch.tensor([[1] * max_length]).to(model.model.device)
        inputs_embeds = inputs_embeds.unsqueeze(0)

        generation_output = model.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=model.max_new_tokens,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True
        )

        sequences = generation_output.sequences
        scores = generation_output.scores[0].softmax(dim=-1)

        option_indices = [319, 350, 315, 360, 382, 383, 402, 379, 306, 435, 476, 365, 341, 405, 438, 349, 660, 390, 317, 323]
        option_names = list('ABCDEFGHIJKLMNOPQRST')
        probs = scores[0, option_indices].cpu().numpy()
        sorted_idx = probs.argsort()[::-1]

        print(f"  生成的 token IDs: {sequences[0].tolist()[-5:]}")  # 最后 5 个 token
        print(f"  生成的文本: '{model.tokenizer.decode(sequences[0], skip_special_tokens=True)}'")
        print(f"  第一个生成 token 的 A-T 概率分布 (Top 5):")
        for i in range(5):
            idx = sorted_idx[i]
            print(f"    {option_names[idx]}: {probs[idx]:.6f}")

    # ============================================================
    # 4. 用 forward() 跑同一个样本（传入正确答案 label）
    # ============================================================
    print("\n" + "=" * 60)
    print("[4/6] forward() 训练模式检查 (传入正确答案)")
    print("=" * 60)
    sample_forward = {'id': ['q1'], 'graph': [graphs], 'question': [query_text], 'label': [gold_letter]}

    model.train()
    with torch.enable_grad():
        loss = model.forward(sample_forward)

    print(f"  传入真实答案 '{gold_letter}' 时的 loss: {loss.item():.4f}")

    # 再试试传入错误答案 'C'
    sample_forward_c = {'id': ['q1'], 'graph': [graphs], 'question': [query_text], 'label': ['C']}
    with torch.enable_grad():
        loss_c = model.forward(sample_forward_c)
    print(f"  传入错误答案 'C' 时的 loss: {loss_c.item():.4f}")

    # 再试试传入错误答案 'A'
    sample_forward_a = {'id': ['q1'], 'graph': [graphs], 'question': [query_text], 'label': ['A']}
    with torch.enable_grad():
        loss_a = model.forward(sample_forward_a)
    print(f"  传入错误答案 'A' 时的 loss: {loss_a.item():.4f}")

    # ============================================================
    # 5. 检查 forward() 和 inference() 的 inputs_embeds 是否一致
    # ============================================================
    print("\n" + "=" * 60)
    print("[5/6] forward() vs inference() 输入一致性检查")
    print("=" * 60)
    # 手动构造 forward 的输入（不含 label）
    model.eval()
    with torch.no_grad():
        questions_f = model.tokenizer(sample_forward['question'], add_special_tokens=False)
        eos_user_tokens_f = model.tokenizer('[/INST]', add_special_tokens=False)
        bos_embeds_f = model.word_embedding(model.tokenizer('<s>[INST]', add_special_tokens=False, return_tensors='pt').input_ids[0])

        graph_embeds_list_f = model.encode_graphs(sample_forward)
        projected_graph_embeds_list_f = [model.projector(g) for g in graph_embeds_list_f]

        input_ids_f = questions_f.input_ids[0] + eos_user_tokens_f.input_ids
        inputs_embeds_f = model.word_embedding(torch.tensor(input_ids_f).to(model.model.device))
        sample_graph_embeds_f = torch.cat([proj.unsqueeze(0) for proj in projected_graph_embeds_list_f[0]], dim=0).mean(dim=0, keepdim=True)
        inputs_embeds_f = torch.cat([bos_embeds_f, sample_graph_embeds_f, inputs_embeds_f], dim=0).to(model.model.dtype)

        # 和之前 inference 手动构造的对比
        diff = (inputs_embeds_f - inputs_embeds[0]).abs().max().item()
        print(f"  forward() 和 inference() 手动构造的 inputs_embeds 最大差异: {diff:.8f}")
        if diff < 1e-5:
            print("  ✅ 输入完全一致")
        else:
            print("  ❌ 输入存在差异！")

    # ============================================================
    # 6. 检查 graph_embed 是否是常量
    # ============================================================
    print("\n" + "=" * 60)
    print("[6/6] Graph embed 常量检查")
    print("=" * 60)
    # 再取第二个样本，比较 graph_embed
    with open("dataset/ML1M/10000_data_id_20.json", 'r') as f:
        data2 = json.load(f)[9001]
    input_text2 = data2['input']
    question2 = data2['questions']
    sequence_id2 = json.loads(data2.get('sequence_ids', '[]'))
    query_text2 = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
## Instruction: Given the user's watching history, selecting a film that is most likely to interest the user from the options. ### Watching history: {input_text2}. ###Options: {question2}. Select a movie from options A to T that the user is most likely to be interested in. Please answer "A" or "B" or "C" or "D" or "E" or "F" or "G" or "H" or "I" or "J" or "K" or "L" or "M" or "N" or "O" or "P" or "Q" or "R" or "S" or "T" only"."""

    retrieve_movies_list2 = retrieval.whether_retrieval(sequence_id2, args.adaptive_ratio * len(sequence_id2))
    graphs2 = retrieval.retrieval_topk(input_text2, retrieve_movies_list2, args.sub_graph_numbers, args.reranking_numbers)

    sample2 = {'id': ['q2'], 'graph': [graphs2], 'question': [query_text2], 'label': ['']}

    with torch.no_grad():
        graph_embeds_list1 = model.encode_graphs(sample_infer)
        graph_embeds_list2 = model.encode_graphs(sample2)

        proj1 = model.projector(graph_embeds_list1[0])
        proj2 = model.projector(graph_embeds_list2[0])

        mean1 = proj1.mean(dim=0, keepdim=True)
        mean2 = proj2.mean(dim=0, keepdim=True)

        print(f"  样本1 graph_embed (after projector+mean) shape: {mean1.shape}")
        print(f"  样本2 graph_embed (after projector+mean) shape: {mean2.shape}")
        print(f"  两样本 graph_embed 的 L2 距离: {torch.norm(mean1 - mean2).item():.4f}")
        print(f"  样本1 graph_embed 均值: {mean1.mean().item():.4f}, 标准差: {mean1.std().item():.4f}")
        print(f"  样本2 graph_embed 均值: {mean2.mean().item():.4f}, 标准差: {mean2.std().item():.4f}")

        if torch.norm(mean1 - mean2).item() < 1e-3:
            print("  ⚠️  两个完全不同样本的 graph_embed 几乎一样！说明 GNN/Projector 输出退化/常量")
        else:
            print("  ✅ 两个样本的 graph_embed 有差异")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
