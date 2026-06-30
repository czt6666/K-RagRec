"""
No-injection direct-prompt baseline: loads the frozen LLM only, no GNN/retrieval.
Uses the same 1000-sample test split [9000:10000] as evaluate.py.
"""
import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model import llama_model_path


ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRST, ")
RECALL_KS = (1, 3, 5)
EXAMPLE_LETTER_POOL = ['C', 'F', 'A', 'P', 'J', 'Q', 'M', 'D', 'S', 'K']


def build_task_content(input_text, question_text, num_answers):
    placeholder_format = ','.join(['<letter>'] * num_answers)
    example_str = ','.join(EXAMPLE_LETTER_POOL[:num_answers])
    return (
        f"""Given the user's watching history, rank the films the user is most likely to be interested in from the options.\n"""
        f"""Watching history: {input_text}.\n"""
        f"""Options: {question_text}.\n"""
        f"""Respond with exactly {num_answers} distinct letters from A to T, separated by commas, ranked from most to least likely. Use this exact format: {placeholder_format} (for example {example_str} -- those are placeholder letters showing the format only, not your actual answer).\n"""
        f"""Do not include any other characters, and do not repeat the question or explain your choice.\n"""
        f"""The {num_answers} ranked letters are:"""
    )


def build_prompt(tokenizer, input_text, question_text, num_answers):
    content = build_task_content(input_text, question_text, num_answers)
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content


def is_valid_response(decoded, expected_count):
    if not decoded or any(c not in ALLOWED_CHARS for c in decoded):
        return False
    parts = [p.strip() for p in decoded.split(',')]
    if len(parts) != expected_count or len(set(parts)) != expected_count:
        return False
    return all(len(p) == 1 and 'A' <= p <= 'T' for p in parts)


def recall_at_k(actual, predicted, ks):
    hits = {k: 0 for k in ks}
    for rank, x in enumerate(predicted, start=1):
        if x in actual:
            for k in ks:
                if rank <= k:
                    hits[k] = 1
            break
    return hits


def main(args):
    model_path = llama_model_path[args.llm_model_name]
    is_qwen = "qwen" in model_path.lower()
    is_llama3 = "llama-3" in model_path.lower() or "llama3" in model_path.lower()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=is_qwen or is_llama3)
    if is_qwen or is_llama3:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    use_bf16 = is_qwen or is_llama3
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Same test split as evaluate.py
    with open(args.data_path, 'r') as f:
        data = json.load(f)[9000:10000]

    recalls = {k: [] for k in RECALL_KS}
    gold_all = []
    pred_top1_all = []
    detailed_logs = []

    for idx, sample in enumerate(data):
        gold_letter = sample['output']
        gold_idx = ord(gold_letter) - ord('A')
        gold_all.append(gold_idx)

        prompt = build_prompt(tokenizer, sample['input'], sample['questions'], args.num_answers)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        decoded = ""
        valid_format = False
        for attempt in range(args.max_retries):
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    do_sample=attempt > 0,
                    temperature=0.7 if attempt > 0 else None,
                    top_p=0.9 if attempt > 0 else None,
                )
            new_tokens = gen[0][inputs.input_ids.shape[1]:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if is_valid_response(decoded, args.num_answers):
                valid_format = True
                break

        parts = [p.strip() for p in decoded.split(',')]
        ranking = []
        seen = set()
        for p in parts:
            if len(p) == 1 and 'A' <= p <= 'T':
                letter_idx = ord(p) - ord('A')
                if letter_idx not in seen:
                    seen.add(letter_idx)
                    ranking.append(letter_idx)
        if len(ranking) < max(RECALL_KS):
            ranking += [j for j in range(20) if j not in seen]

        pred_top1_all.append(ranking[0] if ranking else -1)
        hits = recall_at_k([gold_idx], ranking, RECALL_KS)
        for k in RECALL_KS:
            recalls[k].append(hits[k])

        # Record detailed log
        detailed_logs.append({
            "sample_id": idx,
            "question": sample['questions'],
            "prompt": prompt,
            "watching_history": sample['input'],
            "raw_response": decoded,
            "parsed_ranking": ranking,
            "parsed_letters": [chr(ord('A') + idx) for idx in ranking[:args.num_answers]],
            "gold_answer": gold_idx,
            "gold_letter": gold_letter,
            "valid_format": valid_format,
            "retries_used": attempt + 1,
            "soft_injection_enabled": False,
        })

        if idx % 50 == 0:
            n = len(recalls[RECALL_KS[0]])
            print(f"[{idx}] " + ", ".join(f"Recall@{k}: {sum(recalls[k])/n:.3f}" for k in RECALL_KS))

    n = len(recalls[RECALL_KS[0]])
    acc = sum(1 for g, p in zip(gold_all, pred_top1_all) if g == p) / n

    print("\n=== No-RAG direct-prompt baseline ===")
    print(f"Samples: {n}, Acc: {acc:.3f}")
    for k in RECALL_KS:
        print(f"Recall@{k}: {sum(recalls[k])/n:.3f}")

    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)
    result_path = f"{args.output_dir}/{args.dataset}/norag_results.json"
    results = {
        "method": "no_rag",
        "dataset": args.dataset,
        "llm_model_name": args.llm_model_name,
        "accuracy": acc,
        **{f"recall@{k}": sum(recalls[k]) / n for k in RECALL_KS},
    }
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {result_path}")

    # Save detailed logs
    raw_dir = "output/raw"
    os.makedirs(raw_dir, exist_ok=True)
    log_name = f"{args.llm_model_name}_{args.dataset}_norag_detailed.jsonl"
    log_path = os.path.join(raw_dir, log_name)
    with open(log_path, "w") as f:
        for entry in detailed_logs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Detailed logs saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model_name", type=str, default="llama3_8b_chat")
    parser.add_argument("--dataset", type=str, default="ml1m")
    parser.add_argument("--data_path", type=str, default="dataset/ML1M/10000_data_id_20.json")
    parser.add_argument("--output_dir", type=str, default="output_llama3_norag")
    parser.add_argument("--max_new_tokens", type=int, default=15)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--num_answers", type=int, default=5)
    args = parser.parse_args()
    main(args)
