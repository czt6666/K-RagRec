import os
import torch
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import pandas as pd
from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.model import load_model, llama_model_path
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from sklearn.metrics import classification_report

import numpy as np
import re
from torch_geometric.data import Data
import networkx as nx
from retrieve import GraphRetrieval
import time

def main(args):

    seed_everything(seed=args.seed)
    print(args)

    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args)
    try:
        model = _reload_best_model(model, args)
    except:
        pass
    model.eval()
    retrieval_model=GraphRetrieval(model_name='sbert', path='dataset/fb')
    test_data_path: str = "dataset/ML1M/10000_data_id_20.json"

    # Pool of non-sequential placeholder letters for the prompt example, so
    # the model can't pattern-match a literal "A,B,C,D,..." example.
    EXAMPLE_LETTER_POOL = ['C', 'F', 'A', 'P', 'J', 'Q', 'M', 'D', 'S', 'K']
    NUM_ANSWERS = 5

    def recall_at_k(actual, predicted, ks):
        hits = {k: 0 for k in ks}
        for rank, x in enumerate(predicted, start=1):
            if x in actual:
                for k in ks:
                    if rank <= k:
                        hits[k] = 1
                break
        return hits

    def evaluate(inputs, questions=None, gold=None, graphs=None):
        id=[]
        query=[]
        label=[]

        sample = {'id':id, 'graph':graphs, 'question':query, 'label':label}
        d = 0
        placeholder_format = ','.join(['<letter>'] * NUM_ANSWERS)
        example_str = ','.join(EXAMPLE_LETTER_POOL[:NUM_ANSWERS])
        for input, question in zip(inputs, questions):
            d = d+1
            id.append(f'query{d}')
            query.append(f"""Given the user's watching history, rank the films the user is most likely to be interested in from the options.
Watching history: {input}.
Options: {question}.
Respond with exactly {NUM_ANSWERS} distinct letters from A to T, separated by commas, ranked from most to least likely. Use this exact format: {placeholder_format} (for example {example_str} -- those are placeholder letters showing the format only, not your actual answer).
Do not include any other characters, and do not repeat the question or explain your choice.
The {NUM_ANSWERS} ranked letters are:""")
            label.append('')

        output, raw_texts = model.inference(sample)
        return output.tolist(), query, raw_texts

    from tqdm import tqdm
    RECALL_KS = (1, 3, 5)
    gold = []
    pred = []
    recalls = {k: [] for k in RECALL_KS}
    detailed_logs = []
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)[9000:10000]
        inputs = [_['input'] for _ in test_data]
        questions = [_['questions'] for _ in test_data]
        gold = [0 if _['output']=="A" else 1 if _['output']=="B" else 2 if _['output']=="C" else 3 if _['output']=="D" else 4 if _['output']=="E" else 5 if _['output']=="F" else 6 if _['output']=="G" else 7 if _['output']=="H" else 8 if _['output']=="I" else 9 if _['output']=="J" else 10 if _['output']=="K" else 11 if _['output']=="L" else 12 if _['output']=="M" else 13 if _['output']=="N" else 14 if _['output']=="O" else 15 if _['output']=="P" else 16 if _['output']=="Q" else 17 if _['output']=="R" else 18 if _['output']=="S" else 19 for _ in test_data]
        sequence_ids = [json.loads(_.get('sequence_ids', '[]')) for _ in test_data]

        # Pre-cache all graphs once to avoid repeated CPU retrieval during evaluation
        print("Pre-computing graphs for all test samples...")
        cached_graphs = []
        for inp, sid in tqdm(zip(inputs, sequence_ids), total=len(inputs)):
            retrieve_movies_list = retrieval_model.whether_retrieval(sid, args.adaptive_ratio * len(sid))
            cached_graphs.append(retrieval_model.retrieval_topk(inp, retrieve_movies_list, args.sub_graph_numbers, args.reranking_numbers))
        print("Graph pre-computation done.")

        def batch(list, batch_size=args.eval_batch_size):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]
        for i, batch in enumerate(zip(batch(inputs), batch(questions), batch(gold), batch(cached_graphs), batch(sequence_ids))):
            inputs_b, questions_b, golds, graphs_b, sid_b = batch
            output, prompts, raw_texts = evaluate(inputs_b, questions_b, golds, graphs_b)
            pred.extend(output)
            start_index = len(pred) - len(golds)
            for j in range(len(golds)):
                hits = recall_at_k([golds[j]], output[j], RECALL_KS)
                for k in RECALL_KS:
                    recalls[k].append(hits[k])
                # Record detailed log
                retrieve_movies_list = retrieval_model.whether_retrieval(sid_b[j], args.adaptive_ratio * len(sid_b[j]))
                retrieval_flags = [int(mid in retrieve_movies_list) for mid in sid_b[j]]
                detailed_logs.append({
                    "sample_id": start_index + j,
                    "question": questions_b[j],
                    "prompt": prompts[j],
                    "watching_history": inputs_b[j],
                    "sequence_ids": sid_b[j],
                    "retrieved_movies": retrieve_movies_list,
                    "retrieval_flags": retrieval_flags,
                    "raw_response": raw_texts[j],
                    "parsed_ranking": output[j],
                    "parsed_letters": [chr(ord('A') + idx) for idx in output[j][:NUM_ANSWERS]],
                    "gold_answer": golds[j],
                    "gold_letter": chr(ord('A') + golds[j]),
                    "soft_injection_enabled": True,
                })
            n = len(recalls[RECALL_KS[0]])
            print(", ".join(f"Recall@{k}: {sum(recalls[k]) / n}" for k in RECALL_KS))

    top1_pred = [p[0] if p else -1 for p in pred]
    acc = accuracy_score(gold, top1_pred) if pred else 0.0
    n = len(recalls[RECALL_KS[0]])
    final_results = {
        "method": "baseline",
        "dataset": args.dataset,
        "llm_model_name": args.llm_model_name,
        "gnn_model_name": args.gnn_model_name,
        "accuracy": acc,
        **{f"recall@{k}": sum(recalls[k]) / n for k in RECALL_KS},
    }
    import os
    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)
    result_path = f"{args.output_dir}/{args.dataset}/baseline_results.json"
    with open(result_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Results saved to {result_path}")

    # Save detailed logs
    raw_dir = "output/raw"
    os.makedirs(raw_dir, exist_ok=True)
    log_name = f"{args.llm_model_name}_{args.gnn_model_name}_{args.dataset}_detailed.jsonl"
    log_path = os.path.join(raw_dir, log_name)
    with open(log_path, "w") as f:
        for entry in detailed_logs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Detailed logs saved to {log_path}")


if __name__ == "__main__":
    args = parse_args_llama()

    main(args)
