import os
import sys
import torch
import json

# Ensure we can import from methods/baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'methods', 'baseline'))

from src.model import load_model, llama_model_path
from src.config import parse_args_llama
from src.utils.ckpt import _reload_best_model
from transformers import AutoTokenizer

args = parse_args_llama()
args.llm_model_name = '7b'
args.llm_model_path = llama_model_path['7b']
args.model_name = 'graph_llm'
args.gnn_model_name = 'gt'
args.gnn_num_layers = 4
args.gnn_in_dim = 1024
args.gnn_hidden_dim = 1024
args.gnn_num_heads = 4
args.gnn_dropout = 0.0
args.llm_frozen = 'True'
args.max_txt_len = 512
args.max_new_tokens = 64
args.output_dir = 'output_baseline'
args.dataset = 'ml1m'
args.seed = 9
args.num_query_tokens = 1  # baseline uses 1 (projector + mean)
args.qformer_num_heads = 8
args.qformer_dropout = 0.0
args.llm_hidden_dim = 4096

print('Loading model...')
model = load_model[args.model_name](args=args)
model = _reload_best_model(model, args)
model.eval()

tok = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False)

# Load first 3 test samples
with open('dataset/ML1M/10000_data_id_20.json', 'r') as f:
    data = json.load(f)[9000:9003]

print('\n=== Checking generation for first 3 test samples ===')
for idx, sample_data in enumerate(data):
    print(f"\n--- Sample {idx} ---")
    print(f"Gold answer: {sample_data['output']}")

    # Build sample dict (minimal, no real graph)
    from torch_geometric.data import Data
    dummy_graph = Data(
        x=torch.randn(5, 1024),
        edge_index=torch.tensor([[0,1,2],[1,2,3]]),
        edge_attr=torch.randn(3, 1024)
    )
    sample = {
        'id': ['query1'],
        'graph': [[dummy_graph]],
        'question': [f"Below is an instruction... ### Watching history: {sample_data['input'][:50]}... ###Options: {sample_data['questions'][:80]}..."],
        'label': [sample_data['output']]
    }

    with torch.no_grad():
        result = model.inference(sample)
        top5_preds = result[0][:5].tolist()
        print(f"Top-5 predicted option indices: {top5_preds}")
        print(f"Top-5 predicted letters: {[chr(ord('A') + i) for i in top5_preds]}")

        # Also check what the model actually generates
        questions = tok(sample['question'], add_special_tokens=False)
        eos_user_tokens = tok('[/INST]', add_special_tokens=False)
        bos_embeds = model.word_embedding(tok('<s>[INST]', add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = model.word_embedding(torch.tensor(tok.pad_token_id)).unsqueeze(0)

        graph_embeds_list = model.encode_graphs(sample)
        for i in range(1):
            input_ids = questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = model.word_embedding(torch.tensor(input_ids).to(model.model.device))
            sample_graph_embeds = graph_embeds_list[i]
            sample_graph_embeds = model.projector(sample_graph_embeds)
            sample_graph_embeds = sample_graph_embeds.mean(dim=0, keepdim=True)
            inputs_embeds = torch.cat([bos_embeds, sample_graph_embeds, inputs_embeds], dim=0)
            attention_mask = torch.tensor([[1] * inputs_embeds.shape[0]]).to(model.model.device)

            with model.maybe_autocast():
                inputs_embeds = inputs_embeds.to(model.model.dtype)
                gen_output = model.model.generate(
                    inputs_embeds=inputs_embeds.unsqueeze(0),
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True
                )
            seq = gen_output.sequences[0]
            decoded = tok.decode(seq, skip_special_tokens=True)
            print(f"Generated text: '{decoded[:100]}'")
            first_token_id = seq[len(input_ids) + 1] if len(seq) > len(input_ids) + 1 else seq[-1]
            print(f"First generated token id: {first_token_id}, decoded: '{tok.decode([first_token_id])}'")
