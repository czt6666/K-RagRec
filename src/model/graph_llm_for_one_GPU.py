import contextlib
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from src.model.gnn import load_gnn_model
from torch_geometric.data import Batch
from torch_geometric.data import Data
import time
import re
import torch.nn.functional as F

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


def _llm_load_kwargs():
    """
    device_map/max_memory 传给 from_pretrained。
    - max_memory 不会“变大”显存，只是限制单卡占用；单卡应接近真实容量并留出 GNN/投影/激活余量。
    - 可用环境变量 K_RAGREC_MAX_MEMORY_GPU0 覆盖，例如 8G 卡: set K_RAGREC_MAX_MEMORY_GPU0=6GiB
    """
    revision = "main"
    if not torch.cuda.is_available():
        return {"revision": revision}

    override = os.environ.get("K_RAGREC_MAX_MEMORY_GPU0", "").strip()
    if override:
        return {
            "revision": revision,
            "device_map": "auto",
            "max_memory": {0: override},
        }

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gib = total_bytes / (1024**3)
    # 为 GNN、projector、检索编码与同 batch 激活留余量（过小会触发 CPU offload，极慢）
    reserve_gib = float(os.environ.get("K_RAGREC_GPU_RESERVE_GIB", "2.0"))
    llm_cap = max(total_gib - reserve_gib, 1.0)
    return {
        "revision": revision,
        "device_map": "auto",
        "max_memory": {0: f"{llm_cap:.1f}GiB"},
    }


class GraphLLM(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.model_name = args.llm_model_name

        print('Loading LLAMA')
        kwargs = _llm_load_kwargs()
        print(kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        load_kw = {k: v for k, v in kwargs.items() if k != "revision"}
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            revision=kwargs["revision"],
            **load_kw
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        # 训练时显著降低激活显存；eval 下 HF 仅在 training 时使用 checkpoint
        if torch.cuda.is_available() and os.environ.get("K_RAGREC_NO_GRADIENT_CHECKPOINTING", "").lower() not in ("1", "true", "yes"):
            self.model.gradient_checkpointing_enable()
        print('Finish loading LLAMA!')

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).to(self.model.device)
        #4096 for llama 3584 for qwen
        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # 与 from_pretrained(torch_dtype=float16) 一致；单卡上 bfloat16 易触发 CUBLAS_STATUS_INTERNAL_ERROR
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast("cuda", dtype=dtype)
        else:
            return contextlib.nullcontext()

  

    def encode_graphs(self, samples):
        graphs_list = samples['graph']
        graph_embeds_list = []

        for graphs in graphs_list:
            graphs = Batch.from_data_list(graphs).to(self.model.device)
            graphs = graphs.to(self.model.device)
            n_embeds, _ = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)
            g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
            graph_embeds_list.append(g_embeds)

        return graph_embeds_list

    def forward(self, samples):

        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)
        
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        # encode graphs
        graph_embeds_list = self.encode_graphs(samples)
        projected_graph_embeds_list = [self.projector(g_embeds) for g_embeds in graph_embeds_list]

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            sample_graph_embeds = torch.cat([proj.unsqueeze(0) for proj in projected_graph_embeds_list[i]], dim=0)
            sample_graph_embeds = sample_graph_embeds.mean(dim=0, keepdim=True)
            inputs_embeds = torch.cat([bos_embeds, sample_graph_embeds, inputs_embeds], dim=0)
            # inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
                use_cache=False,
                # do_sample=True
            )

        return outputs.loss


    def inference(self, samples):

        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        # encode graphs
        graph_embeds_list = self.encode_graphs(samples)
        
        

        projected_graph_embeds_list = [self.projector(g_embeds) for g_embeds in graph_embeds_list]
        

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            
            input_ids = questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            sample_graph_embeds = torch.cat([proj.unsqueeze(0) for proj in projected_graph_embeds_list[i]], dim=0)
            sample_graph_embeds = sample_graph_embeds.mean(dim=0, keepdim=True)
            inputs_embeds = torch.cat([bos_embeds, sample_graph_embeds, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            generation_output = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens, 
                attention_mask=attention_mask,
                return_dict_in_generate=True, 
                output_scores=True,
                use_cache=True 
            )
        

        sequences = generation_output.sequences
        scores = generation_output.scores[0].softmax(dim=-1)
        # self.tokenizer.encode('A', add_special_tokens=False) use this to get the index of the option
        if  self.model_name == "7b":
            option_indices = [
                319,
                350,
                315,
                360,
                382,
                383,
                402,
                379,
                306,
                435,
                476,
                365,
                341,
                405,
                438,
                349,
                660,
                390,
                317,
                323
            ]
        else:
            option_indices = [
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51
            ]
        output = self.tokenizer.batch_decode(sequences, skip_special_tokens=True) 
        specific_logits = torch.tensor(scores[:, option_indices], dtype=torch.float32).softmax(dim=-1)
        sorted_indices = specific_logits.argsort(dim=-1, descending=True)
        return sorted_indices
    



    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
