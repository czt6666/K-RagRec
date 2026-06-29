from src.model.graph_llm import GraphLLM


load_model = {
    "graph_llm": GraphLLM,
}

llama_model_path = {
    "7b": "/data/hf_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",
    "qwen2.5_7b_chat": "/data/hf_cache/manual/Qwen2.5-7B-Instruct",
    "llama3_8b_chat": "/data/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    "mistral_7b_chat": "/data/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71",
}
