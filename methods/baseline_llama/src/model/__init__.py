from src.model.graph_llm import GraphLLM


load_model = {
    "graph_llm": GraphLLM,
}

llama_model_path = {
    "7b": "/data/hf_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",
    "qwen2.5_7b_chat": "/data/hf_cache/manual/Qwen2.5-7B-Instruct",
    "llama3_8b_chat": "/data/hf_cache/manual/Llama-3.1-8B-Instruct",
    "mistral_7b_chat": "/data/hf_cache/manual/Mistral-7B-Instruct-v0.3",
}
