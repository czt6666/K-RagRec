# H5 — Relation-aware GNN encoder

> Forked from `methods/baseline/`. Adds two relation-aware GNN encoders
> (RGCN, CompGCN) so the encoder can use the ~177 distinct relation types
> in the Freebase KG, instead of treating `edge_attr` as a single 1024-dim
> blob. See `paper/INNOVATIONS.md` H5 for motivation.

## Files changed vs baseline

| file | what changed |
|---|---|
| `src/model/gnn.py` | + `RGCN` (uses PyG `RGCNConv` over discrete `edge_type` ids) and + `CompGCN` (custom `MessagePassing` with multiplicative composition `phi(h,r)=h*r`). Registered under keys `rgcn` and `compgcn` in `load_gnn_model`. |
| `retrieve.py` | In `__init__`, derive a global `edge_type` tensor by `torch.unique(G.edge_attr, dim=0, return_inverse=True)` (177 unique relation embeddings → 177 ids). In `get_first_order_subgraph`, slice and attach `edge_type` to each `Data`. |
| `src/model/graph_llm.py` | `encode_graphs` inspects the GNN's forward signature; if it accepts `edge_type`, it forwards `graphs.edge_type` (auto-batched by `Batch.from_data_list`). Other GNNs are unaffected. |

## Run (PowerShell)

```powershell
$env:PYTHONPATH = "methods/h5_rel_gnn"

# RGCN encoder
python methods/h5_rel_gnn/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "<llama-2-7b path>" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name rgcn `
    --gnn_num_layers 4 `
    --sub_graph_numbers 3 `
    --reranking_numbers 5 `
    --adaptive_ratio 5

# CompGCN encoder
python methods/h5_rel_gnn/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "<llama-2-7b path>" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name compgcn `
    --gnn_num_layers 4 `
    --sub_graph_numbers 3 `
    --reranking_numbers 5 `
    --adaptive_ratio 5

# Evaluate (matching --gnn_model_name to whichever was trained)
python methods/h5_rel_gnn/evaluate.py `
    --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m `
    --batch_size 5 --gnn_model_name rgcn --gnn_num_layers 4 `
    --sub_graph_numbers 3 --reranking_numbers 5 --adaptive_ratio 5
```

## Smoke test

```powershell
$env:PYTHONPATH = "methods/h5_rel_gnn"
python tools/smoke_h5_gnn.py     # tests every GNN forward on a synthetic batch
python tools/smoke_retrieval.py  # confirms retrieval with edge_type still works
```

Expected to print `[OK] H5 GNN smoke test passed.` and the retrieval test
to also succeed.

## Ablation matrix the paper will report

| --gnn_model_name | source |
|---|---|
| `gcn` | baseline |
| `gat` | baseline |
| `gt` | baseline (paper default) |
| `graphsage` | baseline |
| `rgcn` | **H5 new** |
| `compgcn` | **H5 new** |

Compare each on ML1M / ML20M / Book × LLaMA-2-7B; expect `rgcn` and
`compgcn` to outperform the baseline four by 1-3 % ACC because Freebase
has ~177 relation types whose semantics are otherwise lost in the
edge-as-1024-dim-vector formulation.
