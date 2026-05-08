# H3 — Path-aware multi-source connected-subgraph retrieval

> Forked from `methods/baseline/`. Replaces the baseline's per-item top-k
> cosine retrieval (which emits ~10 disjoint 1-hop subgraphs that lose
> all cross-item linker context) with a single connected subgraph per
> sample, anchored on all 10 history items plus the top-K nodes most
> similar to the user's query. The subgraph naturally captures the
> "shared cast / studio / genre" linkers that connect movies through KG.
>
> Implementation note: the file is named `pcst_retrieval.py` because the
> original plan was to use Prize-Collecting Steiner Tree (G-Retriever's
> trick). The Windows wheel of `pcst_fast 1.0.10` was discovered to
> return corrupt output (verts return all-zeros), and `nx.steiner_tree`
> took 51 minutes per sample on this 14k-node KG. The actual algorithm
> shipped here is a **fast multi-source 2-hop BFS neighborhood union**:
> nodes within 2 hops of multiple terminals are ranked highest, which
> recovers the "linker node" property of Steiner trees in O(|T|·(V+E))
> time (~0.1 s per sample). See `tools/test_steiner.py` and
> `tools/debug_h3_pcst.py` for the diagnostic trail.

## Files changed vs baseline

| file | what changed |
|---|---|
| `src/pcst_retrieval.py` | **new** — `multisource_neighborhood(nxg, terminals, k, max_nodes)` (fast Steiner-alternative) and `pcst_retrieve(retriever, query_text, sequence_ids, fused_x, ...)` (full retrieval entry point including SBERT-similarity prizes and hop-field fused features). Also exposes `precompute_movie_anchors()` to build the ML1M-movie-id → KG-node-id map at retriever init. |
| `retrieve.py` | `__init__` loads `layer3_embeddings_W.pt` (was unused in baseline), computes `self.fused_x` as a 0.5/0.3/0.2 weighted blend of the 3 layers, and pre-computes `self.movie_id_to_anchor` (3883 entries; uses one batched SBERT call). Adds `pcst_retrieval_topk()` method that returns a list with one big connected subgraph. |
| `train.py` / `evaluate.py` | Replace `whether_retrieval` + `retrieval_topk` with one call to `pcst_retrieval_topk(input, sequence_id, ...)`. |
| `src/config.py` | + `--pcst_anchor_prize`, `--pcst_topk_query_prize`, `--pcst_edge_cost`, `--pcst_max_nodes`. (Some are vestigial from the earlier PCST attempt; `--pcst_topk_query_prize` and `--pcst_max_nodes` are the actively used knobs.) |

## Run (PowerShell)

```powershell
$env:PYTHONPATH = "methods/h3_pcst"

python methods/h3_pcst/train.py `
    --model_name graph_llm `
    --llm_model_name 7b `
    --llm_model_path "<llama-2-7b path>" `
    --llm_frozen True `
    --dataset ml1m `
    --batch_size 5 `
    --gnn_model_name gt --gnn_num_layers 4 `
    --pcst_anchor_prize 10.0 `
    --pcst_topk_query_prize 20 `
    --pcst_max_nodes 200

python methods/h3_pcst/evaluate.py `
    --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m `
    --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 `
    --pcst_anchor_prize 10.0 --pcst_topk_query_prize 20 --pcst_max_nodes 200
```

## Smoke test

```powershell
$env:PYTHONPATH = "methods/h3_pcst"
python tools/smoke_h3_pcst.py
```

Expected last-block output:
```
subgraph nodes: 200
subgraph edges: 587
connected components: 1
[OK] H3 PCST smoke test passed.
```

The retriever load takes ~14 s (mostly SBERT-encoding 3883 movie titles
once at startup). PCST per sample is ~0.09 s.

## Ablation grid

| config | what it tests |
|---|---|
| `--pcst_max_nodes 50` | tight subgraph, less context |
| `--pcst_max_nodes 200` | default |
| `--pcst_max_nodes 500` | richer context, larger LLM prefix |
| `--pcst_topk_query_prize 0` | anchors only, no query similarity prizes |
| `--pcst_topk_query_prize 50` | aggressive query expansion |
| baseline retrieval (`methods/baseline/`) | 2 N disjoint top-k subgraphs |

## Why 2-hop BFS instead of full PCST

1. **pcst_fast (G-Retriever's library)**: prebuilt Windows wheel returns
   garbage (`verts: [0,0,0,...]`); reproducible with a 4-node toy graph.
   We ran out of time to debug the C++ extension.
2. **nx.steiner_tree (KMB approximation)**: 51 minutes per sample on
   14 669 nodes; algorithm scales as O(|T|·SP) and SP is `dijkstra_predecessors`
   from each terminal. Unusable for 9 000-sample training.
3. **Multi-source 2-hop BFS**: O(|T|·(V+E)) ≈ 0.1 s per sample. The
   "linker node" intuition is preserved: any node within 2 hops of two
   anchors lies on a 4-or-fewer-hop path between them, and we rank by
   how many anchors a node is shared between. With KG diameter ~6 and
   anchors typically 2 hops apart, this captures most natural Steiner
   structure. Empirically the result is fully connected (1 component).

If you later get a working `pcst_fast` (e.g. on Linux), drop the
multi-source-BFS path and reinstate the prize/cost-based PCST — the
Q-Former and gate downstream don't care which retriever produced the
subgraph.
