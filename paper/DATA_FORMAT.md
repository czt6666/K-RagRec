# Dataset & KG Format Report

> Audit of every file under `dataset/` that the K-RagRec pipeline touches.
> Numbers come from `tools/inspect_data.py` (run on the live data on
> 2026-05-07). Use this as the contract when implementing H1-H5: any new
> code must read the same fields with the same semantics.

---

## 0. Top-level layout

```
dataset/
├── ML1M/                                  ← MovieLens-1M (used by the runs in this repo)
│   ├── 10000_data_id_20.json              ← train + eval prompts (M = 20 candidates)
│   ├── 10000_data_id_10.json              ← prompts with M = 10 candidates (Appendix A.5)
│   ├── all_data.json                      ← 392 MB, presumably full unprocessed prompts (not used by train/eval)
│   ├── movies_id_name.txt                 ← (movie_id, clean_title) lookup
│   ├── ratings_45.txt                     ← ratings ≥ 4, used for popularity sort
│   └── ml1m_raw/                          ← original GroupLens dump + leave-one-out CSVs
│       ├── movies.dat / ratings.dat / users.dat
│       ├── train.csv / valid.csv / test.csv  ← 10-history → 1-target sequential split
│       └── test_filtered.csv
├── ml-20m/                                ← MovieLens-20M (raw; not used in current scripts)
├── book/                                  ← Amazon Book (raw; not used in current scripts)
└── fb/                                    ← processed Freebase KG
    ├── filtered_full_fb.txt               ← 348,979 KG triples (text form)
    ├── movies_with_mids.txt               ← (Freebase MID, ML1M-style title) for 3,883 movies
    ├── fb_entity_names.tsv                ← MID → entity name (multilingual)
    └── graphs/
        ├── 0.pt                           ← PLM (SBERT) node + edge embeddings, no GNN
        ├── layer2_embeddings_W.pt         ← after 1 GCN layer
        └── layer3_embeddings_W.pt         ← after 2 GCN layers (currently unused at runtime)
```

> The non-dotted files inside `dataset/fb/{nodes,edges}/` are produced by
> `index_KG.py` if they don't already exist. **They are absent** in the
> distributed snapshot, so any code path that calls `load_text_data()` /
> `get_first_order_subgraph_edge()` will crash. The normal training/eval
> path does NOT touch them — only `get_first_order_subgraph()`, which uses
> `self.G.x` / `self.G.edge_attr` directly.

---

## 1. `ML1M/10000_data_id_20.json` — the training & eval prompts

JSON list of **10 000 records**. Each record:

| field | type | example | notes |
|---|---|---|---|
| `input` | str | `'"Bram Stoker\'s Dracula", "Crucible, The", ..., "Mariachi, El"'` | comma-separated, double-quoted **history of 10 movie titles** |
| `output` | str (single letter) | `"R"` | ground-truth label, ∈ {A..T} |
| `questions` | str | `'A: Star Trek III..., B: Chairman of the Board, ...'` | the 20-option menu |
| `sequence_ids` | str (JSON-encoded list) | `'[1339, 1366, 17, 159, 509, 3416, 1748, 912, 1203, 3267]'` | list of 10 ML1M `movie_id`s, **same order** as `input` |

Stats:
- `len(records) = 10 000`
- `sequence length` is **always 10** (no padding needed)
- `output` letter distribution is roughly uniform across A-T (459-543 each)
- `train.py:40` slices `[:9000]` for training; `evaluate.py:86` slices `[9000:10000]` for eval

> ⚠ `train.py:43` and `evaluate.py:89` map letters with a 19-deep ternary
> `if`. Using `ord(letter) - ord('A')` would do the same in one line.

---

## 2. `ML1M/movies_id_name.txt` — the id ↔ title map

Tab-separated, no header, **3 883 rows**.

```
1<TAB>Toy Story
2<TAB>Jumanji
3<TAB>Grumpier Old Men
...
3952<TAB>(some title)
```

- `movie_id` is sparse (max=3952, count=3883 → 69 missing IDs).
- Values are clean titles (no year, no genre).
- `retrieve.py:185-192` loads this as `dict[int -> str]` and uses it to turn
  a sequence ID into the SBERT query.

---

## 3. `ML1M/ratings_45.txt` — popularity source

Tab-separated, no header, **575 281 rows**. Filtered subset of ratings.dat where rating ≥ 4.

| col 1 | col 2 | col 3 | col 4 |
|---|---|---|---|
| user_id | movie_id | rating ∈ {4, 5} | unix timestamp |

- 6 038 unique users; 3 533 unique movies (subset of the 3 883 movies).
- Timestamps span 956 703 932 .. 1 046 454 590 (≈ Apr 2000 — Feb 2003).
- Median user has 58 high-rated interactions; max 1 435.
- `retrieve.py:71-83` uses col 1 + col 2 only — **timestamp is currently ignored** (H4 will fix).
- The popularity sort is **ascending by interaction count**, so
  `whether_retrieval(seq_ids, k)` returns the *least popular* k items first
  (i.e. the cold-start ones).

---

## 4. `ML1M/ml1m_raw/{train,test,valid}.csv` — leave-one-out splits

Pandas-readable CSV with header. Each row is one `(history, target)` pair.

| column | type | example |
|---|---|---|
| `user_id` | int | `1` |
| `history_movie_id` | str (Python list literal) | `"['3186', '1270', '1721', '1022', '2340', '1836', '3408', '2804', '1207', '1193']"` |
| `history_rating` | str (Python list literal of 0/1) | `"[1, 1, 1, 1, 0, 1, 1, 1, 1, 1]"` |
| `movie_id` | int | `720` |
| `rating` | int (0 or 1) | `0` |
| `timestamp` | int (unix) | `978300760` |

- `history_movie_id` length is always 10.
- `history_rating` is a binary {1=liked, 0=not liked} signal, but the JSON
  prompts in `10000_data_id_*.json` **list all 10 history items regardless
  of rating** — so this column is not used by current code.
- `timestamp` is the ground-truth interaction time, available but unused.

These CSVs are the source data; `10000_data_id_*.json` is the prompt-format
projection used at training time. The provided `10000_data_id_20.json`
contains **9 000 train + 1 000 test** samples (mixed in one file; the split
happens via index slicing inside the scripts).

---

## 5. `fb/filtered_full_fb.txt` — KG triples (raw text)

TSV, no header, **348 979 rows**. Each row is one Freebase triple:

```
/m/01006ysx<TAB>/film/film/other_crew<TAB>/m/05z5lc
/m/05z5lc  <TAB>/film/film/release_date_s<TAB>/m/010g913j
...
```

- `head_mid \t relation \t tail_mid`.
- Mid-encoded entities (`/m/...`); about ~14 700 distinct entities appear,
  matching the 14 669 nodes in `graphs/0.pt`.
- Distinct relations: ~177 (after counting unique `edge_attr` vectors in
  `0.pt`, see §7). The first 10 k triples already exhibit 112 different
  relations, so most relation diversity shows up early.

---

## 6. `fb/movies_with_mids.txt` — ML1M ↔ Freebase mid bridge

TSV, **3 883 rows**.

```
/m/0dyb1<TAB>Toy Story
/m/09w353<TAB>Jumanji
...
```

- 1:1 with `movies_id_name.txt` by **title**, but **only 98.2 %** of ML1M
  titles have a Freebase MID. The 1.8 % gap is what the retriever
  silently falls back to "no relevant subgraph" for.
- The current code does NOT use this mapping at runtime (it queries the KG
  by SBERT-encoded movie *name*, not by MID). Useful for any future
  improvement that wants to ground a movie directly to a KG node ID.

---

## 7. `fb/graphs/{0,layer2,layer3}.pt` — the actual KG used by `retrieve.py`

All three files are PyG `torch_geometric.data.Data` objects with **identical
graph structure** and **different node features**.

| field | value | dtype |
|---|---|---|
| `num_nodes` | 14 669 | — |
| `x` | tensor `[14669, 1024]` | `float32` — node features |
| `edge_index` | tensor `[2, 63203]` | `int64` — directed edges (head, tail) |
| `edge_attr` | tensor `[63203, 1024]` | `float32` — relation embeddings, one row per edge instance |
| `y` | `None` | — |

Differences in `x`:

| file | node-feature meaning |
|---|---|
| `0.pt` | SBERT (`all-roberta-large-v1`) text embedding of each entity name; **no GNN propagation** |
| `layer2_embeddings_W.pt` | output of one GCN layer (`index_KG.py:104-117`, line 113-114) |
| `layer3_embeddings_W.pt` | output of two GCN layers (line 116-117) |

The "Hop-Field Indexing" of the paper is **only partly implemented**:
`retrieve.py:24-25` loads `0.pt` (as `self.G`) and `layer2_embeddings_W.pt`
(as `self.G1`); `layer3_embeddings_W.pt` is generated but never read.

### About `edge_attr`

Although the tensor has 63 203 rows (one per edge), it has only **177 unique
vectors** — i.e. each unique relation type's SBERT embedding is repeated for
every edge of that type. A relation-aware GNN (H5) can dedupe this to a
`relation_embedding[177, 1024]` table and add a per-edge `relation_id`
tensor.

### About `edge_index` direction

The first 3 columns are `[[0,0,0],[1,2,3]]` — i.e. node 0 has out-edges to
nodes 1, 2, 3. Only the (head, tail) direction is stored (no reverse), so
when `retrieve.py:107-128` builds `networkx.Graph` from this it implicitly
makes the graph undirected via `nx.Graph()`. Any GNN that needs directed
semantics must add reverse edges itself.

---

## 8. Data-flow at training / eval time

```
JSON record
   │
   ├──► sequence_ids (10 ML1M ids)
   │       │
   │       ▼
   │   whether_retrieval()                     ┐
   │       │                                   │  retrieve.py
   │       ▼                                   │
   │   movie_id_to_name → SBERT → q_emb       │
   │       │                                   │
   │       ▼                                   │
   │   retrieval_topk(G, q_emb, k=3)            ┘
   │   retrieval_topk(G1, q_emb, k=3)
   │       │
   │       ▼
   │   re_ranking(global_q = SBERT(input))
   │       │
   │       ▼
   │   list[Data]   (10 PyG subgraphs, each ~ a center node + 1-hop edges)
   │
   ├──► input + questions
   │       │
   │       ▼
   │   prompt template (BOS + INST + watching history + options + EOS_USER)
   │
   └──► output letter
           │
           ▼
       gold answer (used at training as label, at eval as Recall@k target)

graph_llm.py
   list[Data] ─► GNN_Encoding ─► [N, 1024] node feats
                                 │
                                 ▼ scatter-mean per subgraph
                              [N_subg, 1024]
                                 │
                                 ▼ MLP projector to LLM dim
                              [N_subg, 4096]
                                 │
                                 ▼ ★★ mean(dim=0) ★★ ← H1 will replace this
                              [1, 4096]   (single soft prompt token)
                                 │
                                 ▼ prepend to BOS embedding
                              LLM forward → CE loss / argmax over A-T
```

---

## 9. Field invariants every new method must preserve

When implementing H1-H5, **do not** assume things outside this contract:

1. The model gets a **list of PyG `Data` objects** per sample. Each Data has
   `x`, `edge_index`, `edge_attr`, `num_nodes`. No batch dimension, no
   masks; subgraph sizes vary (1-50+ nodes per subgraph based on first
   inspection).
2. Node and edge features are **1024-dim float32 SBERT embeddings**.
3. Candidate selection is exactly **20 multiple-choice letters A-T**, target
   is one of those letters. Recall@k is computed on the LLM's
   `softmax(scores[A..T])` ranking (`graph_llm.py:256-305`).
4. `sequence_ids` are **ML1M movie ids 1..3952** (sparse). To get a KG node
   index, go through `movie_id_to_name` → SBERT → `top-1 cosine over G.x`.
   There is no direct id-to-node lookup table at runtime.
5. `ratings_45.txt` is the only popularity source (sorted ascending → least
   popular first). H2 may swap in a learned gate but must keep the
   "produces a list of items to retrieve for" interface.
6. Edges are **directed and stored once**. Anything that needs neighborhood
   in both directions must symmetrize manually.
7. `dataset/` paths are resolved **relative to repo-root cwd**, not the
   script's location. Don't `cd` into the method directory before running.

---

## 10. Quick reference: numbers to remember

| concept | value |
|---|---|
| KG nodes | 14 669 |
| KG edges (directed) | 63 203 |
| KG unique relation types (in graph) | ~177 |
| KG node feature dim | 1024 (SBERT) |
| ML1M movies in `movies_id_name.txt` | 3 883 |
| ML1M movies with FB MID | 3 814 (98.2%) |
| ML1M users (rating ≥ 4) | 6 038 |
| ML1M ratings ≥ 4 | 575 281 |
| Train records | 9 000 (first 9 000 of `10000_data_id_20.json`) |
| Eval records | 1 000 (last 1 000) |
| History length per record | 10 |
| Candidate count per record | 20 |
| LLaMA-2 hidden dim | 4 096 |
| Qwen2 hidden dim | 3 584 |

These are the constants that any H1-H5 implementation must respect.
