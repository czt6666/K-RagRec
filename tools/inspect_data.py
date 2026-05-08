"""One-shot probe of all dataset files. Run from repo root with
PYTHONPATH=methods/baseline."""
import json
from collections import Counter

import torch
import pandas as pd


def banner(s):
    print(f"\n{'='*8} {s} {'='*8}")


# ---------- KG embeddings ----------
banner("KG embedding .pt files")
for f in ["0.pt", "layer2_embeddings_W.pt", "layer3_embeddings_W.pt"]:
    g = torch.load(f"dataset/fb/graphs/{f}", weights_only=False)
    print(f"-- dataset/fb/graphs/{f} (type={type(g).__name__})")
    print(f"   num_nodes  : {g.num_nodes}")
    print(f"   x          : shape={tuple(g.x.shape)}  dtype={g.x.dtype}")
    print(f"   edge_index : shape={tuple(g.edge_index.shape)}  dtype={g.edge_index.dtype}")
    print(f"   edge_attr  : "
          f"{tuple(g.edge_attr.shape) if g.edge_attr is not None else None}")
    print(f"   x[0][:8]   : {[round(v, 4) for v in g.x[0][:8].tolist()]}")
    print(f"   ei sample  : {g.edge_index[:, :3].tolist()}")
    if hasattr(g, "y"):
        print(f"   y          : {g.y}")

# ---------- ML1M JSON ----------
banner("dataset/ML1M/10000_data_id_20.json")
with open("dataset/ML1M/10000_data_id_20.json") as f:
    arr = json.load(f)
print(f"records       : {len(arr)}")
print(f"keys          : {list(arr[0].keys())}")
seq0 = json.loads(arr[0]["sequence_ids"])
print(f"first record:")
print(f"  input       : {arr[0]['input'][:120]}...")
print(f"  output      : {arr[0]['output']}")
print(f"  questions   : {arr[0]['questions'][:120]}...")
print(f"  sequence_ids: {seq0}  (len={len(seq0)})")
out_dist = Counter(r["output"] for r in arr)
print(f"output letter distribution: {dict(sorted(out_dist.items()))}")
seq_lens = Counter(len(json.loads(r["sequence_ids"])) for r in arr)
print(f"sequence length distribution: {dict(seq_lens)}")

# ---------- ratings_45.txt ----------
banner("dataset/ML1M/ratings_45.txt")
df = pd.read_csv(
    "dataset/ML1M/ratings_45.txt",
    sep="\t",
    header=None,
    names=["user_id", "movie_id", "rating", "timestamp"],
)
print(f"rows           : {len(df):,}")
print(f"unique users   : {df.user_id.nunique():,}")
print(f"unique movies  : {df.movie_id.nunique():,}")
print(f"rating values  : {sorted(df.rating.unique().tolist())}")
print(f"timestamp range: {df.timestamp.min()} .. {df.timestamp.max()}")
print(f"interactions/user: min={df.groupby('user_id').size().min()}, "
      f"median={int(df.groupby('user_id').size().median())}, "
      f"max={df.groupby('user_id').size().max()}")
print(f"head:\n{df.head(3).to_string(index=False)}")

# ---------- movies_id_name.txt ----------
banner("dataset/ML1M/movies_id_name.txt")
m = pd.read_csv(
    "dataset/ML1M/movies_id_name.txt",
    sep="\t",
    header=None,
    names=["movie_id", "name"],
)
print(f"rows: {len(m):,}")
print(f"head:\n{m.head(5).to_string(index=False)}")
print(f"max movie_id: {m.movie_id.max()}  (gap = {m.movie_id.max() - len(m)})")

# ---------- KG raw triples ----------
banner("dataset/fb/filtered_full_fb.txt")
with open("dataset/fb/filtered_full_fb.txt", encoding="utf-8") as f:
    lines = f.readlines()
print(f"triples: {len(lines):,}")
heads, rels, tails = zip(*(l.strip().split("\t") for l in lines[:10000]))
print(f"first 5 triples:")
for ln in lines[:5]:
    print("  " + ln.strip())
print(f"unique relations (first 10k triples): {len(set(rels))}")

# ---------- movies_with_mids ----------
banner("dataset/fb/movies_with_mids.txt")
mw = pd.read_csv(
    "dataset/fb/movies_with_mids.txt",
    sep="\t",
    header=None,
    names=["mid", "name"],
)
print(f"rows: {len(mw):,}")
print(mw.head(5).to_string(index=False))

# ---------- correlation: how many JSON records have ALL items in KG mapping? ----------
banner("ML1M <-> KG mapping coverage")
m_id_to_name = dict(zip(m.movie_id, m.name))
mid_to_name = dict(zip(mw.mid, mw.name))
name_to_mid = {n: mid for mid, n in mid_to_name.items()}
movie_id_has_mid = sum(1 for mid_, name in m.values if name in name_to_mid)
print(f"ML1M movies (n={len(m)}) with FB mid: {movie_id_has_mid}")
print(f"% covered: {100*movie_id_has_mid/len(m):.1f}%")

# Are sequence_ids in the JSON pointing to ML1M movie_ids?
mids_seen = set()
for r in arr[:200]:
    mids_seen.update(json.loads(r["sequence_ids"]))
print(f"sequence_ids unique values (first 200 rec): {len(mids_seen)}")
print(f"  range: min={min(mids_seen)}, max={max(mids_seen)}")
print(f"  all in movies_id_name.txt? {mids_seen.issubset(set(m.movie_id))}")

# ---------- Edge attr in 0.pt ----------
banner("Edge attribute analysis (0.pt)")
g0 = torch.load("dataset/fb/graphs/0.pt", weights_only=False)
print(f"edge_attr shape: {tuple(g0.edge_attr.shape)}")
# edge_attr is a SBERT-encoded *relation text* embedding per edge
# sample similarity between first few edges
ea = g0.edge_attr[:100]
sim = torch.nn.functional.cosine_similarity(ea[:1], ea[1:100])
print(f"cosine similarity edge[0] vs edges[1..99]: "
      f"min={sim.min():.3f}  median={sim.median():.3f}  max={sim.max():.3f}")
# Are there many distinct relation embeddings or are there clusters (one per relation type)?
unique_rows = torch.unique(ea, dim=0)
print(f"unique edge-attr vectors in first 100 edges: {len(unique_rows)}")
# Most likely each relation type has the same SBERT vector everywhere it appears.
ea_full = g0.edge_attr
print(f"unique edge-attr vectors across ALL {len(ea_full)} edges: "
      f"{len(torch.unique(ea_full, dim=0))}  → ≈ number of relation types")
