"""B1: build cross-encoder training pairs from TRAIN data only, excluding the validation S1s.

Rows come from V3 out-of-fold scores (v3_oof_scores.parquet). Keep: all pairs with 0.02 < p < 0.98,
10 % of positives outside that band, 0.2 % of negatives outside it. Uncertain pairs are capped at 1.5 M.
Text is RAW (accents/case/script kept): "name | address".
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

SEED = 42
ap = argparse.ArgumentParser()
ap.add_argument("--oof", required=True)
ap.add_argument("--gt", required=True)
ap.add_argument("--val_ids", required=True)
ap.add_argument("--data_dir", required=True, help="folder holding train_source{1,2,3}.tsv")
ap.add_argument("--out", default="ce_data")
ap.add_argument("--cap", type=int, default=1_500_000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.RandomState(SEED)

val = set(x.strip() for x in open(a.val_ids, encoding="utf-8") if x.strip())
print("val S1 ids", len(val), flush=True)

gt = pd.read_csv(a.gt, sep="\t", dtype=str, keep_default_na=False, quoting=3)
gt = gt[gt.matched_entity_ids != ""]
gt = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
gk = np.sort(pd.util.hash_array((gt.source1_entity_id + "|" + gt.m).values.astype(object)))
del gt
print("gt keys", len(gk), flush=True)

pf = pq.ParquetFile(a.oof)
parts = []
n_all = n_val = 0
for b in pf.iter_batches(batch_size=4_000_000, columns=["source1_entity_id", "candidate_entity_id", "score"]):
    d = b.to_pandas()
    n_all += len(d)
    keep = ~d.source1_entity_id.isin(val).values
    n_val += int((~keep).sum())
    d = d[keep]
    h = pd.util.hash_array((d.source1_entity_id + "|" + d.candidate_entity_id).values.astype(object))
    i = np.searchsorted(gk, h)
    y = (gk[np.minimum(i, len(gk) - 1)] == h).astype(np.int8)
    p = d.score.values
    unc = (p > 0.02) & (p < 0.98)
    r = rng.random_sample(len(d))
    sel = unc | ((y == 1) & (r < 0.10)) | ((y == 0) & (r < 0.002))
    x = d[sel].copy()
    x["y"] = y[sel]
    x["unc"] = unc[sel]
    parts.append(x)
    print(f"  scanned {n_all:,} kept so far {sum(len(q) for q in parts):,}", flush=True)
df = pd.concat(parts, ignore_index=True)
del parts
assert not df.source1_entity_id.isin(val).any(), "LEAK: validation S1 in training pairs"
n_unc = int(df.unc.sum())
if n_unc > a.cap:
    keep_unc = rng.choice(np.where(df.unc.values)[0], a.cap, replace=False)
    m = ~df.unc.values
    m[keep_unc] = True
    df = df[m]
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"rows {len(df):,} uncertain {int(df.unc.sum()):,} pos_rate {df.y.mean():.3f} (dropped val rows: {n_val:,})", flush=True)

# raw text lookup, only for ids we need
need = pa.array(pd.unique(np.concatenate([df.source1_entity_id.values, df.candidate_entity_id.values])))
tabs = []
for k in (1, 2, 3):
    t = pacsv.read_csv(os.path.join(a.data_dir, f"train_source{k}.tsv"),
                       parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                       convert_options=pacsv.ConvertOptions(column_types={"entity_id": pa.string(), "business_name": pa.string(),
                                                                          "business_address": pa.string(), "country": pa.string()}))
    t = t.filter(pc.is_in(t["entity_id"], value_set=need))
    tabs.append(t.select(["entity_id", "business_name", "business_address"]).to_pandas())
    print("source", k, len(tabs[-1]), flush=True)
txt = pd.concat(tabs, ignore_index=True).drop_duplicates("entity_id").set_index("entity_id")
txt = (txt.business_name.fillna("") + " | " + txt.business_address.fillna(""))
df["text_a"] = txt.reindex(df.source1_entity_id.values).fillna(" | ").values
df["text_b"] = txt.reindex(df.candidate_entity_id.values).fillna(" | ").values
df = df.rename(columns={"score": "p_v3"})[["source1_entity_id", "candidate_entity_id", "text_a", "text_b", "y", "p_v3"]]
df.to_parquet(os.path.join(a.out, "train_ce.parquet"), compression="zstd", index=False)
json.dump({"rows": int(len(df)), "pos_rate": float(df.y.mean()), "val_rows_dropped": n_val,
           "val_s1_in_training": 0}, open(os.path.join(a.out, "build_report.json"), "w"), indent=1)
print(df.sample(5, random_state=1)[["text_a", "text_b", "y"]].to_string())
