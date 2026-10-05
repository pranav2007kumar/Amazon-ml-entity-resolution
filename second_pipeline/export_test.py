"""Export raw XGBoost probabilities of every test candidate pair (deliverables A and C). No TSV is written.

score = mean of the two fold-model probabilities, before the threshold and before the one-to-one step.
  A: <out>/model2_test_scores.parquet   source1_entity_id, candidate_entity_id, score   (all candidate pairs)
  C: <out>/model2_test_ourpairs.parquet the given hard pairs; NaN where the pair is not a candidate
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from common import log


def obj(s):
    return np.asarray(pd.Series(s).astype(object).tolist(), dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default=None, help="hard-pairs parquet for deliverable C")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--batch", type=int, default=2_000_000)
    a = ap.parse_args()
    import xgboost as xgb

    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.work, "models", "xgb_meta.json")))
    cols = meta["features"]
    models = []
    for f in (0, 1):
        b = xgb.Booster(); b.load_model(os.path.join(a.work, "models", f"xgb_f{f}.json"))
        b.set_param({"device": a.device}); models.append(b)
    W = os.path.join(a.work, "test")
    s1_ids = obj(pd.read_parquet(os.path.join(W, "s1", "text.parquet"), columns=["entity_id"])["entity_id"])
    ids = {k: obj(pd.read_parquet(os.path.join(W, f"s{k}", "text.parquet"), columns=["entity_id"])["entity_id"])
           for k in (2, 3)}
    path = os.path.join(a.out, "model2_test_scores.parquet")
    writer, total = None, 0
    keys, scores = [], []
    for k in (2, 3):
        d = ds.dataset(os.path.join(W, f"feats_s{k}.parquet"))
        for b in d.to_batches(columns=cols + ["rec", "s1"], batch_size=a.batch):
            df = b.to_pandas()
            p = np.mean([m.inplace_predict(df[cols]) for m in models], axis=0).astype(np.float32)
            rec, s1 = df["rec"].values.astype(np.int64), df["s1"].values.astype(np.int64)
            t = pa.table({"source1_entity_id": pa.array(s1_ids[s1].tolist()),
                          "candidate_entity_id": pa.array(ids[k][rec].tolist()), "score": pa.array(p)})
            writer = writer or pq.ParquetWriter(path, t.schema, compression="zstd")
            writer.write_table(t)
            keys.append(s1 * (1 << 33) + (k - 2) * (1 << 32) + rec); scores.append(p)
            total += len(p)
        log(f"S{k}: done, {total:,} pairs written so far")
    writer.close()
    log(f"A wrote {path}: {total:,} rows")

    if a.pairs:
        key = np.concatenate(keys); sc = np.concatenate(scores); del keys, scores
        o = np.argsort(key); key, sc = key[o], sc[o]
        pv = pd.read_parquet(a.pairs)
        r1 = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
        s1r = r1.reindex(pv["source1_entity_id"].values).values
        k_ = pv["candidate_entity_id"].str[1].astype(int).values
        recr = np.full(len(pv), np.nan)
        for k in (2, 3):
            m = k_ == k
            recr[m] = pd.Series(np.arange(len(ids[k])), index=ids[k]).reindex(pv["candidate_entity_id"].values[m]).values
        ok = ~(np.isnan(s1r) | np.isnan(recr))
        q = np.zeros(len(pv), np.int64)
        q[ok] = s1r[ok].astype(np.int64) * (1 << 33) + (k_[ok] - 2) * (1 << 32) + recr[ok].astype(np.int64)
        pos = np.clip(np.searchsorted(key, q), 0, len(key) - 1)
        hit = ok & (key[pos] == q)
        out = pv[["source1_entity_id", "candidate_entity_id"]].copy()
        out["score"] = np.where(hit, sc[pos], np.nan).astype(np.float32)
        p2 = os.path.join(a.out, "model2_test_ourpairs.parquet")
        out.to_parquet(p2, compression="zstd", index=False)
        log(f"C wrote {p2}: {len(out):,} rows, scored {int(hit.sum()):,} ({100 * hit.mean():.1f}%), NaN {int((~hit).sum()):,}")

    d = pq.ParquetFile(path)
    assert d.metadata.num_rows == total
    first = d.read_row_group(0).to_pandas()
    assert first["source1_entity_id"].str.startswith("S1-").all() and first["candidate_entity_id"].str[:2].isin(["S2", "S3"]).all()
    mn = mx = None
    for i in range(d.num_row_groups):
        s = d.read_row_group(i, columns=["score"])["score"].to_numpy()
        assert not np.isnan(s).any()
        mn = s.min() if mn is None else min(mn, s.min()); mx = s.max() if mx is None else max(mx, s.max())
    assert mn >= 0 and mx <= 1
    log(f"A checks ok: rows {total:,} = pairs scored, no NaN, score range [{mn:.4f}, {mx:.4f}]")


if __name__ == "__main__":
    main()
