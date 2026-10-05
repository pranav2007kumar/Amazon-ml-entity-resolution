"""Export leak-free validation scores of the XGBoost pipeline for the team's validation S1s.

Scores are the OUT-OF-FOLD probabilities of train_xgb.py (each S1 is scored by the fold model that never
trained on it; the dense retriever is out-of-fold the same way). Only S1 that are inside the train sub-world
(--work) have scores.

  python export_val.py --work work50 --val-ids model_export_frommodel2/our_val_s1_ids.txt
        --pairs model_export_frommodel2/pairs_to_score_val.parquet --out model_export_frommodel2/out_model2
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--val-ids", required=True)
    ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    val_ids = set(open(a.val_ids).read().split())
    s1_ids = pd.read_parquet(os.path.join(a.work, "train", "s1", "text.parquet"), columns=["entity_id"])["entity_id"]
    s1_arr = np.asarray(s1_ids.astype(object).tolist(), dtype=object)
    in_val = np.fromiter((x in val_ids for x in s1_arr), bool, len(s1_arr))
    log(f"val S1 inside the {a.work} sub-world: {int(in_val.sum()):,} of {len(val_ids):,} ({100 * in_val.sum() / len(val_ids):.1f}%)")

    oof = pd.read_parquet(os.path.join(a.work, "oof.parquet"), columns=["rec_key", "s1", "label", "prob"])
    oof = oof[in_val[oof["s1"].values]].reset_index(drop=True)
    src = (oof["rec_key"].values // 10 ** 9).astype(np.int8)
    rec = (oof["rec_key"].values % 10 ** 9).astype(np.int64)
    ids = {}
    for k in (2, 3):
        e = pd.read_parquet(os.path.join(a.work, "train", f"s{k}", "text.parquet"), columns=["entity_id"])["entity_id"]
        ids[k] = np.asarray(e.astype(object).tolist(), dtype=object)

    # ---- B.1 all candidate pairs of the val S1s
    path = os.path.join(a.out, "model2_val_scores.parquet")
    w, step = None, 5_000_000
    for i in range(0, len(oof), step):
        sl = slice(i, i + step)
        cand = np.empty(len(oof.iloc[sl]), dtype=object)
        for k in (2, 3):
            m = src[sl] == k
            cand[m] = ids[k][rec[sl][m]]
        t = pa.table({"source1_entity_id": pa.array(s1_arr[oof["s1"].values[sl]].tolist()),
                      "candidate_entity_id": pa.array(cand.tolist()),
                      "score": pa.array(oof["prob"].values[sl].astype(np.float32))})
        w = w or pq.ParquetWriter(path, t.schema, compression="zstd")
        w.write_table(t)
    w.close()
    log(f"B.1 wrote {path}: {len(oof):,} rows")

    # ---- B.2 the team's hard validation pairs
    pv = pd.read_parquet(a.pairs)
    r1 = pd.Series(np.arange(len(s1_arr)), index=s1_arr)
    pv["s1"] = r1.reindex(pv["source1_entity_id"].values).values
    pv["k"] = pv["candidate_entity_id"].str[1].astype(int)
    pv["rk"] = np.nan
    for k in (2, 3):
        rr = pd.Series(np.arange(len(ids[k])), index=ids[k])
        m = pv["k"] == k
        pv.loc[m, "rk"] = rr.reindex(pv.loc[m, "candidate_entity_id"].values).values + k * 10 ** 9
    # OOF covers only val S1; for hard pairs whose S1 is in the world but not in the val list use the full oof
    full = pd.read_parquet(os.path.join(a.work, "oof.parquet"), columns=["rec_key", "s1", "prob"])
    ok = pv["s1"].notna() & pv["rk"].notna()
    q = pv[ok].copy(); q["s1"] = q["s1"].astype(np.int64); q["rk"] = q["rk"].astype(np.int64)
    q = q.merge(full.rename(columns={"rec_key": "rk", "prob": "score"}), on=["rk", "s1"], how="left")
    pv["score"] = np.nan
    pv.loc[pv.index[ok], "score"] = q["score"].values
    out = pv[["source1_entity_id", "candidate_entity_id", "score"]].astype({"score": np.float32})
    p2 = os.path.join(a.out, "model2_val_ourpairs.parquet")
    out.to_parquet(p2, compression="zstd", index=False)
    log(f"B.2 wrote {p2}: {len(out):,} rows, scored {int(out['score'].notna().sum()):,} "
        f"({100 * out['score'].notna().mean():.1f}%), NaN = pair not in our candidate set / S1 outside the sub-world")

    # ---- self-checks
    from sklearn.metrics import roc_auc_score
    for name, d in (("B.1", pd.read_parquet(path, columns=["source1_entity_id", "candidate_entity_id", "score"])), ("B.2", out)):
        assert d["source1_entity_id"].str.startswith("S1-").all() and d["candidate_entity_id"].str[:2].isin(["S2", "S3"]).all()
        assert not d.duplicated(["source1_entity_id", "candidate_entity_id"]).any(), f"{name}: duplicate pairs"
        s = d["score"].dropna()
        assert s.min() >= 0 and s.max() <= 1, f"{name}: score out of [0,1]"
        log(f"{name} checks ok: {len(d):,} rows, no duplicates, scores in [0,1]")
    assert set(pd.read_parquet(path, columns=["source1_entity_id"])["source1_entity_id"].unique()) <= val_ids
    log("leakage check ok: every B.1 S1 is in the validation list and was scored out-of-fold")
    gt = os.path.join(a.work, "..", "dataset", "train", "train_ground_truth.tsv")
    if os.path.exists(gt):
        g = pd.read_csv(gt, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        pos = set(zip(g["source1_entity_id"].repeat(g["matched_entity_ids"].str.count(",") + (g["matched_entity_ids"] != "")),
                      ",".join(g["matched_entity_ids"][g["matched_entity_ids"] != ""]).split(",")))
        d = out.dropna(subset=["score"])
        y = np.fromiter(((s, c) in pos for s, c in zip(d["source1_entity_id"], d["candidate_entity_id"])), bool, len(d))
        log(f"B.2 scored pairs: {len(d):,}, positive rate {y.mean():.4f}, AUC {roc_auc_score(y, d['score']):.4f}")


if __name__ == "__main__":
    main()


