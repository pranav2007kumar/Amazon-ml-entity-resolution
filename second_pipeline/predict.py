"""Stage 5 - score test candidates, one-to-one assignment, write both submission files from ONE run.

output/matching_results.tsv  : every test S1 exactly once, matches = subset of candidates
output/candidate_pairs.tsv   : the exact pairs the model scored (union of sparse + dense candidates)
Also prints a label-free per-country check (France vs US/India shape).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from common import assign_best, log, read_tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tau", type=float, default=None, help="override threshold from training")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-stack", action="store_true", help="ignore the stacking stage")
    ap.add_argument("--save-probs", action="store_true", help="save stage-1 probabilities per source (for blending)")
    a = ap.parse_args()
    import xgboost as xgb

    meta = json.load(open(os.path.join(a.work, "models", "xgb_meta.json")))
    cols, tau = meta["features"], (a.tau if a.tau is not None else meta["tau"])
    models = []
    for f in (0, 1):
        b = xgb.Booster(); b.load_model(os.path.join(a.work, "models", f"xgb_f{f}.json"))
        b.set_param({"device": a.device})
        models.append(b)
    W = os.path.join(a.work, "test")
    t1 = pd.read_parquet(os.path.join(W, "s1", "text.parquet"), columns=["entity_id", "country"])
    ids = {k: pd.read_parquet(os.path.join(W, f"s{k}", "text.parquet"), columns=["entity_id"])["entity_id"].values
           for k in (2, 3)}
    kept, cands, allp = [], [], []
    for k in (2, 3):
        d = ds.dataset(os.path.join(W, f"feats_s{k}.parquet"))
        rec, s1, pr = [], [], []
        for b in d.to_batches(columns=cols + ["rec", "s1"], batch_size=2_000_000):
            df = b.to_pandas()
            p = np.mean([m.inplace_predict(df[cols]) for m in models], axis=0).astype(np.float32)
            rec.append(df["rec"].values); s1.append(df["s1"].values); pr.append(p)
        rec, s1, pr = np.concatenate(rec), np.concatenate(s1), np.concatenate(pr)
        cands.append((k, rec, s1))
        allp.append((k, rec, s1, pr))
        if a.save_probs:
            np.savez(os.path.join(W, f"xgb_probs_s{k}.npz"), rec=rec.astype(np.int32), s1=s1.astype(np.int32), p=pr)
        log(f"S{k}: {len(rec):,} pairs scored by stage-1")

    sm_path = os.path.join(a.work, "models", "stack_meta.json")
    if os.path.exists(sm_path) and json.load(open(sm_path))["use"] and not a.no_stack:
        from stack import stack_features
        sm = json.load(open(sm_path))
        tau = a.tau if a.tau is not None else sm["tau"]
        rk_all = np.concatenate([r.astype(np.int64) + k * 10 ** 9 for k, r, _, _ in allp])
        s1_all_ = np.concatenate([s for _, _, s, _ in allp])
        p_all = np.concatenate([p for _, _, _, p in allp])
        X = stack_features(rk_all, s1_all_, p_all)
        sb = []
        for f in (0, 1):
            b = xgb.Booster(); b.load_model(os.path.join(a.work, "models", f"stack_f{f}.json"))
            b.set_param({"device": a.device}); sb.append(b)
        p_all = np.mean([b.inplace_predict(X) for b in sb], axis=0).astype(np.float32)
        off = 0
        allp = [(k, r, s, p_all[off:(off := off + len(r))]) for k, r, s, _ in allp]
        log(f"stacking applied (tau {tau})")
    for k, rec, s1, pr in allp:
        best = assign_best(rec.astype(np.int64), pr)
        best = best[pr[best] >= tau]
        kept.append(pd.DataFrame({"s1": s1[best], "other": ids[k][rec[best]], "prob": pr[best]}))
        log(f"S{k}: {len(best):,} matches kept (tau {tau})")
    kept = pd.concat(kept, ignore_index=True)

    os.makedirs(a.out_dir, exist_ok=True)
    order = read_tsv(os.path.join(a.data_dir, "test", "test_source1.tsv"), usecols=["entity_id"])["entity_id"]
    row_of = pd.Series(np.arange(len(t1)), index=t1["entity_id"].values)
    m = kept.groupby("s1")["other"].agg(",".join)
    lists = pd.Series("", index=np.arange(len(t1)), dtype=object)
    lists.loc[m.index] = m.values
    with open(os.path.join(a.out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for eid, r in zip(order.values, row_of.reindex(order.values).values):
            fh.write(f"{eid}\t{lists.iat[r]}\n")
    log("wrote matching_results.tsv")

    from write_candidates import write_candidates
    obj = lambda s: np.asarray(pd.Series(s).astype(object).tolist(), dtype=object)
    write_candidates(cands, {k: obj(v) for k, v in ids.items()}, obj(t1["entity_id"].values),
                     order.tolist(), os.path.join(a.out_dir, "candidate_pairs.tsv"))
    log("wrote candidate_pairs.tsv")

    kept["country"] = t1["country"].values[kept["s1"].values]
    n = kept.groupby("s1").size().reindex(np.arange(len(t1))).fillna(0)
    rep = pd.DataFrame({"country": t1["country"].values, "n": n.values}).groupby("country")["n"] \
        .agg(s1="size", empty_pct=lambda x: round(100 * (x == 0).mean(), 2), matches_per_s1="mean")
    rep["median_prob_kept"] = kept.groupby("country")["prob"].median()
    print("\nPer-country shape of the submission (train truth: 5.59% empty, 3.46 matches/S1 in every country;")
    print("with our recall expect ~ the same numbers for US and India, and France should look similar):")
    print(rep.round(4).to_string())


if __name__ == "__main__":
    main()
