"""Unseen-country simulation (leave one country out), stage 1 only, on the dev chunks.

France appears only in the test set. To choose settings for it honestly, we pretend one training
country is unseen: train on country A only, then score the validation S1s of country B.
Reported for B:
  * F0.5 with the threshold tuned on A (what we would do blindly for France)
  * F0.5 with B's own best threshold (oracle; shows how far the threshold shifts)
  * the same after one round of self-training: B's own training rows are labelled by the model
    (confident matches p >= POS_P after one-parent assignment, confident non-matches p <= NEG_P)
    and added to the training set. No true labels of B are used.
Usage: python loco.py            (env: ER_WORK_DIR, ER_FOLDS for the dev chunks)
"""
import json
import os

import numpy as np
import polars as pl

from build_features import VAL_START, s1_groups
from config import WORK_DIR
from labels import positive_pairs
from two_stage import (FOLD_A, FOLD_B, MODEL_DIR, NEG_KEEP, assign, booster, fit_xgb, load, predict, to_dict,
                       tune)

POS_P, NEG_P, NEG_PSEUDO_KEEP = 0.97, 0.02, 0.05


def country_of_pairs(pairs, s1_country):
    return s1_country[pairs["s1"].to_numpy()]


def build(parts, s1_country, country, pseudo=None, seed=0):
    """Training matrix: labelled rows of `country` (+ pseudo-labelled rows)."""
    Xs, ys, ws = [], [], []
    rng = np.random.default_rng(seed)
    for t, (pr, X) in parts.items():
        if t == "val_0":
            continue
        c = country_of_pairs(pr, s1_country)
        y = pr["y"].to_numpy()
        m = c == country
        keep = m & ((y == 1) | (rng.random(len(y)) < NEG_KEEP))
        idx = np.nonzero(keep)[0]
        Xs.append(np.asarray(X[idx])); ys.append(y[idx])
        ws.append(np.where(y[idx] == 1, 1.0, 1.0 / NEG_KEEP).astype(np.float32))
        if pseudo is not None and t in pseudo:
            pidx, py = pseudo[t]
            Xs.append(np.asarray(X[pidx])); ys.append(py)
            ws.append(np.where(py == 1, 1.0, 1.0 / NEG_PSEUDO_KEEP).astype(np.float32))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def score_country(parts, s1_country, models, country):
    pr, X = parts["val_0"]
    m = country_of_pairs(pr, s1_country) == country
    idx = np.nonzero(m)[0]
    p = np.mean([predict(b, X[idx]) for b in models], axis=0)
    return pr[idx], p


def evaluate(pv, p, truth, rows, thr_fixed):
    from two_stage import macro_f05, rule_threshold
    res = tune(pv, p, truth, rows)
    a = assign(pv, p)
    fixed = macro_f05(to_dict(rule_threshold(a, thr_fixed)), truth, rows)
    return {"oracle_thr": res["threshold"]["thr"], "f_oracle": res["threshold"]["f05"], "f_at_source_thr": fixed}


def run():
    os.makedirs(MODEL_DIR, exist_ok=True)
    tags = FOLD_A + FOLD_B + ["val_0"]
    parts = {t: load(t) for t in tags}
    s1_country = pl.read_parquet(os.path.join(WORK_DIR, "train_s1.parquet"), columns=["country"])["country"].to_numpy()
    n1 = len(s1_country)
    val_rows = np.nonzero(s1_groups(n1) >= VAL_START)[0]
    pos = positive_pairs().filter(pl.col("s1").is_in(pl.Series(val_rows.astype(np.int32))))
    out = {}
    for src_c, tgt_c in (("US", "India"), ("India", "US")):
        print(f"\n=== train on {src_c}, unseen = {tgt_c}", flush=True)
        X, y, w = build(parts, s1_country, src_c)
        models = [fit_xgb(X, y, w, f"loco_{src_c}").get_booster()]
        del X
        rows_src = [int(r) for r in val_rows if s1_country[r] == src_c]
        rows_tgt = [int(r) for r in val_rows if s1_country[r] == tgt_c]
        truth_src = to_dict(pos.filter(pl.col("s1").is_in(pl.Series(np.array(rows_src, np.int32)))))
        truth_tgt = to_dict(pos.filter(pl.col("s1").is_in(pl.Series(np.array(rows_tgt, np.int32)))))
        pvs, ps = score_country(parts, s1_country, models, src_c)
        thr_src = tune(pvs, ps, truth_src, rows_src)["threshold"]["thr"]
        pvt, pt = score_country(parts, s1_country, models, tgt_c)
        r0 = evaluate(pvt, pt, truth_tgt, rows_tgt, thr_src)
        r0["source_thr"] = thr_src
        print(f"  no self-training: {r0}", flush=True)
        # self-training: label the target country's TRAIN rows with the model (no true labels)
        pseudo = {}
        rng = np.random.default_rng(3)
        for t in FOLD_A + FOLD_B:
            pr, Xc = parts[t]
            idx = np.nonzero(country_of_pairs(pr, s1_country) == tgt_c)[0]
            p = predict(models[0], Xc[idx])
            a = assign(pr[idx].with_columns(pl.Series("i", idx)), p)  # one parent per record
            best = pr[idx].select("s1", "src", "row").with_columns(pl.Series("p", p), pl.Series("i", idx)) \
                .join(a.select("s1", "src", "row"), on=["s1", "src", "row"], how="semi")
            pos_i = best.filter(pl.col("p") >= POS_P)["i"].to_numpy()
            neg_m = (p <= NEG_P) & (rng.random(len(p)) < NEG_PSEUDO_KEEP)
            neg_i = idx[neg_m]
            pi = np.concatenate([pos_i, neg_i])
            pseudo[t] = (pi, np.concatenate([np.ones(len(pos_i), np.int8), np.zeros(len(neg_i), np.int8)]))
            yt = pr["y"].to_numpy()
            print(f"  pseudo {t}: {len(pos_i):,} pos (true-label precision {yt[pos_i].mean():.4f}), "
                  f"{len(neg_i):,} neg (true-label neg rate {1 - yt[neg_i].mean():.4f})", flush=True)
        X, y, w = build(parts, s1_country, src_c, pseudo)
        models2 = [fit_xgb(X, y, w, f"loco_{src_c}_st").get_booster()]
        del X
        pvt, pt2 = score_country(parts, s1_country, models2, tgt_c)
        r1 = evaluate(pvt, pt2, truth_tgt, rows_tgt, thr_src)
        print(f"  with self-training: {r1}", flush=True)
        out[f"{src_c}->{tgt_c}"] = {"plain": r0, "self_train": r1}
    json.dump(out, open(os.path.join(MODEL_DIR, "loco_results.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
