"""Stage 4b - competition-aware second stage on top of the out-of-fold XGBoost probabilities.

Stage-1 scores each (S2/S3 record, S1) pair alone. Real errors are competition errors: the same record
looks good for two S1, or one S1 has two strong rivals. This stage sees, for every pair, how its stage-1
probability compares with the other candidates of the same record and of the same S1 (per source), then
re-scores. Country is never used. Keep it only if fold scores beat stage-1 (the script prints both).

  python stack.py --work work          (after train_xgb.py)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import assign_best, f05_from_counts, log, stable_fold
from train_xgb import TAUS, score

STACK_COLS = ["p", "logit", "rk_o", "gap_o", "mg_o", "top2_o", "n_o", "rk_s", "gap_s", "mg_s", "n_s",
              "cnt_hi_s", "sum_p_s", "src3", "hi_same_other", "hi_cross", "sum_same_other"]


def _group(key, p):
    o = np.lexsort((-p, key))
    ks, ps = key[o], p[o]
    first = np.ones(len(ks), bool)
    first[1:] = ks[1:] != ks[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(len(ks)), 0))
    rank = np.arange(len(ks)) - start
    size = np.diff(np.append(np.flatnonzero(first), len(ks)))[np.cumsum(first) - 1]
    top1 = ps[start]
    nxt = np.minimum(start + 1, len(ks) - 1)
    top2 = np.where(size > 1, ps[nxt], 0.0)
    out = [np.empty(len(ks), np.float32) for _ in range(4)]
    for arr, v in zip(out, (rank, top1, top2, size)):
        arr[o] = v
    return out


def stack_features(rec_key, s1, p):
    """rec_key = record row + src*10**9 (src 2/3); returns DataFrame of STACK_COLS."""
    p = p.astype(np.float32)
    src = (rec_key // 10 ** 9).astype(np.int64)
    rk_o, t1_o, t2_o, n_o = _group(rec_key, p)
    rk_s, t1_s, t2_s, n_s = _group(s1.astype(np.int64) * 4 + src, p)
    n1 = int(s1.max()) + 1
    hi = np.bincount(s1, weights=(p > 0.5), minlength=n1).astype(np.float32)
    sp = np.bincount(s1, weights=p, minlength=n1).astype(np.float32)
    ks = s1.astype(np.int64) * 4 + src
    hi9 = (p > 0.9).astype(np.float32)
    same_hi = np.bincount(ks, weights=hi9, minlength=n1 * 4 + 4).astype(np.float32)
    all_hi = np.bincount(s1, weights=hi9, minlength=n1).astype(np.float32)
    hi_same_other = same_hi[ks] - hi9
    hi_cross = all_hi[s1] - same_hi[ks]
    sum_same_other = np.bincount(ks, weights=p, minlength=n1 * 4 + 4).astype(np.float32)[ks] - p
    mg_o = np.where(rk_o == 0, p - t2_o, p - t1_o)
    mg_s = np.where(rk_s == 0, p - t2_s, p - t1_s)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return pd.DataFrame({"p": p, "logit": np.log(pc / (1 - pc)), "rk_o": rk_o, "gap_o": t1_o - p,
                         "mg_o": mg_o, "top2_o": t2_o, "n_o": n_o, "rk_s": rk_s, "gap_s": t1_s - p,
                         "mg_s": mg_s, "n_s": n_s, "cnt_hi_s": hi[s1], "sum_p_s": sp[s1],
                         "src3": (src == 3).astype(np.float32), "hi_same_other": hi_same_other,
                         "hi_cross": hi_cross, "sum_same_other": sum_same_other})[STACK_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--rounds", type=int, default=300); ap.add_argument("--depth", type=int, default=7)
    a = ap.parse_args()
    import xgboost as xgb

    oof = pd.read_parquet(os.path.join(a.work, "oof.parquet"))
    t1 = pd.read_parquet(os.path.join(a.work, "train", "s1", "text.parquet"), columns=["entity_id", "country"])
    n1 = len(t1)
    fold1 = stable_fold(t1["entity_id"].values, salt="xgb")
    ngt = np.zeros(n1)
    for k in (2, 3):
        ngt += np.bincount(np.load(os.path.join(a.work, "train", f"gt_s{k}.npz"))["s1"], minlength=n1)
    rk, s1, lb, p0 = (oof["rec_key"].values, oof["s1"].values, oof["label"].values.astype(float),
                      oof["prob"].values)
    X = stack_features(rk, s1, p0)
    fo = fold1[s1]
    prm = dict(objective="binary:logistic", tree_method="hist", device=a.device, max_depth=a.depth,
               learning_rate=0.1, subsample=0.8, colsample_bytree=0.9, min_child_weight=5, seed=7)
    p1 = np.zeros(len(X), np.float32)
    models = []
    for f in (0, 1):
        tr = fo != f
        keep = tr
        bst = xgb.train(prm, xgb.QuantileDMatrix(X[keep], label=lb[keep]), num_boost_round=a.rounds)
        models.append(bst)
        bst.save_model(os.path.join(a.work, "models", f"stack_f{f}.json"))
        p1[fo == f] = bst.inplace_predict(X[fo == f]).astype(np.float32)
        log(f"stack model {f} trained")

    def best_tau(p):
        r = [(t, *score(rk, s1, p, lb, ngt, t)) for t in TAUS]
        return max(r, key=lambda x: x[1])

    b0, b1 = best_tau(p0), best_tau(p1)
    tau = float(b1[0])
    f1s = [score(rk, s1, p1, lb, ngt, tau, fold1 == f)[0] for f in (0, 1)]
    f0s = [score(rk, s1, p0, lb, ngt, float(b0[0]), fold1 == f)[0] for f in (0, 1)]
    print("\n================ STACKING vs STAGE-1 (leaderboard formula) ================")
    print(f"stage-1 : F0.5 {b0[1]:.5f} P {b0[2]:.5f} R {b0[3]:.5f} tau {b0[0]} | folds {f0s[0]:.5f} {f0s[1]:.5f}")
    print(f"stacked : F0.5 {b1[1]:.5f} P {b1[2]:.5f} R {b1[3]:.5f} tau {b1[0]} | folds {f1s[0]:.5f} {f1s[1]:.5f}")
    use = bool(f1s[0] > f0s[0] and f1s[1] > f0s[1])
    print(f"stacking improves BOTH folds: {use}  -> predict.py will {'use' if use else 'skip'} it")
    for c in pd.unique(t1["country"]):
        m = (t1["country"] == c).values
        print(f"country {c}: stage-1 {score(rk, s1, p0, lb, ngt, float(b0[0]), m)[0]:.5f} | "
              f"stacked {score(rk, s1, p1, lb, ngt, tau, m)[0]:.5f}")
    json.dump({"use": use, "tau": tau, "cv_f05": b1[1], "fold_f05": f1s},
              open(os.path.join(a.work, "models", "stack_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
