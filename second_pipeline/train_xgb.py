"""Stage 4 - XGBoost on GPU, 2-fold out-of-fold by S1, one-to-one assignment, threshold for macro F0.5.

Prints the CV macro F0.5 exactly as the leaderboard computes it (all S1 of the sub-world, singletons
included, S1 with no candidate counted too). With --holdout it also trains on one country and scores
the other (US->India, India->US): the gap is the best available estimate of the France penalty.

Saves <work>/models/xgb_f0.json, xgb_f1.json, xgb_meta.json (features, threshold) and oof.parquet.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from common import assign_best, f05_from_counts, log, stable_fold

DROP = {"rec", "s1", "label", "u"}
TAUS = np.round(np.concatenate([np.arange(0.20, 0.951, 0.01), np.arange(0.955, 0.9951, 0.005),
                                np.array([0.997, 0.998, 0.999, 0.9995, 0.9998])]), 4)


def feat_cols(work):
    names = ds.dataset(os.path.join(work, "train", "feats_s2.parquet")).schema.names
    return [c for c in names if c not in DROP]


def load_sample(work, cols, neg_frac, extra_filter=None):
    parts = []
    for k in (2, 3):
        d = ds.dataset(os.path.join(work, "train", f"feats_s{k}.parquet"))
        flt = (ds.field("label") == 1) | (ds.field("u") < neg_frac)
        t = d.to_table(columns=cols + ["s1", "label"], filter=flt).to_pandas()
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def xgb_params(a):
    return dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device=a.device,
                max_depth=a.depth, learning_rate=a.lr, subsample=0.8, colsample_bytree=0.8, max_bin=256,
                min_child_weight=2, seed=42)


def fit(X, y, a, rounds):
    import xgboost as xgb
    dm = xgb.QuantileDMatrix(X, label=y)
    return xgb.train(xgb_params(a), dm, num_boost_round=rounds)


def predict(bst, X):
    import xgboost as xgb
    return bst.inplace_predict(X).astype(np.float32)


_BEST = {}


def score(rec_key, s1, prob, label, ngt, tau, s1_mask=None):
    """macro F0.5 over all S1 (optionally restricted by s1_mask) after one-to-one + threshold."""
    if _BEST.get("p") is not prob:
        _BEST["p"] = prob
        _BEST["b"] = assign_best(rec_key, prob)
    best = _BEST["b"]
    best = best[prob[best] >= tau]
    n1 = len(ngt)
    npred = np.bincount(s1[best], minlength=n1)
    tp = np.bincount(s1[best], weights=label[best], minlength=n1)
    f, p, r = f05_from_counts(tp, npred, ngt)
    if s1_mask is not None:
        f, p, r = f[s1_mask], p[s1_mask], r[s1_mask]
    return f.mean(), p.mean(), r.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rounds", type=int, default=800); ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--neg-frac", type=float, default=0.2, help="share of negatives used for fitting")
    ap.add_argument("--holdout", action="store_true", help="also run US<->India cross-country test")
    a = ap.parse_args()

    cols = feat_cols(a.work)
    t1 = pd.read_parquet(os.path.join(a.work, "train", "s1", "text.parquet"), columns=["entity_id", "country"])
    n1 = len(t1)
    fold1 = stable_fold(t1["entity_id"].values, salt="xgb")
    ngt = np.zeros(n1)
    for k in (2, 3):
        ngt += np.bincount(np.load(os.path.join(a.work, "train", f"gt_s{k}.npz"))["s1"], minlength=n1)
    log(f"{len(cols)} features; {n1:,} S1 in sub-world, {int((ngt == 0).sum()):,} singletons")

    tr = load_sample(a.work, cols, a.neg_frac)
    log(f"training sample: {len(tr):,} rows, {int(tr['label'].sum()):,} positives")
    models = []
    for f in (0, 1):
        m = fold1[tr["s1"].values] != f
        bst = fit(tr.loc[m, cols], tr.loc[m, "label"], a, a.rounds)
        models.append(bst)
        os.makedirs(os.path.join(a.work, "models"), exist_ok=True)
        bst.save_model(os.path.join(a.work, "models", f"xgb_f{f}.json"))
        log(f"fold model {f} trained on {int(m.sum()):,} rows")
    imp = pd.Series(models[0].get_score(importance_type="gain")).sort_values(ascending=False)
    del tr

    # out-of-fold prediction for EVERY candidate pair
    oof = []
    for k in (2, 3):
        d = ds.dataset(os.path.join(a.work, "train", f"feats_s{k}.parquet"))
        for b in d.to_batches(columns=cols + ["rec", "s1", "label"], batch_size=2_000_000):
            df = b.to_pandas()
            fo = fold1[df["s1"].values]
            p = np.zeros(len(df), np.float32)
            for f in (0, 1):
                m = fo == f
                if m.any():
                    p[m] = predict(models[f], df.loc[m, cols])
            oof.append(pd.DataFrame({"rec_key": df["rec"].values.astype(np.int64) + k * 10**9,
                                     "s1": df["s1"].values, "label": df["label"].values, "prob": p}))
    oof = pd.concat(oof, ignore_index=True)
    oof.to_parquet(os.path.join(a.work, "oof.parquet"), index=False)
    rk, s1, pr, lb = oof["rec_key"].values, oof["s1"].values, oof["prob"].values, oof["label"].values.astype(float)

    res = [(t, *score(rk, s1, pr, lb, ngt, t)) for t in TAUS]
    best = max(res, key=lambda x: x[1])
    tau = float(best[0])
    folds_f = [score(rk, s1, pr, lb, ngt, tau, fold1 == f)[0] for f in (0, 1)]
    print("\n================ CROSS-VALIDATION (leaderboard formula) ================")
    print(f"macro F0.5 {best[1]:.5f} | precision {best[2]:.5f} | recall {best[3]:.5f} | threshold {tau}")
    print(f"fold 0: {folds_f[0]:.5f}   fold 1: {folds_f[1]:.5f}")
    for c in pd.unique(t1["country"]):
        m = (t1["country"] == c).values
        print(f"country {c}: F0.5 {score(rk, s1, pr, lb, ngt, tau, m)[0]:.5f}")
    print("\ntop 15 features by gain:")
    print((imp / imp.sum()).head(15).round(4).to_string())

    hold = {}
    if a.holdout:
        print("\n================ UNSEEN-COUNTRY TEST (France proxy) ================")
        cols_c = cols
        sample = load_sample(a.work, cols_c, a.neg_frac * 0.5)
        sample["country"] = t1["country"].values[sample["s1"].values]
        cty = t1["country"].values[s1]
        for c in pd.unique(t1["country"]):
            m = sample["country"] != c
            bst = fit(sample.loc[m, cols_c], sample.loc[m, "label"], a, a.rounds // 2)
            sel = cty == c
            chunks = []
            for k in (2, 3):
                d = ds.dataset(os.path.join(a.work, "train", f"feats_s{k}.parquet"))
                for b in d.to_batches(columns=cols_c + ["s1"], batch_size=2_000_000):
                    df = b.to_pandas()
                    chunks.append(predict(bst, df[cols_c]))
            px = np.concatenate(chunks)
            px[~sel] = 0
            mask = (t1["country"] == c).values
            fx = score(rk, s1, px, lb, ngt, tau, mask)[0]
            fo = max(score(rk, s1, px, lb, ngt, t, mask)[0] for t in TAUS)
            fin = score(rk, s1, pr, lb, ngt, tau, mask)[0]
            hold[c] = {"in_country": fin, "unseen": fx, "unseen_best_tau": fo}
            print(f"{c}: in-country {fin:.5f} | trained without {c}: {fx:.5f} "
                  f"(with its own best threshold {fo:.5f}) | gap {fin - fx:+.5f}")

    json.dump({"features": cols, "tau": tau, "cv_f05": best[1], "fold_f05": folds_f, "holdout": hold},
              open(os.path.join(a.work, "models", "xgb_meta.json"), "w"), indent=2)
    log("saved models/xgb_f0.json, xgb_f1.json, xgb_meta.json, oof.parquet")


if __name__ == "__main__":
    main()
