"""Error analysis of the out-of-fold predictions (India/US validation). No test data.

  python analyze.py --work work [--oof oof.parquet]

Prints per-country / per-source P, R, F0.5, singleton accuracy and where the lost points go:
  FN not in candidates (retrieval), FN below threshold (model), FN lost in one-to-one,
  FP on records with a different true S1 (competition), FP on records matching nothing (distractors),
  each split by duplicate-name crowding, missing address and Indian script.
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import assign_best, f05_from_counts, log
from train_xgb import TAUS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--oof", default="oof.parquet")
    ap.add_argument("--tau", type=float, default=None)
    a = ap.parse_args()
    W = a.work
    oof = pd.read_parquet(os.path.join(W, a.oof))
    t1 = pd.read_parquet(os.path.join(W, "train", "s1", "text.parquet"), columns=["country", "core_cnt"])
    n1 = len(t1)
    rk, s1, pr, lb = (oof["rec_key"].values, oof["s1"].values, oof["prob"].values,
                      oof["label"].values.astype(np.int8))
    ngt = np.zeros(n1)
    owner, recinfo = {}, {}
    for k in (2, 3):
        g = np.load(os.path.join(W, "train", f"gt_s{k}.npz"))
        ngt += np.bincount(g["s1"], minlength=n1)
        tk = pd.read_parquet(os.path.join(W, "train", f"s{k}", "text.parquet"), columns=["indic", "addr"])
        o = np.full(len(tk), -1, np.int64); o[g["rec"]] = g["s1"]
        owner[k] = o
        recinfo[k] = (tk["indic"].values, (tk["addr"].str.len() == 0).values)
    src = (rk // 10 ** 9).astype(np.int8)
    rec = (rk % 10 ** 9).astype(np.int64)
    own = np.where(src == 2, 0, 0).astype(np.int64)
    indic = np.zeros(len(oof), bool); noaddr = np.zeros(len(oof), bool)
    for k in (2, 3):
        m = src == k
        own[m] = owner[k][rec[m]]
        indic[m] = recinfo[k][0][rec[m]]; noaddr[m] = recinfo[k][1][rec[m]]
    best = assign_best(rk, pr)
    cty = t1["country"].values

    def macro(tau, mask=None):
        b = best[pr[best] >= tau]
        npred = np.bincount(s1[b], minlength=n1)
        tp = np.bincount(s1[b], weights=lb[b], minlength=n1)
        f, p, r = f05_from_counts(tp, npred, ngt)
        if mask is not None:
            f, p, r = f[mask], p[mask], r[mask]
        return f.mean(), p.mean(), r.mean(), (npred[mask if mask is not None else slice(None)] == 0)[
            (ngt[mask if mask is not None else slice(None)] == 0)].mean()

    tau = a.tau or max(TAUS, key=lambda t: macro(t)[0])
    print(f"\nthreshold {tau} (searched {TAUS[0]}..{TAUS[-1]})")
    rows = []
    for name, m in [("ALL", None)] + [(c, cty == c) for c in pd.unique(cty)]:
        f, p, r, sg = macro(tau, m)
        rows.append({"slice": name, "F0.5": round(f, 5), "precision": round(p, 5), "recall": round(r, 5),
                     "singleton_acc": round(sg, 5)})
    print(pd.DataFrame(rows).to_string(index=False))

    sel = np.zeros(len(oof), bool); sel[best[pr[best] >= tau]] = True
    tot_gt = int(ngt.sum())
    tp = int((sel & (lb == 1)).sum())
    fn_in_cand_low = int(((lb == 1) & ~sel & (pr < tau)).sum())
    fn_onetoone = int(((lb == 1) & ~sel & (pr >= tau)).sum())
    fn_not_cand = tot_gt - int(lb.sum())
    fp = sel & (lb == 0)
    fp_comp = int((fp & (own >= 0)).sum()); fp_orphan = int((fp & (own < 0)).sum())
    print(f"\ntrue pairs {tot_gt:,}: found {tp:,} | missed: not in candidates {fn_not_cand:,}, "
          f"below threshold {fn_in_cand_low:,}, lost in one-to-one {fn_onetoone:,}")
    print(f"false merges {int(fp.sum()):,}: record truly belongs to another S1 {fp_comp:,}, "
          f"record matches nothing {fp_orphan:,}")

    dup = (t1["core_cnt"].values[s1] > 1)
    rows = []
    for c in pd.unique(cty):
        cm = cty[s1] == c
        for nm, m in [("all", cm), ("dup-name S1", cm & dup), ("unique-name S1", cm & ~dup),
                      ("record has no address", cm & noaddr), ("Indian-script record", cm & indic),
                      ("source S2", cm & (src == 2)), ("source S3", cm & (src == 3))]:
            pos = int(((lb == 1) & m).sum())
            rows.append({"country": c, "slice": nm, "cand_pairs": int(m.sum()), "true_in_cand": pos,
                         "FN_missed": int(((lb == 1) & ~sel & m).sum()),
                         "FP_wrong": int((fp & m).sum()),
                         "FN_rate%": round(100 * ((lb == 1) & ~sel & m).sum() / max(pos, 1), 3),
                         "FP_per_1k_sel": round(1000 * (fp & m).sum() / max((sel & m).sum(), 1), 3)})
    print("\n" + pd.DataFrame(rows).to_string(index=False))

    # export a sample of mistakes for reading
    out = os.path.join(W, "errors_sample.tsv")
    e = pd.DataFrame({"rec_key": rk, "s1": s1, "prob": pr, "label": lb, "selected": sel, "country": cty[s1]})
    e = e[(e["selected"] & (e["label"] == 0)) | ((e["label"] == 1) & ~e["selected"])]
    e.sample(min(len(e), 20000), random_state=0).to_csv(out, sep="\t", index=False)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
