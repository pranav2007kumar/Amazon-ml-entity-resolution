"""Stage 2c - union of sparse top-k and dense top-k = the exact candidate set the model scores.

Writes <work>/<split>/cand_s<k>.parquet (rec, s1, s_rank, d_rank, d_cos, d_cos_name, d_cos_addr [, label]).
On train it prints the recall gate: share of true pairs inside sparse, dense and union candidates,
by country, source and Indian-script records. Your old B2 top-5 was 0.975.
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import log, wdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--sparse-k", type=int, default=5); ap.add_argument("--dense-k", type=int, default=3)
    a = ap.parse_args()
    t1 = pd.read_parquet(os.path.join(a.work, a.split, "s1", "text.parquet"), columns=["country"])
    n1 = len(t1)
    rep = []
    for k in (2, 3):
        sp = np.load(os.path.join(a.work, a.split, f"sparse_s{k}.npz"))
        sd = np.load(os.path.join(a.work, a.split, f"sparse_dense_s{k}.npz"))["d"]
        de = np.load(os.path.join(a.work, a.split, f"dense_s{k}.npz"))
        s = pd.DataFrame({"key": sp["rec"].astype(np.int64) * n1 + sp["s1"], "s_rank": sp["s_rank"],
                          "d_cos": sd[:, 0], "d_cos_name": sd[:, 1], "d_cos_addr": sd[:, 2]})
        d = pd.DataFrame({"key": de["rec"].astype(np.int64) * n1 + de["s1"], "d_rank": de["d_rank"],
                          "d_cos": de["d_cos"], "d_cos_name": de["d_cos_name"], "d_cos_addr": de["d_cos_addr"]})
        s_all, d_all = s, d
        s = s[s["s_rank"] <= a.sparse_k]
        d = d[d["d_rank"] <= a.dense_k]
        u = s.merge(d, on="key", how="outer", suffixes=("", "_d"))
        for c in ("d_cos", "d_cos_name", "d_cos_addr"):
            u[c] = u[c].fillna(u[c + "_d"]).astype(np.float32)
        u = u.drop(columns=["d_cos_d", "d_cos_name_d", "d_cos_addr_d"])
        u["s_rank"] = u["s_rank"].fillna(99).astype(np.int8)
        u["d_rank"] = u["d_rank"].fillna(99).astype(np.int8)
        u["rec"] = (u["key"] // n1).astype(np.int32)
        u["s1"] = (u["key"] % n1).astype(np.int32)
        if a.split == "train":
            g = np.load(os.path.join(a.work, "train", f"gt_s{k}.npz"))
            gkey = g["rec"].astype(np.int64) * n1 + g["s1"]
            u["label"] = np.isin(u["key"].values, gkey).astype(np.int8)
            tk = pd.read_parquet(os.path.join(a.work, "train", f"s{k}", "text.parquet"), columns=["indic"])
            in_s = np.isin(gkey, s["key"].values); in_d = np.isin(gkey, d["key"].values)
            s_rank_true = pd.Series(s_all["s_rank"].values, index=s_all["key"].values).reindex(gkey).fillna(0).values
            d_true = np.load(os.path.join(a.work, "train", f"dense_truerank_s{k}.npy"))[g["rec"]]
            frame = pd.DataFrame({"country": t1["country"].values[g["s1"]], "indic": tk["indic"].values[g["rec"]],
                                  "sparse": in_s, "dense": in_d, "union": in_s | in_d,
                                  "s_rank": s_rank_true, "d_rank": d_true})
            frame["src"] = f"S{k}"
            rep.append(frame)
        u = u.drop(columns="key").sort_values(["rec", "s1"]).reset_index(drop=True)
        u.to_parquet(os.path.join(wdir(a.work, a.split), f"cand_s{k}.parquet"), index=False)
        log(f"{a.split} S{k}: {len(u):,} candidate pairs ({len(u) / max(u['rec'].nunique(), 1):.2f} per record)"
            + (f", positives {int(u['label'].sum()):,}" if "label" in u else ""))
    if rep:
        f = pd.concat(rep, ignore_index=True)
        rows = []
        for name, g in [("ALL", f)] + [(f"country={c}", x) for c, x in f.groupby("country")] + \
                       [(f"src={s}", x) for s, x in f.groupby("src")] + \
                       [(f"indian_script={v}", x) for v, x in f.groupby("indic")]:
            rows.append({"slice": name, "true_pairs": len(g),
                         **{f"sparse@{k}": round(((g.s_rank >= 1) & (g.s_rank <= k)).mean(), 5) for k in (1, 3, 5, 10)},
                         **{f"dense@{k}": round(((g.d_rank >= 1) & (g.d_rank <= k)).mean(), 5) for k in (1, 3, 5, 10, 20)},
                         f"UNION(s{a.sparse_k}+d{a.dense_k})": round(g.union.mean(), 5)})
        r = pd.DataFrame(rows)
        pd.set_option("display.width", 250)
        print("\nCANDIDATE RECALL (share of true pairs the model can still find):")
        print(r.to_string(index=False))
        r.to_csv(os.path.join(a.work, "recall_report.csv"), index=False)


if __name__ == "__main__":
    main()
