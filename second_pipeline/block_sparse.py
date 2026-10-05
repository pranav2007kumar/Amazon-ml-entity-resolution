"""Stage 2a - sparse token blocking (your B2 idea, rebuilt for local CPU).

Tokens per record: name words, joined name, address words, numbers, and the rare combinations
number x address-word, name-word x number, name-word x address-word. Weighted by IDF computed on S1
of the same country label (label-free, so it works for any country incl. unseen ones), cosine
similarity, top-k S1 per S2/S3 record. Tokens shared by more than --max-df S1 records are skipped
for speed; the rare combined tokens carry the identity signal.

Output <work>/<split>/sparse_s<k>.npz : rec, s1, s_cos, s_rank   (row numbers)
       <work>/<split>/idf_<i>.npy      : per-country IDF (used again for pair features)
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import log, wdir

N_BLOCK = 1 << 24


def load_block(work, split, k):
    d = os.path.join(work, split, f"s{k}")
    ip = np.load(os.path.join(d, "blk_indptr.npy"))
    ix = np.load(os.path.join(d, "blk_indices.npy"))
    return sp.csr_matrix((np.ones(len(ix), np.float32), ix, ip), shape=(len(ip) - 1, N_BLOCK))


def weigh(X, idf):
    X = X.copy()
    X.data = idf[X.indices].astype(np.float32)
    n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    n[n == 0] = 1
    return sp.diags(1 / n).dot(X).tocsr()


def topk_rows(R, k):
    R = R.tocsr()
    lens = np.diff(R.indptr)
    rows = np.repeat(np.arange(R.shape[0], dtype=np.int64), lens)
    order = np.lexsort((-R.data, rows))
    srows = rows[order]
    rank = np.arange(len(order)) - R.indptr[srows]
    keep = rank < k
    return srows[keep], R.indices[order][keep], R.data[order][keep], (rank[keep] + 1)


def _chunk(Xq, X1T, k, q0):
    R = Xq @ X1T
    r, c, v, rk = topk_rows(R, k)
    return (r + q0).astype(np.int32), c.astype(np.int32), v.astype(np.float32), rk.astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-df", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    from joblib import Parallel, delayed

    t1 = pd.read_parquet(os.path.join(a.work, a.split, "s1", "text.parquet"), columns=["country"])
    X1all = load_block(a.work, a.split, 1)
    c1 = t1["country"].values
    countries = list(pd.unique(c1))
    pd.Series(countries).to_csv(os.path.join(wdir(a.work, a.split), "countries.csv"), index=False)
    for k in (2, 3):
        tq = pd.read_parquet(os.path.join(a.work, a.split, f"s{k}", "text.parquet"), columns=["country"])
        Xqall = load_block(a.work, a.split, k)
        cq = tq["country"].values
        out = []
        for ci, c in enumerate(countries):
            lo, hi = np.searchsorted(c1, c, "left"), np.searchsorted(c1, c, "right")
            qlo, qhi = np.searchsorted(cq, c, "left"), np.searchsorted(cq, c, "right")
            if qhi <= qlo:
                continue
            X1 = X1all[lo:hi]
            df = np.bincount(X1.indices, minlength=N_BLOCK)
            idf = np.log((hi - lo + 1) / (df + 1)).astype(np.float32)
            np.save(os.path.join(wdir(a.work, a.split), f"idf_{ci}.npy"), idf)
            idf_q = idf.copy()
            idf_q[(df == 0) | (df > a.max_df)] = 0
            X1T = weigh(X1, idf).T.tocsr()
            Xq = weigh(Xqall[qlo:qhi], idf)
            Xq.data[idf_q[Xq.indices] == 0] = 0
            Xq.eliminate_zeros()
            res = Parallel(n_jobs=a.jobs)(delayed(_chunk)(Xq[i:i + a.chunk], X1T, a.topk, qlo + i)
                                          for i in range(0, qhi - qlo, a.chunk))
            for r, cc, v, rk in res:
                out.append((r, cc + lo, v, rk))
            log(f"{a.split} S{k} {c}: {qhi - qlo:,} records blocked against {hi - lo:,} S1")
        rec = np.concatenate([o[0] for o in out]); s1 = np.concatenate([o[1] for o in out]).astype(np.int32)
        np.savez(os.path.join(wdir(a.work, a.split), f"sparse_s{k}.npz"), rec=rec, s1=s1,
                 s_cos=np.concatenate([o[2] for o in out]), s_rank=np.concatenate([o[3] for o in out]))
        log(f"{a.split} S{k}: {len(rec):,} sparse candidate pairs saved")


if __name__ == "__main__":
    main()
