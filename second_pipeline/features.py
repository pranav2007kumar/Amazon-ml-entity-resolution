"""Stage 3 - pair features for every candidate pair (country is never a feature).

Groups:
  retrieval   d_cos, d_cos_name, d_cos_addr (learned model), s_cos (IDF token cosine), ranks
  context     rank / margin of each score among the record's candidates and among the S1's candidates
  name        rapidfuzz ratio, token-set, token-sort, partial, Jaro-Winkler on name / core / joined name
  address     rapidfuzz ratio, token-set, partial on address and address words
  numbers     overlap, Jaccard, first-number equal, digit edit distance, truncation, sibling, gap
  abbrev      country-agnostic abbreviation matching (r~rue, pvt~private) for name and address
  uniqueness  how many S1 share the core name (crowding), Indian-script flag, lengths, missing fields
Written in chunks to <work>/<split>/feats_s<k>.parquet.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from common import log

N_BLOCK = 1 << 24
TEXT_COLS = ["name", "core", "joined", "addr", "awords", "nums", "core_cnt", "indic", "country"]


# ------------------------------------------------------------ python features (per pair)
def _is_abv(s, l):
    if len(s) >= len(l) or s[0] != l[0]:
        return False
    it = iter(l)
    return all(ch in it for ch in s)


def _abv(a, b):
    if not a or not b:
        return 0.0, 0.0, 0, 0
    bset, used, ma, nab = set(b), set(), 0, 0
    for t in a:
        if t in bset and t not in used:
            used.add(t); ma += 1; continue
        for u in b:
            if u not in used and (_is_abv(t, u) or _is_abv(u, t)):
                used.add(u); ma += 1; nab += 1; break
    x, y = a[0], b[0]
    return ma / len(a), len(used) / len(b), nab, int(x == y or _is_abv(x, y) or _is_abv(y, x))


def _lev(a, b):
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(a, b)


def _num(na, nb):
    A, B = na.split(), nb.split()
    if not A or not B:
        return (len(A), len(B), 0, 0.0, 0, 9, 0, 0, 0, -1.0)
    sa, sb = set(A), set(B)
    common = len(sa & sb)
    jac = common / len(sa | sb)
    first = int(A[0] == B[0])
    best = min(_lev(x, y) for x in A[:3] for y in B[:3])
    trunc = int(any(x != y and (x.endswith(y) or y.endswith(x) or x.startswith(y) or y.startswith(x))
                    for x in A[:3] for y in B[:3]))
    sib = int(common == 0 and any(len(x) == len(y) and _lev(x, y) == 1 for x in A[:3] for y in B[:3]))
    try:
        gap = float(np.log1p(abs(int(A[0][:12]) - int(B[0][:12]))))
    except ValueError:
        gap = -1.0
    return (len(A), len(B), common, jac, first, best, trunc, sib, int(sa == sb), gap)


NUM_NAMES = ["n_num_o", "n_num_s", "num_common", "num_jac", "num_first_eq", "num_best_edit",
             "num_trunc", "num_sibling", "num_same_set", "num_log_gap"]
ABV_NAMES = [f"abv_{f}_{x}" for f in ("name", "addr") for x in ("cov_o", "cov_s", "n", "first")]


def _py_work(args):
    no, ns, ao, as_, uo, us, jo, js = args
    out = np.zeros((len(no), len(NUM_NAMES) + len(ABV_NAMES) + 2), np.float32)
    for i in range(len(no)):
        r = list(_num(uo[i], us[i]))
        r += list(_abv(no[i].split(), ns[i].split())) + list(_abv(ao[i].split(), as_[i].split()))
        a, b = jo[i], js[i]
        r += [int(len(a) >= 4 and len(b) >= 4 and (a in b or b in a)), int(a == b and len(a) > 0)]
        out[i] = r
    return out


# ------------------------------------------------------------ sparse cosine for any pair
def load_blk(work, split, k):
    d = os.path.join(work, split, f"s{k}")
    ip = np.load(os.path.join(d, "blk_indptr.npy")); ix = np.load(os.path.join(d, "blk_indices.npy"))
    return sp.csr_matrix((np.ones(len(ix), np.float32), ix, ip), shape=(len(ip) - 1, N_BLOCK))


def rows_weighted(X, rows, idf):
    A = X[rows]
    A.data = idf[A.indices]
    n = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()); n[n == 0] = 1
    return sp.diags(1 / n).dot(A)


def add_context(df, scores):
    for key, suf in (("rec", "o"), ("s1", "s")):
        g = df.groupby(key)
        df[f"n_cand_{suf}"] = g[key].transform("size").astype(np.int16)
        for c in scores:
            v = df[c]
            rk = g[c].rank(method="first", ascending=False)
            best = g[c].transform("max")
            df[f"{c}_rk_{suf}"] = rk.astype(np.int16)
            df[f"{c}_gap_{suf}"] = (best - v).astype(np.float32)
            if suf == "o":
                sec = df.assign(_t=np.where(rk.values == 1, -9.0, v)).groupby(key)["_t"].transform("max")
                df[f"{c}_mg_o"] = np.where(rk.values == 1, v - sec.clip(lower=-1), v - best).astype(np.float32)
    for c in scores:
        df[f"{c}_mutual"] = ((df[f"{c}_rk_o"] == 1) & (df[f"{c}_rk_s"] <= 2)).astype(np.int8)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler

    W = os.path.join(a.work, a.split)
    countries = pd.read_csv(os.path.join(W, "countries.csv")).iloc[:, 0].tolist()
    t1 = pd.read_parquet(os.path.join(W, "s1", "text.parquet"), columns=TEXT_COLS, dtype_backend="pyarrow")
    X1 = load_blk(a.work, a.split, 1)
    rng = np.random.default_rng(0)
    for k in (2, 3):
        tk = pd.read_parquet(os.path.join(W, f"s{k}", "text.parquet"), columns=TEXT_COLS, dtype_backend="pyarrow")
        Xk = load_blk(a.work, a.split, k)
        cand_all = pd.read_parquet(os.path.join(W, f"cand_s{k}.parquet"))
        log(f"{a.split} S{k}: {len(cand_all):,} pairs")
        cc_all = np.asarray(t1["country"].iloc[cand_all["s1"].values].astype(str))
        out_path = os.path.join(W, f"feats_s{k}.parquet")
        writer = None
        pool = Pool(a.jobs)
        for ci, cname in enumerate(countries):
            cand = cand_all[cc_all == cname].reset_index(drop=True)
            if not len(cand):
                continue
            idf = np.load(os.path.join(W, f"idf_{ci}.npy"))
            s_cos = np.zeros(len(cand), np.float32)
            for i in range(0, len(cand), a.chunk):
                A = rows_weighted(Xk, cand["rec"].values[i:i + a.chunk], idf)
                B = rows_weighted(X1, cand["s1"].values[i:i + a.chunk], idf)
                s_cos[i:i + a.chunk] = np.asarray(A.multiply(B).sum(1)).ravel()
            cand["s_cos"] = s_cos
            cand = add_context(cand, ["d_cos", "s_cos", "d_cos_name", "d_cos_addr"])
            log(f"  S{k} {cname}: {len(cand):,} pairs, retrieval + context features done")
            for i in range(0, len(cand), a.chunk):
                c = cand.iloc[i:i + a.chunk].reset_index(drop=True)
                ro, rs = c["rec"].values, c["s1"].values
                o, s = tk.iloc[ro].reset_index(drop=True), t1.iloc[rs].reset_index(drop=True)
                f = {}
                for nm, col, scorers in (("name", "name", ("ratio", "token_set_ratio", "token_sort_ratio", "partial_ratio")),
                                         ("core", "core", ("ratio", "token_set_ratio")),
                                         ("addr", "addr", ("ratio", "token_set_ratio", "partial_ratio")),
                                         ("aw", "awords", ("token_set_ratio",))):
                    q, ch = o[col].tolist(), s[col].tolist()
                    for sc in scorers:
                        f[f"{nm}_{sc}"] = process.cpdist(q, ch, scorer=getattr(fuzz, sc), workers=-1).astype(np.float32)
                for col in ("name", "joined"):
                    f[f"{col}_jw"] = process.cpdist(o[col].tolist(), s[col].tolist(),
                                                    scorer=JaroWinkler.normalized_similarity, workers=-1).astype(np.float32)
                cols = [o["name"].tolist(), s["name"].tolist(), o["addr"].tolist(), s["addr"].tolist(),
                        o["nums"].tolist(), s["nums"].tolist(), o["joined"].tolist(), s["joined"].tolist()]
                step = max(1, len(c) // (a.jobs * 4) + 1)
                py = np.concatenate(pool.map(_py_work, [tuple(x[j:j + step] for x in cols) for j in range(0, len(c), step)]))
                for j, n in enumerate(NUM_NAMES + ABV_NAMES + ["joined_contains", "joined_equal"]):
                    f[n] = py[:, j]
                f["len_name_o"] = np.asarray(o["name"].str.len(), dtype=np.int16)
                f["len_name_s"] = np.asarray(s["name"].str.len(), dtype=np.int16)
                f["addr_empty_o"] = (np.asarray(o["addr"].str.len()) == 0).astype(np.int8)
                f["core_cnt_s"] = np.asarray(s["core_cnt"], dtype=np.int32)
                f["core_cnt_o"] = np.asarray(o["core_cnt"], dtype=np.int32)
                f["indic_o"] = np.asarray(o["indic"], dtype=np.int8)
                f["src3"] = np.full(len(c), int(k == 3), np.int8)
                f["u"] = rng.random(len(c)).astype(np.float32)
                tbl = pa.Table.from_pandas(pd.concat([c, pd.DataFrame(f)], axis=1), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, tbl.schema)
                writer.write_table(tbl)
                log(f"  S{k} {cname}: {min(i + a.chunk, len(cand)):,}/{len(cand):,} pairs featurised")
            del cand
        pool.close(); pool.join()
        if writer is not None:
            writer.close()
        del cand_all
    log(f"{a.split}: features done")


if __name__ == "__main__":
    main()
