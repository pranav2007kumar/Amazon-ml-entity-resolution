"""Stage 1 - normalise text, build sub-world for train, write hashed bags for both retrievers.

Train "sub-world": keep a random 50% of S1 (hash), all S2/S3 records they own, and 50% of the
unmatched S2/S3 records. Test US has exactly half the S1 density of train US (663k vs 1.32M), so the
sub-world has realistic crowding while halving RAM and time.

Outputs per split/source in <work>/<split>/s<k>/:
  text.parquet   entity_id, country, indic, name, core, joined, addr, awords, nums (+ core_cnt)
  name_*.npy / addr_*.npy   hashed char-3-gram+word bags (dense retriever input)
  blk_*.npy                 hashed blocking tokens (sparse retriever input)
Train also gets gt_s2.npz / gt_s3.npz (true pairs as row numbers).
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import has_indic, load_gt_pairs, log, normalise, read_tsv, source_path, stable_hash01, wdir


def in_world(h, a):
    """Sub-world membership: A = hash < s1_frac (original behaviour), B = hash >= 1 - s1_frac (disjoint half)."""
    return h < a.s1_frac if a.s1_side == "A" else h >= 1 - a.s1_frac

N_CHAR, N_WORD = 1 << 20, 1 << 18
N_BLOCK = 1 << 24
WEB = {"www", "http", "https", "com", "net", "org", "biz", "info"}


# ------------------------------------------------------------ hashing (module level for joblib)
def block_tokens(doc):
    core, aw, nums = doc.split("\x01")
    cw = [w for w in core.split() if len(w) > 1][:4]
    aws = [w for w in aw.split() if len(w) > 1][:8]
    ns = nums.split()[:3]
    t = ["n:" + w for w in cw] + ["a:" + w for w in aws] + ["d:" + n for n in ns]
    j = core.replace(" ", "")
    if len(j) >= 4:
        t.append("j:" + j)
    t += ["x:" + n + "_" + w for n in ns for w in aws]
    t += ["y:" + w + "_" + n for w in cw for n in ns]
    t += ["c:" + w + "_" + a for w in cw[:3] for a in aws[:6]]
    return t


def _hash_bags(texts):
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import HashingVectorizer
    hc = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=N_CHAR,
                           alternate_sign=False, norm=None, binary=True, dtype=np.float32)
    hw = HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=N_WORD,
                           alternate_sign=False, norm=None, binary=True, dtype=np.float32)
    X = sp.hstack([hc.transform(texts), hw.transform(texts)], format="csr")
    X.sort_indices()
    return X.indptr.astype(np.int64), X.indices.astype(np.int32)


def _hash_block(docs):
    from sklearn.feature_extraction.text import HashingVectorizer
    h = HashingVectorizer(analyzer=block_tokens, n_features=N_BLOCK, alternate_sign=False,
                          norm=None, binary=True, dtype=np.float32)
    X = h.transform(docs)
    X.sort_indices()
    return X.indptr.astype(np.int64), X.indices.astype(np.int32)


def hash_parallel(fn, texts, jobs, step=100_000):
    from joblib import Parallel, delayed
    texts = list(texts)
    parts = Parallel(n_jobs=jobs)(delayed(fn)(texts[i:i + step]) for i in range(0, len(texts), step))
    indptr, indices, off = [np.zeros(1, np.int64)], [], 0
    for ip, ix in parts:
        indptr.append(ip[1:] + off); indices.append(ix); off += len(ix)
    return np.concatenate(indptr), (np.concatenate(indices) if indices else np.zeros(0, np.int32))


# ------------------------------------------------------------ text
def common_tokens(names, countries, min_share):
    """Tokens appearing in > min_share of names of a country (legal forms, fillers). Label-free."""
    out = {}
    for c in pd.unique(countries):
        s = names[countries == c]
        if len(s) > 1_000_000:
            s = s.sample(1_000_000, random_state=0)
        df = s.str.split().map(lambda t: list(set(t))).explode().value_counts()
        out[c] = set(df[df > min_share * len(s)].index)
    return out


def build_text(raw):
    t = pd.DataFrame({"entity_id": raw["entity_id"].values, "country": raw["country"].values,
                      "indic": has_indic(raw["business_name"]).values,
                      "name": normalise(raw["business_name"]).values,
                      "addr": normalise(raw["business_address"]).values})
    t["name"] = t["name"].str.replace(r"\b\d{7,}\b", " ", regex=True).str.split().str.join(" ")
    tok = t["addr"].str.split()
    t["nums"] = tok.map(lambda l: " ".join(x for x in ("".join(ch for ch in w if ch.isdigit()).lstrip("0")
                                                        for w in l if any(ch.isdigit() for ch in w)) if x))
    t["awords"] = tok.map(lambda l: " ".join(w for w in l if not any(ch.isdigit() for ch in w)))
    return t


def add_core(t, common):
    def core(name, c):
        toks = [w for w in name.split() if w not in WEB]
        k = [w for w in toks if w not in common.get(c, ())]
        return " ".join(k if k else toks)
    t["core"] = [core(n, c) for n, c in zip(t["name"].values, t["country"].values)]
    t["joined"] = t["core"].str.replace(" ", "", regex=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--s1-frac", type=float, default=0.5, help="train sub-world size")
    ap.add_argument("--s1-side", choices=["A", "B"], default="A",
                    help="A: hash < s1_frac (default); B: hash >= 1 - s1_frac (the complementary sub-world, see REPORT_B.md)")
    ap.add_argument("--common-share", type=float, default=0.004)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()

    gt = None
    if a.split == "train":
        _, gt = load_gt_pairs(os.path.join(a.data_dir, "train", "train_ground_truth.tsv"))
        all_other = set(gt["other_id"])

    def finish(k, t):
        d = wdir(a.work, a.split, k)
        t.to_parquet(os.path.join(d, "text.parquet"), index=False)
        for f in ("name", "addr"):
            ip, ix = hash_parallel(_hash_bags, t[f].values, a.jobs)
            np.save(os.path.join(d, f"{f}_indptr.npy"), ip); np.save(os.path.join(d, f"{f}_indices.npy"), ix)
        docs = (t["core"] + "\x01" + t["awords"] + "\x01" + t["nums"]).values
        ip, ix = hash_parallel(_hash_block, docs, a.jobs)
        np.save(os.path.join(d, "blk_indptr.npy"), ip); np.save(os.path.join(d, "blk_indices.npy"), ix)
        log(f"{a.split} s{k}: text + bags written ({len(t):,} rows)")

    # ---- S1 first: it defines the common-token list and the core-name counts
    raw = read_tsv(source_path(a.data_dir, a.split, 1))
    log(f"{a.split} s1: {len(raw):,} rows read")
    if gt is not None:
        raw = raw[in_world(stable_hash01(raw["entity_id"].values, "world"), a)]
        gt = gt[gt["s1_id"].isin(set(raw["entity_id"]))]
        log(f"train sub-world: {len(raw):,} S1, {len(gt):,} true pairs")
    t = build_text(raw); del raw
    common = common_tokens(t["name"], t["country"].values, a.common_share)
    for c, s_ in common.items():
        log(f"country {c}: {len(s_)} common name tokens (legal forms/fillers) kept out of core name, e.g. {sorted(s_)[:10]}")
    t = add_core(t, common).sort_values(["country", "entity_id"], kind="stable").reset_index(drop=True)
    key = t["country"] + "|" + t["core"]
    cnt = key.value_counts()
    t["core_cnt"] = key.map(cnt).astype(np.int32).values
    r1 = pd.Series(np.arange(len(t)), index=t["entity_id"].values)
    finish(1, t); del t, key

    for k in (2, 3):
        raw = read_tsv(source_path(a.data_dir, a.split, k))
        log(f"{a.split} s{k}: {len(raw):,} rows read")
        if gt is not None:
            ids = pd.Index(raw["entity_id"].values)
            keep = ids.isin(set(gt["other_id"])) | (~ids.isin(all_other) &
                                                     in_world(stable_hash01(ids.values, "world"), a))
            raw = raw[keep]
            log(f"train sub-world s{k}: {len(raw):,} records kept")
        t = add_core(build_text(raw), common); del raw
        t = t.sort_values(["country", "entity_id"], kind="stable").reset_index(drop=True)
        t["core_cnt"] = (t["country"] + "|" + t["core"]).map(cnt).fillna(0).astype(np.int32).values
        if gt is not None:
            g = gt[gt["other_id"].str.startswith(f"S{k}-")]
            rk = pd.Series(np.arange(len(t)), index=t["entity_id"].values)
            np.savez(os.path.join(wdir(a.work, "train"), f"gt_s{k}.npz"),
                     rec=rk.reindex(g["other_id"]).values.astype(np.int32),
                     s1=r1.reindex(g["s1_id"]).values.astype(np.int32))
        finish(k, t); del t
    log(f"{a.split}: prep done")


if __name__ == "__main__":
    main()
