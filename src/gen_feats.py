"""Stage 4c: features that tell *noise* (same business, corrupted record) from *decoys*
(a different business that looks alike). Written to <feat>/<tag>_Q.npy (float32, GEN_NAMES).

Numbers  - house numbers of a noisy copy are truncated / padded / typo'd (1901 -> 19, 01901),
           while a neighbouring business has a different number of the same length (2400 vs 2413).
Words    - a noisy copy has typos (close spelling) or a random made-up word; a decoy swaps in a
           different *real* word (Escalators -> Producer). Rarity comes from the Source-1 IDF of
           the country (stats.py): made-up words are unseen (high IDF), real words are common.
Usage: python gen_feats.py --split train|test
"""
import argparse
import glob
import math
import os
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz.distance import JaroWinkler, Levenshtein

from build_features import FEAT_DIR
from features import load_strings
from normalize import squash

COLS = ["name_core", "addr_norm", "nums", "n_idf", "a_idf"]
UNSEEN_IDF = 12.4
GEN_NAMES = ["nf_samelen_diff", "nf_prefix", "nf_suffix", "nf_absdiff_log", "nf_hamming", "num_edit_min_all",
             "num_b_notin_a", "nm_typo_b", "nm_subst_b", "nm_del_a", "nm_subst_min_idf", "nm_subst_max_idf",
             "nm_del_min_idf", "nm_unknown_b", "ad_typo_b", "ad_subst_b", "ad_del_a", "ad_subst_min_idf"]


def _align(at, bt, bidf, aidf):
    """Classify tokens of b missing from a as typo (close spelling) or substitution; tokens of a
    missing from b as deletions. Returns counts and IDF extremes of substituted / deleted tokens."""
    aset, bset = set(at), set(bt)
    typo = subst = 0
    s_idf = []
    for t, w in zip(bt, bidf):
        if t in aset:
            continue
        best = max((JaroWinkler.similarity(t, u) for u in at), default=0.0)
        if best >= 0.85:
            typo += 1
        else:
            subst += 1
            s_idf.append(w)
    d_idf = [w for t, w in zip(at, aidf) if t not in bset]
    return (typo, subst, len(d_idf), min(s_idf) if s_idf else -1.0, max(s_idf) if s_idf else -1.0,
            min(d_idf) if d_idf else -1.0)


def _num_feats(anum, bnum):
    a, b = anum.split(), bnum.split()
    if not a or not b:
        return (-1.0,) * 7
    fa, fb = a[0], b[0]
    same_len_diff = float(len(fa) == len(fb) and fa != fb)
    pre = float(fa != fb and (fa.startswith(fb) or fb.startswith(fa)))
    suf = float(fa != fb and (fa.endswith(fb) or fb.endswith(fa)))
    try:
        absd = math.log1p(abs(int(fa[:12]) - int(fb[:12])))
    except ValueError:
        absd = -1.0
    ham = float(sum(x != y for x, y in zip(fa, fb))) if len(fa) == len(fb) else -1.0
    edit = min(Levenshtein.distance(x, y) for x in a for y in b)
    return (same_len_diff, pre, suf, absd, ham, float(edit), float(len(set(b) - set(a))))


def _chunk(rows):
    out = []
    for an, aa, anum, anidf, aaidf, bn, ba, bnum, bnidf, baidf in rows:
        at, bt = squash(an).split(), squash(bn).split()
        nm = _align(at, bt, bnidf, anidf)
        # share of b's name words never (or once) seen in the country's Source 1: IDF >= ln(N) - 0.01;
        # ln(N) is >= 12.4 for every country here (N >= 259k), so 12.4 marks unseen/made-up words
        unknown = (sum(1 for w in bnidf if w >= UNSEEN_IDF) / len(bnidf)) if len(bnidf) else -1.0
        if aa and ba:
            ad = _align(aa.split(), ba.split(), baidf, aaidf)
            adf = (ad[0], ad[1], ad[2], ad[3])
        else:
            adf = (-1.0,) * 4
        out.append(_num_feats(anum, bnum) + nm + (unknown,) + adf)
    return np.asarray(out, np.float32)


def run(split):
    strings = load_strings(split)
    pat = ["tr_*", "val_0"] if split == "train" else ["te_*"]
    tags = sorted({os.path.basename(p)[:-len("_pairs.parquet")] for q in pat
                   for p in glob.glob(os.path.join(FEAT_DIR, f"{q}_pairs.parquet"))})
    with Pool(int(os.environ.get("ER_WORKERS", 16))) as pool:
        for t in tags:
            out = os.path.join(FEAT_DIR, f"{t}_Q.npy")
            if os.path.exists(out):
                continue
            pr = pl.read_parquet(os.path.join(FEAT_DIR, f"{t}_pairs.parquet"), columns=["s1", "src", "row"])
            Q = np.empty((len(pr), len(GEN_NAMES)), np.float32)
            a_all = strings[1].select(COLS)[pr["s1"].to_numpy()]
            src, rows = pr["src"].to_numpy(), pr["row"].to_numpy()
            for k in (2, 3):
                m = np.nonzero(src == k)[0]
                for j in range(0, len(m), 400000):
                    mm = m[j:j + 400000]
                    a = a_all[mm]
                    b = strings[k].select(COLS)[rows[mm]]
                    tup = list(zip(*[a[c].to_list() for c in COLS], *[b[c].to_list() for c in COLS]))
                    Q[mm] = np.vstack(pool.map(_chunk, [tup[i:i + 20000] for i in range(0, len(tup), 20000)]))
            np.save(out, Q)
            print(f"  {t}: {len(Q):,} pairs", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    run(ap.parse_args().split)
