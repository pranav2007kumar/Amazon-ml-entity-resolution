"""Country-relative feature scaling: rarity-type features as percentiles within the S1 country and split.

Also removes the train/test shift: train has twice as many US S1 as test (1.32M vs 0.66M), so IDF
values (ln N/df) and the rare-token blocking scores (tokens above the document-frequency cap are
dropped) are distributed differently in train and test for the same country. Percentiles are
computed per split, so a value means "rarer than x% of this country's pairs" in both.

IDF-based and count-based features depend on each country's vocabulary: French business names are
built from very common words (club, amicale, ecole, union ...), so raw IDF values in France are much
lower than in the US / India, and a model trained there distrusts every French name. Replacing these
columns by their percentile inside the pair's country makes an unseen country look like the training
countries. Quantile tables come from the split itself (no labels), per country and column.

  python ctry_norm.py --split train|test     writes <feat>/ctry_quantiles_<split>.npz
two_stage.load() applies the mapping when ER_CTRY_NORM=1.
"""
import argparse
import glob
import os

import numpy as np
import polars as pl

from build_features import FEAT_DIR
from config import WORK_DIR
from features import COMP_NAMES, FEAT_NAMES

COLS = COMP_NAMES + FEAT_NAMES
NORM = [c for c in ["sp_score", "sp_gap_q", "sp_margin", "sp_gap_s1",
                    "n_idf_a", "n_idf_b", "n_idf_max_shared", "n_idf_max_miss_a", "n_idf_max_miss_b",
                    "a_idf_a", "a_idf_b", "a_idf_max_miss_a", "a_idf_max_miss_b",
                    "g_s1cnt_a", "g_s1cnt_b", "g_ocnt_a", "a_s1cnt_a"] if c in COLS]
NQ = 201


def _s1_country(split):
    return pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s1.parquet"), columns=["country"])["country"].to_numpy()


def fit(split, per_chunk=300_000, seed=0):
    rng = np.random.default_rng(seed)
    pat = "tr_*_X.npy" if split == "train" else "te_*_X.npy"
    files = sorted(glob.glob(os.path.join(FEAT_DIR, pat)))
    ctry = _s1_country(split)
    idx = [COLS.index(c) for c in NORM]
    vals, cs = [], []
    for f in files:
        X = np.load(f, mmap_mode="r")
        s1 = pl.read_parquet(f.replace("_X.npy", "_pairs.parquet"), columns=["s1"])["s1"].to_numpy()
        take = np.sort(rng.choice(len(X), min(per_chunk, len(X)), replace=False))
        vals.append(np.asarray(X[take][:, idx])); cs.append(ctry[s1[take]])
    V, C = np.vstack(vals), np.concatenate(cs)
    out = {}
    for c in np.unique(C):
        m = C == c
        out[str(c)] = np.quantile(V[m], np.linspace(0, 1, NQ), axis=0).astype(np.float32)  # [NQ, n_cols]
        print(f"  {split} {c}: {m.sum():,} sampled pairs", flush=True)
    np.savez(os.path.join(FEAT_DIR, f"ctry_quantiles_{split}.npz"), **out)


class Normalizer:
    """Maps the NORM columns of a feature block to within-country percentiles in [0, 1]."""

    def __init__(self, split):
        q = np.load(os.path.join(FEAT_DIR, f"ctry_quantiles_{split}.npz"))
        self.q = {k: q[k] for k in q.files}
        self.idx = [COLS.index(c) for c in NORM]
        self.ctry = _s1_country(split)
        self.grid = np.linspace(0, 1, NQ).astype(np.float32)

    def __call__(self, block, s1_rows):
        block = np.array(block, dtype=np.float32, copy=True)
        cs = self.ctry[s1_rows]
        for c in np.unique(cs):
            m = cs == c
            q = self.q.get(str(c))
            if q is None:
                continue
            for j, col in enumerate(self.idx):
                block[m, col] = np.interp(block[m, col], q[:, j], self.grid)
        return block


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    fit(ap.parse_args().split)
