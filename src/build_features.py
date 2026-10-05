"""Stage 4 (cached): build stage-1 pair features once and store them on disk.

Pairs are written in chunks so RAM stays bounded:
  <work>/feat/<tag>_<i>_pairs.parquet   s1, src, row (+ y for train)
  <work>/feat/<tag>_<i>_X.npy           float32 [n_pairs, len(COMP_NAMES)+len(FEAT_NAMES)]

Train S1 rows are grouped by a fixed random value g in [0,1) (train.s1_groups):
  train chunks  : g in [0, TRAIN_END) split into CHUNKS equal ranges
  val           : g in [VAL_START, 1)
Test S1 rows are chunked by s1 % TEST_PARTS.
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import polars as pl

from config import WORK_DIR
from features import COMP_NAMES, competition_features, load_strings, string_features, union_candidates
from labels import positive_pairs

FEAT_DIR = os.path.join(WORK_DIR, "feat")
K, KS = 30, 10  # dense: records with an address have 10 candidates, empty-address ones up to 30
TRAIN_END, CHUNK_W, VAL_START, TEST_PARTS = 0.40, 0.05, 0.96, 20


def s1_groups(n, seed=42):
    """Same deterministic split as train.py."""
    return np.random.default_rng(seed).random(n)


def build_chunk(split, expr, strings, pool, pos, out):
    t = time.time()
    cand = competition_features(union_candidates(split, K, KS, expr), split)
    X = np.hstack([cand.select(COMP_NAMES).to_numpy().astype(np.float32), string_features(cand, strings, pool)])
    pairs = cand.select("s1", "src", "row")
    if pos is not None:
        pairs = pairs.join(pos, on=["s1", "src", "row"], how="left", maintain_order="left").with_columns(
            pl.col("y").fill_null(0))
    pairs.write_parquet(out + "_pairs.parquet")
    np.save(out + "_X.npy", X)
    print(f"  {os.path.basename(out)}: {len(pairs):,} pairs in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--only", nargs="*", help="build only these chunk tags, e.g. val_0 tr_0 tr_1")
    ap.add_argument("--train_end", type=float, default=TRAIN_END,
                    help="train chunks cover g in [0, train_end); 0.96 = all non-validation S1")
    a = ap.parse_args()
    os.makedirs(FEAT_DIR, exist_ok=True)
    strings = load_strings(a.split)
    with Pool(16) as pool:
        if a.split == "train":
            n1 = len(strings[1])
            g = s1_groups(n1)
            pos = positive_pairs().with_columns(pl.lit(1, pl.Int8).alias("y"))
            los = np.arange(0, a.train_end - 1e-9, CHUNK_W)
            jobs = [("val_0", g >= VAL_START)]
            jobs += [(f"tr_{i}", (g >= lo) & (g < min(lo + CHUNK_W, a.train_end))) for i, lo in enumerate(los)]
            for tag, mask in jobs:
                out = os.path.join(FEAT_DIR, tag)
                if os.path.exists(out + "_X.npy") or (a.only and tag not in a.only):
                    continue
                keep = pl.Series(np.nonzero(mask)[0].astype(np.int32))
                build_chunk("train", pl.col("s1").is_in(keep), strings, pool, pos, out)
        else:
            for i in range(TEST_PARTS):
                out = os.path.join(FEAT_DIR, f"te_{i}")
                if os.path.exists(out + "_X.npy") or (a.only and f"te_{i}" not in a.only):
                    continue
                build_chunk("test", pl.col("s1") % TEST_PARTS == i, strings, pool, None, out)
