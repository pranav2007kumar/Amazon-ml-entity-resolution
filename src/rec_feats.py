"""Stage 4b: record-level competition features across ALL chunks of a split.

Every S2/S3 record belongs to at most one S1, but its candidate S1s are spread over many feature
chunks (chunks are cut by S1). For each pair this computes how it compares with the record's best
*other* candidate on several pair features, e.g. "another S1 has a much closer name". The values are
exact only when every chunk of the split exists (all train chunks + val_0, or all te_* chunks).

Writes <feat>/<tag>_R.npy (float32, one row per pair, columns REC_NAMES).
Usage: python rec_feats.py --split train|test
"""
import argparse
import glob
import os

import numpy as np
import polars as pl

from build_features import FEAT_DIR
from features import COMP_NAMES, FEAT_NAMES

COLS = COMP_NAMES + FEAT_NAMES
BASE = ["n_tset", "a_tset", "ph_tset", "n_idf_a", "a_idf_a", "sim"]
REC_NAMES = ["rec_ncand", "rec_exact_other"] + [f"{c}_other" for c in BASE] + [f"{c}_gap" for c in BASE]


def tags(split):
    pat = "tr_*_X.npy" if split == "train" else "te_*_X.npy"
    out = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(FEAT_DIR, pat)))
    return out + (["val_0"] if split == "train" else [])


def extract(tag):
    """Small per-chunk cache of the columns needed here (reading the full X once)."""
    path = os.path.join(FEAT_DIR, f"{tag}_rc.npy")
    if not os.path.exists(path):
        X = np.load(os.path.join(FEAT_DIR, f"{tag}_X.npy"), mmap_mode="r")
        idx = [COLS.index(c) for c in BASE + ["n_exact_glued"]]
        np.save(path, np.ascontiguousarray(X[:, idx]))
    return np.load(path, mmap_mode="r")


def run(split):
    tl = tags(split)
    pairs = {t: pl.read_parquet(os.path.join(FEAT_DIR, f"{t}_pairs.parquet"), columns=["src", "row"]) for t in tl}
    rc = {t: extract(t) for t in tl}
    print(f"{split}: {len(tl)} chunks, {sum(len(p) for p in pairs.values()):,} pairs", flush=True)
    # per record: count, #exact names, and top-2 of each base column (one source at a time for RAM)
    stats = []
    n_slices = 4
    for src in (2, 3):
        for sl in range(n_slices):  # records split by row % n_slices to bound RAM
            parts = []
            for t in tl:
                r = pairs[t]["row"].to_numpy()
                m = (pairs[t]["src"] == src).to_numpy() & (r % n_slices == sl)
                d = {"row": r[m]}
                v = np.asarray(rc[t][m])
                for j, c in enumerate(BASE + ["exact"]):
                    d[c] = v[:, j]
                parts.append(pl.DataFrame(d))
            long = pl.concat(parts)
            del parts
            aggs = [pl.len().alias("rec_ncand"), (pl.col("exact") > 0).sum().alias("rec_nexact")]
            for c in BASE:
                aggs.append(pl.col(c).top_k(2).alias(f"{c}_top"))
            st = long.group_by("row").agg(aggs)
            del long
            st = st.with_columns([pl.col(f"{c}_top").list.get(0).alias(f"{c}_m1") for c in BASE]
                                 + [pl.col(f"{c}_top").list.get(1, null_on_oob=True).alias(f"{c}_m2") for c in BASE]
                                 ).drop([f"{c}_top" for c in BASE])
            stats.append(st.with_columns(pl.lit(src, pl.Int8).alias("src")))
        print(f"  src {src} done", flush=True)
    st = pl.concat(stats)
    for t in tl:
        out = os.path.join(FEAT_DIR, f"{t}_R.npy")
        d = pairs[t].with_columns(pl.int_range(pl.len()).alias("i"))
        v = np.asarray(rc[t])
        d = d.with_columns([pl.Series(c, v[:, j]) for j, c in enumerate(BASE + ["exact"])])
        d = d.join(st, on=["src", "row"], how="left", maintain_order="left")
        cols = [pl.col("rec_ncand").cast(pl.Float32),
                (pl.col("rec_nexact") - (pl.col("exact") > 0).cast(pl.UInt32)).cast(pl.Float32).alias("rec_exact_other")]
        others = []
        for c in BASE:
            # best value among the record's other candidates (ties: the other copy of the max)
            other = pl.when(pl.col(c) >= pl.col(f"{c}_m1")).then(pl.col(f"{c}_m2")).otherwise(pl.col(f"{c}_m1"))
            others.append(other.fill_null(-1.0).cast(pl.Float32).alias(f"{c}_other"))
        d = d.with_columns(cols + others)
        d = d.with_columns([pl.when(pl.col(f"{c}_other") < -0.5).then(pl.lit(99.0))
                              .otherwise(pl.col(c) - pl.col(f"{c}_other")).cast(pl.Float32).alias(f"{c}_gap") for c in BASE])
        np.save(out, d.select(REC_NAMES).to_numpy().astype(np.float32))
        print(f"  wrote {t}_R.npy", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    run(ap.parse_args().split)
