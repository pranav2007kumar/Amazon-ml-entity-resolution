"""Stage 1: read raw TSVs, normalise every record, write parquet.

Output: <work>/<split>_s<k>.parquet with columns
  entity_id, country, name, address          (raw)
  name_core, name_legal, addr_norm, nums, state   (normalised)
  ph (sound key per name token), acr (initials of a multi-word name)
"""
import argparse
import os
from multiprocessing import Pool

import polars as pl

from config import DATA_DIR, WORK_DIR
from normalize import address_tokens, name_tokens, squash
from translit import phonetic


def _norm_chunk(rows):
    """Normalise a list of (name, address) tuples; runs in a worker process."""
    out = []
    for name, addr in rows:
        core, legal = name_tokens(name or "")
        toks, nums, state = address_tokens(addr or "")
        sq = squash(" ".join(core)).split()
        ph = " ".join(p for p in (phonetic(t) for t in sq) if p)
        acr = "".join(t[0] for t in sq if t[0].isalpha()) if len(sq) >= 2 else ""
        out.append((" ".join(core), " ".join(legal), " ".join(toks), " ".join(nums), state, ph, acr))
    return out


def read_source(split, k):
    """Read one raw source file with every column as a string."""
    path = os.path.join(DATA_DIR, split, f"{split}_source{k}.tsv")
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    df = df.rename({"business_name": "name", "business_address": "address"})
    return df.with_columns(pl.col("name").fill_null(""), pl.col("address").fill_null(""))


def process(split, k, pool, chunk=20000):
    df = read_source(split, k)
    pairs = list(zip(df["name"].to_list(), df["address"].to_list()))
    chunks = [pairs[i:i + chunk] for i in range(0, len(pairs), chunk)]
    res = [r for part in pool.imap(_norm_chunk, chunks) for r in part]
    cols = list(zip(*res))
    df = df.with_columns(
        pl.Series("name_core", cols[0]), pl.Series("name_legal", cols[1]),
        pl.Series("addr_norm", cols[2]), pl.Series("nums", cols[3]),
        pl.Series("state", cols[4]), pl.Series("ph", cols[5]), pl.Series("acr", cols[6]),
    )
    out = os.path.join(WORK_DIR, f"{split}_s{k}.parquet")
    df.write_parquet(out)
    print(f"{split} s{k}: {len(df):,} rows -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        for split in a.splits:
            for k in (1, 2, 3):
                process(split, k, pool)
