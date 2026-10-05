"""Stage 1b: rarity statistics per record, computed from the provided files only (per country).

Adds these columns to <work>/<split>_s<k>.parquet:
  g_s1cnt  number of Source-1 records in the same country with the same glued core name
  g_ocnt   number of Source-2/3 records in the same country with the same glued core name
  a_s1cnt  number of Source-1 records in the same country with the same normalised address (0 if empty)
  n_idf    IDF (over Source-1 names of the country) of each squashed core-name token, aligned with
           squash(name_core).split()
  a_idf    IDF (over Source-1 addresses of the country) of each addr_norm token, aligned with
           addr_norm.split()
Rare shared words ('Galaxy') are strong evidence, common ones ('Enterprises') are weak; a name shared
by several Source-1 businesses cannot identify its owner on its own.
"""
import argparse
import os

import polars as pl

from config import WORK_DIR
from normalize import squash


def _path(split, k):
    return os.path.join(WORK_DIR, f"{split}_s{k}.parquet")


def _idf_lists(df: pl.DataFrame, col: str, idf: pl.DataFrame, n1: pl.DataFrame) -> pl.Series:
    """For every row, the list of IDF values of the tokens in `col` (unknown token -> ln N)."""
    t = (df.select("i", "country", pl.col(col).str.split(" ").alias("tok"))
           .explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")))
    t = (t.join(idf, on=["country", "tok"], how="left").join(n1, on="country", how="left")
          .with_columns(pl.col("idf").fill_null(pl.col("ln_n")).cast(pl.Float32)))
    g = t.group_by("i", maintain_order=True).agg(pl.col("idf"))
    out = df.select("i").join(g, on="i", how="left", maintain_order="left")
    return out["idf"].fill_null(pl.lit([], dtype=pl.List(pl.Float32)))


def run(split):
    dfs = {}
    for k in (1, 2, 3):
        df = pl.read_parquet(_path(split, k))
        drop = [c for c in ("g_s1cnt", "g_ocnt", "a_s1cnt", "n_idf", "a_idf", "name_sq") if c in df.columns]
        df = df.drop(drop).with_row_index("i")
        sq = [squash(s) for s in df["name_core"].to_list()]
        dfs[k] = df.with_columns(pl.Series("name_sq", sq), pl.col("name_core").str.replace_all(" ", "").alias("glued"))
    s1 = dfs[1]
    g1 = s1.filter(pl.col("glued") != "").group_by("country", "glued").len("g_s1cnt")
    go = pl.concat([dfs[2].select("country", "glued"), dfs[3].select("country", "glued")]) \
        .filter(pl.col("glued") != "").group_by("country", "glued").len("g_ocnt")
    a1 = s1.filter(pl.col("addr_norm") != "").group_by("country", "addr_norm").len("a_s1cnt")
    n1 = s1.group_by("country").len("n").with_columns(pl.col("n").cast(pl.Float64).log().alias("ln_n")).drop("n")

    def idf_table(col):
        t = (s1.select("i", "country", pl.col(col).str.split(" ").alias("tok")).explode("tok")
               .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).unique(["i", "tok"]))
        df_ = t.group_by("country", "tok").len("df").join(n1, on="country")
        return df_.select("country", "tok", (pl.col("ln_n") - pl.col("df").cast(pl.Float64).log()).alias("idf"))

    idf_n, idf_a = idf_table("name_sq"), idf_table("addr_norm")
    for k in (1, 2, 3):
        df = dfs[k]
        df = (df.join(g1, on=["country", "glued"], how="left", maintain_order="left")
                .join(go, on=["country", "glued"], how="left", maintain_order="left")
                .join(a1, on=["country", "addr_norm"], how="left", maintain_order="left")
                .with_columns(pl.col("g_s1cnt", "g_ocnt", "a_s1cnt").fill_null(0).cast(pl.Int32)))
        df = df.with_columns(_idf_lists(df, "name_sq", idf_n, n1).alias("n_idf"),
                             _idf_lists(df, "addr_norm", idf_a, n1).alias("a_idf"))
        df.drop("i", "glued").write_parquet(_path(split, k))
        print(f"{split} s{k}: stats added ({len(df):,} rows)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    for s in ap.parse_args().splits:
        run(s)
