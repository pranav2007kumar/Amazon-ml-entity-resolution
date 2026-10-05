"""Stage 3b: second blocking pass -- rare-token overlap (inverted index).

Dense SVD vectors blur rare, highly specific tokens (a building name, a locality,
a house number). This pass scores S1 candidates by the summed IDF weight of the
exact tokens they share with the query record:
  address unigrams + adjacent-token bigrams, name core tokens, glued full name.
Tokens that are too common within a country (document frequency > DF_CAP over
S1) are ignored, which keeps the join small and discards uninformative words.

Output: <work>/<split>_cand_sparse.parquet with columns s1, src, row, sp_score, sp_rank
"""
import argparse
import os

import numpy as np
import polars as pl

from config import WORK_DIR

DF_CAP = 200
Q_CHUNK = 60000


def tokens(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    """Unique (id, token-hash) rows for a frame with name_core / addr_norm."""
    a = (df.select(id_col, pl.col("addr_norm").str.split(" ").alias("t"))
           .with_columns(pl.col("t").list.eval(pl.element().filter(pl.element() != "")))
           .explode("t").drop_nulls("t")
           .with_columns(pl.col("t").shift(-1).over(id_col).alias("t2")))
    uni = a.select(id_col, ("a|" + pl.col("t")).alias("tok"))
    bi = a.drop_nulls("t2").select(id_col, ("b|" + pl.col("t") + "_" + pl.col("t2")).alias("tok"))
    n = (df.select(id_col, pl.col("name_core").str.split(" ").alias("t")).explode("t")
           .filter(pl.col("t").str.len_chars() >= 2).select(id_col, ("n|" + pl.col("t")).alias("tok")))
    g = (df.filter(pl.col("name_core") != "")
           .select(id_col, ("g|" + pl.col("name_core").str.replace_all(" ", "")).alias("tok")))
    # sound keys of name words: 'kansalting' (Hindi script) meets 'consulting', typos meet originals
    p = (df.select(id_col, pl.col("ph").str.split(" ").alias("t")).explode("t")
           .filter(pl.col("t").str.len_chars() >= 3).select(id_col, ("p|" + pl.col("t")).alias("tok")))
    # initials: 'Medecins Centre' (mc) meets 'MC'; both sides emit their initials and short glued names
    glued = pl.col("name_core").str.replace_all(" ", "")
    i1 = df.filter(pl.col("acr").str.len_chars() >= 2).select(id_col, ("i|" + pl.col("acr")).alias("tok"))
    i2 = (df.filter((glued.str.len_chars() >= 2) & (glued.str.len_chars() <= 6))
            .select(id_col, ("i|" + glued).alias("tok")))
    out = pl.concat([uni, bi, n, g, p, i1, i2])
    return out.select(id_col, pl.col("tok").hash().alias("h")).unique()


def run(split, k):
    cols = ["country", "name_core", "addr_norm", "ph", "acr"]
    s1 = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s1.parquet"), columns=cols).with_row_index("s1")
    parts = []
    for country in s1["country"].unique().to_list():
        c1 = s1.filter(pl.col("country") == country)
        t1 = tokens(c1, "s1")
        df_ = t1.group_by("h").len("df").filter(pl.col("df") <= DF_CAP)
        n1 = len(c1)
        df_ = df_.with_columns((np.log(n1 / pl.col("df").cast(pl.Float64))).cast(pl.Float32).alias("w"))
        t1 = t1.join(df_.select("h", "w"), on="h")
        for src in (2, 3):
            q = (pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s{src}.parquet"), columns=cols)
                   .with_row_index("row").filter(pl.col("country") == country))
            for i in range(0, len(q), Q_CHUNK):
                tq = tokens(q[i:i + Q_CHUNK], "row")
                sc = (tq.join(t1, on="h").group_by("row", "s1").agg(pl.col("w").sum().alias("sp_score"))
                        .sort(["row", "sp_score"], descending=[False, True])
                        .with_columns(pl.col("s1").cum_count().over("row").alias("sp_rank"))
                        .filter(pl.col("sp_rank") <= k)
                        .with_columns((pl.col("sp_rank") - 1).cast(pl.Int8)))
                parts.append(sc.select(pl.col("s1").cast(pl.Int32), pl.lit(src, pl.Int8).alias("src"),
                                       pl.col("row").cast(pl.Int32), "sp_score", "sp_rank"))
            print(f"{split} S{src} {country}: {len(q):,} queries", flush=True)
    out = pl.concat(parts)
    out.write_parquet(os.path.join(WORK_DIR, f"{split}_cand_sparse.parquet"))
    print(f"{split}: {len(out):,} sparse candidate pairs", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    for s in a.splits:
        run(s, a.k)
