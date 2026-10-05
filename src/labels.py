"""Ground-truth helpers: positive pairs in row-index form, and the F0.5 metric."""
import os

import numpy as np
import polars as pl

from config import DATA_DIR, WORK_DIR


def id_to_row(split, k):
    """DataFrame mapping entity_id -> row index in the <split>_s<k> parquet."""
    ids = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s{k}.parquet"), columns=["entity_id"])
    return ids.with_row_index("row").with_columns(pl.col("row").cast(pl.Int32))


def positive_pairs():
    """All true (s1, src, row) pairs of the training set."""
    gt = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t",
                     quote_char=None, infer_schema=False).fill_null("")
    ex = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != ""))
    s1 = id_to_row("train", 1).rename({"entity_id": "source1_entity_id", "row": "s1"})
    ex = ex.join(s1, on="source1_entity_id")
    out = []
    for k in (2, 3):
        m = id_to_row("train", k).rename({"entity_id": "matched_entity_ids"})
        out.append(ex.join(m, on="matched_entity_ids").select(
            "s1", pl.lit(k, pl.Int8).alias("src"), "row"))
    return pl.concat(out)


def macro_f05(pred: dict, truth: dict, s1_rows) -> float:
    """Macro F0.5 over the given S1 rows. pred/truth: s1 -> set of (src,row)."""
    tot = 0.0
    for s in s1_rows:
        p, t = pred.get(s, set()), truth.get(s, set())
        if not t:
            tot += 1.0 if not p else 0.0
            continue
        if not p:
            continue
        tp = len(p & t)
        if tp == 0:
            continue
        pr, rc = tp / len(p), tp / len(t)
        tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / len(s1_rows)
