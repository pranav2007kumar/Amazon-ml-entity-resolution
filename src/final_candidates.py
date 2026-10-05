"""Write candidate_pairs.tsv as exactly the pairs the FINAL model (the stacker) scored.

Stage-1/2 XGBoost acts as a learned blocking filter: of the ~114 raw blocking candidates per S1, only pairs with our
score >= 1e-3 or the second pipeline's score >= 1e-4 are passed to the stacker (~7.9 per S1). Those are the pairs the final model
scores and the only pairs a match can come from, so they form the candidate set (matches are asserted to be a subset).

  python final_candidates.py <stack prefix> <submission dir>   (rewrites <dir>/candidate_pairs.tsv)
"""
import os
import sys

import numpy as np
import polars as pl

from config import WORK_DIR
from two_stage import MODEL_DIR


def main(pre, out):
    st = pl.scan_parquet(os.path.join(MODEL_DIR, pre + "test_scores_blend.parquet")).filter(pl.col("p_xgb").is_null()) \
        .select("s1", "src", "row").collect()
    ids1 = pl.read_parquet(os.path.join(WORK_DIR, "test_s1.parquet"), columns=["entity_id"])["entity_id"].to_numpy()
    ids = {k: pl.read_parquet(os.path.join(WORK_DIR, f"test_s{k}.parquet"), columns=["entity_id"])["entity_id"].to_numpy() for k in (2, 3)}
    src, row = st["src"].to_numpy(), st["row"].to_numpy()
    c = np.empty(len(st), object)
    for k in (2, 3):
        c[src == k] = ids[k][row[src == k]]
    g = pl.DataFrame({"s1": st["s1"].to_numpy(), "c": c.astype(str)}).group_by("s1").agg(pl.col("c").sort().str.join(","))
    cand = dict(zip(ids1[g["s1"].to_numpy()], g["c"].to_list()))
    lines = open(os.path.join(out, "matching_results.tsv"), encoding="utf-8").read().split("\n")
    o, n = ["source1_entity_id\tcandidate_entity_ids"], 0
    for ln in lines[1:]:
        if not ln:
            continue
        k, _, m = ln.partition("\t")
        cs = cand.get(k, "")
        assert set(x for x in m.split(",") if x) <= set(cs.split(",")), f"match outside candidates for {k}"
        o.append(k + "\t" + cs); n += cs.count(",") + (cs != "")
    dst = os.path.join(out, "candidate_pairs.tsv")
    if os.path.exists(dst):
        os.remove(dst)            # may be a hard link to another submission's file
    open(dst, "w", encoding="utf-8", newline="").write("\n".join(o) + "\n")
    print(f"wrote {dst}: {n:,} candidate pairs, {n / (len(o) - 1):.2f} per S1", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
