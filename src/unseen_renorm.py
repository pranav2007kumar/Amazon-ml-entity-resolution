"""Unseen-country decision with one-owner renormalisation.

For every S2/S3 record of a country unseen in training, the stack probabilities of its candidate S1s are turned into
ownership shares q_i = o_i / (1 + sum_j o_j) (o = p / (1 - p)); a record with two strong competing S1s (co-located
businesses) then no longer gives both a high score. Records are assigned to their best S1 by q, the validation group
thresholds are raised by one delta chosen so that the unseen countries' mean matches per S1 equals TARGET, and the
seen-country rows are copied unchanged from the base submission.

  python unseen_renorm.py <stack prefix> <base_out_dir> <new_out_dir> <target mean matches per S1>
"""
import json
import os
import shutil
import sys

import numpy as np
import polars as pl

from config import WORK_DIR
from two_stage import MODEL_DIR, assign


def main(pre, base, out, target):
    target = float(target)
    thr = json.load(open(os.path.join(MODEL_DIR, pre + "val_results.json")))["blended"]["group_threshold"]["thr"]
    thr = {int(k): float(v) for k, v in thr.items()}
    seen = set(pl.read_parquet(os.path.join(WORK_DIR, "train_s1.parquet"), columns=["country"])["country"].unique().to_list())
    s1c = pl.read_parquet(os.path.join(WORK_DIR, "test_s1.parquet"), columns=["country"])["country"].to_numpy()
    n1, unseen_mask = len(s1c), ~np.isin(s1c, list(seen))
    unseen = pl.Series(np.nonzero(unseen_mask)[0].astype(np.int32))
    sc = pl.scan_parquet(os.path.join(MODEL_DIR, pre + "test_scores_blend.parquet")).filter(pl.col("s1").is_in(unseen) & (pl.col("p") > 0)) \
        .select("s1", "src", "row", "p", "grp").collect()
    pc = pl.col("p").clip(1e-6, 1 - 1e-6)
    sc = sc.with_columns((pc / (1 - pc)).alias("o"))
    sc = sc.with_columns((pl.col("o") / (1 + pl.col("o").sum().over(["src", "row"]))).alias("q"))
    a = assign(sc, sc["q"].to_numpy()).join(sc.select("s1", "src", "row", "grp"), on=["s1", "src", "row"])
    t = pl.col("grp").cast(pl.Int64).replace_strict(thr, default=0.99)
    best = None
    for d in np.arange(-0.3, 0.6, 0.005):
        keep = a.filter(pl.col("p") >= pl.min_horizontal(t + d, pl.lit(0.995)))
        mu = float(np.bincount(keep["s1"].to_numpy(), minlength=n1)[unseen_mask].mean())
        if best is None or abs(mu - target) < abs(best[1] - target):
            best = (round(float(d), 3), mu)
    d = best[0]
    keep = a.filter(pl.col("p") >= pl.min_horizontal(t + d, pl.lit(0.995)))
    cnt = np.bincount(keep["s1"].to_numpy(), minlength=n1)[unseen_mask]
    print(f"unseen: delta {d:+.3f} -> mean {cnt.mean():.3f} matches/S1 (target {target}), empty {(cnt == 0).mean():.4f}", flush=True)
    ids1 = pl.read_parquet(os.path.join(WORK_DIR, "test_s1.parquet"), columns=["entity_id"])["entity_id"].to_numpy()
    ids = {k: pl.read_parquet(os.path.join(WORK_DIR, f"test_s{k}.parquet"), columns=["entity_id"])["entity_id"].to_numpy() for k in (2, 3)}
    srcs, rws = keep["src"].to_numpy(), keep["row"].to_numpy()
    other = np.empty(len(keep), object)
    for k in (2, 3):
        other[srcs == k] = ids[k][rws[srcs == k]]
    lists = pl.DataFrame({"s1": keep["s1"].to_numpy(), "m": other}).group_by("s1").agg(pl.col("m").sort().str.join(","))
    newmap = dict(zip(ids1[lists["s1"].to_numpy()], lists["m"].to_list()))
    unseen_ids = set(ids1[unseen_mask].tolist())
    os.makedirs(out, exist_ok=True)
    lines = open(os.path.join(base, "matching_results.tsv"), encoding="utf-8").read().split("\n")
    o, changed = [lines[0]], 0
    for ln in lines[1:]:
        if not ln:
            continue
        k, _, old = ln.partition("\t")
        if k in unseen_ids:
            new = newmap.get(k, "")
            changed += set(old.split(",")) != set(new.split(","))
            ln = k + "\t" + new
        o.append(ln)
    open(os.path.join(out, "matching_results.tsv"), "w", encoding="utf-8", newline="").write("\n".join(o) + "\n")
    dst = os.path.join(out, "candidate_pairs.tsv")
    if not os.path.exists(dst):
        try:
            os.link(os.path.join(base, "candidate_pairs.tsv"), dst)
        except OSError:
            shutil.copy(os.path.join(base, "candidate_pairs.tsv"), dst)
    print(f"wrote {out}: {changed:,} unseen-country S1 rows changed vs base", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
