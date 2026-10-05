"""Label-free threshold calibration for countries unseen in training (open set: no country is named).

In the training data the number of true matches per S1 has the same distribution in every country (US and India
agree to within 0.01 matches per S1). On seen countries our decision rule produces a certain operating point
(mean predicted matches per S1 and share of empty S1). For every unseen country, the group thresholds are raised by
the smallest delta that brings its predicted mean down to the seen countries' predicted mean.

  python unseen_calib.py <prefix> <base_out_dir> <new_out_dir>
     prefix       e.g. ce_sv_ (uses <MODEL_DIR>/<prefix>test_scores_blend.parquet and <prefix>val_results.json)
     base_out_dir submission whose rows are kept for seen countries and whose candidate file is reused
"""
import json
import os
import shutil
import sys

import numpy as np
import polars as pl

from config import WORK_DIR
from two_stage import MODEL_DIR, assign


def main(prefix, base, out):
    thr = json.load(open(os.path.join(MODEL_DIR, prefix + "val_results.json")))["blended"]["group_threshold"]["thr"]
    thr = {int(k): float(v) for k, v in thr.items()}
    seen = set(pl.read_parquet(os.path.join(WORK_DIR, "train_s1.parquet"), columns=["country"])["country"].unique().to_list())
    s1c = pl.read_parquet(os.path.join(WORK_DIR, "test_s1.parquet"), columns=["country"])["country"].to_numpy()
    n1 = len(s1c)
    sc = pl.scan_parquet(os.path.join(MODEL_DIR, prefix + "test_scores_blend.parquet")).filter(pl.col("p") >= 0.3) \
        .select("s1", "src", "row", "p", "grp").collect()
    a = assign(sc, sc["p"].to_numpy()).join(sc.select("s1", "src", "row", "grp"), on=["s1", "src", "row"])
    a = a.with_columns(pl.Series("unseen", ~np.isin(s1c[a["s1"].to_numpy()], list(seen))))
    t = pl.col("grp").cast(pl.Int64).replace_strict(thr, default=0.99)
    CAP = 0.995   # a raised threshold must stay reachable: at >= 1.0 a whole record group would vanish

    def raised(delta):
        return pl.when(pl.col("unseen")).then(pl.min_horizontal(t + delta, pl.lit(CAP))).otherwise(t)

    def stats(keep, mask_s1):
        cnt = np.bincount(keep["s1"].to_numpy(), minlength=n1)[mask_s1]
        return float(cnt.mean()), float((cnt == 0).mean())

    seen_mask = np.isin(s1c, list(seen))
    base_keep = a.filter(pl.col("p") >= t)
    target_mean, target_empty = stats(base_keep, seen_mask)
    print(f"seen countries {sorted(seen)}: mean {target_mean:.3f} matches/S1, empty {target_empty:.4f}")
    res = []
    for c in sorted(set(s1c) - seen):
        m = s1c == c
        prev = None
        for delta in np.arange(0.0, 0.30, 0.01):
            keep = a.filter(pl.col("p") >= raised(delta))
            mu, em = stats(keep, m)
            if prev is not None and prev - mu > 0.1:   # sudden collapse: stop before it
                print(f"  {c} delta {delta:.2f}: mean {mu:.3f} -> collapse, stop")
                break
            res.append((c, round(float(delta), 2), mu, em))
            prev = mu
            if mu <= target_mean:
                break
    for r in res:
        print(f"  {r[0]} delta {r[1]:.2f}: mean {r[2]:.3f}, empty {r[3]:.4f}")
    # closest to the seen-country mean (never below it by more than 0.02), per unseen country; one delta for all
    delta = max(min((r for r in res if r[0] == c and r[2] >= target_mean - 0.02), key=lambda r: abs(r[2] - target_mean))[1]
                for c in {r[0] for r in res})
    if os.environ.get("ER_CALIB_DELTA"):   # reproduce an earlier run exactly (v6ce6 used 0.16)
        delta = float(os.environ["ER_CALIB_DELTA"])
    print(f"chosen delta for unseen countries: {delta:.2f}")
    keep = a.filter(pl.col("p") >= raised(delta))
    chk = stats(keep, ~seen_mask)
    assert chk[0] > target_mean - 0.2, f"unseen mean collapsed to {chk[0]:.3f}"
    print(f"unseen countries after calibration: mean {chk[0]:.3f}, empty {chk[1]:.4f}")
    # write: unseen S1 rows from the new rule, seen rows unchanged from base; candidates reused from base
    ids = {k: pl.read_parquet(os.path.join(WORK_DIR, f"test_s{k}.parquet"), columns=["entity_id"])["entity_id"].to_numpy() for k in (1, 2, 3)}
    kn = keep.filter(pl.col("unseen"))
    srcs, rows = kn["src"].to_numpy(), kn["row"].to_numpy()
    other = np.empty(len(kn), object)
    for k in (2, 3):
        other[srcs == k] = ids[k][rows[srcs == k]]
    lists = pl.DataFrame({"s1": kn["s1"].to_numpy(), "m": other}).group_by("s1").agg(pl.col("m").sort().str.join(","))
    newmap = dict(zip(ids[1][lists["s1"].to_numpy()], lists["m"].to_list()))
    unseen_ids = set(ids[1][~seen_mask].tolist())
    os.makedirs(out, exist_ok=True)
    lines = open(os.path.join(base, "matching_results.tsv"), encoding="utf-8").read().split("\n")
    o, n = [lines[0]], 0
    for ln in lines[1:]:
        if not ln:
            continue
        k = ln.partition("\t")[0]
        if k in unseen_ids:
            o.append(k + "\t" + newmap.get(k, "")); n += 1
        else:
            o.append(ln)
    open(os.path.join(out, "matching_results.tsv"), "w", encoding="utf-8", newline="").write("\n".join(o) + "\n")
    src_c = os.path.join(base, "candidate_pairs.tsv")
    dst_c = os.path.join(out, "candidate_pairs.tsv")
    if not os.path.exists(dst_c):
        try:
            os.link(src_c, dst_c)
        except OSError:
            shutil.copy(src_c, dst_c)
    print(f"wrote {out}: {n:,} unseen-country rows re-decided (delta {delta:.2f})")


if __name__ == "__main__":
    main(*sys.argv[1:4])
