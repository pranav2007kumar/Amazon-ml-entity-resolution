"""Write a small copy of the dataset (same folder layout) to smoke-test every stage in minutes.

Train: a random fraction of S1, all their matches, and the same fraction of unmatched S2/S3.
Test : the same fraction of every source (keeps France).  Only for testing code, never for the final run.
"""
import argparse
import os

from common import load_gt_pairs, read_tsv, source_path, stable_hash01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.01)
    a = ap.parse_args()
    for sp in ("train", "test"):
        os.makedirs(os.path.join(a.out, sp), exist_ok=True)
    gt, pairs = load_gt_pairs(os.path.join(a.data_dir, "train", "train_ground_truth.tsv"))
    s1 = read_tsv(source_path(a.data_dir, "train", 1))
    s1 = s1[stable_hash01(s1["entity_id"].values, "sample") < a.frac]
    keep1 = set(s1["entity_id"])
    own = set(pairs.loc[pairs["s1_id"].isin(keep1), "other_id"])
    owned_any = set(pairs["other_id"])
    s1.to_csv(source_path(a.out, "train", 1), sep="\t", index=False, quoting=3)
    gt[gt["source1_entity_id"].isin(keep1)].to_csv(os.path.join(a.out, "train", "train_ground_truth.tsv"),
                                                    sep="\t", index=False, quoting=3)
    for k in (2, 3):
        r = read_tsv(source_path(a.data_dir, "train", k))
        ids = r["entity_id"]
        m = ids.isin(own) | (~ids.isin(owned_any) & (stable_hash01(ids.values, "sample") < a.frac))
        r[m].to_csv(source_path(a.out, "train", k), sep="\t", index=False, quoting=3)
    for k in (1, 2, 3):
        r = read_tsv(source_path(a.data_dir, "test", k))
        r[stable_hash01(r["entity_id"].values, "sample") < a.frac * 2].to_csv(
            source_path(a.out, "test", k), sep="\t", index=False, quoting=3)
    print("sample written to", a.out)


if __name__ == "__main__":
    main()
