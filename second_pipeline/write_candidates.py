"""Fast writer for output/candidate_pairs.tsv (the exact pairs the model scored).

  python write_candidates.py --work work50 --data-dir dataset --out-dir output
reads only (rec, s1) from <work>/test/feats_s2|3.parquet, so it can be re-run without re-scoring.
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from common import log, read_tsv


def _ids(work, k):
    s = pd.read_parquet(os.path.join(work, "test", f"s{k}", "text.parquet"), columns=["entity_id"])["entity_id"]
    return np.asarray(s.astype(object).tolist(), dtype=object)


def write_candidates(cands, ids, t1_ids, order, path):
    """cands: list of (src, rec_rows, s1_rows); ids: {src: object array}; order: S1 ids in file order."""
    n1 = len(t1_ids)
    per = {}
    for k, rec, s1 in cands:
        key = s1.astype(np.int64) * (1 << 32) + rec.astype(np.int64)
        key.sort()
        s1s, recs = (key >> 32).astype(np.int32), (key & 0xFFFFFFFF).astype(np.int64)
        per[k] = (ids[k][recs], np.searchsorted(s1s, np.arange(n1), "left"), np.searchsorted(s1s, np.arange(n1), "right"))
        del key, s1s, recs
        log(f"  S{k}: candidate lists indexed")
    row_of = pd.Series(np.arange(n1), index=t1_ids)
    rows = row_of.reindex(order).values
    a2, s2, e2 = per[2]
    a3, s3, e3 = per[3]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        buf = []
        for eid, r in zip(order, rows):
            buf.append(eid + "\t" + ",".join(a2[s2[r]:e2[r]].tolist() + a3[s3[r]:e3[r]].tolist()) + "\n")
            if len(buf) >= 100_000:
                fh.write("".join(buf)); buf = []
        fh.write("".join(buf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="output")
    a = ap.parse_args()
    t1 = pd.read_parquet(os.path.join(a.work, "test", "s1", "text.parquet"), columns=["entity_id"])["entity_id"]
    ids = {k: _ids(a.work, k) for k in (2, 3)}
    cands = []
    for k in (2, 3):
        t = ds.dataset(os.path.join(a.work, "test", f"feats_s{k}.parquet")).to_table(columns=["rec", "s1"])
        cands.append((k, t["rec"].to_numpy(), t["s1"].to_numpy()))
        del t
    order = read_tsv(os.path.join(a.data_dir, "test", "test_source1.tsv"), usecols=["entity_id"])["entity_id"].tolist()
    os.makedirs(a.out_dir, exist_ok=True)
    write_candidates(cands, ids, np.asarray(t1.astype(object).tolist(), dtype=object), order,
                     os.path.join(a.out_dir, "candidate_pairs.tsv"))
    log("wrote candidate_pairs.tsv")


if __name__ == "__main__":
    main()
