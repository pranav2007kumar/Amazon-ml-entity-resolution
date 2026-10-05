"""Stage 3: candidate generation (blocking) with exact GPU nearest-neighbour search.

For every Source-2 / Source-3 record we retrieve the top-K most similar Source-1
records *of the same country* by cosine similarity of the blocking vectors
(embed.py). Because every S2/S3 record belongs to at most one S1 entity, the
"reverse" direction (S2/S3 -> S1) is the natural one: each record only has to
find its own parent. The search is brute-force fp16 matrix multiplication on the
GPU, so there is no approximate-index recall loss.

Output: <work>/<split>_cand.parquet  with columns
  s1 (row in s1 parquet), src (2|3), row (row in src parquet), sim, rank
"""
import argparse
import os

import numpy as np
import polars as pl
import torch

from config import WORK_DIR


# Which part of the blocking vector to search with (embed.py layout: name 0:128 | address 128:256 |
# numbers 256:320). "name" and "addr" are extra routes: a record whose address changed a lot (or whose
# name was replaced) can drop out of the combined top-k but still be close on the other part.
PARTS = {"all": slice(0, 320), "name": slice(0, 128), "addr": slice(128, 320)}


class _Part:
    """Row-sliceable view of the vectors restricted to one part, L2-renormalised on read."""

    def __init__(self, vec, part):
        self.vec, self.sl = vec, PARTS[part]
        self.full = part == "all"

    def __len__(self):
        return len(self.vec)

    def __getitem__(self, idx):
        v = np.asarray(self.vec[idx][:, self.sl], dtype=np.float32)
        if not self.full:
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
        return v.astype(np.float16)


def load(split, k, part="all"):
    df = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s{k}.parquet"), columns=["country", "addr_norm"])
    vec = np.load(os.path.join(WORK_DIR, f"{split}_s{k}_vec.npy"), mmap_mode="r")
    return df["country"].to_numpy(), (df["addr_norm"] == "").to_numpy(), _Part(vec, part)


@torch.no_grad()
def topk_search(keys, queries, k, batch=1024):
    """Return (idx, sim) of the k best keys for every query row (both on GPU)."""
    kt = keys.T.contiguous()
    idx_out, sim_out = [], []
    for i in range(0, queries.shape[0], batch):
        s = queries[i:i + batch] @ kt
        v, j = torch.topk(s, min(k, kt.shape[1]), dim=1)
        idx_out.append(j.int().cpu())
        sim_out.append(v.float().cpu())
    return torch.cat(idx_out).numpy(), torch.cat(sim_out).numpy()


def run(split, k, k_empty, part="all"):
    """Records with an empty address can only be matched by name, and a name alone is less
    specific, so they get a deeper candidate list (k_empty)."""
    c1, _, v1 = load(split, 1, part)
    parts = []
    for src in (2, 3):
        cs, empty, vs = load(split, src, part)
        if part == "addr":
            empty_q = empty
            empty = np.zeros_like(empty)  # address route: skip records without an address below
        else:
            empty_q = None
        for country in np.unique(c1):  # open set: whatever labels the data has
            kr = np.nonzero(c1 == country)[0]
            if len(kr) == 0:
                continue
            keys = torch.from_numpy(np.ascontiguousarray(v1[kr])).cuda()
            base_m = (cs == country) & (~empty_q if empty_q is not None else True)
            for qr, kq in ((np.nonzero(base_m & ~empty)[0], k), (np.nonzero(base_m & empty)[0], k_empty)):
                if len(qr) == 0:
                    continue
                out_i, out_s = [], []
                for j in range(0, len(qr), 500000):  # stream queries from disk
                    q = torch.from_numpy(np.ascontiguousarray(vs[qr[j:j + 500000]])).cuda()
                    ii, ss = topk_search(keys, q, kq)
                    out_i.append(ii); out_s.append(ss)
                    del q
                ii, ss = np.vstack(out_i), np.vstack(out_s)
                kk = ii.shape[1]
                parts.append(pl.DataFrame({
                    "s1": kr[ii.ravel()].astype(np.int32),
                    "src": np.full(ii.size, src, np.int8),
                    "row": np.repeat(qr, kk).astype(np.int32),
                    "sim": ss.ravel().astype(np.float32),
                    "rank": np.tile(np.arange(kk, dtype=np.int8), len(qr)),
                }))
                print(f"{split} S{src} {country}: {len(qr):,} queries (k={kk}) vs {len(kr):,} S1", flush=True)
            del keys
            torch.cuda.empty_cache()
    cand = pl.concat(parts)
    suffix = "" if part == "all" else f"_{part}"
    cand.write_parquet(os.path.join(WORK_DIR, f"{split}_cand{suffix}.parquet"))
    print(f"{split}: {len(cand):,} candidate pairs ({part})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--k_empty", type=int, default=30)
    ap.add_argument("--part", choices=list(PARTS), default="all")
    a = ap.parse_args()
    for s in a.splits:
        run(s, a.k, a.k_empty, a.part)
