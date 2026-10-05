"""Stage 2: dense vectors for blocking (character n-gram TF-IDF -> SVD, on GPU).

Each record gets three L2-normalised blocks that are concatenated with weights:
  name    : char 2-4 gram TF-IDF of the squashed core name, SVD -> NAME_DIM
  address : char 2-4 gram TF-IDF of the normalised address, SVD -> ADDR_DIM
  numbers : signed feature-hashing of number tokens -> NUM_DIM
so that cosine(u, v) = w_n*name_sim + w_a*addr_sim + w_d*number_sim.

The SVD is fitted without labels on a sample drawn from both train and test
(so unseen countries such as France contribute their own character patterns).
"""
import os
import pickle
from multiprocessing import Pool

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils.extmath import randomized_svd

from config import WORK_DIR
from normalize import squash

NAME_DIM, ADDR_DIM, NUM_DIM = 128, 128, 64
N_FEAT = 2 ** 20
W_NAME, W_ADDR, W_NUM = 0.60, 0.65, 0.45

_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=N_FEAT,
                        alternate_sign=False, norm=None, dtype=np.float32)
_HV_NUM = HashingVectorizer(analyzer=str.split, n_features=NUM_DIM, alternate_sign=True,
                            norm="l2", dtype=np.float32)


def name_text(core: str) -> str:
    return squash(core)


def _hash_chunk(args):
    """Worker: hash one chunk of names / addresses / numbers into sparse matrices."""
    names, addrs, nums = args
    xn = _HV.transform([name_text(s) for s in names])
    xa = _HV.transform([squash(s) for s in addrs])
    xd = _HV_NUM.transform(nums)
    for x in (xn, xa):
        x.data = np.log1p(x.data)
    return xn, xa, xd


def hash_frame(df, pool, chunk=50000):
    names, addrs, nums = df["name_core"].to_list(), df["addr_norm"].to_list(), df["nums"].to_list()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk], nums[i:i + chunk]) for i in range(0, len(names), chunk)]
    parts = pool.map(_hash_chunk, jobs)
    return [sp.vstack([p[j] for p in parts]).tocsr() for j in range(3)]


def _tfidf_l2_inplace(x, idf):
    """In place: multiply CSR columns by idf, then L2-normalise rows (low memory)."""
    x.data *= idf[x.indices]
    rows = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
    norms = np.sqrt(np.bincount(rows, weights=x.data.astype(np.float64) ** 2, minlength=x.shape[0])) + 1e-9
    x.data /= norms[rows].astype(np.float32)
    return x


def fit(pool, sample_per_file=120000, seed=0):
    """Fit IDF weights and SVD projections on a label-free sample."""
    frames = []
    for split in ("train", "test"):
        for k in (1, 2, 3):
            df = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s{k}.parquet"),
                                 columns=["name_core", "addr_norm", "nums"])
            frames.append(df.sample(min(sample_per_file, len(df)), seed=seed))
    df = pl.concat(frames)
    xn, xa, _ = hash_frame(df, pool)
    model = {}
    for key, x, dim in (("name", xn, NAME_DIM), ("addr", xa, ADDR_DIM)):
        dfreq = np.bincount(x.indices, minlength=N_FEAT)
        idf = (np.log((1 + x.shape[0]) / (1 + dfreq)) + 1).astype(np.float32)
        xt = _tfidf_l2_inplace(x, idf)
        _, _, vt = randomized_svd(xt, dim, n_iter=4, random_state=seed)
        del xt, x
        model[key] = {"idf": idf, "vt": vt.astype(np.float32)}
        print(f"fitted {key}: sample {len(df):,}", flush=True)
    with open(os.path.join(WORK_DIR, "embed_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    return model


def _project(x, idf, vt_gpu, batch=200000):
    """TF-IDF weight + L2 + SVD projection on the GPU; returns float16 numpy."""
    x = _tfidf_l2_inplace(x.tocsr(), idf)
    out = []
    for i in range(0, x.shape[0], batch):
        b = x[i:i + batch]
        t = torch.sparse_csr_tensor(torch.from_numpy(b.indptr.astype(np.int64)),
                                    torch.from_numpy(b.indices.astype(np.int64)),
                                    torch.from_numpy(b.data), size=b.shape, device="cuda")
        y = torch.nn.functional.normalize(t @ vt_gpu, dim=1)
        out.append(y.half().cpu().numpy())
    return np.vstack(out)


def embed_file(split, k, model, pool):
    """Write <split>_s<k>_vec.npy (float16, rows aligned with the parquet)."""
    df = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s{k}.parquet"),
                         columns=["name_core", "addr_norm", "nums"])
    vt_n = torch.from_numpy(model["name"]["vt"].T.copy()).cuda()
    vt_a = torch.from_numpy(model["addr"]["vt"].T.copy()).cuda()
    vec = np.lib.format.open_memmap(os.path.join(WORK_DIR, f"{split}_s{k}_vec.npy"), mode="w+",
                                    dtype=np.float16, shape=(len(df), NAME_DIM + ADDR_DIM + NUM_DIM))
    step = 200000
    for i in range(0, len(df), step):  # chunked to bound RAM
        xn, xa, xd = hash_frame(df[i:i + step], pool)
        vn = _project(xn, model["name"]["idf"], vt_n)
        va = _project(xa, model["addr"]["idf"], vt_a)
        has_addr = (np.asarray(xa.sum(1)).ravel() > 0)[:, None]
        va = va * has_addr  # empty address -> zero block (no evidence either way)
        vd = xd.toarray().astype(np.float16)
        vec[i:i + len(vn)] = np.hstack([W_NAME * vn, W_ADDR * va, W_NUM * vd])
    vec.flush()
    print(f"embedded {split} s{k}: {vec.shape}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    with Pool(8) as pool:  # each worker holds a chunk of sparse features; 8 keeps RAM < 16 GB
        mp = os.path.join(WORK_DIR, "embed_model.pkl")
        model = fit(pool) if a.refit or not os.path.exists(mp) else pickle.load(open(mp, "rb"))
        for split in a.splits:
            for k in (1, 2, 3):
                embed_file(split, k, model, pool)
