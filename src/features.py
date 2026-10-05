"""Stage 4: pairwise features for (Source-1 record, candidate record).

All features are country-agnostic string/number similarities, plus
"competition" features from the blocking step (how the pair ranks against the
other candidates of the same record). No country one-hot is used, so the model
applies unchanged to countries unseen in training (France).
"""
import os
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from config import WORK_DIR
from normalize import squash

STR_COLS = ["name_core", "name_legal", "addr_norm", "nums", "state", "name", "address",
            "ph", "acr", "n_idf", "a_idf", "g_s1cnt", "g_ocnt", "a_s1cnt"]


def _jacc(a, b):
    if not a and not b:
        return -1.0
    return len(a & b) / max(1, len(a | b))


def _nonlatin(s):
    return float(any(ord(c) > 0x24F for c in s))


def _idf_overlap(toks, idf, other):
    """(shared IDF / total IDF, max IDF of a shared token, max IDF of a token missing from other)."""
    tot = shared = mx_sh = mx_miss = 0.0
    seen = set()
    for t, w in zip(toks, idf):
        if t in seen:
            continue
        seen.add(t)
        tot += w
        if t in other:
            shared += w
            mx_sh = max(mx_sh, w)
        else:
            mx_miss = max(mx_miss, w)
    return (shared / tot if tot > 0 else -1.0), mx_sh, mx_miss


def _extra_feats(asq, bsq, ag, bg, aa, ba, a_x, b_x):
    """v5 features: sound keys, initials, IDF-weighted overlaps, name/address frequencies."""
    aph, aacr, anidf, aaidf, ag1, ago, aa1 = a_x
    bph, bacr, bnidf, baidf, bg1, _, _ = b_x
    apg, bpg = aph.replace(" ", ""), bph.replace(" ", "")
    at, bt = asq.split(), bsq.split()
    ats, bts = set(at), set(bt)
    n_a = _idf_overlap(at, anidf, bts)
    n_b = _idf_overlap(bt, bnidf, ats)
    aat, bat = aa.split(), ba.split()
    a_a = _idf_overlap(aat, aaidf, set(bat)) if ba else (-1.0, -1.0, -1.0)
    a_b = _idf_overlap(bat, baidf, set(aat)) if ba else (-1.0, -1.0, -1.0)
    return (
        fuzz.ratio(apg, bpg) if apg and bpg else -1.0,
        fuzz.token_set_ratio(aph, bph) if aph and bph else -1.0,
        fuzz.token_sort_ratio(aph, bph) if aph and bph else -1.0,
        float(bool(aacr) and len(bg) <= 6 and (bg == aacr or bg == aacr[:len(bg)] and len(bg) >= 2)),
        float(bool(bacr) and len(ag) <= 6 and ag == bacr),
        n_a[0], n_b[0], n_a[1], n_a[2], n_b[2],
        a_a[0], a_b[0], a_a[2], a_b[2],
        float(ag1), float(bg1), float(ago), float(aa1),
        float(aa == ba) if aa and ba else -1.0,
    )


EXTRA_NAMES = ["ph_ratio", "ph_tset", "ph_tsort", "acr_b_of_a", "acr_a_of_b",
               "n_idf_a", "n_idf_b", "n_idf_max_shared", "n_idf_max_miss_a", "n_idf_max_miss_b",
               "a_idf_a", "a_idf_b", "a_idf_max_miss_a", "a_idf_max_miss_b",
               "g_s1cnt_a", "g_s1cnt_b", "g_ocnt_a", "a_s1cnt_a", "addr_exact"]


def _pair_feats(r):
    """Features for one pair; r = strings of record A (S1) then record B."""
    n = len(STR_COLS)
    an, al, aa, anum, ast, araw, _ = r[:7]
    bn, bl, ba, bnum, bst, braw, braddr = r[n:n + 7]
    a_x, b_x = r[7:n], r[n + 7:]
    asq, bsq = squash(an), squash(bn)
    ag, bg = asq.replace(" ", ""), bsq.replace(" ", "")
    at, bt = set(asq.split()), set(bsq.split())
    aat, bat = set(aa.split()), set(ba.split())
    anu, bnu = set(anum.split()), set(bnum.split())
    a_first = anum.split()[0] if anum else ""
    b_first = bnum.split()[0] if bnum else ""
    pins_a = {x for x in anu if len(x) >= 5}
    pins_b = {x for x in bnu if len(x) >= 5}
    return (
        # name
        fuzz.ratio(asq, bsq), fuzz.token_sort_ratio(asq, bsq), fuzz.token_set_ratio(asq, bsq),
        fuzz.partial_ratio(asq, bsq), JaroWinkler.similarity(ag, bg), fuzz.ratio(ag, bg),
        fuzz.partial_ratio(ag, bg) if min(len(ag), len(bg)) >= 4 else -1.0,
        _jacc(at, bt), len(at & bt), abs(len(at) - len(bt)), len(asq), len(bsq),
        float(asq == bsq), float(ag == bg),
        _jacc(set(al.split()), set(bl.split())), float(bool(al)), float(bool(bl)),
        _nonlatin(braw), float("." in braw and " " not in braw.strip()),
        # address
        fuzz.ratio(aa, ba) if ba else -1.0, fuzz.token_set_ratio(aa, ba) if ba else -1.0,
        fuzz.token_sort_ratio(aa, ba) if ba else -1.0,
        fuzz.partial_ratio(aa, ba) if ba else -1.0,
        _jacc(aat, bat) if ba else -1.0, len(aat & bat), len(bat),
        float(not braddr.strip() or ba == ""),
        # numbers
        _jacc(anu, bnu), len(anu & bnu), len(anu), len(bnu),
        float(a_first == b_first) if a_first and b_first else -1.0,
        float(bnu <= anu) if bnu else -1.0,
        float(bool(pins_a & pins_b)) if pins_a and pins_b else -1.0,
        (Levenshtein.normalized_similarity(a_first, b_first) if a_first and b_first else -1.0),
        # state
        float(ast == bst) if ast and bst else -1.0,
    ) + _extra_feats(asq, bsq, ag, bg, aa, ba, a_x, b_x)


FEAT_NAMES = [
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw_glued", "n_ratio_glued", "n_partial_glued",
    "n_jacc", "n_common", "n_lendiff", "n_len_a", "n_len_b", "n_exact", "n_exact_glued",
    "legal_jacc", "legal_a", "legal_b", "b_nonlatin", "b_domain",
    "a_ratio", "a_tset", "a_tsort", "a_partial", "a_jacc", "a_common", "a_len_b", "b_addr_empty",
    "num_jacc", "num_common", "num_a", "num_b", "num_first_eq", "num_b_subset", "pin_eq", "num_first_lev",
    "state_eq",
] + EXTRA_NAMES


def _chunk(rows):
    return np.asarray([_pair_feats(r) for r in rows], dtype=np.float32)


def load_strings(split):
    """Normalised + raw string columns of every source, as python lists."""
    out = {}
    for k in (1, 2, 3):
        df = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s{k}.parquet"), columns=STR_COLS)
        out[k] = df
    return out


def string_features(pairs: pl.DataFrame, strings: dict, pool: Pool, chunk=20000) -> np.ndarray:
    """Compute FEAT_NAMES for every row of pairs (columns s1, src, row)."""
    s1 = strings[1][pairs["s1"].to_numpy()]  # aligned with pairs
    feats = np.empty((len(pairs), len(FEAT_NAMES)), np.float32)
    src = pairs["src"].to_numpy()
    rows = pairs["row"].to_numpy()
    for k in (2, 3):
        m = np.nonzero(src == k)[0]
        if len(m) == 0:
            continue
        for j in range(0, len(m), 400000):  # bounded RAM: build python tuples per block
            mm = m[j:j + 400000]
            a = s1[mm]
            b = strings[k][rows[mm]]
            tuples = list(zip(*[a[c].to_list() for c in STR_COLS], *[b[c].to_list() for c in STR_COLS]))
            jobs = [tuples[i:i + chunk] for i in range(0, len(tuples), chunk)]
            feats[mm] = np.vstack(pool.map(_chunk, jobs))
    return feats


def _dense_sim(split, pairs: pl.DataFrame) -> np.ndarray:
    """Cosine of blocking vectors for arbitrary pairs (GPU dot products)."""
    import torch
    out = np.zeros(len(pairs), np.float32)
    v1 = np.load(os.path.join(WORK_DIR, f"{split}_s1_vec.npy"), mmap_mode="r")
    for k in (2, 3):
        m = np.nonzero(pairs["src"].to_numpy() == k)[0]
        if len(m) == 0:
            continue
        vk = np.load(os.path.join(WORK_DIR, f"{split}_s{k}_vec.npy"), mmap_mode="r")
        s1 = pairs["s1"].to_numpy()[m]
        rw = pairs["row"].to_numpy()[m]
        for i in range(0, len(m), 1_000_000):
            a = torch.from_numpy(v1[np.sort(s1[i:i + 1_000_000])]).cuda()  # sorted = faster disk reads
            order = np.argsort(np.argsort(s1[i:i + 1_000_000]))
            a = a[torch.from_numpy(order).cuda()]
            b = torch.from_numpy(vk[rw[i:i + 1_000_000]]).cuda()
            out[m[i:i + 1_000_000]] = (a.float() * b.float()).sum(1).cpu().numpy()
    return out


def union_candidates(split, kd, ks, s1_keep=None) -> pl.DataFrame:
    """Merge dense top-kd and sparse top-ks candidates (optionally for a subset of S1)."""
    dense = pl.scan_parquet(os.path.join(WORK_DIR, f"{split}_cand.parquet")).filter(pl.col("rank") < kd)
    sparse = pl.scan_parquet(os.path.join(WORK_DIR, f"{split}_cand_sparse.parquet")).filter(pl.col("sp_rank") < ks)
    if s1_keep is not None:
        dense, sparse = dense.filter(s1_keep), sparse.filter(s1_keep)
    cand = dense.collect().join(sparse.collect(), on=["s1", "src", "row"], how="full", coalesce=True)
    miss = cand["sim"].is_null().to_numpy()
    if miss.any():
        sims = cand["sim"].to_numpy().copy()
        sims[miss] = _dense_sim(split, cand.filter(pl.Series(miss)))
        cand = cand.with_columns(pl.Series("sim", sims, dtype=pl.Float32))
    return cand.with_columns(pl.col("rank").fill_null(99), pl.col("sp_rank").fill_null(99),
                             pl.col("sp_score").fill_null(0.0))


_Q_CACHE = {}


def competition_features(cand: pl.DataFrame, split: str) -> pl.DataFrame:
    """Blocking-rank context: margins vs. other candidates of the same record.

    Per-record (S2/S3) best / second-best values come from the full blocking
    output, so they are correct even when cand is restricted to some S1 rows.
    """
    if split not in _Q_CACHE:  # per-record tables are the same for every chunk: load once
        full = pl.scan_parquet(os.path.join(WORK_DIR, f"{split}_cand.parquet"))
        sfull = pl.scan_parquet(os.path.join(WORK_DIR, f"{split}_cand_sparse.parquet"))
        q = (full.filter(pl.col("rank") <= 1)
                 .select("src", "row", pl.col("sim").max().over("src", "row").alias("q_best"),
                         pl.col("sim").min().over("src", "row").alias("q_second"))
                 .unique(["src", "row"]))
        p = (sfull.filter(pl.col("sp_rank") <= 1)
                  .select("src", "row", pl.col("sp_score").max().over("src", "row").alias("q_sp_best"),
                          pl.col("sp_score").min().over("src", "row").alias("q_sp_second"),
                          pl.len().over("src", "row").alias("_n"))
                  .unique(["src", "row"])
                  .with_columns(pl.when(pl.col("_n") > 1).then(pl.col("q_sp_second")).otherwise(None)
                                  .alias("q_sp_second")).drop("_n"))
        _Q_CACHE[split] = q.collect().join(p.collect(), on=["src", "row"], how="full", coalesce=True)
    cand = cand.join(_Q_CACHE[split], on=["src", "row"], how="left")
    cand = cand.with_columns(
        pl.col("sim").max().over("s1").alias("s1_best"),
        pl.len().over("s1").alias("s1_ncand"),
        pl.col("sim").rank("ordinal", descending=True).over("s1").alias("s1_rank"),
        pl.col("sp_score").max().over("s1").alias("s1_sp_best"),
    )
    return cand.with_columns(
        (pl.col("sim") - pl.col("q_best")).alias("gap_q_best"),
        (pl.col("q_best") - pl.col("q_second").fill_null(0)).alias("q_margin"),
        (pl.col("sim") - pl.col("s1_best")).alias("gap_s1_best"),
        (pl.col("sp_score") - pl.col("q_sp_best").fill_null(0)).alias("sp_gap_q"),
        (pl.col("q_sp_best").fill_null(0) - pl.col("q_sp_second").fill_null(0)).alias("sp_margin"),
        (pl.col("sp_score") - pl.col("s1_sp_best")).alias("sp_gap_s1"),
    )


COMP_NAMES = ["sim", "rank", "gap_q_best", "q_margin", "gap_s1_best", "s1_ncand", "s1_rank", "src",
              "sp_score", "sp_rank", "sp_gap_q", "sp_margin", "sp_gap_s1"]
