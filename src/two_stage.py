"""v4: two-stage matcher on the cached features from build_features.py.

stage 1  Two XGBoost models on disjoint S1 folds (A = tr_0..3, B = tr_4..7).
         Out-of-fold scores for training rows; the average of both for val / test.
stage 2  Adds S1-level "group" features computed from the stage-1 scores of all
         candidates of the same S1 (how many confident matches it has, the rank of
         this candidate, and how similar this record is to the S1's best other
         candidate, the "anchor"). All candidates of an S1 live in one chunk, so
         these features are exact for train, val and test alike.
decision One-parent assignment, then either a global threshold or, per S1, the
         candidate subset with the highest expected F0.5.

Usage:  python two_stage.py fit          (train both stages, tune decision on val)
        python two_stage.py predict      (score cached test chunks, write output/)
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from build_features import FEAT_DIR, TEST_PARTS
from config import OUT_DIR, WORK_DIR
from features import load_strings
from labels import macro_f05, positive_pairs
from normalize import squash

# Experiment settings (env vars so several experiments can live side by side):
#   ER_MODEL_TAG  sub-folder of <work> for models / results (default "v4")
#   ER_FOLDS      train chunk ids for fold A and fold B, e.g. "0,1|2,3" (default "0,1,2,3|4,5,6,7")
MODEL_TAG = os.environ.get("ER_MODEL_TAG", "v4")
_folds = os.environ.get("ER_FOLDS", "0,1,2,3|4,5,6,7").split("|")
FOLD_A = [f"tr_{i}" for i in _folds[0].split(",")]
FOLD_B = [f"tr_{i}" for i in _folds[1].split(",")]
NEG_KEEP = float(os.environ.get("ER_NEG_KEEP", 0.30))    # stage 1
NEG_KEEP2 = float(os.environ.get("ER_NEG_KEEP2", 0.15))  # stage 2: more columns -> fewer negatives (8 GB VRAM)
MODEL_DIR = os.path.join(WORK_DIR, MODEL_TAG)
G_SUFFIX = "" if MODEL_TAG == "v4" else f"_{MODEL_TAG}"  # group features depend on the stage-1 models
XGB_PARAMS = dict(n_estimators=1500, max_depth=9, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, tree_method="hist", device="cuda", eval_metric="logloss",
                  early_stopping_rounds=60, max_bin=256)
GROUP_NAMES = ["p1", "s1_max_p", "p_gap_s1max", "p_rank_s1", "s1_cnt90_other", "s1_cnt50_other",
               "s1_sum_other", "s1_ncand_assigned", "anc_p", "anc_same_src", "anc_n_ratio", "anc_n_tset",
               "anc_a_tset", "anc_num_jacc", "anc_num_first_eq", "has_anchor"]


USE_R = os.environ.get("ER_USE_R", "0") == "1"  # add record-level features from rec_feats.py
FAMILY = os.environ.get("ER_FAMILY", "0") == "1"  # add second-sibling "family" group features (v6)


USE_Q = os.environ.get("ER_USE_Q", "0") == "1"  # add noise-vs-decoy features from gen_feats.py


class Cat:
    """Row-indexable side-by-side view of on-disk arrays (pair features | extra feature blocks)."""

    def __init__(self, a, *more):
        self.a, self.more = a, more
        self.shape = (a.shape[0], a.shape[1] + sum(m.shape[1] for m in more))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return np.hstack([np.asarray(self.a[idx])] + [np.asarray(m[idx]) for m in self.more])

    def __array__(self, dtype=None, copy=None):
        return self[:]


def base(X):
    return X.a if isinstance(X, Cat) else X


CTRY_NORM = os.environ.get("ER_CTRY_NORM", "0") == "1"  # rarity features as within-country percentiles
_NORMALIZERS = {}


class NormX:
    """Pair-feature block whose rarity columns are mapped to within-country percentiles on read."""

    def __init__(self, X, s1, norm):
        self.X, self.s1, self.norm = X, s1, norm
        self.shape = X.shape
        self.cols = set(norm.idx)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, tuple):  # (rows, column): only map when the column is a normalised one
            rows, col = idx
            raw = np.asarray(self.X[idx], dtype=np.float32)
            if isinstance(col, (int, np.integer)) and col in self.cols:
                j = self.norm.idx.index(col)
                s1 = self.s1[rows]
                out = raw.copy()
                for c in np.unique(self.norm.ctry[s1]):
                    m = self.norm.ctry[s1] == c
                    q = self.norm.q.get(str(c))
                    if q is not None:
                        out[m] = np.interp(raw[m], q[:, j], self.norm.grid)
                return out
            return raw
        return self.norm(np.asarray(self.X[idx]), self.s1[idx])

    def __array__(self, dtype=None, copy=None):
        return self[:]


def load(tag):
    X = np.load(os.path.join(FEAT_DIR, f"{tag}_X.npy"), mmap_mode="r")
    if CTRY_NORM:
        split = "test" if tag.startswith("te_") else "train"
        if split not in _NORMALIZERS:
            from ctry_norm import Normalizer
            _NORMALIZERS[split] = Normalizer(split)
        s1 = pl.read_parquet(os.path.join(FEAT_DIR, f"{tag}_pairs.parquet"), columns=["s1"])["s1"].to_numpy()
        X = NormX(X, s1, _NORMALIZERS[split])
    more = [np.load(os.path.join(FEAT_DIR, f"{tag}_{b}.npy"), mmap_mode="r")
            for b, use in (("R", USE_R), ("Q", USE_Q)) if use]
    if more:
        X = Cat(X, *more)
    return pl.read_parquet(os.path.join(FEAT_DIR, f"{tag}_pairs.parquet")), X


NEG_EASY = float(os.environ.get("ER_NEG_EASY", 0))  # >0: keep-rate for easy negatives (hard ones use NEG_KEEP)


def hard_mask(X):
    """Negatives worth keeping more often: some name / sound / address similarity."""
    from features import COMP_NAMES, FEAT_NAMES
    cols = COMP_NAMES + FEAT_NAMES
    X = base(X)
    h = np.asarray(X[:, cols.index("n_tset")]) >= 60
    h |= np.asarray(X[:, cols.index("a_tset")]) >= 60
    if "ph_tset" in cols:
        h |= np.asarray(X[:, cols.index("ph_tset")]) >= 70
    return h


def subsample(y, keep, seed, hard=None):
    """Indices of all positives + a random share of negatives, and their weights (1 / keep-rate).
    With `hard` and NEG_EASY > 0, hard negatives are kept at rate `keep` and easy ones at NEG_EASY."""
    r = np.random.default_rng(seed).random(len(y))
    rate = np.full(len(y), keep, np.float32)
    if hard is not None and NEG_EASY > 0:
        rate = np.where(hard, keep, NEG_EASY).astype(np.float32)
    idx = np.nonzero((y == 1) | (r < rate))[0]
    w = np.where(y[idx] == 1, 1.0, 1.0 / rate[idx]).astype(np.float32)
    return idx, w


def fit_xgb(X, y, w, name):
    import xgboost as xgb
    n = len(y)
    # 5% early-stopping rows: copied out (small) and given weight 0 in training, so the big
    # training matrix is never copied (a 95% fancy-index copy would double RAM on the full run)
    es = np.sort(np.random.default_rng(1).permutation(n)[int(n * 0.95):])
    w_tr = w.copy()
    w_tr[es] = 0.0
    m = xgb.XGBClassifier(**XGB_PARAMS)
    t = time.time()
    m.fit(X, y, sample_weight=w_tr, eval_set=[(X[es], y[es])], sample_weight_eval_set=[w[es]], verbose=False)
    print(f"  {name}: {n - len(es):,} rows, {m.best_iteration} trees, {time.time() - t:.0f}s", flush=True)
    m.save_model(os.path.join(MODEL_DIR, f"{name}.json"))
    return m


def booster(name):
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(os.path.join(MODEL_DIR, f"{name}.json"))
    b.set_param({"device": "cuda"})
    return b


def predict(b, X, batch=2_000_000):
    return np.concatenate([np.asarray(b.inplace_predict(np.ascontiguousarray(X[i:i + batch])), np.float32)
                           for i in range(0, len(X), batch)])


# ---------------------------------------------------------------- self-training
SELF_TRAIN = os.environ.get("ER_SELF_TRAIN", "0") == "1"
ST_POS, ST_NEG, ST_NEG_KEEP = 0.97, 0.02, 0.05


def self_train_sets(bA, bB):
    """Pseudo-labelled test pairs for countries that never occur in training (open set: whatever
    labels are new, e.g. France). One-parent assignment first; the kept parent with p >= ST_POS
    becomes a positive, pairs with p <= ST_NEG a (sampled) negative. No true labels are used."""
    tr_c = set(pl.read_parquet(os.path.join(WORK_DIR, "train_s1.parquet"), columns=["country"])["country"].unique())
    te_c = pl.read_parquet(os.path.join(WORK_DIR, "test_s1.parquet"), columns=["country"])["country"].to_numpy()
    unseen = sorted(set(te_c) - tr_c)
    print(f"self-training on unseen countries: {unseen}", flush=True)
    if not unseen:
        return {}
    frames = []
    for i in range(TEST_PARTS):
        t = f"te_{i}"
        pr, X = load(t)
        idx = np.nonzero(np.isin(te_c[pr["s1"].to_numpy()], unseen))[0]
        p = (predict(bA, X[idx]) + predict(bB, X[idx])) / 2
        frames.append(pr[idx].select("s1", "src", "row").with_columns(
            pl.Series("p", p), pl.Series("i", idx), pl.lit(t).alias("t")))
    d = pl.concat(frames)
    best = d.sort("p", descending=True).unique(["src", "row"], keep="first")
    pos = best.filter(pl.col("p") >= ST_POS).select("t", "i")
    rng = np.random.default_rng(11)
    neg = d.filter((pl.col("p") <= ST_NEG) & pl.Series(rng.random(len(d)) < ST_NEG_KEEP)).select("t", "i")
    out = {}
    for t in d["t"].unique().to_list():
        pi = pos.filter(pl.col("t") == t)["i"].to_numpy()
        ni = neg.filter(pl.col("t") == t)["i"].to_numpy()
        idx = np.concatenate([pi, ni])
        order = np.argsort(idx)
        y = np.concatenate([np.ones(len(pi), np.int8), np.zeros(len(ni), np.int8)])[order]
        w = np.where(y == 1, 1.0, 1.0 / ST_NEG_KEEP).astype(np.float32)
        out[t] = (idx[order], y, w)
    print(f"  pseudo labels: {len(pos):,} positives, {len(neg):,} sampled negatives "
          f"(of {len(d):,} unseen-country pairs)", flush=True)
    return out


# ---------------------------------------------------------------- group features
def _anchor_chunk(rows):
    out = []
    for bn, ba, bnum, cn, ca, cnum in rows:
        if cn is None:
            out.append((-1.0,) * 5)
            continue
        bs, cs = squash(bn), squash(cn)
        nb, nc = set(bnum.split()), set(cnum.split())
        fb, fc = (bnum.split() or [""])[0], (cnum.split() or [""])[0]
        out.append((fuzz.ratio(bs, cs), fuzz.token_set_ratio(bs, cs),
                    fuzz.token_set_ratio(ba, ca) if ba and ca else -1.0,
                    len(nb & nc) / max(1, len(nb | nc)) if nb or nc else -1.0,
                    float(fb == fc) if fb and fc else -1.0))
    return out


def _gather(strings, srcs, rows, cols=("name_core", "addr_norm", "nums")):
    """name/address/number strings for records given as (src, row) arrays, in order."""
    out = [np.empty(len(rows), object) for _ in cols]
    for k in (2, 3):
        m = np.nonzero(srcs == k)[0]
        if len(m):
            df = strings[k].select(cols)[rows[m]]
            for j, c in enumerate(cols):
                out[j][m] = df[c].to_list()
    return out


def group_features(pairs: pl.DataFrame, p1: np.ndarray, strings, pool) -> np.ndarray:
    """S1-level context from stage-1 scores (+ similarity of each record to its S1 anchor)."""
    d = pairs.select("s1", "src", "row").with_columns(pl.Series("p1", p1), pl.int_range(pl.len()).alias("i"))
    # only the S1's own assignment matters for the anchor: a record whose best S1 is elsewhere
    # cannot be an anchor here
    d = d.with_columns(
        pl.col("p1").max().over("s1").alias("s1_max_p"),
        pl.col("p1").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("p_rank_s1"),
        ((pl.col("p1") > 0.9).sum().over("s1") - (pl.col("p1") > 0.9)).cast(pl.Float32).alias("s1_cnt90_other"),
        ((pl.col("p1") > 0.5).sum().over("s1") - (pl.col("p1") > 0.5)).cast(pl.Float32).alias("s1_cnt50_other"),
        (pl.col("p1").sum().over("s1") - pl.col("p1")).alias("s1_sum_other"),
        pl.len().over("s1").cast(pl.Float32).alias("s1_ncand_assigned"),
    ).with_columns((pl.col("p1") - pl.col("s1_max_p")).alias("p_gap_s1max"))
    # anchor = best other candidate of the same S1 with p1 > 0.5
    top = (d.sort(["s1", "p1"], descending=[False, True])
             .group_by("s1", maintain_order=True)
             .agg(pl.col("i").head(2).alias("ti"), pl.col("p1").head(2).alias("tp")))
    top = top.with_columns(pl.col("ti").list.get(0, null_on_oob=True).alias("i0"),
                           pl.col("ti").list.get(1, null_on_oob=True).alias("i1"),
                           pl.col("tp").list.get(0, null_on_oob=True).alias("p0"),
                           pl.col("tp").list.get(1, null_on_oob=True).alias("pp1")).drop("ti", "tp")
    d = d.join(top, on="s1", how="left", maintain_order="left").with_columns(
        pl.when(pl.col("i") == pl.col("i0")).then(pl.col("i1")).otherwise(pl.col("i0")).alias("ai"),
        pl.when(pl.col("i") == pl.col("i0")).then(pl.col("pp1")).otherwise(pl.col("p0")).alias("anc_p"))
    d = d.with_columns(pl.when(pl.col("anc_p") > 0.5).then(pl.col("ai")).otherwise(None).alias("ai"))
    src = d["src"].to_numpy(); row = d["row"].to_numpy(); ai = d["ai"].to_numpy()
    has = ~np.isnan(ai.astype(np.float64)) if ai.dtype.kind == "f" else ~d["ai"].is_null().to_numpy()
    ai_int = np.where(has, np.nan_to_num(ai.astype(np.float64), nan=0), 0).astype(np.int64)
    a_src = np.where(has, src[ai_int], 0); a_row = np.where(has, row[ai_int], 0)

    feats = np.full((len(d), 5), -1.0, np.float32)
    idx = np.nonzero(has)[0]
    for j in range(0, len(idx), 400000):  # gather strings per batch to bound RAM
        ii = idx[j:j + 400000]
        b = _gather(strings, src[ii], row[ii])
        c = _gather(strings, a_src[ii], a_row[ii])
        rows = list(zip(b[0], b[1], b[2], c[0], c[1], c[2]))
        res = pool.map(_anchor_chunk, [rows[x:x + 20000] for x in range(0, len(rows), 20000)])
        feats[ii] = np.asarray([r for part in res for r in part], np.float32)
    anc_same = np.where(has, (a_src == src).astype(np.float32), -1.0)
    G = np.column_stack([d.select(GROUP_NAMES[:9]).fill_null(-1).to_numpy().astype(np.float32),
                         anc_same, feats, has.astype(np.float32)])
    if not FAMILY:
        return G
    # "family" features: a second confident sibling of the same S1, sound-key similarity to both
    # siblings, and how many confident siblings (p1 > 0.8) the S1 has besides this record
    top3 = (d.sort(["s1", "p1"], descending=[False, True]).group_by("s1", maintain_order=True)
              .agg(pl.col("i").head(3).alias("ti"), pl.col("p1").head(3).alias("tp")))
    d3 = d.select("s1", "i", "ai").join(top3, on="s1", how="left", maintain_order="left")
    ti, tp, own, first = d3["ti"].to_list(), d3["tp"].to_list(), d3["i"].to_list(), d3["ai"].to_list()
    a2 = np.full(len(d3), -1, np.int64)
    for k in range(len(d3)):  # second anchor: best confident candidate that is neither self nor anchor 1
        for cand, pc in zip(ti[k] or [], tp[k] or []):
            if cand != own[k] and cand != first[k] and pc > 0.5:
                a2[k] = cand
                break
    has2 = a2 >= 0
    a2i = np.where(has2, a2, 0)
    feats2 = np.full((len(d), 5), -1.0, np.float32)
    ph = np.full((len(d), 2), -1.0, np.float32)
    for anchor_has, a_idx, target in ((has, ai_int, 0), (has2, a2i, 1)):
        idx = np.nonzero(anchor_has)[0]
        for j in range(0, len(idx), 400000):
            ii = idx[j:j + 400000]
            b = _gather(strings, src[ii], row[ii], cols=("name_core", "addr_norm", "nums", "ph"))
            c = _gather(strings, src[a_idx[ii]], row[a_idx[ii]], cols=("name_core", "addr_norm", "nums", "ph"))
            ph[ii, target] = np.asarray([fuzz.ratio(x.replace(" ", ""), y.replace(" ", "")) if x and y else -1.0
                                         for x, y in zip(b[3], c[3])], np.float32)
            if target == 1:
                rows = list(zip(b[0], b[1], b[2], c[0], c[1], c[2]))
                res = pool.map(_anchor_chunk, [rows[x:x + 20000] for x in range(0, len(rows), 20000)])
                feats2[ii] = np.asarray([r for part in res for r in part], np.float32)
    cnt80 = d.select(((pl.col("p1") > 0.8).sum().over("s1") - (pl.col("p1") > 0.8)).cast(pl.Float32))
    return np.column_stack([G, feats2, has2.astype(np.float32), ph, cnt80.to_numpy()])


# ---------------------------------------------------------------- decision rules
def assign(pairs: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """Keep each (src,row) only for its best S1."""
    d = pairs.select("s1", "src", "row").with_columns(pl.Series("p", p))
    return d.sort("p", descending=True).unique(["src", "row"], keep="first")


def rule_threshold(a: pl.DataFrame, thr: float) -> pl.DataFrame:
    return a.filter(pl.col("p") >= thr)


def rule_group_threshold(a: pl.DataFrame, thr: dict) -> pl.DataFrame:
    """One threshold per record group (source x empty-address); `a` needs a column 'grp'."""
    t = pl.col("grp").replace_strict({int(k): v for k, v in thr.items()}, default=0.99)
    return a.filter(pl.col("p") >= t)


def record_groups(X) -> np.ndarray:
    """Group id per pair: 0/1 = S2 with/without address, 2/3 = S3 with/without address."""
    from features import COMP_NAMES, FEAT_NAMES
    cols = COMP_NAMES + FEAT_NAMES
    X = base(X)
    src = np.asarray(X[:, cols.index("src")])
    empty = np.asarray(X[:, cols.index("b_addr_empty")]) > 0
    return ((src == 3) * 2 + empty).astype(np.int8)


def rule_expected_f(a: pl.DataFrame, floor: float, t_extra: float) -> pl.DataFrame:
    """Per S1 choose the top-m candidates maximising expected F0.5 (vs. predicting empty)."""
    a = a.filter(pl.col("p") >= floor * 0.5).sort(["s1", "p"], descending=[False, True])
    a = a.with_columns(
        pl.col("p").cum_sum().over("s1").alias("ctp"),
        pl.int_range(1, pl.len() + 1).over("s1").alias("m"),
        (pl.col("p").sum().over("s1") + t_extra).alias("T"),
        (1 - pl.col("p")).clip(1e-6, 1).log().sum().over("s1").exp().alias("p_empty"),
    ).with_columns(
        (1.25 * pl.col("ctp") / (1.25 * pl.col("ctp") + 0.25 * (pl.col("T") - pl.col("ctp"))
                                 + (pl.col("m") - pl.col("ctp")))).alias("ef"))
    a = a.with_columns(pl.col("ef").max().over("s1").alias("ef_best"))
    m_best = a.filter(pl.col("ef") == pl.col("ef_best")).group_by("s1").agg(pl.col("m").min().alias("m_best"))
    a = a.join(m_best, on="s1").filter((pl.col("m") <= pl.col("m_best")) & (pl.col("ef_best") > pl.col("p_empty"))
                                       & (pl.col("p") >= floor))
    return a.select("s1", "src", "row", "p")


def to_dict(k: pl.DataFrame):
    out = {}
    for s, sr, r in zip(k["s1"].to_list(), k["src"].to_list(), k["row"].to_list()):
        out.setdefault(s, set()).add((sr, r))
    return out


def tune(pairs, p, truth, val_rows, grp=None):
    a = assign(pairs, p)
    res = {}
    best_t = max(((macro_f05(to_dict(rule_threshold(a, t)), truth, val_rows), t) for t in np.arange(0.5, 0.96, 0.025)))
    res["threshold"] = {"f05": best_t[0], "thr": float(best_t[1])}
    if grp is not None:
        # coordinate ascent: start from the global threshold, tune each group in turn (2 passes)
        ag = a.join(pairs.select("s1", "src", "row").with_columns(pl.Series("grp", grp)), on=["s1", "src", "row"])
        thr = {g: float(best_t[1]) for g in range(4)}
        best = best_t[0]
        for _ in range(2):
            for g in range(4):
                for t in np.arange(0.5, 0.96, 0.025):
                    cand = {**thr, g: float(t)}
                    f = macro_f05(to_dict(rule_group_threshold(ag, cand)), truth, val_rows)
                    if f > best + 1e-6:
                        best, thr = f, cand
        res["group_threshold"] = {"f05": best, "thr": {str(k): v for k, v in thr.items()}}
    best_e = max(((macro_f05(to_dict(rule_expected_f(a, fl, te)), truth, val_rows), fl, te)
                  for fl in (0.2, 0.3, 0.4, 0.5) for te in (0.0, 0.1, 0.25)))
    res["expected_f"] = {"f05": best_e[0], "floor": best_e[1], "t_extra": best_e[2]}
    return res


# ---------------------------------------------------------------- main
def fit():
    os.makedirs(MODEL_DIR, exist_ok=True)
    parts = {t: load(t) for t in FOLD_A + FOLD_B + ["val_0"]}

    def stack(tags, seed, keep=NEG_KEEP, extra=None, pseudo=None):
        """Sampled rows of several chunks in ONE pre-allocated array (no vstack copy: RAM).
        pseudo: {test tag: (row idx, pseudo label, weight)} rows added from self-training."""
        sel = []
        for t in tags:
            pr, X = parts[t]
            y = pr["y"].to_numpy()
            idx, w = subsample(y, keep, seed, hard_mask(X) if NEG_EASY > 0 else None)
            sel.append((t, idx, y[idx], w))
        for t, (idx, y, w) in (pseudo or {}).items():
            if t not in parts:
                parts[t] = load(t)
            sel.append((t, idx, y, w))
        n = sum(len(s[1]) for s in sel)
        width = parts[tags[0]][1].shape[1] + (extra[tags[0]].shape[1] if extra else 0)
        out = np.empty((n, width), np.float32)
        pos = 0
        for t, idx, _, _ in sel:
            block = np.asarray(parts[t][1][idx])
            if extra:
                block = np.hstack([block, np.asarray(extra[t][idx])])
            out[pos:pos + len(idx)] = block
            pos += len(idx)
            del block
        print(f"  stacked {n:,} rows x {width} features ({out.nbytes / 1e9:.1f} GB)", flush=True)
        return out, np.concatenate([s[2] for s in sel]), np.concatenate([s[3] for s in sel])

    print("stage 1", flush=True)
    base = "_base" if SELF_TRAIN else ""
    for fold, tags in (("s1_A", FOLD_A), ("s1_B", FOLD_B)):
        if not os.path.exists(os.path.join(MODEL_DIR, f"{fold}{base}.json")):
            X, y, w = stack(tags, 0)
            fit_xgb(X, y, w, fold + base)
            del X
    if SELF_TRAIN:
        # round 2: add confident, model-labelled test pairs of countries unseen in training
        pseudo = self_train_sets(booster("s1_A_base"), booster("s1_B_base"))
        for fold, tags in (("s1_A", FOLD_A), ("s1_B", FOLD_B)):
            if not os.path.exists(os.path.join(MODEL_DIR, f"{fold}.json")):
                X, y, w = stack(tags, 0, pseudo=pseudo)
                fit_xgb(X, y, w, fold)
                del X
        for t in [t for t in parts if t.startswith("te_")]:
            del parts[t]
    bA, bB = booster("s1_A"), booster("s1_B")
    p1 = {}
    for t in FOLD_A:
        p1[t] = predict(bB, parts[t][1])
    for t in FOLD_B:
        p1[t] = predict(bA, parts[t][1])
    p1["val_0"] = (predict(bA, parts["val_0"][1]) + predict(bB, parts["val_0"][1])) / 2

    # validation truth
    n1 = pl.scan_parquet(os.path.join(WORK_DIR, "train_s1.parquet")).select(pl.len()).collect().item()
    from build_features import VAL_START, s1_groups
    val_rows = np.nonzero(s1_groups(n1) >= VAL_START)[0]
    vpos = positive_pairs().filter(pl.col("s1").is_in(pl.Series(val_rows.astype(np.int32))))
    truth = to_dict(vpos)
    val_rows = val_rows.tolist()
    pv = parts["val_0"][0]
    r1 = tune(pv, p1["val_0"], truth, val_rows)
    print(f"stage 1 on val: {json.dumps(r1)}", flush=True)

    print("stage 2 features", flush=True)
    strings = load_strings("train")
    G = {}
    with Pool(16) as pool:
        for t in FOLD_A + FOLD_B + ["val_0"]:
            gp = os.path.join(FEAT_DIR, f"{t}_G{G_SUFFIX}.npy")
            if not os.path.exists(gp):
                np.save(gp, group_features(parts[t][0], p1[t], strings, pool))
            G[t] = np.load(gp, mmap_mode="r")  # on disk: keeps RAM free
            print(f"  group features {t}", flush=True)
    del strings
    s2_tags = FOLD_A + FOLD_B
    s2_max = int(os.environ.get("ER_S2_CHUNKS", 0))  # >0: train stage 2 on fewer chunks (RAM / VRAM)
    if s2_max:
        s2_tags = [t for pair in zip(FOLD_A, FOLD_B) for t in pair][:s2_max]
    X2, y2, w2 = stack(s2_tags, 7, NEG_KEEP2, G)
    m2 = fit_xgb(X2, y2, w2, "s2")
    del X2
    p2 = m2.predict_proba(np.hstack([np.asarray(parts["val_0"][1]), G["val_0"]]))[:, 1]
    np.save(os.path.join(MODEL_DIR, "val_p2.npy"), p2.astype(np.float32))
    r2 = tune(pv, p2, truth, val_rows, record_groups(parts["val_0"][1]))
    print(f"stage 2 on val: {json.dumps(r2)}", flush=True)
    json.dump({"stage1": r1, "stage2": r2}, open(os.path.join(MODEL_DIR, "val_results.json"), "w"), indent=2)


def predict_test():
    from predict import write_lists
    res = json.load(open(os.path.join(MODEL_DIR, "val_results.json")))
    use2 = max(res["stage2"].values(), key=lambda r: r["f05"])["f05"] >= max(
        res["stage1"].values(), key=lambda r: r["f05"])["f05"]
    stage = "stage2" if use2 else "stage1"
    rule = max(res[stage], key=lambda k: res[stage][k]["f05"])
    cfg = res[stage][rule]
    print(f"using {stage} with rule {rule}: {cfg}", flush=True)
    bA, bB = booster("s1_A"), booster("s1_B")
    b2 = booster("s2") if use2 else None
    strings = load_strings("test") if use2 else None
    scored = []
    with Pool(16) as pool:
        for i in range(TEST_PARTS):
            pr, X = load(f"te_{i}")
            pp = os.path.join(FEAT_DIR, f"te_{i}_p_{stage}{G_SUFFIX}.npy")
            if os.path.exists(pp):
                p = np.load(pp)
            else:
                p = (predict(bA, X) + predict(bB, X)) / 2
                if use2:
                    G = group_features(pr, p, strings, pool)
                    p = predict(b2, np.hstack([np.asarray(X), G]))
                np.save(pp, p)
            scored.append(pr.select("s1", "src", "row").with_columns(pl.Series("p", p), pl.Series("grp", record_groups(X))))
            print(f"  test part {i + 1}/{TEST_PARTS}", flush=True)
    scored = pl.concat(scored)
    scored.write_parquet(os.path.join(MODEL_DIR, "test_scores.parquet"))
    delta = float(os.environ.get("ER_UNSEEN_DELTA", 0))
    if delta:
        # countries never seen in training get a threshold shifted by delta (estimated by loco.py);
        # shifting p of all pairs of such a country is the same and leaves one-parent choices intact
        tr_c = set(pl.read_parquet(os.path.join(WORK_DIR, "train_s1.parquet"), columns=["country"])["country"].unique())
        te_c = pl.read_parquet(os.path.join(WORK_DIR, "test_s1.parquet"), columns=["country"])["country"].to_numpy()
        unseen = np.isin(te_c[scored["s1"].to_numpy()], sorted(set(te_c) - tr_c))
        scored = scored.with_columns(pl.Series("p", np.where(unseen, scored["p"].to_numpy() - delta,
                                                               scored["p"].to_numpy()).astype(np.float32)))
        print(f"unseen-country threshold shift {delta:+.3f} on {unseen.sum():,} pairs", flush=True)
    a = assign(scored, scored["p"].to_numpy()).join(scored.select("s1", "src", "row", "grp"), on=["s1", "src", "row"])
    if rule == "threshold":
        keep = rule_threshold(a, cfg["thr"])
    elif rule == "group_threshold":
        keep = rule_group_threshold(a, cfg["thr"])
    else:
        keep = rule_expected_f(a, cfg["floor"], cfg["t_extra"])
    ids = {k: pl.read_parquet(os.path.join(WORK_DIR, f"test_s{k}.parquet"), columns=["entity_id"])
               .with_row_index("row").with_columns(pl.col("row").cast(pl.Int32)) for k in (1, 2, 3)}
    eid = pl.concat([ids[k].rename({"entity_id": "eid"}).with_columns(pl.lit(k, pl.Int8).alias("src")) for k in (2, 3)])
    write_lists(scored.join(eid, on=["src", "row"]), ids[1], "candidate_entity_ids",
                os.path.join(OUT_DIR, "candidate_pairs.tsv"))
    write_lists(keep.join(eid, on=["src", "row"]), ids[1], "matched_entity_ids",
                os.path.join(OUT_DIR, "matching_results.tsv"))


if __name__ == "__main__":
    {"fit": fit, "predict": predict_test}[sys.argv[1]]()
