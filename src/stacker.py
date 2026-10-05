"""Stage 8: stack our pipeline (v6 XGBoost + cross-encoder m2 blend) with the second pipeline's model (model2v1).

Candidate set = our candidates UNION the second pipeline's candidates. Per pair: our blended score, our XGBoost score,
the second pipeline's score, record-level competition features of both systems (rank of this S1 among the record's
candidates, best other S1 score) and, for pairs only second pipeline found (with his score >= CE_HIS_MIN), our
cross-encoder score. A small XGBoost stacker is fitted on the validation S1s second pipeline scored out-of-fold
(2-fold by S1 for an honest estimate), then the one-parent + group-threshold decision is re-tuned.

Both systems' tiny scores are floored (pairs below the floor on both sides are dropped: they can never
be matches) identically on validation and test.

  python stacker.py val     -> <MODEL_DIR>/ce_sv_val_results.json + stacker
  python stacker.py test    -> <MODEL_DIR>/ce_sv_test_scores_blend.parquet
  then  ER_CE_NAME=sv python ce.py write
"""
import json
import os
import pickle
import sys

import numpy as np
import polars as pl

from config import WORK_DIR
from two_stage import MODEL_DIR

VDIR = os.environ.get("ER_M2_DIR", r"model2_scores")
# ER_VCE_DIR: a France-trained cross-encoder scores (ce_fr_val/test_scores.parquet). When set, our
# blended score of the uncertain band is recomputed with that cross-encoder instead of m2 (prefix ce_sv2_).
VCE_DIR = os.environ.get("ER_VCE_DIR", "")
# ER_CE_EXT: several external cross-encoders, "name=val.parquet,test.parquet;name2=..." (columns source1_entity_id,
# candidate_entity_id, ce_score, on the uncertain band pairs). All of them + our m2 re-blend the band score, and each
# raw score is also a stacker feature (NaN outside the band).
CE_EXT = [(n, *p.split(",")) for n, p in (x.split("=", 1) for x in os.environ.get("ER_CE_EXT", "").split(";") if x)]
PFX = os.environ.get("ER_SV_PFX", "ce_sv3_" if CE_EXT else ("ce_sv2_" if VCE_DIR else "ce_sv_"))
EXT_NAMES = [n for n, _, _ in CE_EXT] + (["m2r"] if CE_EXT else [])
OURS_MIN = 1e-3        # our blended score floor
HIS_MIN = 1e-4         # second pipeline score floor
CE_HIS_MIN = float(os.environ.get("ER_CE_HIS_MIN", 0.3))  # his-only pairs scored by our cross-encoder
FEATS = ["z_ours", "z_xgb", "z_his", "z_ce", "has_ours", "has_his", "rk_ours", "rk_his",
         "oth_ours", "oth_his", "gap_ours", "gap_his", "n_ours", "n_his", "b_empty", "src3"]


def logit(x):
    x = np.clip(x, 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


def id_maps(split):
    s1 = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s1.parquet"), columns=["entity_id"]).with_row_index("s1") \
        .with_columns(pl.col("s1").cast(pl.Int32)).rename({"entity_id": "source1_entity_id"})
    oth = pl.concat([pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s{k}.parquet"), columns=["entity_id"]).with_row_index("row")
                     .with_columns(pl.col("row").cast(pl.Int32), pl.lit(k, pl.Int8).alias("src")) for k in (2, 3)]) \
        .rename({"entity_id": "candidate_entity_id"})
    return s1, oth


def his_keys(path, split, s1_filter=None):
    s1, oth = id_maps(split)
    q = pl.scan_parquet(path).filter(pl.col("score") >= HIS_MIN)
    d = q.collect().join(s1, on="source1_entity_id").join(oth, on="candidate_entity_id") \
        .select("s1", "src", "row", pl.col("score").alias("his"))
    if s1_filter is not None:
        d = d.filter(pl.col("s1").is_in(s1_filter))
    return d


VB_DIR = os.environ.get("ER_M2B_DIR", "")   # the second pipeline's second sub-world (model2B_val/test_scores.parquet)


def his_all(split):
    """second pipeline scores: sub-world A, plus sub-world B when available (val: the S1 sets are disjoint -> union;
    test: the same candidate pairs -> mean of the A and B models)."""
    a = his_keys(os.path.join(VDIR, "model2_val_scores.parquet" if split == "train" else "model2_test_scores.parquet"), split)
    if not VB_DIR:
        return a
    b = his_keys(os.path.join(VB_DIR, "model2B_val_scores.parquet" if split == "train" else "model2B_test_scores.parquet"), split)
    if split == "train":
        b = b.join(a.select("s1").unique(), on="s1", how="anti")
        print(f"  second pipeline B adds {b['s1'].n_unique():,} validation S1", flush=True)
        return pl.concat([a, b])
    m = a.join(b, on=["s1", "src", "row"], how="full", coalesce=True, suffix="_b")
    return m.with_columns(((pl.col("his").fill_null(0.0) + pl.col("his_b").fill_null(0.0)) / 2).alias("his")) \
        .filter(pl.col("his") >= HIS_MIN).select("s1", "src", "row", "his")


def empty_flags(split):
    out = {}
    for k in (2, 3):
        a = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s{k}.parquet"), columns=["addr_norm"])["addr_norm"]
        out[k] = (a.fill_null("") == "").to_numpy()
    return out


def build(ours, his, split):
    """ours: s1,src,row,ours,xgb (floored already); his: s1,src,row,his. Returns feature frame."""
    d = ours.join(his, on=["s1", "src", "row"], how="full", coalesce=True)
    d = d.with_columns(pl.col("ours").is_not_null().cast(pl.Int8).alias("has_ours"),
                       pl.col("his").is_not_null().cast(pl.Int8).alias("has_his"))
    d = d.with_columns(pl.col("ours").fill_null(OURS_MIN * 0.5), pl.col("xgb").fill_null(OURS_MIN * 0.5),
                       pl.col("his").fill_null(HIS_MIN * 0.5))
    for c in ("ours", "his"):
        d = d.with_columns(pl.col(c).rank("ordinal", descending=True).over("src", "row").cast(pl.Int16).alias(f"rk_{c}"),
                           pl.len().over("src", "row").cast(pl.Int16).alias("n_rec"))
        top2 = d.group_by("src", "row").agg(pl.col(c).top_k(2).alias("_t"))
        top2 = top2.with_columns(pl.col("_t").list.get(0).alias("_b1"), pl.col("_t").list.get(1, null_on_oob=True).fill_null(0.0).alias("_b2"))
        d = d.join(top2.select("src", "row", "_b1", "_b2"), on=["src", "row"])
        # best OTHER S1 score for this record, and the margin of this pair over it
        d = d.with_columns(pl.when(pl.col(f"rk_{c}") == 1).then(pl.col("_b2")).otherwise(pl.col("_b1")).alias(f"oth_{c}")) \
             .with_columns((pl.col(c) - pl.col(f"oth_{c}")).alias(f"gap_{c}")).drop("_b1", "_b2")
    d = d.with_columns(pl.col("has_ours").sum().over("src", "row").cast(pl.Int16).alias("n_ours"),
                       pl.col("has_his").sum().over("src", "row").cast(pl.Int16).alias("n_his")).drop("n_rec")
    emp = empty_flags(split)
    src, row = d["src"].to_numpy(), d["row"].to_numpy()
    e = np.zeros(len(d), bool)
    for k in (2, 3):
        e[src == k] = emp[k][row[src == k]]
    d = d.with_columns(pl.Series("b_empty", e.astype(np.int8)), (pl.col("src") == 3).cast(pl.Int8).alias("src3"))
    return d


def add_ce(d, split):
    """Our cross-encoder score for second pipeline-only pairs with a meaningful second pipeline score."""
    from ce import ce_scores
    m = (d["has_ours"] == 0) & (d["his"] >= CE_HIS_MIN)
    sub = d.filter(m).select("s1", "src", "row")
    print(f"  {split}: cross-encoder on {len(sub):,} second pipeline-only pairs", flush=True)
    pc = ce_scores(sub, split) if len(sub) else np.zeros(0, np.float32)
    ce = sub.with_columns(pl.Series("ce", pc))
    d = d.join(ce, on=["s1", "src", "row"], how="left")
    return d


def matrix(d):
    z = {"z_ours": logit(d["ours"].to_numpy()), "z_xgb": logit(d["xgb"].to_numpy()), "z_his": logit(d["his"].to_numpy()),
         "z_ce": np.where(d["ce"].is_null().to_numpy(), np.nan, logit(d["ce"].fill_null(0.5).to_numpy()))}
    cols = [z[c] if c in z else d[c].to_numpy().astype(np.float32) for c in FEATS]
    for n in EXT_NAMES:
        if n in d.columns:
            v = d[n].to_numpy()
            cols.append(np.where(np.isnan(v.astype(np.float64)), np.nan, logit(np.nan_to_num(v, nan=0.5))))
    return np.column_stack(cols).astype(np.float32)


def fit_xgb(X, y, seed=0):
    import xgboost as xgb
    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
                  max_depth=5, eta=0.05, subsample=0.9, colsample_bytree=0.9, min_child_weight=5, seed=seed)
    return xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=400)


def pred(b, X):
    import xgboost as xgb
    return b.predict(xgb.DMatrix(X)).astype(np.float32)


def ext_keys(split):
    """All external cross-encoders + our raw m2, keyed by (s1, src, row); one column per model."""
    s1, oth = id_maps(split)
    out = None
    for n, fv, ft in CE_EXT:
        e = pl.read_parquet(fv if split == "train" else ft).join(s1, on="source1_entity_id").join(oth, on="candidate_entity_id") \
            .select("s1", "src", "row", pl.col("ce_score").alias(n))
        out = e if out is None else out.join(e, on=["s1", "src", "row"], how="full", coalesce=True)
    m2 = pl.read_parquet(os.path.join(WORK_DIR, "ce", f"cache_{split}_m2.parquet")).rename({"pc": "m2r"})
    return out.join(m2, on=["s1", "src", "row"], how="left")


def with_ext(ours, split, y=None):
    from sklearn.linear_model import LogisticRegression
    ours = ours.join(ext_keys(split), on=["s1", "src", "row"], how="left")
    m = ours[EXT_NAMES[0]].is_not_null().to_numpy()
    cols = [logit(ours["xgb"].to_numpy()[m])] + [logit(ours[n].fill_null(0.5).to_numpy()[m]) for n in EXT_NAMES]
    Z = np.column_stack(cols)
    path = os.path.join(MODEL_DIR, PFX + "ext_blend.pkl")
    if y is not None:
        pickle.dump(LogisticRegression(C=1.0, max_iter=2000).fit(Z, y[m]), open(path, "wb"))
    lr = pickle.load(open(path, "rb"))
    print(f"  {split}: band re-blend weights {dict(zip(['xgb'] + EXT_NAMES, np.round(lr.coef_[0], 2).tolist()))}", flush=True)
    new = ours["ours"].to_numpy().copy()
    new[m] = lr.predict_proba(Z)[:, 1]
    return ours.with_columns(pl.Series("ours", new.astype(np.float32)))


def vce_keys(split):
    s1, oth = id_maps(split)
    f = os.path.join(VCE_DIR, "ce_fr_val_scores.parquet" if split == "train" else "ce_fr_test_scores.parquet")
    return pl.read_parquet(f).join(s1, on="source1_entity_id").join(oth, on="candidate_entity_id") \
        .select("s1", "src", "row", pl.col("ce_score").alias("vce"))


def _vz(xgb, vce):
    a, b = logit(xgb), logit(vce)
    return np.column_stack([a, b, a * b])


def with_vce(ours, split, y=None):
    """Replace the band pairs' blended score by a blend of XGB and the France-trained cross-encoder."""
    from sklearn.linear_model import LogisticRegression
    ours = ours.join(vce_keys(split), on=["s1", "src", "row"], how="left")
    m = ours["vce"].is_not_null().to_numpy()
    Z = _vz(ours["xgb"].to_numpy()[m], ours["vce"].to_numpy()[m])
    path = os.path.join(MODEL_DIR, PFX + "vce_blend.pkl")
    if y is not None:
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z, y[m])
        pickle.dump(lr, open(path, "wb"))
    lr = pickle.load(open(path, "rb"))
    new = ours["ours"].to_numpy().copy()
    new[m] = lr.predict_proba(Z)[:, 1]
    print(f"  {split}: {m.sum():,} band pairs re-blended with the France-trained cross-encoder", flush=True)
    return ours.with_columns(pl.Series("ours", new.astype(np.float32))).drop("vce")


def cmd_val():
    from build_features import VAL_START, s1_groups
    from labels import positive_pairs
    from two_stage import to_dict, tune
    his = his_all("train")
    cover = his["s1"].unique()
    print(f"val: second pipeline covers {cover.len():,} S1 ({len(his):,} pairs >= {HIS_MIN})", flush=True)
    pv = pl.read_parquet(os.path.join(WORK_DIR, "feat", "val_0_pairs.parquet"), columns=["s1", "src", "row"])
    pb = np.load(os.path.join(MODEL_DIR, "ce_m2_val_pb.npy"))
    p2 = np.load(os.path.join(MODEL_DIR, "val_p2.npy"))
    ours_all = pv.with_columns(pl.Series("ours", pb), pl.Series("xgb", p2))
    if CE_EXT or VCE_DIR:
        yall = pv.join(positive_pairs().with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "src", "row"], how="left")["y"].fill_null(0).to_numpy()
        ours_all = with_ext(ours_all, "train", yall) if CE_EXT else with_vce(ours_all, "train", yall)
    ours_all = ours_all.filter(pl.col("s1").is_in(cover))
    ours = ours_all.filter(pl.col("ours") >= OURS_MIN)
    d = add_ce(build(ours, his, "train"), "train")
    pos = positive_pairs().filter(pl.col("s1").is_in(cover)).with_columns(pl.lit(1, pl.Int8).alias("y"))
    d = d.join(pos.select("s1", "src", "row", "y"), on=["s1", "src", "row"], how="left").with_columns(pl.col("y").fill_null(0))
    X, y = matrix(d), d["y"].to_numpy()
    print(f"  stack rows {len(d):,}, positives {int(y.sum()):,} (true pairs of covered S1: {len(pos):,})", flush=True)
    s1 = d["s1"].to_numpy()
    fold = ((s1.astype(np.int64) * 2654435761) % 1000003) % 2
    oof = np.zeros(len(d), np.float32)
    for f in (0, 1):
        b = fit_xgb(X[fold != f], y[fold != f])
        oof[fold == f] = pred(b, X[fold == f])
    from sklearn.metrics import log_loss
    print(f"  logloss ours {log_loss(y, np.clip(d['ours'].to_numpy(), 1e-6, 1 - 1e-6)):.4f}  "
          f"his {log_loss(y, np.clip(d['his'].to_numpy(), 1e-6, 1 - 1e-6)):.4f}  stack(oof) {log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)):.4f}", flush=True)
    d.select("s1", "src", "row", "y", "ours", "his", "b_empty").with_columns(pl.Series("oof", oof)).write_parquet(os.path.join(MODEL_DIR, PFX + "val_oof.parquet"))
    final = fit_xgb(X, y, seed=1)
    pickle.dump(final, open(os.path.join(MODEL_DIR, PFX + "stacker.pkl"), "wb"))
    # decision on the covered S1 only: ours alone vs stacked (out-of-fold)
    val_rows = sorted(int(x) for x in cover.to_list())
    truth = to_dict(pos)
    grp = ((d["src"].to_numpy() == 3) * 2 + d["b_empty"].to_numpy()).astype(np.int8)
    keys = d.select("s1", "src", "row")
    r_st = tune(keys, oof, truth, val_rows, grp)
    ok = ours_all
    emp = empty_flags("train")
    src, row = ok["src"].to_numpy(), ok["row"].to_numpy()
    e = np.zeros(len(ok), bool)
    for k in (2, 3):
        e[src == k] = emp[k][row[src == k]]
    r_ours = tune(ok.select("s1", "src", "row"), ok["ours"].to_numpy(), truth, val_rows, ((src == 3) * 2 + e).astype(np.int8))
    r_his = tune(keys, d["his"].to_numpy(), truth, val_rows, grp)
    out = {"blended": r_st, "ours_only": r_ours, "ce_fr_only": r_his, "covered_s1": len(val_rows)}
    print(json.dumps(out), flush=True)
    json.dump(out, open(os.path.join(MODEL_DIR, PFX + "val_results.json"), "w"), indent=2)


def cmd_test():
    his = his_all("test")
    print(f"test: second pipeline pairs >= {HIS_MIN}: {len(his):,}", flush=True)
    src_f = os.path.join(MODEL_DIR, "ce_m2_test_scores_blend.parquet")
    ours = pl.scan_parquet(src_f).filter((pl.col("p") >= OURS_MIN) | ((pl.col("p_xgb") > 0.002) & (pl.col("p_xgb") < 0.998))) \
        .select("s1", "src", "row", pl.col("p").alias("ours"), pl.col("p_xgb").fill_nan(None).alias("xgb")).collect()
    ours = ours.with_columns(pl.col("xgb").fill_null(pl.col("ours")))
    if CE_EXT:
        ours = with_ext(ours, "test")
    elif VCE_DIR:
        ours = with_vce(ours, "test")
    ours = ours.filter(pl.col("ours") >= OURS_MIN)
    d = add_ce(build(ours, his, "test"), "test")
    b = pickle.load(open(os.path.join(MODEL_DIR, PFX + "stacker.pkl"), "rb"))
    p = pred(b, matrix(d))
    grp = ((d["src"].to_numpy() == 3) * 2 + d["b_empty"].to_numpy()).astype(np.int8)
    stacked = d.select("s1", "src", "row").with_columns(pl.Series("p", p), pl.Series("grp", grp), pl.lit(None, pl.Float32).alias("p_xgb"))
    del d
    # candidate set = all our scored pairs (tiny ones keep a tiny score) + second pipeline-only pairs above his floor
    rest = pl.scan_parquet(src_f).filter(pl.col("p") < OURS_MIN).select("s1", "src", "row", "p", "grp", "p_xgb").collect()
    rest = rest.join(stacked.select("s1", "src", "row"), on=["s1", "src", "row"], how="anti").with_columns(pl.lit(0.0, pl.Float32).alias("p"))
    out = pl.concat([stacked.with_columns(pl.col("grp").cast(rest["grp"].dtype), pl.col("p_xgb").cast(rest["p_xgb"].dtype)), rest])
    out.write_parquet(os.path.join(MODEL_DIR, PFX + "test_scores_blend.parquet"))
    print(f"wrote {len(out):,} test pairs ({len(stacked):,} stacked)", flush=True)


if __name__ == "__main__":
    {"val": cmd_val, "test": cmd_test}[sys.argv[1]]()
