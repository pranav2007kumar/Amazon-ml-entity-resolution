"""Stage 6: cross-encoder re-ranker (fine-tuned multilingual transformer) for uncertain pairs.

The XGBoost pipeline decides the easy pairs. Pairs it is unsure about are re-scored by
microsoft/mdeberta-v3-base (MIT licence, 280M parameters) fine-tuned on the provided training pairs
only: input = "[S1 name | S1 address] [SEP] [candidate name | candidate address]" in raw text, so the
model sees scripts, accents, casing and formatting directly. Both scores are then blended.

  python ce.py build   training pairs: XGB-uncertain pairs of some train chunks (+ some easy pairs,
                       + confident pseudo-labelled pairs of countries unseen in training)
  python ce.py train   fine-tune on the GPU (bf16), time-limited
  python ce.py score   re-score uncertain val/test pairs, fit the blend on val, write blended scores
Env: ER_WORK_DIR, ER_MODEL_TAG (the XGB run whose scores are used, e.g. v6), ER_CE_MODEL (base weights dir)
"""
import json
import math
import os
import sys
import time

import numpy as np
import polars as pl

from build_features import FEAT_DIR, TEST_PARTS
from config import WORK_DIR
from two_stage import MODEL_DIR, booster, load, predict

BASE = os.environ.get("ER_CE_MODEL", r"C:\Amazon_ml\models\mdeberta-v3-base")
CE_DIR = os.path.join(WORK_DIR, "ce")
LO = float(os.environ.get("ER_CE_LO", 0.02))   # "uncertain" band of the XGB probability
HI = float(os.environ.get("ER_CE_HI", 0.98))
MAX_LEN = 96
LR = float(os.environ.get("ER_CE_LR", 2e-5))
# ER_CE_NAME names a fine-tuned model (dir CE_DIR/<name>) with its own score cache and output files;
# ER_CE_INIT starts training from those weights instead of BASE; ER_CE_SEED reshuffles the pair order
CE_NAME = os.environ.get("ER_CE_NAME", "model")
CE_INIT = os.environ.get("ER_CE_INIT", "")
CE_SEED = os.environ.get("ER_CE_SEED", "")
SFX = "" if CE_NAME == "model" else f"_{CE_NAME}"
PFX = "ce" + SFX + "_"
BATCH = int(os.environ.get("ER_CE_BATCH", 32))   # 64 overflows the 8 GB GPU (DeBERTa attention buffers)
TRAIN_MINUTES = float(os.environ.get("ER_CE_MINUTES", 100))
CE_CHUNKS = os.environ.get("ER_CE_CHUNKS", "1,3,5,7,9,11").split(",")


def _rows_text(split, k, rows: np.ndarray) -> np.ndarray:
    """'name | address' for the given rows of one source only (low RAM: no full text table)."""
    uniq, inv = np.unique(rows, return_inverse=True)
    df = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s{k}.parquet"), columns=["name", "address"])
    sub = df[uniq]
    del df
    txt = (sub["name"].fill_null("") + " | " + sub["address"].fill_null("")).to_numpy()
    return txt[inv]


def pair_text(pairs: pl.DataFrame, split):
    """(S1 text, candidate text) for every pair, reading only the rows that are needed."""
    a = _rows_text(split, 1, pairs["s1"].to_numpy())
    src, row = pairs["src"].to_numpy(), pairs["row"].to_numpy()
    b = np.empty(len(pairs), object)
    for k in (2, 3):
        m = src == k
        if m.any():
            b[m] = _rows_text(split, k, row[m])
    return a, b


# ------------------------------------------------------------------ build
def build():
    os.makedirs(CE_DIR, exist_ok=True)
    rng = np.random.default_rng(5)
    bA, bB = booster("s1_A"), booster("s1_B")
    folds = os.environ.get("ER_FOLDS", "").split("|")
    in_a = set(folds[0].split(",")) if folds and folds[0] else set()
    parts = []
    for c in CE_CHUNKS:
        pr, X = load(f"tr_{c}")
        p = predict(bB if c in in_a else bA, X)  # out-of-fold stage-1 score
        y = pr["y"].to_numpy()
        unsure = (p > LO) & (p < HI)
        easy_pos = (y == 1) & ~unsure & (rng.random(len(y)) < 0.10)
        easy_neg = (y == 0) & ~unsure & (rng.random(len(y)) < 0.002)
        m = unsure | easy_pos | easy_neg
        parts.append(pr.filter(pl.Series(m)).select("s1", "src", "row", "y").with_columns(pl.lit("train").alias("split")))
        print(f"  tr_{c}: {unsure.sum():,} unsure, {m.sum():,} kept, pos share {y[m].mean():.3f}", flush=True)
    d = pl.concat(parts)
    # pseudo-labelled pairs of unseen test countries (confident XGB decisions, no true labels)
    sc_path = os.path.join(MODEL_DIR, "test_scores.parquet")
    if os.path.exists(sc_path):
        tr_c = set(pl.read_parquet(os.path.join(WORK_DIR, "train_s1.parquet"), columns=["country"])["country"].unique())
        te_c = pl.read_parquet(os.path.join(WORK_DIR, "test_s1.parquet"), columns=["country"])["country"]
        unseen = sorted(set(te_c.unique()) - tr_c)
        sc = pl.read_parquet(sc_path, columns=["s1", "src", "row", "p"])
        sc = sc.filter(te_c.gather(sc["s1"]).is_in(unseen))
        best = sc.sort("p", descending=True).unique(["src", "row"], keep="first")
        pos = best.filter(pl.col("p") >= 0.98)
        pos = pos.sample(min(150_000, len(pos)), seed=1)
        neg = sc.filter((pl.col("p") <= 0.02) & (pl.col("p") >= 0.0005))
        neg = neg.sample(min(150_000, len(neg)), seed=2)
        ps = pl.concat([pos.with_columns(pl.lit(1, pl.Int8).alias("y")), neg.with_columns(pl.lit(0, pl.Int8).alias("y"))])
        d = pl.concat([d, ps.select("s1", "src", "row", "y").with_columns(pl.lit("test").alias("split"))])
        print(f"  pseudo pairs for {unseen}: {len(pos):,} pos, {len(neg):,} neg", flush=True)
    d = d.sample(fraction=1.0, shuffle=True, seed=3)
    d.write_parquet(os.path.join(CE_DIR, "train_pairs.parquet"))
    print(f"CE training pairs: {len(d):,} (pos share {d['y'].mean():.3f})", flush=True)


# ------------------------------------------------------------------ model helpers
def _tok_model(path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    # weights must be float32: the checkpoint is stored in float16, and AdamW on float16 weights
    # turns every parameter into NaN after one step (bf16 is only used inside autocast)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=torch.float32).float()
    return tok, model


def _batches(a, b, tok, order, bs):
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        enc = tok(list(a[j]), list(b[j]), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        yield j, enc


def train():
    import torch
    d = pl.read_parquet(os.path.join(CE_DIR, "train_pairs.parquet"))
    a = np.empty(len(d), object); b = np.empty(len(d), object)
    for sp in ("train", "test"):
        m = (d["split"] == sp).to_numpy()
        if m.any():
            aa, bb = pair_text(d.filter(pl.Series(m)), sp)
            a[m], b[m] = aa, bb
    y = d["y"].to_numpy().astype(np.float32)
    tok, model = _tok_model(CE_INIT or BASE)
    # freeze the 250k-token word embedding (192M of the 280M parameters): without its AdamW state
    # the model fits in 8 GB of GPU memory instead of spilling into shared memory (10x slower)
    model.get_input_embeddings().weight.requires_grad_(False)
    model.cuda().train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = math.ceil(len(d) / BATCH)
    warm = int(0.04 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.05, 1 - s / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    t0, seen, run = time.time(), 0, 0.0
    order = np.random.default_rng(int(CE_SEED)).permutation(len(d)) if CE_SEED else np.arange(len(d))
    for step, (j, enc) in enumerate(_batches(a, b, tok, order, BATCH)):
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), torch.from_numpy(y[j]).cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        seen += len(j); run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 500 == 0:
            el = time.time() - t0
            print(f"  step {step:,}/{steps:,}  pairs {seen:,}  loss {run:.4f}  {seen / max(el, 1):.0f} pairs/s", flush=True)
        if time.time() - t0 > TRAIN_MINUTES * 60:
            print(f"  time limit reached after {seen:,} pairs", flush=True)
            break
    model.save_pretrained(os.path.join(CE_DIR, CE_NAME))
    print(f"saved fine-tuned model ({(time.time() - t0) / 60:.0f} min)", flush=True)


def ce_scores(pairs, split):
    """Cross-encoder probability for pairs; cached per (split, s1, src, row) so later runs only
    score pairs that were never scored before."""
    cache_p = os.path.join(CE_DIR, f"cache_{split}{SFX}.parquet")
    keys = pairs.select("s1", "src", "row").with_columns(pl.int_range(pl.len()).alias("_i"))
    out = np.full(len(keys), np.nan, np.float32)
    cache = pl.read_parquet(cache_p) if os.path.exists(cache_p) else None
    if cache is not None:
        hit = keys.join(cache, on=["s1", "src", "row"], how="inner")
        out[hit["_i"].to_numpy()] = hit["pc"].to_numpy()
    todo = np.nonzero(np.isnan(out))[0]
    print(f"    {len(keys) - len(todo):,} cached, {len(todo):,} to score", flush=True)
    if len(todo):
        sub = keys[todo]
        out[todo] = _ce_raw(sub, split)
        new = sub.select("s1", "src", "row").with_columns(pl.Series("pc", out[todo]))
        (pl.concat([cache, new]) if cache is not None else new).write_parquet(cache_p)
    return out


def _ce_raw(pairs, split):
    """Cross-encoder probability for pairs (length-sorted batches for speed)."""
    import torch
    tok, model = _tok_model(os.path.join(CE_DIR, CE_NAME))
    model.cuda().eval()
    a, b = pair_text(pairs, split)
    lens = np.fromiter((len(x) + len(z) for x, z in zip(a, b)), np.int32, len(a))
    order = np.argsort(lens)
    out = np.empty(len(a), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for n, (j, enc) in enumerate(_batches(a, b, tok, order, 64)):
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[j] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            if n % 1600 == 0:
                print(f"    scored {min((n + 1) * 64, len(a)):,}/{len(a):,} ({(n + 1) * 64 / max(time.time() - t0, 1):.0f}/s)", flush=True)
    return out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend_fit(px, pc, y):
    """Logistic regression on (logit XGB, logit CE) for uncertain pairs."""
    from sklearn.linear_model import LogisticRegression
    Z = np.column_stack([_logit(px), _logit(pc), _logit(px) * _logit(pc)])
    return LogisticRegression(C=1.0, max_iter=500).fit(Z, y)


def blend_apply(m, px, pc):
    Z = np.column_stack([_logit(px), _logit(pc), _logit(px) * _logit(pc)])
    return m.predict_proba(Z)[:, 1].astype(np.float32)


# ------------------------------------------------------------------ score
def score():
    import pickle
    from build_features import VAL_START, s1_groups
    from labels import positive_pairs
    from two_stage import record_groups, to_dict, tune
    os.makedirs(CE_DIR, exist_ok=True)
    # validation: stage-2 scores of the XGB run
    pv, Xv = load("val_0")
    p2 = np.load(os.path.join(MODEL_DIR, "val_p2.npy"))
    unsure = (p2 > LO) & (p2 < HI)
    print(f"val: {unsure.sum():,} uncertain pairs of {len(p2):,}", flush=True)
    pc = ce_scores(pv.filter(pl.Series(unsure)), "train")
    y = pv["y"].to_numpy()[unsure]
    # 2-fold check of the blend on val (by S1) before fitting it on all val pairs
    s1u = pv["s1"].to_numpy()[unsure]
    half = (s1u % 2) == 0
    from sklearn.metrics import log_loss
    for tr_m, te_m in ((half, ~half), (~half, half)):
        m = blend_fit(p2[unsure][tr_m], pc[tr_m], y[tr_m])
        pb_h = blend_apply(m, p2[unsure][te_m], pc[te_m])
        print(f"  held-out half: logloss XGB {log_loss(y[te_m], np.clip(p2[unsure][te_m], 1e-6, 1 - 1e-6)):.4f}  "
              f"CE {log_loss(y[te_m], np.clip(pc[te_m], 1e-6, 1 - 1e-6)):.4f}  blend {log_loss(y[te_m], pb_h):.4f}", flush=True)
    blend = blend_fit(p2[unsure], pc, y)
    pickle.dump(blend, open(os.path.join(MODEL_DIR, PFX + "blend.pkl"), "wb"))
    pb = p2.copy()
    pb[unsure] = blend_apply(blend, p2[unsure], pc)
    n1 = pl.scan_parquet(os.path.join(WORK_DIR, "train_s1.parquet")).select(pl.len()).collect().item()
    val_rows = np.nonzero(s1_groups(n1) >= VAL_START)[0]
    truth = to_dict(positive_pairs().filter(pl.col("s1").is_in(pl.Series(val_rows.astype(np.int32)))))
    grp = record_groups(Xv)
    r_x = tune(pv, p2, truth, val_rows.tolist(), grp)
    r_b = tune(pv, pb, truth, val_rows.tolist(), grp)
    res = {"xgb_only": r_x, "blended": r_b}
    print(json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(MODEL_DIR, PFX + "val_results.json"), "w"), indent=2)
    np.save(os.path.join(MODEL_DIR, PFX + "val_pb.npy"), pb)
    # test: re-score uncertain pairs of the XGB test scores and write blended scores
    sc = pl.read_parquet(os.path.join(MODEL_DIR, "test_scores.parquet"))
    p = sc["p"].to_numpy()
    u = (p > LO) & (p < HI)
    print(f"test: {u.sum():,} uncertain pairs of {len(p):,}", flush=True)
    pct = ce_scores(sc.filter(pl.Series(u)), "test")
    p_new = p.copy()
    p_new[u] = blend_apply(blend, p[u], pct)
    sc.with_columns(pl.Series("p", p_new), pl.Series("p_xgb", p)).write_parquet(os.path.join(MODEL_DIR, PFX + "test_scores_blend.parquet"))
    print("wrote blended test scores", flush=True)


def write():
    """Submission files from the blended test scores, with the decision rule tuned on blended val."""
    from config import OUT_DIR
    from predict import write_lists
    from two_stage import assign, rule_group_threshold, rule_threshold
    res = json.load(open(os.path.join(MODEL_DIR, PFX + "val_results.json")))["blended"]
    rule = max(("threshold", "group_threshold"), key=lambda k: res[k]["f05"] if k in res else -1)
    cfg = res[rule]
    print(f"using {rule}: {cfg}", flush=True)
    sc = pl.read_parquet(os.path.join(MODEL_DIR, PFX + "test_scores_blend.parquet"))
    a = assign(sc, sc["p"].to_numpy()).join(sc.select("s1", "src", "row", "grp"), on=["s1", "src", "row"])
    keep = rule_threshold(a, cfg["thr"]) if rule == "threshold" else rule_group_threshold(a, cfg["thr"])
    ids = {k: pl.read_parquet(os.path.join(WORK_DIR, f"test_s{k}.parquet"), columns=["entity_id"])
               .with_row_index("row").with_columns(pl.col("row").cast(pl.Int32)) for k in (1, 2, 3)}
    eid = pl.concat([ids[k].rename({"entity_id": "eid"}).with_columns(pl.lit(k, pl.Int8).alias("src")) for k in (2, 3)])
    write_lists(sc.join(eid, on=["src", "row"]), ids[1], "candidate_entity_ids", os.path.join(OUT_DIR, "candidate_pairs.tsv"))
    write_lists(keep.join(eid, on=["src", "row"]), ids[1], "matched_entity_ids", os.path.join(OUT_DIR, "matching_results.tsv"))


if __name__ == "__main__":
    {"build": build, "train": train, "score": score, "write": write}[sys.argv[1]]()
