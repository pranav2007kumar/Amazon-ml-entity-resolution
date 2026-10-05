"""Round 2 for the cross-encoder (microsoft/mdeberta-v3-base, MIT, 278M): France-style noise robustness.

Fine-tunes the BASE model (model_base) from scratch on
  * train_hard_pairs.parquet   ~1.8M labelled training pairs (US / India) from the 70% of the training data the
                               first team model never saw (hard pairs chosen by the team's XGBoost)
  * test_pseudo_pairs.parquet  ~400k test pairs (France, US, India) whose label comes from confident, agreeing
                               decisions of the team's XGBoost + cross-encoder (no true test labels are used)
then scores score_val.parquet (labelled validation pairs, used only for the self-check) and score_test.parquet.

  python ce_hard_pairs.py check                  environment + GPU check (1 min)
  python ce_hard_pairs.py train  [--minutes 210] fine-tune (checkpoint every 20 min, resumable)
  python ce_hard_pairs.py score                  score val + test with the trained model, write outputs, self-check
  python ce_hard_pairs.py all                    check, train, score

All paths are relative to this folder. Outputs go to ./out/.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import polars as pl
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
M1 = os.path.join(HERE, "model_base")        # tokenizer (same as before)
INIT = os.environ.get("JCE_INIT", os.path.join(HERE, "out", "model_j"))   # round-1 fine-tuned model = start point
OUT = os.path.join(HERE, "out")
CKPT = os.path.join(OUT, "ckpt2")
FINAL = os.path.join(OUT, "model_j2")
MAX_LEN, BATCH, SCORE_BATCH, LR, SEED = 96, 32, 64, 1e-5, 2


def dev_dtype():
    assert torch.cuda.is_available(), "No CUDA GPU visible to PyTorch - install a CUDA 12.8+ build of PyTorch"
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load(path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M1)
    # weights must stay float32 (bf16/fp16 only inside autocast), otherwise AdamW produces NaN
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=torch.float32).float()
    return tok, model


def batches(a, b, tok, order, bs):
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        yield j, tok(list(a[j]), list(b[j]), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")


def score_pairs(model, tok, a, b):
    dt = dev_dtype()
    model.cuda().eval()
    order = np.argsort(np.fromiter((len(x) + len(z) for x, z in zip(a, b)), np.int32, len(a)))
    out = np.empty(len(a), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for n, (j, enc) in enumerate(batches(a, b, tok, order, SCORE_BATCH)):
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=dt):
                out[j] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            if n % 2000 == 0:
                done = min((n + 1) * SCORE_BATCH, len(a))
                print(f"    scored {done:,}/{len(a):,} ({done / max(time.time() - t0, 1):.0f}/s)", flush=True)
    return out


def metrics(y, p):
    from sklearn.metrics import log_loss, roc_auc_score
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "pos_rate": float(y.mean()), "logloss": float(log_loss(y, p)),
            "auc": float(roc_auc_score(y, p)), "acc@0.5": float(((p > 0.5) == (y == 1)).mean())}


def cmd_check():
    import transformers
    print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "polars", pl.__version__)
    print("GPU", torch.cuda.get_device_name(0), "bf16" if torch.cuda.is_bf16_supported() else "fp16",
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    for f in ("france_aug_pairs.parquet", "train_hard_pairs.parquet", "score_val.parquet", "score_test.parquet"):
        print(f, pl.scan_parquet(os.path.join(HERE, f)).select(pl.len()).collect().item(), "rows")
    tok, model = load(M1)
    enc = tok(["Roux & Fils SAS | 45 Avenue des Fauvettes, Merignac"], ["Roux & Fils | 45 AVE DES FAUVETTES, MERIGNAC"],
              return_tensors="pt")
    model.cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=dev_dtype()):
        p = torch.sigmoid(model(**{k: v.cuda() for k, v in enc.items()}).logits.float()).item()
    print(f"sample output {p:.3f} (untrained model: any finite number between 0 and 1 is fine)")
    assert math.isfinite(p)
    print("CHECK OK")


def cmd_baseline():
    os.makedirs(OUT, exist_ok=True)
    v = pl.read_parquet(os.path.join(HERE, "score_val.parquet"))
    tok, model = load(M1)
    p = score_pairs(model, tok, v["text_a"].to_numpy(), v["text_b"].to_numpy())
    m = metrics(v["y"].to_numpy(), p)
    json.dump(m, open(os.path.join(OUT, "baseline_val_metrics.json"), "w"), indent=1)
    print("BASELINE (model_m1) on val:", m, flush=True)


def cmd_train(minutes):
    os.makedirs(CKPT, exist_ok=True)
    fr = pl.read_parquet(os.path.join(HERE, "france_aug_pairs.parquet"))
    rp = pl.read_parquet(os.path.join(HERE, "train_hard_pairs.parquet")).sample(300_000, seed=5)   # replay: no forgetting
    d = pl.concat([fr, rp]).sample(fraction=1.0, shuffle=True, seed=SEED)
    a, b, y = d["text_a"].to_numpy(), d["text_b"].to_numpy(), d["y"].to_numpy().astype(np.float32)
    steps = math.ceil(len(d) / BATCH)
    state_p = os.path.join(CKPT, "state.json")
    start = json.load(open(state_p))["step"] if os.path.exists(state_p) else 0
    tok, model = load(CKPT if start else INIT)
    model.get_input_embeddings().weight.requires_grad_(False)  # fits 8 GB GPUs without spilling (much faster)
    model.cuda().train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    warm = int(0.04 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.05, 1 - s / steps))
    for _ in range(start):
        sched.step()
    dt = dev_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    lossf = torch.nn.BCEWithLogitsLoss()
    order = np.arange(len(d))[start * BATCH:]
    print(f"training {len(d):,} pairs ({len(fr):,} France-noise + {len(rp):,} replay), steps {steps:,}, start at {start:,}", flush=True)
    t0 = last_ck = time.time()
    run = None
    step = start
    for i, (j, enc) in enumerate(batches(a, b, tok, order, BATCH)):
        step = start + i
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=dt):
            logit = model(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), torch.from_numpy(y[j]).cuda())
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss is {loss.item()} at step {step} - stop and report")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 500 == 0:
            el = time.time() - t0
            print(f"  step {step:,}/{steps:,}  loss {run:.4f}  {(i + 1) * BATCH / max(el, 1):.0f} pairs/s  "
                  f"gpu {torch.cuda.max_memory_reserved() / 1e9:.1f} GB", flush=True)
        if time.time() - last_ck > 20 * 60:
            model.save_pretrained(CKPT); json.dump({"step": step + 1}, open(state_p, "w")); last_ck = time.time()
            print(f"  checkpoint at step {step + 1:,}", flush=True)
        if time.time() - t0 > minutes * 60:
            print(f"  time limit reached at step {step:,}", flush=True)
            break
    model.save_pretrained(FINAL)
    model.save_pretrained(CKPT); json.dump({"step": step + 1}, open(state_p, "w"))
    print(f"saved {FINAL} ({(time.time() - t0) / 60:.0f} min)", flush=True)


def cmd_score():
    os.makedirs(OUT, exist_ok=True)
    tok, model = load(FINAL)
    v = pl.read_parquet(os.path.join(HERE, "score_val.parquet"))
    pv = score_pairs(model, tok, v["text_a"].to_numpy(), v["text_b"].to_numpy())
    m = metrics(v["y"].to_numpy(), pv)
    base = os.path.join(OUT, "baseline_val_metrics.json")
    rep = {"val": m}
    print("VAL:", json.dumps(rep, indent=1), flush=True)
    v.select("source1_entity_id", "candidate_entity_id").with_columns(pl.Series("ce_score", pv)) \
        .write_parquet(os.path.join(OUT, "ce_r2_val_scores.parquet"), compression="zstd")
    del v
    t = pl.read_parquet(os.path.join(HERE, "score_test.parquet"))
    pt = score_pairs(model, tok, t["text_a"].to_numpy(), t["text_b"].to_numpy())
    out = t.select("source1_entity_id", "candidate_entity_id").with_columns(pl.Series("ce_score", pt))
    out.write_parquet(os.path.join(OUT, "ce_r2_test_scores.parquet"), compression="zstd")
    # self-checks
    assert len(out) == len(t) and out["ce_score"].is_nan().sum() == 0, "test output incomplete / NaN"
    assert out["ce_score"].min() >= 0 and out["ce_score"].max() <= 1
    rep["test_rows"] = len(out)
    rep["test_mean_score"] = float(pt.mean())
    json.dump(rep, open(os.path.join(OUT, "report2.json"), "w"), indent=1)
    print("DONE: send out/ce_r2_val_scores.parquet, out/ce_r2_test_scores.parquet, out/report2.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "baseline", "train", "score", "all"])
    ap.add_argument("--minutes", type=float, default=100)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    if a.cmd in ("check", "all"):
        cmd_check()
    if a.cmd == "baseline":
        cmd_baseline()
    if a.cmd in ("train", "all"):
        cmd_train(a.minutes)
    if a.cmd in ("score", "all"):
        cmd_score()
