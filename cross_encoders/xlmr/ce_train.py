"""B3: fine-tune an XLM-RoBERTa cross-encoder (MIT licence) on raw "name | address" pairs.
float32 weights + fp16 autocast + GradScaler. 1 epoch, seed 42, linear warm-up/decay.
The LR schedule follows max(step progress, wall-clock progress) so a time cap still ends with decayed LR.
Exit code 3 = diverged (loss not < 0.5 after 2000 micro-steps) -> the driver retries with a lower LR.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="xlm-roberta-base")
ap.add_argument("--data", required=True)
ap.add_argument("--val_ids", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--warmup", type=float, default=0.04)
ap.add_argument("--max_hours", type=float, default=3.0)
ap.add_argument("--max_len", type=int, default=96)
ap.add_argument("--check_div", type=int, default=0)
a = ap.parse_args()

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
os.makedirs(a.out, exist_ok=True)
df = pd.read_parquet(a.data)
val = set(x.strip() for x in open(a.val_ids, encoding="utf-8") if x.strip())
assert not df.source1_entity_id.isin(val).any(), "LEAK: validation S1 in training data"
print(f"train pairs {len(df):,} pos_rate {df.y.mean():.3f} leak-check passed", flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, torch_dtype=torch.float32).cuda()
print("params", sum(p.numel() for p in model.parameters()), flush=True)


class DS(Dataset):
    def __init__(s, d):
        s.a, s.b, s.y = d.text_a.values, d.text_b.values, d.y.values.astype(np.float32)

    def __len__(s):
        return len(s.y)

    def __getitem__(s, i):
        return s.a[i], s.b[i], s.y[i]


def collate(batch):
    ta, tb, y = zip(*batch)
    enc = tok(list(ta), list(tb), truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor(y)
    return enc


dl = DataLoader(DS(df), batch_size=a.bs, shuffle=True, collate_fn=collate, num_workers=2,
                generator=torch.Generator().manual_seed(SEED), drop_last=True)
total = len(dl) // a.accum
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")
lossf = torch.nn.BCEWithLogitsLoss()
budget = a.max_hours * 3600
t0 = time.time(); last_ck = t0
step = 0; run = []; seen = 0
model.train()
for mi, enc in enumerate(dl):
    y = enc.pop("labels").cuda()
    enc = {k: v.cuda() for k, v in enc.items()}
    with torch.autocast("cuda", dtype=torch.float16):
        logit = model(**enc).logits.squeeze(-1)
    loss = lossf(logit.float(), y)
    scaler.scale(loss / a.accum).backward()
    run.append(loss.item()); seen += len(y)
    if (mi + 1) % a.accum == 0:
        prog = max(step / total, (time.time() - t0) / budget)
        lr = a.lr * (prog / a.warmup if prog < a.warmup else max(0.0, (1 - prog) / (1 - a.warmup)))
        for g in opt.param_groups:
            g["lr"] = lr
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        step += 1
    if (mi + 1) % 500 == 0:
        el = time.time() - t0
        print(f"micro {mi+1} step {step}/{total} loss {np.mean(run[-500:]):.4f} lr {lr:.2e} {seen/el:.1f} pairs/s "
              f"elapsed {el/60:.1f} min", flush=True)
    if a.check_div and mi + 1 == a.check_div and np.mean(run[-500:]) > 0.5:
        print("DIVERGED", flush=True)
        raise SystemExit(3)
    if time.time() - last_ck > 1800:
        torch.save({k: v.half() for k, v in model.state_dict().items()}, os.path.join(a.out, "ckpt.pt"))
        last_ck = time.time()
    if time.time() - t0 > budget:
        print("time budget reached", flush=True)
        break
if not np.isfinite(np.mean(run[-500:])):
    raise SystemExit("NaN loss")
model.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
print(f"saved after {(time.time()-t0)/60:.1f} min, {seen:,} pairs seen", flush=True)
