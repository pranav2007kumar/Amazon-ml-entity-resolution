"""B4: score candidate pairs with a fine-tuned cross-encoder (eval mode, no_grad, fp16 autocast, batch 128,
sorted by text length). One shard per process: rows i::n of the input, pinned to one GPU by the driver."""
import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", required=True)
ap.add_argument("--pairs", required=True)
ap.add_argument("--data_dir", required=True)
ap.add_argument("--split", required=True, choices=["train", "test"])
ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--max_len", type=int, default=96)
a = ap.parse_args()

pr = pd.read_parquet(a.pairs)
pr = pr.iloc[a.shard::a.nshards].reset_index(drop=True)
need = pa.array(pd.unique(np.concatenate([pr.source1_entity_id.values, pr.candidate_entity_id.values])))
tabs = []
for k in (1, 2, 3):
    t = pacsv.read_csv(os.path.join(a.data_dir, f"{a.split}_source{k}.tsv"),
                       parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                       convert_options=pacsv.ConvertOptions(column_types={"entity_id": pa.string(), "business_name": pa.string(),
                                                                          "business_address": pa.string(), "country": pa.string()}))
    t = t.filter(pc.is_in(t["entity_id"], value_set=need))
    tabs.append(t.select(["entity_id", "business_name", "business_address"]).to_pandas())
txt = pd.concat(tabs, ignore_index=True).drop_duplicates("entity_id").set_index("entity_id")
txt = (txt.business_name.fillna("") + " | " + txt.business_address.fillna(""))
ta = txt.reindex(pr.source1_entity_id.values).fillna(" | ").values
tb = txt.reindex(pr.candidate_entity_id.values).fillna(" | ").values
order = np.argsort(np.fromiter((len(x) + len(y) for x, y in zip(ta, tb)), np.int32, len(ta)), kind="stable")
print(f"shard {a.shard}/{a.nshards}: {len(pr):,} pairs", flush=True)

tok = AutoTokenizer.from_pretrained(a.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(a.model_dir, torch_dtype=torch.float32).cuda().eval()
out = np.empty(len(pr), np.float32)
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(order), a.bs):
        ix = order[i:i + a.bs]
        enc = tok(list(ta[ix]), list(tb[ix]), truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(**enc).logits.squeeze(-1)
        out[ix] = torch.sigmoid(lg.float()).cpu().numpy()
        if (i // a.bs) % 500 == 0:
            print(f"  {i:,}/{len(order):,} {i/max(time.time()-t0,1):.0f} pairs/s", flush=True)
assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
pr["ce_score"] = out
pr.to_parquet(a.out, compression="zstd", index=False)
print("done", a.out, f"{time.time()-t0:.0f}s", flush=True)
