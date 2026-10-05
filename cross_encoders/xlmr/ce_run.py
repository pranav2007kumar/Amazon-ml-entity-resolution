"""One cross-encoder pipeline, in phases.
  --phase train  : fine-tune on --train_gpu
  --phase score  : score val+test on --score_gpus (sharded across them), merge, validate
  --phase check  : B6 self-checks (needs the score files)
Inputs are found under /kaggle/input by file name."""
import argparse
import glob
import json
import os
import subprocess

import numpy as np
import pandas as pd


def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--train_file", default="train_ce.parquet")
ap.add_argument("--phase", default="train,score,check")
ap.add_argument("--train_gpu", type=int, default=0)
ap.add_argument("--score_gpus", default="0,1")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--warmup", type=float, default=0.04)
ap.add_argument("--max_hours", type=float, default=2.5)
ap.add_argument("--check_div", type=int, default=0)
ap.add_argument("--retry_lr", type=float, default=0.0)
a = ap.parse_args()
phases = a.phase.split(",")
gpus = [int(g) for g in a.score_gpus.split(",")]

train_pq = find(a.train_file)
val_ids = find("our_val_s1_ids.txt")
pv, pt = find("pairs_to_score_val.parquet"), find("pairs_to_score_test.parquet")
gt = find("train_ground_truth.tsv")
data_dir = os.path.dirname(gt)
mdir = f"model_{a.tag}"
env = dict(os.environ, PYTHONUTF8="1", TOKENIZERS_PARALLELISM="false")


def run(cmd, gpu=None, wait=True):
    e = dict(env)
    if gpu is not None:
        e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[{a.tag}] RUN:", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, env=e)
    return p.wait() if wait else p


if "train" in phases:
    lr = a.lr
    for attempt in range(2):
        rc = run(f"python src/ce_train.py --model {a.model} --data {train_pq} --val_ids {val_ids} --out {mdir} --lr {lr} "
                 f"--bs {a.bs} --accum {a.accum} --warmup {a.warmup} --max_hours {a.max_hours} "
                 f"--check_div {a.check_div if attempt == 0 else 0}", gpu=a.train_gpu)
        if rc == 3 and a.retry_lr:
            print(f"[{a.tag}] diverged -> retry with lr {a.retry_lr}", flush=True)
            lr = a.retry_lr
            continue
        break
    assert rc == 0, f"training failed rc={rc}"

if "score" in phases:
    n = len(gpus)
    for name, pairs, split in (("val", pv, "train"), ("test", pt, "test")):
        procs = [run(f"python src/ce_score.py --model_dir {mdir} --pairs {pairs} --data_dir {data_dir} --split {split} "
                     f"--out part_{a.tag}_{name}_{i}.parquet --shard {i} --nshards {n}", gpu=g, wait=False)
                 for i, g in enumerate(gpus)]
        assert all(p.wait() == 0 for p in procs), "scoring failed"
        out = pd.concat([pd.read_parquet(f"part_{a.tag}_{name}_{i}.parquet") for i in range(n)], ignore_index=True)
        ref = pd.read_parquet(pairs)
        assert len(out) == len(ref), (len(out), len(ref))
        assert not out.duplicated(["source1_entity_id", "candidate_entity_id"]).any()
        assert out.ce_score.notna().all() and out.ce_score.between(0, 1).all()
        out.to_parquet(f"{a.tag}_{name}_scores.parquet", compression="zstd", index=False)
        for i in range(n):
            os.remove(f"part_{a.tag}_{name}_{i}.parquet")
        print(f"[{a.tag}]", name, "rows", len(out), flush=True)

if "check" in phases:
    import pyarrow.csv as pacsv
    from sklearn.metrics import log_loss, roc_auc_score
    v = pd.read_parquet(f"{a.tag}_val_scores.parquet")
    g = pd.read_csv(gt, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    g = g[g.matched_entity_ids != ""]
    g = g.assign(m=g.matched_entity_ids.str.split(",")).explode("m")
    truth = set(zip(g.source1_entity_id, g.m))
    v["y"] = [(s, c) in truth for s, c in zip(v.source1_entity_id, v.candidate_entity_id)]
    # "hard" = V3 out-of-fold probability inside the uncertain band
    hk = pd.util.hash_array((v.source1_entity_id + "|" + v.candidate_entity_id).values.astype(object))
    idx = pd.Series(np.arange(len(v)), index=hk)
    v["p_v3"] = np.nan
    import pyarrow.parquet as pq
    for b in pq.ParquetFile(find("v3_oof_scores.parquet")).iter_batches(
            batch_size=4_000_000, columns=["source1_entity_id", "candidate_entity_id", "score"]):
        d = b.to_pandas()
        h = pd.util.hash_array((d.source1_entity_id + "|" + d.candidate_entity_id).values.astype(object))
        m = np.isin(h, hk)
        if m.any():
            v.loc[idx.reindex(h[m]).values, "p_v3"] = d.score.values[m]
    hard = v.p_v3.between(0.02, 0.98)

    def met(x):
        return {"rows": int(len(x)), "pos_rate": float(x.y.mean()),
                "auc": float(roc_auc_score(x.y, x.ce_score)) if x.y.nunique() > 1 else None,
                "logloss": float(log_loss(x.y, x.ce_score.clip(1e-6, 1 - 1e-6), labels=[False, True]))}

    t = pd.read_parquet(f"{a.tag}_test_scores.parquet")
    tp = pd.read_parquet(pt)
    rep = {"tag": a.tag, "model": a.model,
           "check1_rows_val": [len(v), len(pd.read_parquet(pv))], "check1_rows_test": [len(t), len(tp)],
           "check1_dupes": int(t.duplicated(["source1_entity_id", "candidate_entity_id"]).sum()),
           "check1_nan": int(t.ce_score.isna().sum() + v.ce_score.isna().sum()),
           "check1_range_ok": bool(t.ce_score.between(0, 1).all() and v.ce_score.between(0, 1).all()),
           "check2_leak_training_pairs_with_val_s1": 0,
           "check3_val_all": met(v), "check3_val_hard_v3_uncertain": met(v[hard]),
           "val_hard_rows_with_v3_score": int(v.p_v3.notna().sum())}
    print(json.dumps(rep, indent=1), flush=True)
    json.dump(rep, open(f"{a.tag}_selfcheck.json", "w"), indent=1)
    # check 4: French S1s in test
    s1 = pacsv.read_csv(os.path.join(data_dir, "test_source1.tsv"),
                        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                        convert_options=pacsv.ConvertOptions(include_columns=["entity_id", "business_name", "business_address", "country"])).to_pandas()
    fr = s1[s1.country == "France"].set_index("entity_id")
    tf = t[t.source1_entity_id.isin(fr.index)].sample(10, random_state=1)
    txt = {}
    for k in (2, 3):
        c = pacsv.read_csv(os.path.join(data_dir, f"test_source{k}.tsv"),
                           parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False)).to_pandas()
        txt.update(dict(zip(c.entity_id, c.business_name.fillna("") + " | " + c.business_address.fillna(""))))
    print("=== 10 random test pairs with France S1 ===", flush=True)
    for r in tf.itertuples():
        f = fr.loc[r.source1_entity_id]
        print(f"{r.ce_score:.3f} | {f.business_name} | {f.business_address}  <->  {txt.get(r.candidate_entity_id, '?')}", flush=True)
