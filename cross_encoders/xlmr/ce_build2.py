"""Cross-encoder training set v2 (train data only, validation S1s excluded, raw strings kept).

Differences from ce_build.py (which follows the brief exactly):
  * ALL of V3's confident mistakes are kept: p>0.98 with y=0 and p<0.02 with y=1 (the brief's random 10 % / 0.2 %
    sampling drops most of them).
  * exact duplicate (text_a, text_b) pairs are dropped.
  * label-preserving noise augmentation on a copy of ~12 % of rows (generic, no country rules), imitating the noise
    seen in unseen-country data: acronym-only names, fake diacritics, "(NN)" number tokens, street abbreviations.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

SEED = 42
LEGAL = {"inc", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company", "llp", "plc", "sa",
         "sarl", "sas", "gmbh", "ag", "the", "and", "of", "&", "de", "du", "des", "la", "le", "les"}
ABBR = [("street", "st"), ("avenue", "ave"), ("road", "rd"), ("boulevard", "blvd"), ("drive", "dr"), ("suite", "ste"),
        ("rue", "r"), ("avenue", "av"), ("boulevard", "bd"), ("place", "pl"), ("saint", "st")]
ACC = {"a": "à", "e": "é", "i": "î", "o": "ô", "u": "ù", "c": "ç", "A": "Â", "E": "É", "O": "Ô"}


def acronym(name, rng):
    toks = [t for t in re.split(r"[\s,.]+", name) if t]
    core = [t for t in toks if t.lower().strip(".") not in LEGAL] or toks
    ac = "".join(t[0] for t in core if t[0].isalnum()).upper()
    return ac if len(ac) >= 2 else name


def diacritics(s, rng):
    out = []
    for ch in s:
        out.append(ACC[ch] if ch in ACC and rng.random_sample() < 0.15 else ch)
    return "".join(out)


def abbrev(s, rng):
    for a, b in ABBR:
        s = re.sub(rf"\b{a}\b", b, s, flags=re.I) if rng.random_sample() < 0.5 else s
    return s


def augment(text, rng):
    """text = 'name | address'. Applies one random generic noise to one record."""
    name, _, addr = text.partition(" | ")
    k = rng.randint(4)
    if k == 0:
        name = acronym(name, rng)
    elif k == 1:
        name, addr = diacritics(name, rng), diacritics(addr, rng)
    elif k == 2:
        addr = f"{addr} ({rng.randint(1, 99)})" if addr else addr
    else:
        addr = abbrev(addr, rng)
    return f"{name} | {addr}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--val_ids", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="ce_data2")
    ap.add_argument("--cap_unc", type=int, default=1_500_000)
    ap.add_argument("--aug_frac", type=float, default=0.12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.RandomState(SEED)

    val = set(x.strip() for x in open(a.val_ids, encoding="utf-8") if x.strip())
    gt = pd.read_csv(a.gt, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    gt = gt[gt.matched_entity_ids != ""]
    gt = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    gk = np.sort(pd.util.hash_array((gt.source1_entity_id + "|" + gt.m).values.astype(object)))
    del gt

    parts, n_val, n_all = [], 0, 0
    for b in pq.ParquetFile(a.oof).iter_batches(batch_size=4_000_000,
                                                columns=["source1_entity_id", "candidate_entity_id", "score"]):
        d = b.to_pandas()
        n_all += len(d)
        keep = ~d.source1_entity_id.isin(val).values
        n_val += int((~keep).sum())
        d = d[keep]
        h = pd.util.hash_array((d.source1_entity_id + "|" + d.candidate_entity_id).values.astype(object))
        y = (gk[np.minimum(np.searchsorted(gk, h), len(gk) - 1)] == h).astype(np.int8)
        p = d.score.values
        unc = (p > 0.02) & (p < 0.98)
        err = ((p >= 0.98) & (y == 0)) | ((p <= 0.02) & (y == 1))     # V3 confident mistakes: keep all
        r = rng.random_sample(len(d))
        sel = unc | err | ((y == 1) & ~err & (r < 0.05)) | ((y == 0) & ~err & (r < 0.002))
        x = d[sel].copy()
        x["y"], x["unc"], x["err"] = y[sel], unc[sel], err[sel]
        parts.append(x)
        print(f"  scanned {n_all:,} kept {sum(len(q) for q in parts):,}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    del parts
    assert not df.source1_entity_id.isin(val).any(), "LEAK: validation S1 in training pairs"
    if int(df.unc.sum()) > a.cap_unc:
        drop = rng.choice(np.where(df.unc.values & ~df.err.values)[0], int(df.unc.sum()) - a.cap_unc, replace=False)
        m = np.ones(len(df), bool)
        m[drop] = False
        df = df[m]

    need = pa.array(pd.unique(np.concatenate([df.source1_entity_id.values, df.candidate_entity_id.values])))
    tabs = []
    for k in (1, 2, 3):
        t = pacsv.read_csv(os.path.join(a.data_dir, f"train_source{k}.tsv"),
                           parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                           convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in
                                                                              ("entity_id", "business_name", "business_address", "country")}))
        t = t.filter(pc.is_in(t["entity_id"], value_set=need))
        tabs.append(t.select(["entity_id", "business_name", "business_address"]).to_pandas())
    txt = pd.concat(tabs, ignore_index=True).drop_duplicates("entity_id").set_index("entity_id")
    txt = (txt.business_name.fillna("") + " | " + txt.business_address.fillna(""))
    df["text_a"] = txt.reindex(df.source1_entity_id.values).fillna(" | ").values
    df["text_b"] = txt.reindex(df.candidate_entity_id.values).fillna(" | ").values
    n0 = len(df)
    df = df.drop_duplicates(["text_a", "text_b"])
    print(f"dropped {n0-len(df):,} exact duplicate text pairs", flush=True)

    # label-preserving noise copies (candidate side, or S1 side half of the time)
    aug = df.sample(frac=a.aug_frac, random_state=SEED).copy()
    side_b = rng.random_sample(len(aug)) < 0.75
    aug["text_b"] = [augment(t, rng) if s else t for t, s in zip(aug.text_b.values, side_b)]
    aug["text_a"] = [t if s else augment(t, rng) for t, s in zip(aug.text_a.values, side_b)]
    df = pd.concat([df, aug], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df = df.rename(columns={"score": "p_v3"})[["source1_entity_id", "candidate_entity_id", "text_a", "text_b", "y", "p_v3"]]
    assert not df.source1_entity_id.isin(val).any()
    df.to_parquet(os.path.join(a.out, "train_ce2.parquet"), compression="zstd", index=False)
    json.dump({"rows": int(len(df)), "pos_rate": float(df.y.mean()), "aug_rows": int(len(aug)),
               "val_rows_dropped": n_val, "val_s1_in_training": 0},
              open(os.path.join(a.out, "build_report.json"), "w"), indent=1)
    print(df.sample(8, random_state=3)[["text_a", "text_b", "y"]].to_string())
