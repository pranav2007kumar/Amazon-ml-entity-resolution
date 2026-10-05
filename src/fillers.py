"""Stage 1a: filler words, detected per country without labels, moved out of the core name.

The noisy sources (2 and 3) add filler words to names: 'Services', 'Holdings', 'DBA', 'Northside' in
the US, 'Shri', 'Smt' in India, 'Et Fils', 'Groupe', 'Participations', 'Développement' in France.
The lists differ by country, so a model trained on US/India alone would read an unseen country's
fillers as real name changes. A word is a filler of a country when it is much more frequent in
Source 2/3 names than in Source 1 names of that country:
    rate_S2S3 >= MIN_RATE  and  rate_S2S3 / rate_S1 >= MIN_RATIO
Only Latin-script names are counted, so transliterations of Indic names (never in Source 1) are not
mistaken for fillers. Filler tokens are moved from name_core to name_legal (like 'Pvt', 'SARL');
sound keys (ph) and initials (acr) are recomputed. Fillers are computed per split (train / test).
Usage: python fillers.py
"""
import json
import os

import polars as pl

from config import WORK_DIR
from normalize import squash
from translit import phonetic

MIN_RATE, MIN_RATIO = 0.002, 1.5
_NONLATIN = r"[ऀ-෿]"


def detect(split):
    counts, n = {}, {}
    for k in (1, 2, 3):
        df = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s{k}.parquet"), columns=["country", "name", "name_core"])
        df = df.filter(~pl.col("name").str.contains(_NONLATIN))
        n[k] = df.group_by("country").len().rename({"len": f"N{k}"})
        # number of records of each country whose core name contains the token (once per record)
        counts[k] = (df.select("country", pl.col("name_core").str.split(" ").list.unique().alias("t")).explode("t")
                       .filter((pl.col("t").str.len_chars() >= 2) & ~pl.col("t").str.contains(r"^\d+$"))
                       .group_by("country", "t").len().rename({"len": f"n{k}"}))
    d = counts[1].join(counts[2], on=["country", "t"], how="full", coalesce=True) \
        .join(counts[3], on=["country", "t"], how="full", coalesce=True).fill_null(0)
    for k in (1, 2, 3):
        d = d.join(n[k], on="country")
    d = d.with_columns(((pl.col("n2") + pl.col("n3")) / (pl.col("N2") + pl.col("N3"))).alias("r23"),
                       (pl.col("n1") / pl.col("N1")).alias("r1"))
    d = d.filter((pl.col("r23") >= MIN_RATE) & ((pl.col("r23") + 1e-5) / (pl.col("r1") + 1e-5) >= MIN_RATIO))
    out = {}
    for c, t in zip(d["country"].to_list(), d["t"].to_list()):
        out.setdefault(c, set()).add(t)
    return out


def apply(split, fill):
    for k in (1, 2, 3):
        path = os.path.join(WORK_DIR, f"{split}_s{k}.parquet")
        df = pl.read_parquet(path)
        core, legal, ph, acr = [], [], [], []
        n_moved = 0
        for c, nc, nl in zip(df["country"].to_list(), df["name_core"].to_list(), df["name_legal"].to_list()):
            f = fill.get(c, ())
            toks = nc.split()
            keep = [t for t in toks if t not in f]
            if not keep:  # never empty a name completely
                keep = toks
            moved = [t for t in toks if t not in keep]
            n_moved += bool(moved)
            core.append(" ".join(keep))
            legal.append(" ".join([x for x in [nl] + moved if x]))
            sq = squash(" ".join(keep)).split()
            ph.append(" ".join(p for p in (phonetic(t) for t in sq) if p))
            acr.append("".join(t[0] for t in sq if t[0].isalpha()) if len(sq) >= 2 else "")
        df = df.with_columns(pl.Series("name_core", core), pl.Series("name_legal", legal),
                             pl.Series("ph", ph), pl.Series("acr", acr))
        df.write_parquet(path)
        print(f"  {split} s{k}: fillers moved in {n_moved:,} of {len(df):,} rows", flush=True)


if __name__ == "__main__":
    out = {}
    for split in ("train", "test"):
        fill = detect(split)
        out[split] = {c: sorted(v) for c, v in fill.items()}
        for c, v in out[split].items():
            print(f"{split} {c}: {len(v)} fillers: {', '.join(v[:60])}", flush=True)
        apply(split, fill)
    json.dump(out, open(os.path.join(WORK_DIR, "fillers.json"), "w"), indent=1)
