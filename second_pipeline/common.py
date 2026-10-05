"""Shared helpers for the local pipeline.

Rules followed everywhere:
  * country is only an exact blocking label, never a model feature, never special-cased
  * everything learned is learned from train; test is only transformed
  * records are addressed by integer row numbers (sorted by country, entity_id) to keep RAM low
"""
import hashlib
import os
import re
import time
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

TSV_KW = dict(sep="\t", dtype=str, keep_default_na=False, na_values=[], quoting=3)
INDIC_SCRIPTS = ("DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA",
                 "TAMIL", "TELUGU", "KANNADA", "MALAYALAM")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_tsv(path, usecols=None):
    return pd.read_csv(path, usecols=usecols, **TSV_KW)


def source_path(data_dir, split, k):
    return os.path.join(data_dir, split, f"{split}_source{k}.tsv")


def wdir(work, split, k=None):
    p = os.path.join(work, split) if k is None else os.path.join(work, split, f"s{k}")
    os.makedirs(p, exist_ok=True)
    return p


def stable_hash01(ids, salt):
    """Deterministic value in [0,1) per id (identical on every machine and run)."""
    return np.array([int(hashlib.md5((salt + i).encode()).hexdigest()[:8], 16) / 2 ** 32 for i in ids])


def stable_fold(ids, n_folds=2, salt="fold"):
    return (stable_hash01(ids, salt) * n_folds).astype(np.int8)


# ---------------------------------------------------------------- transliteration (9 Indic scripts)
_VOWEL = {"A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "E": "e", "EE": "e",
          "AI": "ai", "O": "o", "OO": "o", "AU": "au", "VOCALIC R": "ri", "VOCALIC RR": "ri",
          "VOCALIC L": "li", "CANDRA E": "e", "CANDRA O": "o", "SHORT E": "e", "SHORT O": "o"}
_CONS_FIX = {"ca": "ch", "cha": "chh", "tta": "t", "ttha": "th", "dda": "d", "ddha": "dh",
             "nna": "n", "lla": "l", "llla": "l", "rra": "r", "ssa": "sh", "sha": "sh",
             "nya": "ny", "nga": "ng", "nnna": "n", "khandata": "t",
             "va": "v", "ya": "y", "zha": "zh", "tha": "th", "dha": "dh"}


@lru_cache(maxsize=None)
def _indic_kind(ch):
    try:
        nm = unicodedata.name(ch)
    except ValueError:
        return (None, ch)
    script = nm.split(" ")[0]
    if script not in INDIC_SCRIPTS:
        return (None, ch)
    rest = nm[len(script) + 1:]
    if rest.startswith("DIGIT"):
        return ("D", str(unicodedata.digit(ch, 0)))
    if rest.startswith("VOWEL SIGN "):
        return ("M", _VOWEL.get(rest[11:], ""))
    if rest == "SIGN VIRAMA":
        return ("X", "")
    if rest in ("AI LENGTH MARK", "AU LENGTH MARK", "SIGN NUKTA", "ADDAK", "SIGN ADDAK"):
        return ("Z0", "")
    if rest in ("SIGN ANUSVARA", "SIGN CANDRABINDU", "SIGN TIPPI", "SIGN BINDI", "SIGN INVERTED CANDRABINDU"):
        return ("N", "n")
    if rest == "SIGN VISARGA":
        return ("H", "h")
    if rest.startswith("LETTER CHILLU "):
        c = rest[14:].lower()
        return ("V0", _CONS_FIX.get(c + "a", c))
    if rest.startswith("LETTER "):
        v = rest[7:]
        if v in _VOWEL:
            return ("V", _VOWEL[v])
        c = v.lower().replace(" ", "")
        return ("C", _CONS_FIX.get(c, c[:-1] if c.endswith("a") else c))
    return ("Z", " ")


def transliterate(s):
    if s.isascii():
        return s
    out, pend = [], False
    for ch in s:
        kind, r = _indic_kind(ch)
        if kind == "Z0":
            continue
        if kind == "M":
            out.append(r); pend = False; continue
        if kind == "X":
            pend = False; continue
        if pend:
            if kind in ("C", "N", "H", "V0", "V"):
                out.append("a")
            pend = False
        out.append(r)
        if kind == "C":
            pend = True
    return "".join(out)


_ZW = re.compile("[​-‏⁠﻿]")


def _clean_one(s):
    s = unicodedata.normalize("NFKC", s)
    s = _ZW.sub("", s)
    s = transliterate(s)
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


_OCR = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "6": "g", "8": "b"})


def _ocr_fix(m):
    return m.group(0).translate(_OCR)


def normalise(series):
    """lower-case ASCII words and digits; country-agnostic."""
    s = series.fillna("").astype(str).str.replace("<NULL>", " ", regex=False)
    s = s.map(_clean_one).str.lower()
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(r"(?<=[a-z])['’`](?=[a-z])", "", regex=True)
    s = s.str.replace(r"[^0-9a-z]+", " ", regex=True)
    # OCR digits inside words: de1hi -> delhi ; leading digit before 3+ letters: 5ervice -> service
    s = s.str.replace(r"(?<=[a-z])[013568]+(?=[a-z])", _ocr_fix, regex=True)
    s = s.str.replace(r"\b[01568](?=[a-z]{3,})", _ocr_fix, regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def has_indic(series):
    return series.fillna("").astype(str).str.contains("[ऀ-෿]", regex=True)


# ---------------------------------------------------------------- ground truth / metric
def load_gt_pairs(gt_path):
    gt = read_tsv(gt_path)
    lists = gt["matched_entity_ids"].str.split(",")
    pairs = gt[["source1_entity_id"]].assign(other_id=lists).explode("other_id")
    pairs = pairs[pairs["other_id"].fillna("").str.len() > 0]
    return gt, pairs.rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "other_id"]].reset_index(drop=True)


def f05_from_counts(tp, npred, ngt):
    """Vectorised per-S1 F0.5 with the leaderboard's singleton rule."""
    tp, npred, ngt = (np.asarray(x, dtype=np.float64) for x in (tp, npred, ngt))
    p = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    r = np.divide(tp, ngt, out=np.zeros_like(tp), where=ngt > 0)
    f = np.divide(1.25 * p * r, 0.25 * p + r, out=np.zeros_like(tp), where=(0.25 * p + r) > 0)
    f[(npred == 0) & (ngt == 0)] = 1.0
    prec = np.where(npred > 0, p, np.where(ngt == 0, 1.0, 0.0))
    rec = np.where(ngt > 0, r, np.where(npred == 0, 1.0, 0.0))
    return f, prec, rec


def assign_best(rec_key, prob):
    """Index of the best row per record (one-to-one: each S2/S3 record keeps one S1)."""
    order = np.lexsort((-prob, rec_key))
    rk = rec_key[order]
    first = np.ones(len(rk), bool)
    first[1:] = rk[1:] != rk[:-1]
    return order[first]
