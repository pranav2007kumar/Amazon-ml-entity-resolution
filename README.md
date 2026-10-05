# 🔗 Business Entity Resolution at Scale — Amazon ML Challenge 2026

**Team #Scavengers.** Matching 1.7 million businesses against 10 million noisy records across 3 sources, 3 countries and
10 scripts. One of the countries (France) was never seen in training.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU-EC6B23)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-mDeBERTa%20%7C%20XLM--R-FFD21E)
![Polars](https://img.shields.io/badge/Polars-1.44-CD792C)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

| Metric | Value |
|---|---|
| 🏆 Leaderboard macro F0.5 (final) | **0.986367** |
| ✅ Validation macro F0.5 (87,790 held-out businesses, US + India) | **0.9926** |
| 🎯 Candidate pairs per business in the final set | **7.9** (raw blocking: 113.7) |
| 🧮 Test scale | 1,732,544 S1 businesses × 10 M S2/S3 records |
| 💻 Hardware | consumer laptops (RTX 4060 / 4070 / 5050, 8 GB) + Kaggle 2×T4 |

---

## 🛑 The Problem

A company holds the same businesses in three independent sources. **Source 1** is the clean, deduplicated reference;
**Sources 2 and 3** hold noisy copies. For every Source-1 business we must list **all** of its copies, and nothing else.

| Challenge | Reality in the data |
|---|---|
| 📏 **Scale** | Comparing all pairs is 1.7 M × 10 M = 17 trillion pairs, so blocking is mandatory |
| 🔀 **Heavy noise** | legal-form swaps (`Pvt Ltd` ↔ `Private Limited`, `SAS` ↔ `SARL`), filler words (`Services`, `& Fils`, `Groupe`), typos and digit look-alikes (`R0cky`), acronyms (`Art College SARL` → `AC`), web-domain names (`lyceensecolesas.com`), reordered or abbreviated addresses (`Rue` → `R.`), dropped house numbers, empty addresses |
| 🔤 **10 scripts** | names in Latin plus 9 Indic scripts (Devanagari, Tamil, Telugu, Bengali, …) |
| 🪤 **Decoys** | ~40% of test records match no business, and many are **co-located businesses** (a different company at the same address) or chain branches |
| 🌍 **Unseen country** | training covers **US + India**; the test adds **France**, with no French labels at all |
| ⚖️ **Precision-first metric** | macro F0.5 per business (precision weighted 2× over recall), singletons included: one wrong merge costs more than one missed copy |

---

## ✅ Our Solution at a Glance

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        ENTITY RESOLUTION PIPELINE (per country)                  │
│                                                                                  │
│  Raw TSV (S1, S2, S3)                                                            │
│        │                                                                         │
│        ▼                                                                         │
│  ┌──────────────┐   ┌──────────────────────────┐   ┌───────────────────────────┐ │
│  │ Normalise    │──▶│ Blocking (GPU)           │──▶│ Features (~150)           │ │
│  │ transliterate│   │ dense kNN + rare tokens  │   │ string · number · context │ │
│  │ legal/filler │   │ + sound / initials keys  │   │ competition · record      │ │
│  └──────────────┘   │ ≈ 114 candidates / S1    │   └─────────────┬─────────────┘ │
│                     └──────────────────────────┘                 │               │
│        ┌──────────────────────────────────────────────────────────┘              │
│        ▼                                                                         │
│  ┌──────────────────┐   ┌──────────────────────────┐   ┌──────────────────────┐  │
│  │ Two-stage        │──▶│ Cross-encoder re-ranking │──▶│ STACKER (final model)│  │
│  │ XGBoost (GPU)    │   │ mDeBERTa on the uncertain│   │ ours ∪ 2nd XGBoost   │  │
│  │ learned filter   │   │ band 0.002 < p < 0.998   │   │ (2 halves) ∪ 5 CEs   │  │
│  └──────────────────┘   └──────────────────────────┘   │ ≈ 7.9 pairs / S1     │  │
│                                                        └──────────┬───────────┘  │
│        ┌──────────────────────────────────────────────────────────┘              │
│        ▼                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │ Decision: one parent per record · group thresholds tuned on macro F0.5   │    │
│  │ Unseen country: label-free count calibration + one-owner renormalisation │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│        │                                                                         │
│        ▼                                                                         │
│  matching_results.tsv  +  candidate_pairs.tsv                                    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏛️ How One Record Gets Matched

```
S2 record: "Lille Compagnie SAS | 35 R. BONTE POLLET, LILLE"
        │
        ▼
Normalise ──▶ name core "lille compagnie", legal "sas", street "rue bonte pollet", number 35, city lille
        │
        ▼
Blocking ──▶ 20 nearest S1 in France (dense) ∪ S1 sharing rare tokens (sparse) ∪ same sound key
        │
        ▼
XGBoost ──▶ p = 0.604 for S1 "Lille Foyer SAS | 35 Rue Bonte Pollet, Lille"   (uncertain band)
        │
        ▼
Cross-encoders ──▶ 6 transformer opinions on the text pair
        │
        ▼
Stacker ──▶ combines our model, the 2nd pipeline, the cross-encoders and competition features
        │
        ▼
Decision ──▶ best S1 for this record? above its group threshold? another strong S1 competing (co-located)?
        │
        └──▶ reject: probably a different business at the same address (a French decoy)
```

---

## 🧩 Pipeline Breakdown

### 1. 🧹 Normalisation (`preprocess.py`, `normalize.py`, `translit.py`, `stats.py`)
| Step | What it does |
|---|---|
| Transliteration | our own transliterator for 9 Indic scripts (no GPL code); vowel folding makes spellings agree (`baalaajii` → `balaji`) |
| Name split | name core vs legal form vs filler words; fillers are learned per country without labels (`fillers.py`) |
| Address | abbreviation canonicalisation, state/region extraction, numbers extracted with leading zeros stripped |
| Rarity | token document frequencies per country (open set: France gets its own vocabulary) |

### 2. 🔎 Blocking (`embed.py`, `blocking.py`, `blocking_sparse.py`)
| Channel | How |
|---|---|
| Dense | char 2–4-gram TF-IDF → SVD (name + address + hashed numbers), **exact** top-k cosine on the GPU within each country |
| Sparse | inverted index over rare name/address tokens, bigrams and glued names; IDF-weighted overlap |
| Keys | phonetic sound keys and name initials (acronyms); larger k for empty-address records |

### 3. 🧮 Features + Two-Stage XGBoost (`features.py`, `build_features.py`, `rec_feats.py`, `two_stage.py`)
- ~150 features:
  - string similarities (Levenshtein, Jaro-Winkler, token-set, glued names);
  - number and house-number agreement (language-independent);
  - **competition features**: the rank and gap of a pair among the record's S1 candidates and among the S1's records;
  - record-level "family" features.
- Stage 1 filters candidates, and stage 2 re-scores the survivors. Training uses **96% of the training businesses**, in two fold groups.
- **Leave-one-country-out simulation** (`loco.py`): train on the US and test on India, and the reverse. It decides whether
  self-training helps an unseen country.

### 4. 🤖 Cross-Encoder Re-ranking (`ce.py`)
- `microsoft/mdeberta-v3-base` (MIT, 278M), fine-tuned on hard training pairs. Word embeddings are frozen and bf16
  autocast is used, so it fits an 8 GB laptop GPU at ~186 pairs/s.
- It re-scores only the XGBoost's **uncertain band** (0.002 < p < 0.998) and is blended by logistic regression.

### 5. 🧱 Stacking (`stacker.py`)
The final model is an XGBoost stacker over:
- **our** blended score and raw XGBoost score;
- an **independent second pipeline** (`second_pipeline/`): a different blocking (learned bi-encoder) and features,
  trained on **two disjoint halves** of the training businesses (`prep.py --s1-side A|B`), so its scores on our
  validation set are out-of-fold;
- **five more cross-encoders** (`cross_encoders/`): mDeBERTa fine-tunes on hard pairs and French-style data, plus
  XLM-R base/large;
- competition features for both models.

It is trained 2-fold out-of-fold on the held-out businesses. Candidate set = the **13.7 M pairs the stacker scores (7.9 per business)**.

### 6. 🎯 Decision Layer (`two_stage.py`, `unseen_calib.py`, `unseen_renorm.py`, `final_candidates.py`)
| Rule | Why |
|---|---|
| One parent per record | each S2/S3 record belongs to at most one business |
| 4 group thresholds (source × empty address) | tuned directly on macro F0.5, singletons included |
| Unseen-country count calibration | France thresholds raised until its matches per business sit just below the seen countries' mean; France's "no match" share then equals the training singleton rate (5.6%) |
| One-owner renormalisation | q = o / (1 + Σ o) over a record's competing businesses: two strong co-located candidates no longer both look certain |

---

## 🇫🇷 The France Story (the unseen country)

The label-free analysis showed why France is the hard part:

| Fact | US | India | **France** |
|---|---|---|---|
| S1 businesses sharing their **exact address** with another S1 | 4.6% | 5.0% | **16.7%** |
| Records the models are unsure about (0.01 < p < 0.95) | 3.7% | 2.7% | **5.7%** |
| Labels available | ✅ | ✅ | ❌ |

- France has **dense co-located businesses** and a small, generic name vocabulary (city + type word + legal form).
- In US/India, "same address + similar name" is a real copy **97% of the time**. In France that rule breaks, and there
  are no French labels to learn the new one. What worked were **open-set, label-free decision rules**; the leaderboard
  confirmed each step:

| Change (France rows only) | Leaderboard |
|---|---|
| stack, no calibration | 0.985134 |
| count calibration (+0.16) | 0.985270 |
| + more models | 0.986321 |
| stricter calibration (+0.22) | 0.986338 |
| **+ one-owner renormalisation** | **0.986367** |

---

## 📈 Results Journey

| # | Version | What changed | Leaderboard F0.5 |
|---|---|---|---|
| 1 | v3 / v4 | first pipelines | 0.9600 / 0.9623 |
| 2 | v6 | two-stage XGBoost on 96% of train, self-training | 0.973456 |
| 3 | v6ce | + mDeBERTa cross-encoder | 0.980543 |
| 4 | v6ce2 | + longer training, wider band | 0.981992 |
| 5 | v6ce4a | + **stack with the independent pipeline** | 0.985134 |
| 6 | v6ce9 | + 2nd half of that pipeline + 4 more cross-encoders | 0.986274 |
| 7 | v6ce10 | + France-noise cross-encoder | 0.986321 |
| 8 | v6ce15 | + stricter unseen-country calibration | 0.986338 |
| 9 | **v6ce16** | + **one-owner renormalisation** + 7.9-pair candidate file | **0.986367** 🏆 |

### 🧪 What we tried that did *not* help (measured on validation or the leaderboard)
| Idea | Result |
|---|---|
| Cross-encoders deciding **all** French pairs | −0.0007 on the leaderboard: the stack is a better judge of France |
| Self-training cross-encoders on pseudo-labelled French text | +0.00005 only |
| Rescuing French same-address pairs | −0.0005: they were co-located decoys |
| LightGBM stacker / XGBoost + LightGBM | 0.99259 / 0.9926 on validation (no gain) |
| Sibling-record similarity features | 0.99242 on validation (no gain) |
| Load-balancing ambiguous records between businesses | 0.99259 (no gain) |
| Too-strict France (3.28 matches per business) | 0.986193 (worse) |

---

## 💻 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data | Polars 1.44 + PyArrow | fast columnar processing of 25 M records on 16 GB RAM |
| Retrieval | scikit-learn TF-IDF + SVD, PyTorch GPU matmul | exact dense kNN within country |
| String features | RapidFuzz | Levenshtein / Jaro-Winkler / token ratios |
| Classifiers | XGBoost 3.4 (GPU `hist`) | two-stage matcher + stacker |
| Transformers | 🤗 Transformers 5.17, PyTorch 2.11 (CUDA 12.8) | mDeBERTa-v3-base, XLM-R base/large cross-encoders |
| Second pipeline | EmbeddingBag bi-encoder + XGBoost (`second_pipeline/`) | diversity for the stack |
| Validation | macro F0.5 on held-out **whole** businesses, singletons included | the exact leaderboard metric |

All pretrained models are **MIT-licensed and ≤ 560M parameters**. No external data, APIs or lookups.

---

## 📁 Repository Structure

```
entity-resolution/
├── README.md                    ← you are here
├── LICENSE
├── requirements.txt             ← pinned versions
├── docs/
│   └── methodology.md           ← full methodology write-up
├── src/                         ← main pipeline (run from here)
│   ├── config.py                ← paths (ER_DATA_DIR, ER_WORK_DIR, ER_OUT_DIR)
│   ├── preprocess.py · normalize.py · translit.py · stats.py · fillers.py · ctry_norm.py   ← normalisation
│   ├── embed.py · blocking.py · blocking_sparse.py                                         ← blocking
│   ├── features.py · build_features.py · rec_feats.py · gen_feats.py · labels.py           ← features
│   ├── two_stage.py             ← two-stage XGBoost, assignment, threshold tuning
│   ├── loco.py · decide_loco.py ← leave-one-country-out simulation
│   ├── ce.py                    ← mDeBERTa cross-encoder: train / score / blend / write
│   ├── stacker.py               ← final stacker
│   └── unseen_calib.py · unseen_renorm.py · final_candidates.py   ← final decision + candidate file
├── second_pipeline/             ← independent pipeline: learned bi-encoder blocking + XGBoost (sub-worlds A/B)
└── cross_encoders/
    ├── ce_hard_pairs.py         ← mDeBERTa on hard training pairs
    ├── ce_france_noise.py       ← + French-style noise augmentation
    ├── ce_france_siblings.py    ← + exact-label French "sibling" pairs
    ├── ce_france_pseudo.py      ← France-adapted on confident test pairs
    ├── ce_france_hard.py        ← + hard French negatives
    └── xlmr/                    ← XLM-R base / large cross-encoders
```

---

## ⚙️ Quick Start

### Prerequisites
| Item | Requirement |
|---|---|
| OS | Windows / Linux |
| Python | 3.12 |
| GPU | NVIDIA, 8 GB+, CUDA 12.8 driver |
| RAM / disk | 16 GB / ~250 GB free for intermediate files |
| Data | the challenge dataset (`dataset/train`, `dataset/test`): not included in this repo |

### 1. Install
```bash
git clone https://github.com/<your-account>/entity-resolution.git
cd entity-resolution
python -m venv .venv && .venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2. Point to the data
```bash
set ER_DATA_DIR=C:\path\to\dataset          # Linux: export ER_DATA_DIR=/path/to/dataset
set ER_WORK_DIR=C:\path\to\work
set ER_OUT_DIR=C:\path\to\output
set ER_CE_MODEL=C:\path\to\models\mdeberta-v3-base   # microsoft/mdeberta-v3-base from Hugging Face
```

### 3. Run (from `src/`)
| # | Command | Time* |
|---|---|---|
| 1 | `python preprocess.py && python stats.py && python embed.py --refit` | ~20 min |
| 2 | `python blocking.py --splits train` · `python blocking_sparse.py --splits train` (then `--splits test`) | ~60 min |
| 3 | `python build_features.py --split train --train_end 0.96` · `--split test` · `python rec_feats.py --split train` / `test` | ~3.5 h |
| 4 | `python two_stage.py fit` · `python two_stage.py predict` | ~2 h |
| 5 | `python ce.py train` (m1), then m2 with `ER_CE_NAME=m2 ER_CE_INIT=<work>/ce/model ER_CE_LR=1e-5 ER_CE_LO=0.002 ER_CE_HI=0.998`: `python ce.py train && python ce.py score && python ce.py write` | ~4 h |
| 6 | `second_pipeline/` (run with `--s1-side A` and `--s1-side B`) + `cross_encoders/` | parallel GPUs |
| 7 | `python stacker.py val && python stacker.py test` (score files passed via `ER_CE_EXT`, `ER_M2_DIR`, `ER_M2B_DIR`) | ~20 min |
| 8 | `python ce.py write` → `unseen_calib.py` → `final_candidates.py` → `unseen_renorm.py` | ~15 min |

*RTX 4060 Laptop. `docs/methodology.md` has the full methodology.

---

## 🧠 Lessons Learned

1. **Diversity beats depth.** Stacking a second, independently built pipeline trained on disjoint data gave the biggest
   single jump (+0.3 on the leaderboard); more epochs of the same model did not.
2. **Validate on whole held-out businesses with the exact metric.** Pair-level metrics hid the singleton and one-parent effects.
3. **For an unseen domain, prefer label-free decision rules to pseudo-labels.** Training on our own French guesses
   re-learned our own mistakes; calibrating counts and resolving competing owners did not.
4. **Measure before you upload.** Every idea above was checked on 87,790 labelled businesses first, which saved
   uploads for the ones that mattered.

---

## 👥 Team

Built by **Team #Scavengers** for the Amazon ML Challenge 2026.

---

## 📄 License

MIT: see [LICENSE](LICENSE). Pretrained weights keep their own MIT licences (mDeBERTa-v3-base, XLM-RoBERTa).
The challenge dataset is **not** redistributed here.

<p align="center"><b>Built for the Amazon ML Challenge 2026</b><br>
<i>Match every copy, merge nothing wrong.</i></p>
