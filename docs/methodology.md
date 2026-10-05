# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

**Team:** #Scavengers  
**Date:** 27 September 2026

---

## 1. Executive Summary
We block candidates per country with a GPU dense kNN over character n-gram TF-IDF/SVD vectors, a rare-token IDF index, and
sound/initials keys. A two-stage GPU XGBoost scores the pairs; its uncertain band is re-scored by a fine-tuned multilingual
cross-encoder (mdeberta-v3-base, MIT). The final model is an XGBoost stacker over our model, an independently built
XGBoost pipeline (two disjoint training halves) and five cross-encoders. It is followed by a one-parent-per-record
assignment tuned for macro F0.5. For the unseen country (France), a label-free count calibration and a one-owner
renormalisation for co-located businesses are applied. Macro F0.5: **0.9926** on 87,790 held-out US/India S1s, and
**0.986367** on the leaderboard, with **7.9 candidate pairs per S1**.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Structure:** every S2/S3 record belongs to at most one S1 (one parent), and all true pairs share the `country` label.
  A S1 has 0–11 matches (mean 3.46, 5.6% singletons), and several matches can come from the same source.
- **Noise (train, US/India):** legal-suffix changes and reordering, filler words, typos and digit look-alikes, spurious
  accents, website/acronym names, names in 9 Indic scripts; address reordering, abbreviations, state name/code/native
  script, `NULL` tokens, leading zeros, dropped house numbers, ~3% empty addresses.
- **Train/test shift:** the test has more S2/S3 records per S1 (5.8 vs 4.7), i.e. more distractors.
- **France (test only):** label-free analysis showed that 16.7% of French S1s share their exact address with another S1
  (US 4.6%, India 5.0%), and French names come from a small generic vocabulary (city + type word + legal form). Co-located
  businesses therefore look alike, and "same address + similar name" is much weaker evidence than in US/India, where it is
  97–98% true.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier + cross-encoder re-ranking + stacking + constrained assignment (hybrid).  
**Core Innovation:** A learned-blocking cascade: cheap exact GPU retrieval with ~114 candidates/S1, then a stage-1 XGBoost
filter, then an expensive stacker on only 7.9 pairs/S1. Precision-first decoding enforces one parent per record and uses
thresholds per record group, tuned directly on macro F0.5. For the unseen country, open-set rules learned without labels
are applied (no country name in the code).

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:**
  1. dense: name char 2–4-gram TF-IDF → SVD, address TF-IDF → SVD, hashed number tokens; exact top-k cosine on the GPU
     within each country;
  2. sparse: an inverted index over rare name and address tokens, bigrams and glued names (IDF-weighted overlap, very
     frequent tokens dropped);
  3. keys: phonetic sound keys and name initials (acronyms), with a larger k for empty-address records.
  All vector and IDF fitting is label-free and done per country, so France gets its own vocabulary.
- **Candidate pairs generated:** 197.0M raw blocking pairs on test (113.7/S1). The stage-1/2 XGBoost acts as a learned
  filter, and only pairs with score ≥ 1e-3 (or ≥ 1e-4 in the second model) reach the final stacker. **Final
  `candidate_pairs.tsv`: 13,692,420 pairs = 7.9 per S1**, exactly the set the final model scored. Every predicted match is
  inside it (asserted in `final_candidates.py`).
- **How we ensured true matches were not lost:** recall was measured on held-out training S1s. The dense + sparse union
  reached 98.3% of true pairs in the dev study, and the independent second pipeline reaches 99.45% out-of-fold (0.90 for
  empty addresses). The union of both candidate sets is scored, so a pair found by either pipeline can be matched.

---

## 4. Matching Model

**Features used:**
- Name features: Levenshtein / token-sort / token-set / partial ratios, Jaro-Winkler, glued-name equality, token Jaccard,
  IDF-weighted overlap and rarity of shared tokens, legal-form handling, filler words learned per country without labels,
  acronym and domain flags, script flags, sound keys.
- Address features: ratios and token overlap, house-number and postcode agreement (numbers are language-independent),
  state/region agreement, empty-address flags.
- Other: competition features (rank and gap of the pair among the record's S1 candidates and among the S1's records,
  margins, candidate counts), record-level family features, dense/sparse retrieval scores, source (S2/S3).

**Model type:**
- Two-stage XGBoost (GPU `hist`), trained on 96% of the training S1s in two fold groups, with self-training decided by a
  leave-one-country-out simulation.
- mdeberta-v3-base cross-encoder (MIT, 278M; embeddings frozen, bf16 autocast), fine-tuned on hard training pairs and
  applied to the XGBoost uncertain band (0.002 < p < 0.998), blended by logistic regression.
- **Final model:** an XGBoost stacker (depth 5) over our blended score, an independent XGBoost pipeline (`second_pipeline/`,
  trained on two disjoint halves of the training S1s, whose out-of-fold scores cover our validation S1s), five further
  cross-encoders (mdeberta and XLM-R base/large, all MIT) and competition features. It is trained 2-fold out-of-fold on the
  held-out S1s.

**Threshold selection method:** Each S2/S3 record keeps only its best S1 (one parent). Four group thresholds
(source × empty address) are grid-searched to maximise macro F0.5 (singletons included) on held-out S1s. For countries
unseen in training (open set, not named in code):
- the thresholds are raised until the mean matches per S1 is slightly below the seen countries' mean. France's empty
  share then equals the training singleton rate of 5.6%;
- records with several strong competing S1s (co-located businesses) are decided with one-owner renormalised
  probabilities, q = o / (1 + Σ o).

---

## 5. Results & Error Analysis

| Stage | Validation macro F0.5 (held-out US/India S1s) | Leaderboard |
|---|---|---|
| v6 two-stage XGBoost | 0.9822 | 0.9735 |
| + cross-encoder m1 | 0.9869 | 0.9805 |
| + cross-encoder m2, wider band | 0.9879 | 0.9820 |
| + stack with the independent XGBoost | 0.9925 | 0.9851 |
| + unseen-country calibration | = | 0.9853 |
| + second model half + 5 cross-encoders | 0.99265 | 0.98627 |
| + stricter unseen-country calibration | = | 0.98634 |
| **+ one-owner renormalisation (final)** | 0.99262 (same rule on validation) | **0.986367** |

- **F_0.5 Score (macro):** 0.9926 validation (US/India); 0.986367 leaderboard. Diagnostic uploads without France imply
  US/India ≈ validation and France ≈ 0.95 on the leaderboard.
- **Common false positives (wrong merges):**
  - co-located businesses: a different business at the same address whose name shares a generic or city word
    ("Lille Foyer SAS" vs "Lille Compagnie SAS"). This is the dominant French error;
  - chain branches with identical names;
  - empty-address records next to several S1s with the same name.
- **Common false negatives (missed matches):**
  - copies whose name was replaced by an acronym or a web domain and whose address was dropped;
  - heavily transliterated names with an empty address;
  - French copies with a swapped generic word, where the model cannot tell them from co-located decoys.
- **Tried and rejected on validation or the leaderboard:**
  - French-noise self-training of the cross-encoders;
  - cross-encoders deciding all French pairs;
  - rescue rules for same-address pairs;
  - sibling-record features;
  - candidate-set union rules.

---

## 6. Conclusion
The biggest gains came from model diversity: a cross-encoder on top of the XGBoost, and stacking with an independently
built pipeline trained on disjoint data (+0.5 on the leaderboard). The unseen country is the remaining gap. Its dense
co-located businesses break the "same address ⇒ same business" regularity learned on US/India. Label-free, open-set
decision rules (count calibration, one-owner renormalisation) helped, while training on pseudo-labelled French text did
not. A learned blocking cascade keeps the final candidate set at 7.9 pairs per S1.

---

## Appendix

### A. Code Artefacts
- **Pipeline (`src/`):** `preprocess.py` → `stats.py` → `embed.py` → `blocking.py` + `blocking_sparse.py` →
  `build_features.py` / `rec_feats.py` → `two_stage.py` (fit, predict) → `ce.py` (train, score, write) → `stacker.py`
  (val, test) → `ce.py write` → `unseen_calib.py` → `final_candidates.py` → `unseen_renorm.py`.
- **Second pipeline:** `second_pipeline/`.
- **Cross-encoders:** `cross_encoders/`.

`README.md` lists every command, environment variable and timing, and `requirements.txt` pins the versions.

### B. Additional Results
- Candidate set: 197.0M raw → 13.69M final pairs (7.9 per S1); 393 S1s have no candidates (all predicted singletons).
- Unseen-country statistics (label-free): matches per S1 3.33 (seen countries 3.39); empty share 5.6% (training
  singleton rate 5.6%).

**Compliance:**
- Only the provided dataset is used: no external data, APIs, geocoding or lookups.
- Pretrained models: mdeberta-v3-base and xlm-roberta-base/large, all MIT, ≤ 560M parameters.
- Classifiers: XGBoost (Apache 2.0).
- No GPL code; the transliterator is our own.
- Country is treated as an open set.
