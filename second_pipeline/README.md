# Business Entity Resolution - local CPU+GPU pipeline (Scavengers)

One Windows machine (CPU + NVIDIA RTX 5050). No external data, no pretrained models.
Libraries: PyTorch (BSD), XGBoost (Apache-2.0), scikit-learn (BSD), rapidfuzz (MIT).

| # | Script | Does | Device |
|---|---|---|---|
| 1 | `prep.py` | normalise text (NFKC, accents, 9 Indian scripts, OCR digits), core/joined name, numbers, hashed bags; train sub-world | CPU |
| 2 | `block_sparse.py` | IDF token cosine incl. combined tokens, top-10 S1 per S2/S3 record, exact country-label blocking | CPU |
| 3 | `dense.py` | learned bi-encoder (char-3-gram + word hashing, contrastive, hard negatives), out-of-fold on train | GPU |
| 4 | `candidates.py` | union sparse top-k + dense top-k = the candidate set; prints the recall gate | CPU |
| 5 | `features.py` | ~70 pair features incl. learned similarity, context ranks, numbers, abbreviations, crowding | CPU |
| 6 | `train_xgb.py` | XGBoost GPU, 2-fold by S1, one-to-one assignment, F0.5 threshold, `--holdout` = unseen-country test | GPU |
| 7 | `stack.py` | competition-aware 2nd stage on OOF probabilities; used only if it wins on BOTH folds | GPU |
| 8 | `predict.py` | test scoring, both TSVs from one run | GPU |

Country is only an exact blocking label, never a feature; no rule mentions any country, so France is processed like the rest.
`make_sample.py` writes a 1% copy of the data for a smoke test of every stage.
