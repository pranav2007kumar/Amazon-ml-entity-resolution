# XLM-RoBERTa cross-encoders (base 278M / large 560M, MIT)

Two cross-encoders re-score the uncertain candidate pairs; their scores are features of the final stacker (`src/stacker.py`).
Trained on Kaggle (2 × T4, fp16 autocast), training data only (validation S1s excluded), raw `"name | address"` text.

| file | role |
|---|---|
| `ce_build.py` | training pairs: all uncertain pairs of the first-stage model + a sample of confident ones |
| `ce_build2.py` | v2 training set: keeps every confident mistake, drops duplicates, adds generic label-preserving noise (acronym names, fake diacritics, `(NN)` numbers, street abbreviations) |
| `ce_train.py` | 1-epoch fine-tuning (fp32 weights + fp16 autocast, linear warm-up/decay, divergence guard) |
| `ce_score.py` | length-sorted batched scoring, one shard per GPU |
| `ce_run.py` | train / score / self-check phases |
| `ce_both.py` | runs base on GPU 0 and large on GPU 1 in one kernel |

Output: `<name>_val_scores.parquet`, `<name>_test_scores.parquet` with `source1_entity_id, candidate_entity_id, ce_score`.
