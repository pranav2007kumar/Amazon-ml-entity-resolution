"""Turn the unseen-country simulation (loco.py) into settings for unseen test countries.
Prints JSON: {"self_train": "0"|"1", "delta": <threshold shift>}.
  self-training is used only if it helped in BOTH directions (US->India and India->US) at the
  threshold a real unseen country would get (the one tuned on the known country);
  the threshold shift is used only if both directions want a shift of the same sign and it gains.
"""
import json
import os
import sys

from config import WORK_DIR

if __name__ == "__main__":
    path = os.path.join(WORK_DIR, "loco", "loco_results.json")
    if not os.path.exists(path):
        print(json.dumps({"self_train": "0", "delta": "0"}))
        sys.exit(0)
    r = json.load(open(path))
    gains = [v["self_train"]["f_at_source_thr"] - v["plain"]["f_at_source_thr"] for v in r.values()]
    st = all(g > 0 for g in gains)
    mode = "self_train" if st else "plain"
    shifts = [v[mode]["oracle_thr"] - v["plain"]["source_thr"] for v in r.values()]
    gain_thr = [v[mode]["f_oracle"] - v[mode]["f_at_source_thr"] for v in r.values()]
    same_sign = all(s > 0 for s in shifts) or all(s < 0 for s in shifts)
    delta = sum(shifts) / len(shifts) if same_sign and min(gain_thr) > 0.001 else 0.0
    delta = max(-0.1, min(0.15, delta))
    print(json.dumps({"self_train": "1" if st else "0", "delta": f"{delta:.3f}",
                      "detail": {"st_gains": gains, "shifts": shifts, "thr_gains": gain_thr}}))
