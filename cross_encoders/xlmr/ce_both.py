"""B3b: run xlm-roberta-base (GPU 0) and xlm-roberta-large (GPU 1) as two independent processes in ONE kernel.
Base trains and scores on GPU 0 while large trains on GPU 1; once both are done, large is scored on both GPUs."""
import subprocess
import sys

TF = sys.argv[1] if len(sys.argv) > 1 else 'train_ce.parquet'
SUF = sys.argv[2] if len(sys.argv) > 2 else ''

env = "PYTHONPATH=src PYTHONUTF8=1"
base = subprocess.Popen(
    f"python src/ce_run.py --tag xlmr{SUF} --train_file {TF} --model xlm-roberta-base --phase train,score,check --train_gpu 0 --score_gpus 0 "
    "--lr 2e-5 --bs 32 --accum 1 --warmup 0.04 --max_hours 2.5", shell=True)
large = subprocess.Popen(
    f"python src/ce_run.py --tag xlmrL{SUF} --train_file {TF} --model xlm-roberta-large --phase train --train_gpu 1 "
    "--lr 1e-5 --bs 16 --accum 2 --warmup 0.06 --max_hours 5 --check_div 2000 --retry_lr 7e-6", shell=True)
rc_l = large.wait()
rc_b = base.wait()
print("base rc", rc_b, "large-train rc", rc_l, flush=True)
if rc_l == 0:
    rc = subprocess.call(f"python src/ce_run.py --tag xlmrL{SUF} --train_file {TF} --model xlm-roberta-large --phase score,check --score_gpus 0,1",
                         shell=True)
    print("large score rc", rc, flush=True)
    sys.exit(rc_b or rc)
sys.exit(rc_b or rc_l)
