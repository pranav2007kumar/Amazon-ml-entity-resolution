"""Verify CPU/RAM/GPU and that XGBoost + PyTorch can actually use the RTX 5050."""
import os
import platform
import sys

print("python   :", sys.version.split()[0], platform.platform())
print("cpu cores:", os.cpu_count())
try:
    import psutil
    print(f"ram      : {psutil.virtual_memory().total / 2**30:.1f} GiB total, "
          f"{psutil.virtual_memory().available / 2**30:.1f} GiB free")
except ImportError:
    print("ram      : install psutil to see RAM")

try:
    import torch
    print("torch    :", torch.__version__, "| cuda build:", torch.version.cuda)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu      : {p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}")
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        print("torch gpu matmul ok:", float((x @ x).float().mean()) == float((x @ x).float().mean()))
    else:
        print("torch    : CUDA NOT available -> reinstall torch from the cu128 index")
except Exception as e:
    print("torch    : ERROR", repr(e))

try:
    import numpy as np
    import xgboost as xgb
    X = np.random.rand(20000, 20).astype("float32")
    y = (X[:, 0] + X[:, 1] > 1).astype("int32")
    m = xgb.XGBClassifier(n_estimators=20, tree_method="hist", device="cuda")
    m.fit(X, y)
    print("xgboost  :", xgb.__version__, "| GPU training ok")
except Exception as e:
    print("xgboost  : GPU training FAILED ->", repr(e))
