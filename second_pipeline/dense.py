"""Stage 2b - learned dense retriever (bi-encoder), trained from scratch on train pairs, GPU.

Per field (name, address): hashed char-3-grams + words -> EmbeddingBag -> residual MLP -> unit vector.
score = w_name * cos_name + w_addr * cos_addr (w learned; address ignored when empty).
InfoNCE loss, in-batch negatives from the same country + the record's best wrong sparse candidate.

It learns which spellings the data generator treats as the same (transliteration variants,
abbreviations, OCR digits, legal-suffix moves) from 7.6M examples instead of fixed IDF weights.

Leak-free: model_f trains only on pairs whose S1 is in fold f; train records owned by the other fold
are scored by it, so all train dense scores are out-of-fold. Test uses both models (score = mean).

  python dense.py train    --work W --fold 0 --out W/models/dense_f0.pt
  python dense.py train    --work W --fold 1 --out W/models/dense_f1.pt
  python dense.py retrieve --work W --split train --models W/models/dense_f0.pt W/models/dense_f1.pt
  python dense.py retrieve --work W --split test  --models W/models/dense_f0.pt W/models/dense_f1.pt
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import log, stable_fold, wdir

N_BUCKETS = (1 << 20) + (1 << 18)
FIELDS = ("name", "addr")


class Bags:
    def __init__(self, work, split, k):
        d = os.path.join(work, split, f"s{k}")
        self.meta = pd.read_parquet(os.path.join(d, "text.parquet"), columns=["entity_id", "country"])
        self.ip = {f: np.load(os.path.join(d, f"{f}_indptr.npy"), mmap_mode="r") for f in FIELDS}
        self.ix = {f: np.load(os.path.join(d, f"{f}_indices.npy"), mmap_mode="r") for f in FIELDS}
        self.n = len(self.meta)

    def gather(self, rows, f):
        ip, ix = self.ip[f], self.ix[f]
        rows = np.asarray(rows, dtype=np.int64)
        s, e = np.asarray(ip[rows]), np.asarray(ip[rows + 1])
        ln = (e - s).astype(np.int64)
        off = np.zeros(len(rows), np.int64)
        if len(rows) > 1:
            np.cumsum(ln[:-1], out=off[1:])
        pos = np.arange(int(ln.sum()), dtype=np.int64) - np.repeat(off, ln) + np.repeat(s, ln)
        return np.asarray(ix[pos], dtype=np.int64), off, ln

    def span(self, lo, hi, f):
        ip, ix = self.ip[f], self.ix[f]
        s, e = int(ip[lo]), int(ip[hi])
        p = np.asarray(ip[lo:hi + 1])
        return np.asarray(ix[s:e], dtype=np.int64), (p[:-1] - s).astype(np.int64), np.diff(p)


def build_model(d):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.EmbeddingBag(N_BUCKETS, d, mode="mean", sparse=True)
            nn.init.normal_(self.emb.weight, std=0.1)
            self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(self, idx, off):
            h = self.emb(idx, off)
            return F.normalize(h + self.ff(h), dim=-1)

    class BiEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.name, self.addr = Tower(), Tower()
            self.w_logit = nn.Parameter(torch.zeros(2))
            self.log_t = nn.Parameter(torch.tensor(np.log(0.05), dtype=torch.float32))
            self.d = d

        def weights(self):
            return torch.softmax(self.w_logit, 0)

        def encode(self, bags):
            n = self.name(bags["name"][0], bags["name"][1])
            m = (bags["addr"][2] > 0).float().unsqueeze(1)
            return torch.cat([n, self.addr(bags["addr"][0], bags["addr"][1]) * m], 1)

        def fold_query(self, q):
            w = self.weights()
            return torch.cat([q[:, :self.d] * w[0], q[:, self.d:] * w[1]], 1)

    return BiEncoder()


def to_t(g, dev):
    import torch
    return tuple(torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True) for x in g)


def gather_bags(b, rows, dev):
    return {f: to_t(b.gather(rows, f), dev) for f in FIELDS}


def get_device():
    import torch
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    log("WARNING: CUDA not available -> CPU (very slow). Fix torch install first.")
    return torch.device("cpu")


def s1_folds(work):
    ids = pd.read_parquet(os.path.join(work, "train", "s1", "text.parquet"), columns=["entity_id"])["entity_id"].values
    return stable_fold(ids)


# ============================================================ train
def cmd_train(a):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = get_device()
    B = {k: Bags(a.work, "train", k) for k in (1, 2, 3)}
    f1 = s1_folds(a.work)
    parts = []
    for k in (2, 3):
        g = np.load(os.path.join(a.work, "train", f"gt_s{k}.npz"))
        m = f1[g["s1"]] == a.fold
        parts.append(pd.DataFrame({"src": k, "ro": g["rec"][m], "r1": g["s1"][m]}))
    pairs = pd.concat(parts, ignore_index=True)
    pairs["country"] = B[1].meta["country"].values[pairs["r1"].values]
    hard = {}
    for k in (2, 3):   # best wrong sparse candidate per record = hard negative
        p = os.path.join(a.work, "train", f"sparse_s{k}.npz")
        if os.path.exists(p):
            s = np.load(p)
            owner = np.full(B[k].n, -1, np.int64)
            gk = np.load(os.path.join(a.work, "train", f"gt_s{k}.npz"))
            owner[gk["rec"]] = gk["s1"]
            wrong = s["s1"] != owner[s["rec"]]
            df = pd.DataFrame({"rec": s["rec"][wrong], "s1": s["s1"][wrong], "r": s["s_rank"][wrong]})
            df = df.sort_values(["rec", "r"]).drop_duplicates("rec")
            hard[k] = pd.Series(df["s1"].values, index=df["rec"].values)
    if len(pairs) > a.max_pairs:
        pairs = pairs.sample(a.max_pairs, random_state=a.seed)
    pairs = pairs.reset_index(drop=True)
    val = pairs.sample(min(20000, len(pairs) // 20), random_state=a.seed + 1)
    tr = pairs.drop(val.index)
    log(f"fold {a.fold}: {len(tr):,} train pairs, {len(val):,} val pairs, hard negatives: {bool(hard)}")

    model = build_model(a.dim).to(dev)
    opt_s = torch.optim.SparseAdam([model.name.emb.weight, model.addr.emb.weight], lr=a.lr_sparse)
    opt_d = torch.optim.AdamW([p for n, p in model.named_parameters() if "emb" not in n], lr=a.lr, weight_decay=1e-4)

    def batches(df, shuffle=True):
        out = []
        for _, g in df.groupby("country"):
            idx = rng.permutation(g.index.values) if shuffle else g.index.values
            out += [idx[i:i + a.batch] for i in range(0, len(idx), a.batch) if len(idx[i:i + a.batch]) > 32]
        return [out[i] for i in rng.permutation(len(out))] if shuffle else out

    def step(g, use_hard):
        order, embs = [], []
        for k in (2, 3):
            m = g["src"].values == k
            if m.any():
                embs.append(model.encode(gather_bags(B[k], g["ro"].values[m], dev)))
                order.append(np.where(m)[0])
        q = torch.cat(embs, 0)[torch.from_numpy(np.argsort(np.concatenate(order))).to(dev)]
        keys = g["r1"].values
        if use_hard and hard:
            hk = np.concatenate([hard[k].reindex(g["ro"].values[g["src"].values == k]).dropna().values
                                 for k in hard]).astype(np.int64)
            keys = np.concatenate([keys, hk])
        kv = model.encode(gather_bags(B[1], keys, dev))
        s = model.fold_query(q) @ kv.T / model.log_t.exp()
        n = len(g)
        same = torch.from_numpy(keys[None, :] == g["r1"].values[:, None]).to(dev)
        same[torch.arange(n), torch.arange(n)] = False
        s = s.masked_fill(same, -1e4)
        tgt = torch.arange(n, device=dev)
        loss = F.cross_entropy(s, tgt) + F.cross_entropy(s[:, :n].T, tgt)
        return loss, (s.argmax(1) == tgt).float().mean()

    for ep in range(a.epochs):
        model.train()
        bl = batches(tr)
        tot = acc = 0.0
        for i, b in enumerate(bl):
            loss, ac = step(tr.loc[b], True)
            opt_s.zero_grad(); opt_d.zero_grad(); loss.backward(); opt_s.step(); opt_d.step()
            tot += float(loss); acc += float(ac)
            if (i + 1) % 200 == 0:
                log(f"  ep{ep} step {i + 1}/{len(bl)} loss {tot / (i + 1):.4f} acc {acc / (i + 1):.4f}")
        model.eval()
        with torch.no_grad():
            vb = batches(val, shuffle=False)
            vacc = np.mean([float(step(val.loc[b], False)[1]) for b in vb]) if vb else float("nan")
        w = model.weights().detach().cpu().numpy()
        log(f"epoch {ep}: loss {tot / max(len(bl), 1):.4f} | val in-batch acc {vacc:.4f} | "
            f"w_name {w[0]:.2f} w_addr {w[1]:.2f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"state": model.state_dict(), "dim": a.dim, "fold": a.fold}, a.out)
    log(f"saved {a.out}")


# ============================================================ retrieve
def load_models(paths, dev):
    import torch
    ms = []
    for p in paths:
        ck = torch.load(p, map_location=dev)
        m = build_model(ck["dim"]).to(dev)
        m.load_state_dict(ck["state"]); m.eval(); m.fold = ck["fold"]
        ms.append(m)
    return ms


def encode_rows(m, b, rows, dev, chunk=65536):
    import torch
    rows = np.asarray(rows)
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), chunk):
            r = rows[i:i + chunk]
            if r[-1] - r[0] + 1 == len(r):
                g = {f: to_t(b.span(int(r[0]), int(r[-1]) + 1, f), dev) for f in FIELDS}
            else:
                g = gather_bags(b, r, dev)
            e = m.encode(g)
            out.append(e.half() if dev.type == "cuda" else e)
    return torch.cat(out, 0)


def cmd_retrieve(a):
    import torch
    dev = get_device()
    B = {k: Bags(a.work, a.split, k) for k in (1, 2, 3)}
    ms = load_models(a.models, dev)
    d = ms[0].d
    c1 = B[1].meta["country"].values

    if a.split == "train":
        f1 = s1_folds(a.work)
        groups = [([i], 1 - m.fold) for i, m in enumerate(ms)]
    else:
        groups = [(list(range(len(ms))), None)]

    for k in (2, 3):
        meta = B[k].meta
        cq = meta["country"].values
        sp = np.load(os.path.join(a.work, a.split, f"sparse_s{k}.npz"))
        s_order = np.argsort(sp["rec"], kind="stable")
        s_rec, s_s1 = sp["rec"][s_order], sp["s1"][s_order]
        s_out = np.zeros((len(s_rec), 3), np.float32)
        top = []
        true_rank = None
        if a.split == "train":
            g = np.load(os.path.join(a.work, "train", f"gt_s{k}.npz"))
            owner = np.full(B[k].n, -1, np.int64); owner[g["rec"]] = g["s1"]
            rfold = np.full(B[k].n, -1, np.int64); rfold[g["rec"]] = f1[g["s1"]]
            un = rfold < 0
            rfold[un] = stable_fold(meta["entity_id"].values[un])
            true_rank = np.full(B[k].n, -1, np.int16)
        for mids, qfold in groups:
            sub = [ms[i] for i in mids]
            M = len(sub)
            K = torch.cat([encode_rows(m, B[1], np.arange(B[1].n), dev) for m in sub], 1)
            W = [m.weights().detach() for m in sub]
            for c in pd.unique(cq):
                lo, hi = np.searchsorted(c1, c, "left"), np.searchsorted(c1, c, "right")
                qrows = np.arange(np.searchsorted(cq, c, "left"), np.searchsorted(cq, c, "right"))
                if qfold is not None:
                    qrows = qrows[rfold[qrows] == qfold]
                if hi <= lo or len(qrows) == 0:
                    log(f"  S{k} {c}: no S1 with this label, skipped"); continue
                Kc = K[lo:hi]
                kk = min(a.topk, hi - lo); ks = min(a.save_k, kk)
                for i in range(0, len(qrows), a.qchunk):
                    if (i // a.qchunk) % 400 == 0:
                        log(f"    S{k} {c}: {i:,}/{len(qrows):,} queries")
                    r = qrows[i:i + a.qchunk]
                    with torch.no_grad():
                        Q = torch.cat([encode_rows(m, B[k], r, dev, chunk=len(r)) for m in sub], 1)
                        Qw = torch.cat([torch.cat([Q[:, j * 2 * d:j * 2 * d + d] * W[j][0],
                                                   Q[:, j * 2 * d + d:(j + 1) * 2 * d] * W[j][1]], 1)
                                        for j in range(M)], 1).to(K.dtype) / M
                        val, idx = torch.topk(Qw @ Kc.T, kk, dim=1)
                        idx_np = idx.cpu().numpy().astype(np.int64) + lo
                        if true_rank is not None:
                            t = owner[r]
                            hit = idx_np == t[:, None]
                            true_rank[r] = np.where(t >= 0, np.where(hit.any(1), hit.argmax(1) + 1, 0), -1)
                        gk = K[torch.from_numpy(idx_np[:, :ks].reshape(-1)).to(dev)].float().view(len(r), ks, -1)
                        Qf = Q.float().unsqueeze(1)
                        cn = sum((Qf[..., j*2*d:j*2*d+d] * gk[..., j*2*d:j*2*d+d]).sum(-1) for j in range(M)) / M
                        ca = sum((Qf[..., j*2*d+d:(j+1)*2*d] * gk[..., j*2*d+d:(j+1)*2*d]).sum(-1) for j in range(M)) / M
                        top.append((np.repeat(r, ks).astype(np.int32), idx_np[:, :ks].reshape(-1).astype(np.int32),
                                    val[:, :ks].float().cpu().numpy().reshape(-1), cn.cpu().numpy().reshape(-1),
                                    ca.cpu().numpy().reshape(-1), np.tile(np.arange(1, ks + 1, dtype=np.int8), len(r))))
                        a0, a1 = np.searchsorted(s_rec, r[0], "left"), np.searchsorted(s_rec, r[-1], "right")
                        sel = a0 + np.where(np.isin(s_rec[a0:a1], r))[0]
                        if len(sel):
                            pos = torch.from_numpy(np.searchsorted(r, s_rec[sel])).to(dev)
                            kv = K[torch.from_numpy(s_s1[sel].astype(np.int64)).to(dev)].float()
                            q = Q[pos].float()
                            s_out[sel, 0] = (Qw[pos].float() * kv).sum(1).cpu().numpy()
                            s_out[sel, 1] = (sum((q[:, j*2*d:j*2*d+d] * kv[:, j*2*d:j*2*d+d]).sum(1) for j in range(M)) / M).cpu().numpy()
                            s_out[sel, 2] = (sum((q[:, j*2*d+d:(j+1)*2*d] * kv[:, j*2*d+d:(j+1)*2*d]).sum(1) for j in range(M)) / M).cpu().numpy()
                log(f"  S{k} {c}: {len(qrows):,} records retrieved (models {mids})")
            del K
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        cols = [np.concatenate([t[i] for t in top]) for i in range(6)]
        np.savez(os.path.join(wdir(a.work, a.split), f"dense_s{k}.npz"), rec=cols[0], s1=cols[1], d_cos=cols[2],
                 d_cos_name=cols[3], d_cos_addr=cols[4], d_rank=cols[5])
        inv = np.empty_like(s_order); inv[s_order] = np.arange(len(s_order))
        np.savez(os.path.join(wdir(a.work, a.split), f"sparse_dense_s{k}.npz"), d=s_out[inv])
        if true_rank is not None:
            np.save(os.path.join(wdir(a.work, a.split), f"dense_truerank_s{k}.npy"), true_rank)
        log(f"{a.split} S{k}: {len(cols[0]):,} dense candidates saved, {len(s_rec):,} sparse pairs scored")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--work", required=True); p.add_argument("--fold", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dim", type=int, default=128); p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4096); p.add_argument("--max-pairs", type=int, default=3_000_000)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--lr-sparse", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("retrieve")
    p.add_argument("--work", required=True); p.add_argument("--split", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--topk", type=int, default=20); p.add_argument("--save-k", type=int, default=5)
    p.add_argument("--qchunk", type=int, default=1024)
    a = ap.parse_args()
    {"train": cmd_train, "retrieve": cmd_retrieve}[a.cmd](a)


if __name__ == "__main__":
    main()
