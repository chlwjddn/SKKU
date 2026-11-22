"""
Classic item-item baselines on the HW2 setup.
  --model itemknn : cosine item-item similarity (high-rated co-occurrence),
                    shrinkage, top-k truncation per item; S = X_w @ sim
  --model slim    : SLIM (Ning & Karypis, ICDM 2011) via per-item ElasticNet
                    (coordinate descent on GPU, W >= 0, diag 0); S = X_w @ W
Input rows X_w: high-rated (>= 4), optional time weighting (same as EASE v2).
Usage:
  python classic.py --data ~/RS/data --model itemknn --tag iknn
  python classic.py --data ~/RS/data --model slim --tag slim
"""
import argparse, os, time
import numpy as np
import pandas as pd
import torch
from ease_v2 import temporal_split, dense, ndcg, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--thr", type=float, default=4.0)
ap.add_argument("--model", choices=["itemknn", "slim"], default="itemknn")
ap.add_argument("--topk", type=int, default=50, help="itemknn: neighbors kept per item (0 = all)")
ap.add_argument("--shrink", type=float, default=10.0, help="itemknn: shrinkage")
ap.add_argument("--l1", type=float, default=1e-4, help="slim: L1")
ap.add_argument("--l2", type=float, default=1e-2, help="slim: L2")
ap.add_argument("--iters", type=int, default=60, help="slim: coordinate-descent sweeps")
ap.add_argument("--time_beta", type=float, default=1.0, help="input recency weight (0 = off)")
ap.add_argument("--tag", default="iknn")
args = ap.parse_args()
THR = args.thr


def inputs(hist):
    h = hist.copy()
    n = h.groupby("userId")["time_idx"].transform("size")
    h["rel"] = (h.groupby("userId")["time_idx"].rank(method="first") - 1) / (n - 1).clip(lower=1)
    hi = h[h.rating >= THR]
    w = np.exp(-args.time_beta * (1 - hi.rel.values)) if args.time_beta > 0 else np.ones(len(hi))
    Xw = dense(hi, n_users, w)                      # weighted input for scoring
    Xb = dense(hi, n_users, np.ones(len(hi)))       # binary for similarity / SLIM fit
    seen = dense(h, n_users, np.ones(len(h))) > 0
    return Xw, Xb, seen


def itemknn(Xb):
    C = Xb.T @ Xb                                            # co-occurrence
    d = torch.sqrt(torch.diag(C)).clamp(min=1e-8)
    sim = C / (d[:, None] * d[None, :] + args.shrink)
    sim.fill_diagonal_(0)
    if args.topk > 0:
        kth = torch.topk(sim, args.topk, dim=1).values[:, -1:]
        sim = sim * (sim >= kth)
    return sim


def slim(Xb):
    """Per-item ElasticNet via synchronous coordinate descent on GPU.
    W: (N, N), column j = weights reconstructing item j from the others."""
    X = Xb
    G = X.T @ X                                              # Gram (N, N)
    diagG = torch.diag(G).clone()
    W = torch.zeros(N_ITEMS, N_ITEMS, device=dev)
    denom = diagG + args.l2
    for it in range(args.iters):
        # residual-based update for all (k, j) at once:  r_kj = G_kj - (G W)_kj + G_kk W_kj
        GW = G @ W
        num = G - GW + diagG[:, None] * W
        num.fill_diagonal_(0)
        Wn = torch.clamp(num - args.l1, min=0) / denom[:, None]
        Wn.fill_diagonal_(0)
        delta = (Wn - W).abs().max().item()
        W = Wn
        if delta < 1e-5:
            break
    print(f"  slim converged at sweep {it+1} (delta {delta:.2e}, nnz/col {(W > 0).float().sum(0).mean():.0f})",
          flush=True)
    return W


def run(hist):
    Xw, Xb, seen = inputs(hist)
    M = itemknn(Xb) if args.model == "itemknn" else slim(Xb)
    return Xw @ M, seen


d = os.path.expanduser(args.data)
tr = pd.read_csv(os.path.join(d, "train.csv")); te = pd.read_csv(os.path.join(d, "test.csv"))
n_users = int(tr.userId.max()) + 1
os.makedirs("scores", exist_ok=True)
print("device:", dev, "|", vars(args), flush=True)

inner, val = temporal_split(tr)
gt_df = val[val.rating >= THR]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
t = time.time()
S, seen_in = run(inner)
torch.save(S.cpu(), f"scores/{args.tag}_val.pt")
Sm = S.clone(); Sm[seen_in] = -float("inf")
print(f"[{args.model}] val NDCG@20 {ndcg(torch.topk(Sm[vusers], K, 1).indices, vusers, gt_mat):.5f}"
      f"  ({time.time()-t:.0f}s)")

S, seen_full = run(tr)
torch.save(S.cpu(), f"scores/{args.tag}_test.pt")
tusers = torch.as_tensor(te.userId.values, device=dev)
St = S[tusers].clone(); St[seen_full[tusers]] = -float("inf")
recs = torch.topk(St, K, 1).indices.cpu().numpy()
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(f"submission_{args.tag}.csv", index=False)
print(f"saved scores/{args.tag}_{{val,test}}.pt, submission_{args.tag}.csv")
