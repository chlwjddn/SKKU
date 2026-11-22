"""
BiRank (He et al., TKDE 2017) for personalized recommendation on the HW2 setup.
  W: user-item high-rated (>= thr) matrix, optionally weighted by rating x recency
  S = Du^-1/2 W Dp^-1/2
  per user (all users at once, matrix form):
    P <- alpha * S^T U + (1-alpha) * P0 ;  U <- beta * S P + (1-beta) * U0
  P0 = user's own weighted row (query), U0 = one-hot(user)
  iterate to convergence -> item scores P
Grid over alpha/beta, evaluates NDCG@20 on the temporal validation split, saves scores + submission.
Put next to ease_v2.py.
Usage: python birank.py --data ~/RS/data --tag birank
"""
import argparse, os, time
import numpy as np
import pandas as pd
import torch
from ease_v2 import temporal_split, dense, ndcg, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--thr", type=float, default=4.0)
ap.add_argument("--alphas", default="0.8,0.85,0.9,0.95")
ap.add_argument("--betas", default="0.9,1.0")
ap.add_argument("--rating_w", default="4:0.7,5:1", help='edge/query weights; "" for binary')
ap.add_argument("--time_beta", type=float, default=1.25, help="recency weight on edges/query; 0 = off")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--tag", default="birank")
args = ap.parse_args()


def map_w(spec, r):
    kv = sorted((float(a), float(b)) for a, b in (p.split(":") for p in spec.split(",")))
    keys, vals = np.array([k for k, _ in kv]), np.array([v for _, v in kv])
    return vals[np.clip(np.searchsorted(keys, r, side="right") - 1, 0, len(keys) - 1)]


def weighted_mat(hist):
    h = hist.copy()
    n = h.groupby("userId")["time_idx"].transform("size")
    h["rel"] = (h.groupby("userId")["time_idx"].rank(method="first") - 1) / (n - 1).clip(lower=1)
    hi = h[h.rating >= args.thr]
    w = map_w(args.rating_w, hi.rating.values) if args.rating_w else np.ones(len(hi))
    if args.time_beta:
        w = w * np.exp(-args.time_beta * (1 - hi.rel.values))
    W = dense(hi, n_users, w)
    seen = dense(h, n_users, np.ones(len(h))) > 0
    return W, seen


def birank_scores(W, alpha, beta):
    du = W.sum(1); dp = W.sum(0)
    Su = torch.where(du > 0, du.pow(-0.5), torch.zeros_like(du))
    Sp = torch.where(dp > 0, dp.pow(-0.5), torch.zeros_like(dp))
    S = Su[:, None] * W * Sp[None, :]                 # (n_users, N_ITEMS)
    P0 = torch.nn.functional.normalize(W, p=1, dim=1)  # L1-normalised query (paper: L1 norm)
    U0 = torch.eye(n_users, device=dev)
    P = P0.clone(); U = U0.clone()
    for _ in range(args.iters):
        U = beta * P @ S.T + (1 - beta) * U0          # rows = target users
        P = alpha * U @ S + (1 - alpha) * P0
    return P


d = os.path.expanduser(args.data)
tr = pd.read_csv(os.path.join(d, "train.csv")); te = pd.read_csv(os.path.join(d, "test.csv"))
n_users = int(tr.userId.max()) + 1
os.makedirs("scores", exist_ok=True)

inner, val = temporal_split(tr)
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
Wv, seen_v = weighted_mat(inner)

best = (-1, None)
for a in map(float, args.alphas.split(",")):
    for b in map(float, args.betas.split(",")):
        t = time.time()
        S = birank_scores(Wv, a, b)
        Sm = S[vusers].clone(); Sm[seen_v[vusers]] = -float("inf")
        sc = ndcg(torch.topk(Sm, K, 1).indices, vusers, gt_mat)
        print(f"alpha {a:.2f} beta {b:.2f}  val NDCG@20 {sc:.5f}  ({time.time()-t:.0f}s)", flush=True)
        if sc > best[0]:
            best = (sc, (a, b), S)
sc, (a, b), Sv = best
print(f"best: alpha {a} beta {b}  val {sc:.5f}")
torch.save(Sv.cpu(), f"scores/{args.tag}_val.pt")

Wt, seen_t = weighted_mat(tr)
St = birank_scores(Wt, a, b)
torch.save(St.cpu(), f"scores/{args.tag}_test.pt")
tusers = torch.as_tensor(te.userId.values, device=dev)
Sm = St[tusers].clone(); Sm[seen_t[tusers]] = -float("inf")
recs = torch.topk(Sm, K, 1).indices.cpu().numpy()
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(f"submission_{args.tag}.csv", index=False)
print(f"saved submission_{args.tag}.csv")
