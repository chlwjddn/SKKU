"""
Saves score matrices for blending:
  scores/ease_{val,test}.pt   EASE, high-rated input (alpha=0), lam=100, recency weight beta=1 at scoring
  scores/trans_{val,test}.pt  transition model (b follows a within w=10 steps, high-rated sequence)
val = trained on inner split, test = trained on full train.
Usage: python base_scores.py --data ~/RS/data
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
from ease_v2 import temporal_split, dense, ease_fit, N_ITEMS, dev

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--thr", type=float, default=4.0)
ap.add_argument("--lam", type=float, default=100)
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--w", type=int, default=10)
args = ap.parse_args()


def hist_scores(hist, n_users):
    hist = hist.sort_values(["userId", "time_idx"]).reset_index(drop=True)
    n = hist.groupby("userId")["movieId"].transform("size")
    rel = (hist.groupby("userId").cumcount() / (n - 1).clip(lower=1)).values
    m = (hist.rating >= args.thr).values
    hi = hist[m].reset_index(drop=True)
    X = dense(hi, n_users, np.ones(len(hi)))
    Xw = dense(hi, n_users, np.exp(-args.beta * (1 - rel[m])))
    ease = Xw @ ease_fit(X, X, args.lam)

    it = torch.as_tensor(hi.movieId.values, device=dev)
    us = torch.as_tensor(hi.userId.values, device=dev)
    cnt = torch.zeros(N_ITEMS * N_ITEMS, device=dev)
    for k in range(1, args.w + 1):
        same = us[:-k] == us[k:]
        cnt += torch.bincount(it[:-k][same] * N_ITEMS + it[k:][same], minlength=N_ITEMS * N_ITEMS).float()
    T = cnt.view(N_ITEMS, N_ITEMS)
    T = T / (T.sum(1, keepdim=True) + 1.0)
    trans = X @ T
    return ease.cpu(), trans.cpu()


tr = pd.read_csv(os.path.join(os.path.expanduser(args.data), "train.csv"))
n_users = int(tr.userId.max()) + 1
inner, _ = temporal_split(tr)
os.makedirs("scores", exist_ok=True)
for split, hist in [("val", inner), ("test", tr)]:
    e, t = hist_scores(hist, n_users)
    torch.save(e, f"scores/ease_{split}.pt")
    torch.save(t, f"scores/trans_{split}.pt")
    print(f"saved scores/ease_{split}.pt, scores/trans_{split}.pt")
