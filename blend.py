"""
Blend saved score matrices (scores/{tag}_val.pt, scores/{tag}_test.pt).
Each matrix is z-normalized per user (seen items excluded), weights found by
coordinate ascent on validation NDCG@20, then applied to test scores.
Usage: python blend.py --data ~/RS/data --tags ease,trans,lgcn_d64_l3 --out submission_blend.csv
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
from ease_v2 import temporal_split, dense, ndcg, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--tags", required=True)
ap.add_argument("--thr", type=float, default=4.0)
ap.add_argument("--out", default="submission_blend.csv")
ap.add_argument("--save_tag", default="", help="save blended scores as scores/{save_tag}_{val,test}.pt")
ap.add_argument("--norm", choices=["z", "minmax"], default="z",
                help="per-user normalization before the weighted sum: z-score or 0..1 min-max")
args = ap.parse_args()
tags = args.tags.split(",")

d = os.path.expanduser(args.data)
tr = pd.read_csv(os.path.join(d, "train.csv"))
te = pd.read_csv(os.path.join(d, "test.csv"))
n_users = int(tr.userId.max()) + 1
inner, val = temporal_split(tr)
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
seen_v = dense(inner, n_users, np.ones(len(inner))) > 0
seen_t = dense(tr, n_users, np.ones(len(tr))) > 0


def z(S, seen):
    S = S.to(dev).float().clone()
    if args.norm == "minmax":
        S[seen] = float("nan")
        lo = torch.nanquantile(S, 0.0, dim=1, keepdim=True)
        hi = torch.nanquantile(S, 1.0, dim=1, keepdim=True)
        S = (S - lo) / (hi - lo + 1e-8)
        S[seen.to(S.device)] = 0
        return S
    S[seen] = 0
    return (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-8)


Zv = [z(torch.load(f"scores/{t}_val.pt"), seen_v) for t in tags]
Zt = [z(torch.load(f"scores/{t}_test.pt"), seen_t) for t in tags]


def ev(ws):
    S = sum(w * Z for w, Z in zip(ws, Zv))[vusers]
    S[seen_v[vusers]] = -float("inf")
    return ndcg(torch.topk(S, K, 1).indices, vusers, gt_mat)


for i, t in enumerate(tags):
    print(f"{t:20s} alone: {ev([1.0 if j == i else 0.0 for j in range(len(tags))]):.5f}")

grid = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
ws = [1.0] + [0.0] * (len(tags) - 1)
best = ev(ws)
for p in range(3):
    for j in range(len(tags)):
        for g in grid:
            cand = ws.copy(); cand[j] = g
            if sum(cand) == 0:
                continue
            s = ev(cand)
            if s > best + 1e-6:
                best, ws = s, cand
    print(f"pass {p+1}: val {best:.5f}  weights {dict(zip(tags, ws))}")

if args.save_tag:
    torch.save(sum(w * Z for w, Z in zip(ws, Zv)).cpu(), f"scores/{args.save_tag}_val.pt")
S = sum(w * Z for w, Z in zip(ws, Zt))
if args.save_tag:
    torch.save(S.cpu(), f"scores/{args.save_tag}_test.pt")
tusers = torch.as_tensor(te.userId.values, device=dev)
St = S[tusers].clone()
St[seen_t[tusers]] = -float("inf")
recs = torch.topk(St, K, 1).indices.cpu().numpy()
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(args.out, index=False)
print(f"saved {args.out}  (val {best:.5f})")