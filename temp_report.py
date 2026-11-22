"""
One-row-per-model summary on validation: NDCG@20, hit-rate by GT popularity bucket,
mean popularity of recommendations, coverage. Includes EASE as reference.
Usage: python temp_report.py --data ~/RS/data --tags t005,t01,t02
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
from ease_v2 import temporal_split, dense, ease_fit, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--tags", required=True)
ap.add_argument("--thr", type=float, default=4.0)
args = ap.parse_args()

tr = pd.read_csv(os.path.join(os.path.expanduser(args.data), "train.csv"))
n_users = int(tr.userId.max()) + 1
inner, val = temporal_split(tr)
hi = inner[inner.rating >= args.thr]
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
seen = dense(inner, n_users, np.ones(len(inner))) > 0
X = dense(hi, n_users, np.ones(len(hi)))
pop = X.sum(0)
gt_v = gt_mat[vusers]
gp = pop[None, :].expand_as(gt_v)[gt_v]
disc = 1.0 / torch.log2(torch.arange(2, K + 2, device=dev, dtype=torch.float32))
bins = [(5, 20, "hit 6-20"), (20, 50, "hit 21-50"), (50, 1e9, "hit 51+")]

models = {"EASE": X @ ease_fit(X, X, 100.0)}
for t in args.tags.split(","):
    p = f"scores/{t}_val.pt"
    if os.path.exists(p):
        models[t] = torch.load(p).to(dev).float()
    else:
        print(f"missing {p}")

rows = []
for name, S in models.items():
    S = S[vusers].clone()
    S[seen[vusers]] = -float("inf")
    R = torch.topk(S, K, 1).indices
    hits = gt_v.gather(1, R).float()
    n_gt = gt_v.sum(1).clamp(max=K).long()
    nd = ((hits * disc).sum(1) / torch.cumsum(disc, 0)[n_gt - 1]).mean().item()
    Rd = torch.zeros_like(gt_v).scatter_(1, R, True)[gt_v]
    row = {"model": name, "NDCG@20": round(nd, 5)}
    for a, b, lab in bins:
        sel = (gp > a) & (gp <= b)
        row[lab] = round(Rd[sel].float().mean().item(), 3)
    row["rec pop"] = round(pop[R].mean().item(), 1)
    row["coverage"] = R.unique().numel()
    rows.append(row)

pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))
