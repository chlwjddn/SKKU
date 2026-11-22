"""
Hypothesis check (no training): for users whose taste drifted, are GT items stuck at rank 21-150,
and does scoring from the RECENT k items only lift them?
Usage: python diag_recent.py --data ~/RS/data --tag tb1.25
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
from ease_v2 import temporal_split, dense, ease_fit, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--tag", default="tb1.25")
args = ap.parse_args()
pd.set_option("display.width", 220)

tr = pd.read_csv(os.path.join(os.path.expanduser(args.data), "train.csv"))
n_users = int(tr.userId.max()) + 1
inner, val = temporal_split(tr)
inner = inner.sort_values(["userId", "time_idx"])
hi = inner[inner.rating >= 4]
gt_df = val[val.rating >= 4]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vu = np.sort(gt_df.userId.unique()); vusers = torch.as_tensor(vu, device=dev)
seen = dense(inner, n_users, np.ones(len(inner))) > 0
X = dense(hi, n_users, np.ones(len(hi)))
logpop = torch.log1p(X.sum(0))
B = ease_fit(X, X, 100.0)
gt_v = gt_mat[vusers]
disc = 1.0 / torch.log2(torch.arange(2, K + 2, device=dev, dtype=torch.float32))


def recent_X(k):
    last = hi[hi.groupby("userId").cumcount(ascending=False) < k]
    return dense(last, n_users, np.ones(len(last)))


def evaluate(S):
    S = S[vusers].clone(); S[seen[vusers]] = -float("inf")
    order = torch.argsort(S, 1, descending=True)
    rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(N_ITEMS, device=dev).expand(len(vu), -1))
    rank = rank + 1
    hits = gt_v.gather(1, order[:, :K]).float()
    nd = (hits * disc).sum(1) / torch.cumsum(disc, 0)[gt_v.sum(1).clamp(max=K).long() - 1]
    return rank, nd


models = {
    "LightGCN (full)": torch.load(f"scores/{args.tag}_val.pt").to(dev).float(),
    "EASE (full history)": X @ B,
    "EASE (recent 10 only)": recent_X(10) @ B,
    "EASE (recent 20 only)": recent_X(20) @ B,
}
res = {m: evaluate(S) for m, S in models.items()}

# drift groups (same definition as case_study2)
nh = X[vusers].sum(1)
all_lp = ((X[vusers] * logpop).sum(1) / nh.clamp(min=1)).cpu().numpy()
last10 = hi[hi.groupby("userId").cumcount(ascending=False) < 10]
rec_lp = last10.assign(lp=logpop.cpu().numpy()[last10.movieId.values]).groupby("userId").lp.mean().reindex(vu).values
drift = rec_lp - all_lp
grp = pd.qcut(drift, 4, labels=["strong->niche", "->niche", "stable", "->mainstream"]).astype(str)

rows = []
for g in ["strong->niche", "->niche", "stable", "->mainstream"]:
    um = torch.as_tensor(grp == g, device=dev)
    for m, (rank, nd) in res.items():
        r = rank[um][gt_v[um]]
        rows.append({"group": g, "model": m, "users": int(um.sum()), "NDCG": round(nd[um].mean().item(), 4),
                     "GT 1-20": f"{(r <= 20).float().mean():.1%}",
                     "GT 21-150": f"{((r > 20) & (r <= 150)).float().mean():.1%}",
                     "GT >150": f"{(r > 150).float().mean():.1%}"})
print(pd.DataFrame(rows).to_string(index=False))

# for GT stuck at 21-150 under LightGCN: does recent-only scoring lift them into top 20?
print("\nGT items ranked 21-150 by LightGCN, where do they go under recent-only EASE?")
rl = res["LightGCN (full)"][0]
for g in ["strong->niche", "->niche", "stable", "->mainstream"]:
    um = torch.as_tensor(grp == g, device=dev)
    stuck = gt_v & (rl > 20) & (rl <= 150) & um[:, None]
    line = f"  {g:14s} stuck={stuck.sum().item():4d}"
    for m in ["EASE (full history)", "EASE (recent 10 only)", "EASE (recent 20 only)"]:
        rr = res[m][0][stuck]
        line += f" | {m.split('(')[1][:-1]}: top20 {(rr <= 20).float().mean():.1%}"
    print(line)
