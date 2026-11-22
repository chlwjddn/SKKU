"""
Why are GT items stuck at rank 21-150?  Compare four groups of (user, item) pairs:
  HIT   : GT ranked 1-20          STUCK : GT ranked 21-150
  FP    : non-GT ranked 1-20      FAR   : GT ranked >150
Usage: python diag_stuck.py --data ~/RS/data --tag tb1.25
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
val = val.sort_values(["userId", "time_idx"]).copy()
nv = val.groupby("userId")["movieId"].transform("size")
val["vpos"] = val.groupby("userId").cumcount() / (nv - 1).clip(lower=1)
hi = inner[inner.rating >= 4]
gt_df = val[val.rating >= 4]
vu = np.sort(gt_df.userId.unique()); vusers = torch.as_tensor(vu, device=dev)
gt_v = (dense(gt_df, n_users, np.ones(len(gt_df))) > 0)[vusers]
gt_r = dense(gt_df, n_users, gt_df.rating.values)[vusers]
gt_t = dense(gt_df, n_users, gt_df.vpos.values + 1e-3)[vusers]
seen = (dense(inner, n_users, np.ones(len(inner))) > 0)[vusers]
X = dense(hi, n_users, np.ones(len(hi)))
pop = X.sum(0); logpop = torch.log1p(pop)
Xv = X[vusers]


def rank_of(S):
    S = S[vusers].clone(); S[seen] = -float("inf")
    order = torch.argsort(S, 1, descending=True)
    rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(N_ITEMS, device=dev).expand(len(vu), -1))
    return S, rank + 1


S_l, R_l = rank_of(torch.load(f"scores/{args.tag}_val.pt").to(dev).float())
_, R_e = rank_of(X @ ease_fit(X, X, 100.0))
s20 = S_l.gather(1, (R_l == 20).float().argmax(1, keepdim=True))       # 20th score per user
sd = S_l.masked_fill(seen, float("nan")); sd = torch.sqrt(torch.nanmean((sd - torch.nanmean(sd, 1, keepdim=True)) ** 2, 1, keepdim=True))
gap = (S_l - s20) / (sd + 1e-8)                                        # in user-level std units

# co-occurrence similarity of each item to the user's high-rated history
Xn = X / X.norm(dim=0, keepdim=True).clamp(min=1e-8)
Cos = Xn.T @ Xn; Cos.fill_diagonal_(0)
max_sim = torch.zeros(len(vu), N_ITEMS, device=dev); mean_sim = torch.zeros_like(max_sim); frac_co = torch.zeros_like(max_sim)
for k in range(len(vu)):
    h = torch.nonzero(Xv[k]).squeeze(1)
    if len(h) == 0:
        continue
    c = Cos[:, h]
    max_sim[k] = c.max(1).values; mean_sim[k] = c.mean(1); frac_co[k] = (c > 0).float().mean(1)
del Cos

groups = {
    "HIT":   gt_v & (R_l <= 20),
    "STUCK": gt_v & (R_l > 20) & (R_l <= 150),
    "FP":    (~gt_v) & (R_l <= 20),
    "FAR":   gt_v & (R_l > 150),
}
rows = []
for g, m in groups.items():
    row = {"group": g, "pairs": int(m.sum()),
           "item pop": round(pop[None, :].expand_as(m)[m].mean().item(), 1),
           "max_sim": round(max_sim[m].mean().item(), 3), "mean_sim": round(mean_sim[m].mean().item(), 4),
           "hist co-occur": f"{frac_co[m].mean():.1%}",
           "gap to #20 (std)": round(gap[m].median().item(), 2),
           "EASE rank<=20": f"{(R_e[m] <= 20).float().mean():.1%}",
           "EASE rank (med)": int(R_e[m].float().median().item())}
    if g != "FP":
        row["5-star"] = f"{(gt_r[m] == 5).float().mean():.1%}"
        row["late in window"] = f"{(gt_t[m] > 0.667).float().mean():.1%}"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

st = groups["STUCK"]
print("\nSTUCK breakdown")
r = R_l[st]
for a, b in [(20, 30), (30, 50), (50, 100), (100, 150)]:
    sel = (r > a) & (r <= b)
    print(f"  rank {a+1:3d}-{b:3d}: {sel.float().mean():5.1%} of STUCK | EASE puts {((R_e[st][sel]) <= 20).float().mean():5.1%} of them in top-20")
print("  popularity of STUCK items:",
      " ".join(f"{lab}:{((pop[None,:].expand_as(st)[st] > a) & (pop[None,:].expand_as(st)[st] <= b)).float().mean():.0%}"
               for a, b, lab in [(-1, 5, "<=5"), (5, 20, "6-20"), (20, 50, "21-50"), (50, 1e9, "51+")]))
