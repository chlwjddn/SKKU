"""
Failure-case analysis on validation: saved model (scores/{tag}_val.pt) vs EASE.
Put next to ease_v2.py and run:  python failure.py --data ~/RS/data --tag lgcn_ce
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
from ease_v2 import temporal_split, dense, ease_fit, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--tag", default="lgcn_ce")
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
pop = X.sum(0)  # high-rated popularity in inner
gt_v = gt_mat[vusers]
disc = 1.0 / torch.log2(torch.arange(2, K + 2, device=dev, dtype=torch.float32))

models = {
    args.tag: torch.load(f"scores/{args.tag}_val.pt").to(dev).float(),
    "EASE": X @ ease_fit(X, X, 100.0),
}


def analyze(S):
    S = S[vusers].clone()
    S[seen[vusers]] = -float("inf")
    R = torch.topk(S, K, 1).indices
    hits = gt_v.gather(1, R).float()
    n_gt = gt_v.sum(1).clamp(max=K).long()
    nd = (hits * disc).sum(1) / torch.cumsum(disc, 0)[n_gt - 1]
    Rd = torch.zeros_like(gt_v).scatter_(1, R, True)
    return R, nd, Rd


out = {m: analyze(S) for m, S in models.items()}
names = list(models)

print("=== 1. users ===")
for m in names:
    _, nd, _ = out[m]
    print(f"{m:12s} mean NDCG {nd.mean():.4f} | users with 0 hits {(nd == 0).float().mean():.1%}")
act = X[vusers].sum(1)
q = torch.quantile(act, torch.tensor([0.25, 0.5, 0.75], device=dev))
edges = [-1, q[0].item(), q[1].item(), q[2].item(), 1e9]
print("by user activity (# high-rated in history):")
for a, b in zip(edges[:-1], edges[1:]):
    msk = (act > a) & (act <= b)
    line = f"  ({max(a,0):5.0f}, {b if b < 1e9 else act.max().item():5.0f}]  n={msk.sum().item():3d}"
    for m in names:
        line += f"  {m}={out[m][1][msk].mean():.4f}"
    print(line)

print("\n=== 2. GT items by popularity ===")
gp = pop[None, :].expand_as(gt_v)[gt_v]
bins = [(-1, 0, "cold (0)"), (0, 5, "1-5"), (5, 20, "6-20"), (20, 50, "21-50"), (50, 1e9, "51+")]
for a, b, lab in bins:
    sel = (gp > a) & (gp <= b)
    line = f"  {lab:9s} share of GT {sel.float().mean():6.1%}"
    for m in names:
        hit_items = out[m][2][gt_v]
        line += f"  {m} hit-rate {hit_items[sel].float().mean():.3f}"
    print(line)

print("\n=== 3. recommendation behaviour ===")
top100 = torch.topk(pop, 100).indices
print(f"mean popularity of GT items: {gp.mean():.1f}")
for m in names:
    R = out[m][0]
    print(f"{m:12s} mean pop of recs {pop[R].mean():.1f} | recs in top-100 popular {torch.isin(R, top100).float().mean():.1%}"
          f" | coverage {R.unique().numel()} items")

print("\n=== 4. complementarity ===")
a, b = names
overlap = (out[a][2] & out[b][2]).sum(1).float() / K
print(f"rec overlap {a} vs EASE: {overlap.mean():.1%}")
na, nb = out[a][1], out[b][1]
print(f"{a} better on {(na > nb + 1e-9).float().mean():.1%} users | EASE better on {(nb > na + 1e-9).float().mean():.1%} | tie {(na == nb).float().mean():.1%}")
print(f"oracle per-user max: {torch.maximum(na, nb).mean():.4f}  (vs {a} {na.mean():.4f}, EASE {nb.mean():.4f})")
lose = nb > na + 0.1
print(f"users where {a} loses by >0.1: {lose.sum().item()} | their mean activity {act[lose].mean():.1f} (all {act.mean():.1f})")