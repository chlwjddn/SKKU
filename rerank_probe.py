"""
Test failure-case hypotheses by re-ranking saved validation scores (no retraining).
  A. lateness : share of an item's high ratings that fall in the last 20% of users' histories
  B. calib    : prefer items whose popularity matches the user's RECENT taste (last k high-rated)
  C. skipped  : penalise popular items for users with long histories (they probably skipped them)
score' = z(model) + g * z(feature)
Usage: python rerank_probe.py --data ~/RS/data --tag ce_t0.1_d128
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
from ease_v2 import temporal_split, dense, ease_fit, ndcg, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--tag", default="ce_t0.1_d128")
ap.add_argument("--thr", type=float, default=4.0)
args = ap.parse_args()

tr = pd.read_csv(os.path.join(os.path.expanduser(args.data), "train.csv"))
n_users = int(tr.userId.max()) + 1
inner, val = temporal_split(tr)
inner = inner.sort_values(["userId", "time_idx"]).reset_index(drop=True)
n = inner.groupby("userId")["movieId"].transform("size")
inner["rel"] = inner.groupby("userId").cumcount() / (n - 1).clip(lower=1)
hi = inner[inner.rating >= args.thr]
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
seen = dense(inner, n_users, np.ones(len(inner))) > 0
X = dense(hi, n_users, np.ones(len(hi)))
pop = X.sum(0)
logpop = torch.log1p(pop)


def ev(S):
    S = S[vusers].clone()
    S[seen[vusers]] = -float("inf")
    return ndcg(torch.topk(S, K, 1).indices, vusers, gt_mat)


def zrow(S):
    S = S.clone()
    S[seen] = 0
    return (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-8)


# ---- features ----
late = torch.as_tensor(np.bincount(hi[hi.rel >= 0.8].movieId.values, minlength=N_ITEMS), device=dev).float()
lateness = torch.log((late + 1) / (pop + 5))                       # smoothed late share
F_A = lateness[None, :].expand(n_users, -1)

feats = {"A lateness": F_A}
for k in [5, 10, 20]:
    last = hi[hi.groupby("userId").cumcount(ascending=False) < k]
    ul = torch.zeros(n_users, device=dev).index_add_(
        0, torch.as_tensor(last.userId.values, device=dev), logpop[torch.as_tensor(last.movieId.values, device=dev)])
    uc = torch.bincount(torch.as_tensor(last.userId.values, device=dev), minlength=n_users).clamp(min=1)
    feats[f"B calib k={k}"] = -(logpop[None, :] - (ul / uc)[:, None]).abs()
n_hist = seen.sum(1).float()
feats["C skipped"] = -(logpop[None, :] * torch.log1p(n_hist)[:, None])

models = {args.tag: torch.load(f"scores/{args.tag}_val.pt").to(dev).float(), "EASE": X @ ease_fit(X, X, 100.0)}
grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

for name, S in models.items():
    Z = zrow(S)
    print(f"\n=== {name}: base {ev(Z):.5f} ===")
    best = {}
    for fname, F in feats.items():
        Fz = zrow(F)
        res = [(ev(Z + g * Fz), g) for g in grid]
        b = max(res)
        best[fname] = (b, Fz)
        print(f"{fname:16s} " + "  ".join(f"g={g}:{s:.5f}" for s, g in res) + f"   -> best {b[0]:.5f}")
    # combine best A + best B
    bA = best["A lateness"]
    bB = max((v for k_, v in best.items() if k_.startswith("B")), key=lambda v: v[0][0])
    combo = max((ev(Z + bA[0][1] * bA[1] + g * bB[1]), g) for g in grid)
    print(f"A(g={bA[0][1]}) + best B: {combo[0]:.5f} (g_B={combo[1]})")