"""
Popularity baselines (no personalization; same ranking for every user except seen-masking).
  --mode count   : # of high ratings (>= thr)
  --mode rating  : sum of rating weights (5-star counts more)
  --mode recent  : recency-weighted count  w = exp(-beta * (1 - rel))  (rel = position in user's timeline)
  --mode late    : count within each user's last `late_frac` of history only
Scores saved to scores/{tag}_{val,test}.pt for blending.
Usage: python pop.py --data ~/RS/data --mode recent --tag pop_recent
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
from ease_v2 import temporal_split, dense, ndcg, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--thr", type=float, default=4.0)
ap.add_argument("--mode", choices=["count", "rating", "recent", "late"], default="count")
ap.add_argument("--beta", type=float, default=2.0, help="recent: recency weight")
ap.add_argument("--late_frac", type=float, default=0.2, help="late: use each user's last X of history")
ap.add_argument("--tag", default="pop")
args = ap.parse_args()


def pop_scores(hist):
    h = hist.copy()
    n = h.groupby("userId")["time_idx"].transform("size")
    h["rel"] = (h.groupby("userId")["time_idx"].rank(method="first") - 1) / (n - 1).clip(lower=1)
    hi = h[h.rating >= args.thr]
    if args.mode == "count":
        w = np.ones(len(hi))
    elif args.mode == "rating":
        w = np.where(hi.rating.values >= 5, 1.0, 0.7)
    elif args.mode == "recent":
        w = np.exp(-args.beta * (1 - hi.rel.values))
    elif args.mode == "late":
        hi = hi[hi.rel >= 1 - args.late_frac]
        w = np.ones(len(hi))
    cnt = np.bincount(hi.movieId.values, weights=w, minlength=N_ITEMS)
    s = torch.tensor(np.log1p(cnt), dtype=torch.float32, device=dev)
    return s[None, :].expand(n_users, -1).contiguous()     # same ranking for all users


d = os.path.expanduser(args.data)
tr = pd.read_csv(os.path.join(d, "train.csv")); te = pd.read_csv(os.path.join(d, "test.csv"))
n_users = int(tr.userId.max()) + 1
os.makedirs("scores", exist_ok=True)

inner, val = temporal_split(tr)
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
seen_in = dense(inner, n_users, np.ones(len(inner))) > 0

S = pop_scores(inner)
torch.save(S.cpu(), f"scores/{args.tag}_val.pt")
Sm = S.clone(); Sm[seen_in] = -float("inf")
print(f"[{args.mode}] val NDCG@20 {ndcg(torch.topk(Sm[vusers], K, 1).indices, vusers, gt_mat):.5f}")

S = pop_scores(tr)
torch.save(S.cpu(), f"scores/{args.tag}_test.pt")
seen_full = dense(tr, n_users, np.ones(len(tr))) > 0
tusers = torch.as_tensor(te.userId.values, device=dev)
St = S[tusers].clone(); St[seen_full[tusers]] = -float("inf")
recs = torch.topk(St, K, 1).indices.cpu().numpy()
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(f"submission_{args.tag}.csv", index=False)
print(f"saved scores/{args.tag}_{{val,test}}.pt, submission_{args.tag}.csv")
