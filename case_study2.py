"""
More failure angles for a saved model vs EASE (validation).
Usage: python case_study2.py --data ~/RS/data --tag w_p4
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
from ease_v2 import temporal_split, dense, ease_fit, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--tag", default="w_p4")
ap.add_argument("--thr", type=float, default=4.0)
args = ap.parse_args()
pd.set_option("display.width", 220)
L = args.tag

tr = pd.read_csv(os.path.join(os.path.expanduser(args.data), "train.csv"))
n_users = int(tr.userId.max()) + 1
inner, val = temporal_split(tr)
inner = inner.sort_values(["userId", "time_idx"])
val = val.sort_values(["userId", "time_idx"]).copy()
nv = val.groupby("userId")["movieId"].transform("size")
val["vpos"] = val.groupby("userId").cumcount() / (nv - 1).clip(lower=1)
hi = inner[inner.rating >= args.thr]
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vu = np.sort(gt_df.userId.unique())
vusers = torch.as_tensor(vu, device=dev)
seen = dense(inner, n_users, np.ones(len(inner))) > 0
X = dense(hi, n_users, np.ones(len(hi)))
pop = X.sum(0); popn = pop.cpu().numpy(); logpop = torch.log1p(pop)
gt_v = gt_mat[vusers]
disc = 1.0 / torch.log2(torch.arange(2, K + 2, device=dev, dtype=torch.float32))


def run(S):
    S = S[vusers].clone()
    S[seen[vusers]] = -float("inf")
    order = torch.argsort(S, dim=1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(N_ITEMS, device=dev).expand(len(vu), -1))
    R = order[:, :K]
    Rd = torch.zeros_like(gt_v).scatter_(1, R, True)
    hits = gt_v.gather(1, R).float()
    nd = (hits * disc).sum(1) / torch.cumsum(disc, 0)[gt_v.sum(1).clamp(max=K).long() - 1]
    return {"R": R, "Rd": Rd, "rank": rank + 1, "nd": nd}


M = {L: run(torch.load(f"scores/{L}_val.pt").to(dev).float()), "EASE": run(X @ ease_fit(X, X, 100.0))}


def hit_line(sel):
    return "  ".join(f"{m}={(M[m]['Rd'] & sel).sum().item() / max(sel.sum().item(), 1):.3f}" for m in M)


print("=== A. hit-rate by GT rating ===")
gr = dense(gt_df, n_users, gt_df.rating.values)[vusers]
for r in [4.0, 4.5, 5.0]:
    sel = gt_v & (gr == r)
    print(f"  rating {r}: n={sel.sum().item():5d}  {hit_line(sel)}")

print("\n=== B. hit-rate by time distance inside the val window ===")
gb = dense(gt_df, n_users, np.digitize(gt_df.vpos.values, [1 / 3, 2 / 3]) + 1)[vusers]
for b, lab in [(1, "early (near future)"), (2, "middle"), (3, "late (far future)")]:
    sel = gt_v & (gb == b)
    print(f"  {lab:20s} n={sel.sum().item():5d}  {hit_line(sel)}")

U = pd.DataFrame({"user": vu, f"nd_{L}": M[L]["nd"].cpu().numpy(), "nd_EASE": M["EASE"]["nd"].cpu().numpy()})
U["mean_rating"] = inner.groupby("userId").rating.mean().reindex(vu).values
nh = X[vusers].sum(1)
all_lp = ((X[vusers] * logpop).sum(1) / nh.clamp(min=1)).cpu().numpy()
last = hi[hi.groupby("userId").cumcount(ascending=False) < 10]
rec_lp = last.assign(lp=logpop.cpu().numpy()[last.movieId.values]).groupby("userId").lp.mean().reindex(vu).values
U["drift"] = rec_lp - all_lp  # <0: recent taste more niche than overall

print("\n=== C. NDCG by user mean rating ===")
U["rq"] = pd.qcut(U.mean_rating, 4)
print(U.groupby("rq", observed=True)[[f"nd_{L}", "nd_EASE"]].agg(["mean", "size"]).round(4).to_string())

print("\n=== D. NDCG by taste drift (recent 10 vs all high-rated, mean log-popularity) ===")
U["dq"] = pd.qcut(U.drift, 4, labels=["strong->niche", "->niche", "stable", "->mainstream"])
print(U.groupby("dq", observed=True).agg(users=("user", "size"), drift=("drift", "mean"),
                                         **{f"nd_{L}": (f"nd_{L}", "mean")}, nd_EASE=("nd_EASE", "mean")).round(4).to_string())


def show(k, title):
    u = int(vu[k])
    h = hi[hi.userId == u]
    g = torch.nonzero(gt_v[k]).squeeze(1)
    g = g[torch.argsort(M[L]["rank"][k, g])]
    hitset = set(torch.nonzero(gt_v[k] & M[L]["Rd"][k]).squeeze(1).tolist())
    print(f"\n-- {title}: user {u} | NDCG {L} {M[L]['nd'][k]:.3f} / EASE {M['EASE']['nd'][k]:.3f} | high-rated {len(h)}"
          f" | mean rating {U.mean_rating[k]:.2f} | drift {U.drift[k]:+.2f} | GT {len(g)}")
    print("   last 10 high-rated (item:pop):", " ".join(f"{i}:{int(popn[i])}" for i in h.movieId.values[-10:]))
    print(f"   GT sorted by {L} rank (item:pop:rank, * = hit):",
          " ".join(f"{i}:{int(popn[i])}:{M[L]['rank'][k, i].item()}{'*' if i in hitset else ''}" for i in g.tolist()[:12]))
    print(f"   {L} top-10 (item:pop):", " ".join(f"{i}:{int(popn[i])}" for i in M[L]["R"][k, :10].tolist()))


print("\n=== E. examples ===")
nd = M[L]["nd"]
for k in torch.argsort(nd, descending=True)[:2].tolist():
    show(k, "SUCCESS")
best_gt_rank = torch.where(gt_v, M[L]["rank"], torch.full_like(M[L]["rank"], 10 ** 9)).min(1).values
near = torch.nonzero((nd == 0) & (best_gt_rank >= 21) & (best_gt_rank <= 30)).squeeze(1)
for k in near[torch.argsort(gt_v[near].sum(1), descending=True)][:2].tolist():
    show(k, "NEAR MISS")