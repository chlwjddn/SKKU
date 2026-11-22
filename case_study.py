"""
Case-level failure study on validation for a saved model vs EASE.
Put next to ease_v2.py and run:  python case_study.py --data ~/RS/data --tag ce_t0.1_d128
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
from ease_v2 import temporal_split, dense, ease_fit, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--tag", default="ce_t0.1_d128")
ap.add_argument("--thr", type=float, default=4.0)
ap.add_argument("--n_examples", type=int, default=3)
args = ap.parse_args()
pd.set_option("display.width", 220)

tr = pd.read_csv(os.path.join(os.path.expanduser(args.data), "train.csv"))
n_users = int(tr.userId.max()) + 1
inner, val = temporal_split(tr)
inner = inner.sort_values(["userId", "time_idx"])
hi = inner[inner.rating >= args.thr]
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vu = np.sort(gt_df.userId.unique())
vusers = torch.as_tensor(vu, device=dev)
seen = dense(inner, n_users, np.ones(len(inner))) > 0
X = dense(hi, n_users, np.ones(len(hi)))
pop = X.sum(0)
popn = pop.cpu().numpy()
gt_v = gt_mat[vusers]
disc = 1.0 / torch.log2(torch.arange(2, K + 2, device=dev, dtype=torch.float32))
L = args.tag


def run(S):
    S = S[vusers].clone()
    S[seen[vusers]] = -float("inf")
    order = torch.argsort(S, dim=1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(N_ITEMS, device=dev).expand(len(vu), -1))
    R = order[:, :K]
    Rd = torch.zeros_like(gt_v).scatter_(1, R, True)
    hits = gt_v.gather(1, R).float()
    n_gt = gt_v.sum(1).clamp(max=K).long()
    nd = (hits * disc).sum(1) / torch.cumsum(disc, 0)[n_gt - 1]
    return {"R": R, "Rd": Rd, "rank": rank + 1, "nd": nd}


M = {L: run(torch.load(f"scores/{L}_val.pt").to(dev).float()), "EASE": run(X @ ease_fit(X, X, 100.0))}

# ---------- 1. where do GT items land in the full ranking? ----------
print("=== 1. rank of GT items in the full ranking (share of GT items) ===")
reach = pop >= 6
for name, m in M.items():
    r = m["rank"][gt_v]
    rr = m["rank"][gt_v & reach[None, :]]
    def dist(x):
        return " ".join(f"{lab}:{((x > a) & (x <= b)).float().mean():5.1%}"
                        for a, b, lab in [(0, 20, "1-20"), (20, 50, "21-50"), (50, 100, "51-100"),
                                          (100, 500, "101-500"), (500, 1e9, ">500")])
    print(f"{name:14s} all GT      {dist(r)}")
    print(f"{'':14s} reachable  {dist(rr)}   (reachable = item pop >= 6)")

# ---------- 2. item level ----------
print("\n=== 2. item level ===")
gtc = gt_v.sum(0)
items = pd.DataFrame({
    "item": np.arange(N_ITEMS), "pop": popn, "in_GT": gtc.cpu().numpy(),
    f"hit_{L}": (gt_v & M[L]["Rd"]).sum(0).cpu().numpy(),
    "hit_EASE": (gt_v & M["EASE"]["Rd"]).sum(0).cpu().numpy(),
    f"rec_{L}": M[L]["Rd"].sum(0).cpu().numpy(),
    "rec_EASE": M["EASE"]["Rd"].sum(0).cpu().numpy(),
})
items["missed"] = items.in_GT - items[f"hit_{L}"]
print("-- most frequently missed GT items --")
print(items.sort_values("missed", ascending=False).head(12).to_string(index=False))
items["false_pos"] = items[f"rec_{L}"] - items[f"hit_{L}"]
print(f"\n-- most over-recommended items by {L} (recommended but not in GT) --")
fp = items.sort_values("false_pos", ascending=False).head(12).copy()
fp["precision"] = (fp[f"hit_{L}"] / fp[f"rec_{L}"].clip(lower=1)).round(3)
print(fp[["item", "pop", "in_GT", f"rec_{L}", f"hit_{L}", "precision", "rec_EASE"]].to_string(index=False))
items["ease_minus"] = items.hit_EASE - items[f"hit_{L}"]
print(f"\n-- items EASE hits but {L} misses most --")
print(items.sort_values("ease_minus", ascending=False).head(10)
      [["item", "pop", "in_GT", f"hit_{L}", "hit_EASE", f"rec_{L}", "rec_EASE"]].to_string(index=False))

# ---------- 3. user level ----------
print("\n=== 3. user level ===")
nh = X[vusers].sum(1)
U = pd.DataFrame({
    "user": vu,
    "hist_high": nh.cpu().numpy(),
    "hist_all": seen[vusers].sum(1).cpu().numpy(),
    "mean_rating": inner.groupby("userId").rating.mean().reindex(vu).values,
    "mainstream": ((X[vusers] * pop).sum(1) / nh.clamp(min=1)).cpu().numpy(),
    "n_gt": gt_v.sum(1).cpu().numpy(),
    "n_reach": (gt_v & reach[None, :]).sum(1).cpu().numpy(),
    "gt_pop": ((gt_v * pop).sum(1) / gt_v.sum(1)).cpu().numpy(),
    f"nd_{L}": M[L]["nd"].cpu().numpy(),
    "nd_EASE": M["EASE"]["nd"].cpu().numpy(),
})
U["zero"] = U[f"nd_{L}"] == 0
U["group"] = np.where(~U.zero, "hit>0", np.where(U.n_reach == 0, "zero: nothing reachable", "zero: TRUE FAILURE"))
print(U.groupby("group").agg(users=("user", "size"), hist_high=("hist_high", "mean"),
                             mean_rating=("mean_rating", "mean"), mainstream=("mainstream", "mean"),
                             n_gt=("n_gt", "mean"), n_reach=("n_reach", "mean"),
                             gt_pop=("gt_pop", "mean"), nd_EASE=("nd_EASE", "mean")).round(2).to_string())
print("\nNDCG by mainstreamness (mean popularity of the user's high-rated history):")
U["mq"] = pd.qcut(U.mainstream, 4, labels=["niche", "q2", "q3", "mainstream"])
print(U.groupby("mq", observed=True)[[f"nd_{L}", "nd_EASE", "gt_pop", "n_reach"]].mean().round(4).to_string())
print("\nNDCG by # GT items:")
U["gq"] = pd.qcut(U.n_gt, 4, duplicates="drop")
print(U.groupby("gq", observed=True)[[f"nd_{L}", "nd_EASE"]].mean().round(4).to_string())

# ---------- 4. example true-failure users ----------
print(f"\n=== 4. example TRUE FAILURE users (0 hits for {L}, most reachable GT) ===")
ex = U[U.group == "zero: TRUE FAILURE"].sort_values("n_reach", ascending=False).head(args.n_examples)
for _, row in ex.iterrows():
    u = int(row.user); k = int(np.where(vu == u)[0][0])
    print(f"\n-- user {u} | high-rated history {int(row.hist_high)} (all {int(row.hist_all)}) | mean rating {row.mean_rating:.2f}"
          f" | mainstream {row.mainstream:.1f} | GT {int(row.n_gt)} (reachable {int(row.n_reach)}) | EASE NDCG {row.nd_EASE:.3f}")
    h = hi[hi.userId == u].tail(10)
    print("   last 10 high-rated (item:pop):", " ".join(f"{i}:{int(popn[i])}" for i in h.movieId.values))
    g = torch.nonzero(gt_v[k]).squeeze(1)
    g = g[torch.argsort(pop[g], descending=True)]
    print(f"   GT (item:pop:rank_{L}/rank_EASE):",
          " ".join(f"{i}:{int(popn[i])}:{M[L]['rank'][k, i].item()}/{M['EASE']['rank'][k, i].item()}"
                   for i in g.tolist()[:12]))
    print(f"   {L} top-10 recs (item:pop):", " ".join(f"{i}:{int(popn[i])}" for i in M[L]["R"][k, :10].tolist()))
    print("   EASE top-10 recs (item:pop):", " ".join(f"{i}:{int(popn[i])}" for i in M["EASE"]["R"][k, :10].tolist()))