"""
Two-stage reranking for skku-esw-4024-recommender-system.

Stage 1: LightGCN (current SOTA config) -> top-C candidates per user
Stage 2: LightGBM LambdaRank over per (user, item) features

Leakage-free time layout
  H2 = first 64% of each user's train  -> base models -> candidates/features
       labels = 64-80% window (rating >= 4)            -> TRAIN the ranker
  H1 = first 80% (= our usual inner)    -> base models -> candidates/features
       labels = 80-100% window (= our val GT)          -> EVALUATE the ranker
  H0 = full train                       -> base models -> candidates/features -> submission
Put next to ease_v2.py.   pip install lightgbm --break-system-packages (if missing)
Usage: python rerank.py --data ~/RS/data --out submission_rerank.csv
"""
import argparse, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from ease_v2 import temporal_split, dense, ease_fit, ndcg, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--out", default="submission_rerank.csv")
ap.add_argument("--C", type=int, default=100, help="candidates per user")
ap.add_argument("--lgcn_epochs", type=int, default=260)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--temp", type=float, default=0.1)
ap.add_argument("--time_beta", type=float, default=1.25)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
THR = 4.0


def rel_of(df):
    n = df.groupby("userId")["time_idx"].transform("size")
    return ((df.groupby("userId")["time_idx"].rank(method="first") - 1) / (n - 1).clip(lower=1)).values


# ---------------- base model 1: LightGCN (SOTA config) ----------------
def lgcn(hist):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    h = hist.copy(); h["rel"] = rel_of(h)
    pos = h[h.rating >= THR]
    N = n_users + N_ITEMS
    u = pos.userId.values; i = pos.movieId.values + n_users
    deg = np.bincount(np.concatenate([u, i]), minlength=N).astype(np.float32)
    d = np.zeros_like(deg); d[deg > 0] = deg[deg > 0] ** -0.5
    rows = np.concatenate([u, i]); cols = np.concatenate([i, u])
    A = torch.sparse_coo_tensor(torch.as_tensor(np.vstack([rows, cols])),
                                torch.as_tensor(d[rows] * d[cols]), (N, N)).coalesce().to(dev)
    emb = nn.Embedding(N, args.dim).to(dev); nn.init.normal_(emb.weight, std=0.1)
    opt = torch.optim.Adam(emb.parameters(), lr=5e-3)
    pu = torch.as_tensor(pos.userId.values, device=dev); pi = torch.as_tensor(pos.movieId.values, device=dev)
    rw = np.where(pos.rating.values >= 5, 1.0, 0.7)
    pw = torch.as_tensor((rw * np.exp(-args.time_beta * (1 - pos.rel.values))).astype(np.float32), device=dev)
    pmask = dense(pos, n_users, np.ones(len(pos))) > 0

    def prop():
        e = emb.weight; out = [e]
        for _ in range(args.layers):
            e = torch.sparse.mm(A, e); out.append(e)
        e = torch.stack(out).mean(0)
        return e[:n_users], e[n_users:]

    for ep in range(args.lgcn_epochs):
        perm = torch.randperm(len(pu), device=dev)
        for s in range(0, len(perm), 4096):
            b = perm[s: s + 4096]; uu, ii, w = pu[b], pi[b], pw[b]
            U, I = prop()
            lg = F.normalize(U[uu], dim=1) @ F.normalize(I, dim=1).T / args.temp
            m = pmask[uu].clone(); m[torch.arange(len(uu), device=dev), ii] = False
            lg = lg.masked_fill(m, float("-inf"))
            loss = (F.cross_entropy(lg, ii, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        U, I = prop()
        return F.normalize(U, dim=1), F.normalize(I, dim=1)


# ---------------- features for one history ----------------
def build(hist, label_df):
    t = time.time()
    h = hist.sort_values(["userId", "time_idx"]).copy(); h["rel"] = rel_of(h)
    hi = h[h.rating >= THR]
    seen = dense(h, n_users, np.ones(len(h))) > 0
    X = dense(hi, n_users, np.ones(len(hi)))
    pop = X.sum(0); logpop = torch.log1p(pop)
    # base scores
    Un, In = lgcn(h)
    S_l = Un @ In.T
    Xw = dense(hi, n_users, np.exp(-1.0 * (1 - hi.rel.values)))
    S_e = Xw @ ease_fit(X, X, 100.0)
    it = torch.as_tensor(hi.movieId.values, device=dev); us = torch.as_tensor(hi.userId.values, device=dev)
    cnt = torch.zeros(N_ITEMS * N_ITEMS, device=dev)
    for k in range(1, 11):
        same = us[:-k] == us[k:]
        cnt += torch.bincount(it[:-k][same] * N_ITEMS + it[k:][same], minlength=N_ITEMS * N_ITEMS).float()
    T = cnt.view(N_ITEMS, N_ITEMS); T = T / (T.sum(1, keepdim=True) + 1.0)
    S_t = X @ T; del cnt, T
    # item / user side stats
    late = torch.as_tensor(np.bincount(hi[hi.rel >= 0.8].movieId.values, minlength=N_ITEMS), device=dev).float()
    lateness = torch.log((late + 1) / (pop + 5))
    last = hi[hi.groupby("userId").cumcount(ascending=False) < 10]
    lu = torch.as_tensor(last.userId.values, device=dev); li = torch.as_tensor(last.movieId.values, device=dev)
    lc = torch.bincount(lu, minlength=n_users).clamp(min=1).float()
    u_recent_lp = torch.zeros(n_users, device=dev).index_add_(0, lu, logpop[li]) / lc
    u_all_lp = (X * logpop).sum(1) / X.sum(1).clamp(min=1)
    Ir = torch.zeros(n_users, args.dim, device=dev).index_add_(0, lu, In[li]) / lc[:, None]   # recent-item mean
    u_hist_hi = torch.log1p(X.sum(1)); u_hist_all = torch.log1p(seen.sum(1).float())
    u_mean_r = torch.as_tensor(h.groupby("userId").rating.mean().reindex(np.arange(n_users)).fillna(3.5).values,
                               dtype=torch.float32, device=dev)
    # candidates: LightGCN top-C
    S_m = S_l.clone(); S_m[seen] = -float("inf")
    cand = torch.topk(S_m, args.C, dim=1).indices                       # (n_users, C)
    g = lambda S: S.gather(1, cand)

    def zr(S):
        S = S.clone(); S[seen] = float("nan")
        mu = torch.nanmean(S, 1, keepdim=True); sd = torch.sqrt(torch.nanmean((S - mu) ** 2, 1, keepdim=True))
        return (S - mu) / (sd + 1e-8)
    e_rank = torch.argsort(torch.argsort(torch.where(seen, -1e9, S_e), 1, descending=True), 1).float()
    cE = In[cand]                                                        # (n_users, C, d)
    feats = {
        "lgcn": g(S_l), "lgcn_rank": torch.arange(args.C, device=dev).float().expand(n_users, -1),
        "ease_z": g(zr(S_e)), "ease_rank": torch.log1p(g(e_rank)), "trans_z": g(zr(S_t)),
        "item_logpop": logpop[cand], "item_late": lateness[cand],
        "popdiff_recent": logpop[cand] - u_recent_lp[:, None],
        "recent_sim": (cE * Ir[:, None, :]).sum(-1),
        "u_hist_hi": u_hist_hi[:, None].expand(-1, args.C), "u_hist_all": u_hist_all[:, None].expand(-1, args.C),
        "u_mean_r": u_mean_r[:, None].expand(-1, args.C), "u_recent_lp": u_recent_lp[:, None].expand(-1, args.C),
        "u_drift": (u_recent_lp - u_all_lp)[:, None].expand(-1, args.C),
    }
    names = list(feats)
    Fm = torch.stack([feats[k] for k in names], -1).cpu().numpy()      # (n_users, C, F)
    if label_df is not None:
        lab = label_df[label_df.rating >= THR]
        Gm = dense(lab, n_users, np.ones(len(lab))) > 0
        Y = Gm.gather(1, cand).cpu().numpy().astype(int)
        recall = (Gm.gather(1, cand).sum(1).float() / Gm.sum(1).clamp(min=1)).cpu().numpy()
        users_lab = np.nonzero(Gm.sum(1).cpu().numpy() > 0)[0]
    else:
        Gm = Y = recall = users_lab = None
    print(f"  built features in {time.time()-t:.0f}s", flush=True)
    return dict(F=Fm, Y=Y, cand=cand, names=names, Gm=Gm, S_l=S_l, seen=seen, recall=recall, users=users_lab)


def ndcg_from_order(B, order_scores, users):
    """order_scores: (n_users, C) numpy; rank candidates, compute NDCG@20 vs full GT."""
    sc = torch.as_tensor(order_scores, device=dev)
    top = torch.topk(sc, K, 1).indices
    recs = B["cand"].gather(1, top)
    u = torch.as_tensor(users, device=dev)
    return ndcg(recs[u], u, B["Gm"])


d = os.path.expanduser(args.data)
tr = pd.read_csv(os.path.join(d, "train.csv")); te = pd.read_csv(os.path.join(d, "test.csv"))
n_users = int(tr.userId.max()) + 1
H1, V1 = temporal_split(tr)             # inner / val
H2, V2 = temporal_split(H1)             # inner of inner / its last 20%
print("device:", dev, "| C =", args.C, flush=True)

print("[H2] features for ranker training", flush=True)
B2 = build(H2, V2)
print("[H1] features for evaluation", flush=True)
B1 = build(H1, V1)
print(f"candidate recall@{args.C}: H2 {np.mean(B2['recall'][B2['users']]):.3f} | H1 {np.mean(B1['recall'][B1['users']]):.3f}")

base = ndcg_from_order(B1, B1["F"][..., 0], B1["users"])
print(f"H1 val NDCG@20  LightGCN only: {base:.5f}")


def to_rank_data(B, users):
    Xr = B["F"][users].reshape(-1, B["F"].shape[-1]); yr = B["Y"][users].reshape(-1)
    return Xr, yr, [args.C] * len(users)


rng = np.random.default_rng(args.seed)
u2 = rng.permutation(B2["users"]); cut = int(len(u2) * 0.85)
Xtr, ytr, gtr = to_rank_data(B2, np.sort(u2[:cut])); Xes, yes, ges = to_rank_data(B2, np.sort(u2[cut:]))
ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=2000, learning_rate=0.03, num_leaves=31,
                        min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                        lambdarank_truncation_level=40, verbose=-1, random_state=args.seed)
ranker.fit(Xtr, ytr, group=gtr, eval_set=[(Xes, yes)], eval_group=[ges], eval_at=[20],
           callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"ranker best iteration {ranker.best_iteration_}")
imp = sorted(zip(B2["names"], ranker.booster_.feature_importance("gain")), key=lambda x: -x[1])
print("feature importance (gain):", ", ".join(f"{n}:{v:.0f}" for n, v in imp))

P1 = ranker.predict(B1["F"].reshape(-1, B1["F"].shape[-1]), num_iteration=ranker.best_iteration_).reshape(n_users, -1)
print(f"H1 val NDCG@20  reranked     : {ndcg_from_order(B1, P1, B1['users']):.5f}")

print("[H0] features for submission", flush=True)
B0 = build(tr, None)
P0 = ranker.predict(B0["F"].reshape(-1, B0["F"].shape[-1]), num_iteration=ranker.best_iteration_).reshape(n_users, -1)
top = torch.topk(torch.as_tensor(P0, device=dev), K, 1).indices
recs = B0["cand"].gather(1, top).cpu().numpy()
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, recs[u])) for u in te.userId.values]}).to_csv(args.out, index=False)
print(f"saved {args.out}")
