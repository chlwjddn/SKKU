"""
EASE v2 for skku-esw-4024-recommender-system
- Validation: per-user temporal split (last 20% of each user's train by time_idx)
- Val ground truth: only ratings >= --thr (test appears to use rating >= 4)
- Grid over:
    alpha  : input weight of low-rated (< thr) interactions (0 = drop, 1 = treat as normal)
    target : 'all'  -> reconstruct every interaction (classic EASE, Y = X)
             'high' -> predict only high-rated interactions (Y = X_high)
    lam    : L2 regularization
- Seen items (any rating) are always masked at recommendation time.
Usage:  python ease_v2.py --data ~/RS/data --out submission_ease_v2.csv
"""
import argparse, itertools, os, time
import numpy as np
import pandas as pd
import torch

K = 20
N_ITEMS = 9724  # MovieLens-small id range (0..9723)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def temporal_split(df, ratio=0.2):
    df = df.sort_values(["userId", "time_idx"])
    rank = df.groupby("userId").cumcount()
    cnt = df.groupby("userId")["movieId"].transform("size")
    is_tr = rank < np.floor(cnt * (1 - ratio))
    return df[is_tr], df[~is_tr]


def dense(df, n_users, values):
    M = torch.zeros(n_users, N_ITEMS, device=dev)
    u = torch.as_tensor(df.userId.values, device=dev)
    i = torch.as_tensor(df.movieId.values, device=dev)
    M[u, i] = torch.as_tensor(np.asarray(values, dtype=np.float32), device=dev)
    return M


def build(df, n_users, thr, alpha):
    high = (df.rating.values >= thr).astype(np.float32)
    X = dense(df, n_users, high + alpha * (1 - high))   # weighted input
    Xh = dense(df, n_users, high)                         # high-rated only
    seen = dense(df, n_users, np.ones(len(df))) > 0
    return X, Xh, seen


def ease_fit(X, Y, lam):
    """min ||Y - XB||^2 + lam||B||^2  s.t. diag(B) = 0  (closed form)."""
    G = X.T @ X
    G.diagonal().add_(lam)
    P = torch.linalg.inv(G)
    C = P @ (X.T @ Y)
    mu = C.diagonal() / P.diagonal()
    return C - P * mu[None, :]


def topk(X, B, seen, users, k=K):
    S = X[users] @ B
    S[seen[users]] = -float("inf")
    return torch.topk(S, k, dim=1).indices


def ndcg(recs, users, gt_mat, k=K):
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=dev, dtype=torch.float32))
    hits = gt_mat[users].gather(1, recs).float()
    dcg = (hits * disc).sum(1)
    n_gt = gt_mat[users].sum(1).clamp(max=k).long()
    idcg = torch.cumsum(disc, 0)[n_gt - 1]
    return (dcg / idcg).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/RS/data")
    ap.add_argument("--out", default="submission_ease_v2.csv")
    ap.add_argument("--thr", type=float, default=4.0)
    ap.add_argument("--alphas", default="0,0.25,0.5,1")
    ap.add_argument("--targets", default="all,high")
    ap.add_argument("--lams", default="100,200,300,500,800,1200,2000")
    args = ap.parse_args()
    d = os.path.expanduser(args.data)
    print("device:", dev)

    tr = pd.read_csv(os.path.join(d, "train.csv"))
    te = pd.read_csv(os.path.join(d, "test.csv"))
    n_users = int(tr.userId.max()) + 1

    # ---------- validation ----------
    inner, val = temporal_split(tr)
    gt_df = val[val.rating >= args.thr]
    gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
    vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
    print(f"inner {len(inner)} | val GT (rating>={args.thr}) {len(gt_df)} | {len(vusers)} users")

    results = []
    for alpha in map(float, args.alphas.split(",")):
        X, Xh, seen = build(inner, n_users, args.thr, alpha)
        for target, lam in itertools.product(args.targets.split(","), map(float, args.lams.split(","))):
            t = time.time()
            B = ease_fit(X, X if target == "all" else Xh, lam)
            s = ndcg(topk(X, B, seen, vusers), vusers, gt_mat)
            results.append((s, alpha, target, lam))
            print(f"alpha={alpha:<5} target={target:<4} lam={lam:6.0f}  NDCG@20={s:.5f}  ({time.time()-t:.1f}s)")

    results.sort(reverse=True)
    print("\n=== top 10 ===")
    for s, a, tg, l in results[:10]:
        print(f"NDCG@20={s:.5f}  alpha={a}  target={tg}  lam={l:.0f}")
    s, alpha, target, lam = results[0]

    # ---------- full train -> submission ----------
    X, Xh, seen = build(tr, n_users, args.thr, alpha)
    B = ease_fit(X, X if target == "all" else Xh, lam)
    tusers = torch.as_tensor(te.userId.values, device=dev)
    recs = topk(X, B, seen, tusers).cpu().numpy()
    pd.DataFrame({
        "id": te.id.values,
        "recommendations": [" ".join(map(str, r)) for r in recs],
    }).to_csv(args.out, index=False)
    print(f"\nsaved {args.out}  (alpha={alpha}, target={target}, lam={lam:.0f}, val={s:.5f})")


if __name__ == "__main__":
    main()
