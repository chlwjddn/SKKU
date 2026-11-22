"""
EASE baseline for skku-esw-4024-recommender-system
- Validation: per-user temporal split (last 20% of each user's train by time_idx),
  mirroring how test appears to be built.
- Metric: NDCG@20 over the full item set, seen items masked.
Usage:  python ease_baseline.py --data ~/RS/data --out submission_ease.csv
"""
import argparse, os, time
import numpy as np
import pandas as pd
import scipy.sparse as sp

K = 20


def temporal_split(df, ratio=0.2):
    """Per user: first (1-ratio) by time_idx -> train, rest -> val."""
    df = df.sort_values(["userId", "time_idx"])
    rank = df.groupby("userId").cumcount()
    cnt = df.groupby("userId")["movieId"].transform("size")
    is_tr = rank < np.floor(cnt * (1 - ratio))
    return df[is_tr], df[~is_tr]


def to_csr(df, n_users, n_items):
    return sp.csr_matrix(
        (np.ones(len(df), dtype=np.float64), (df.userId.values, df.movieId.values)),
        shape=(n_users, n_items),
    )


def ease_fit(X, lam):
    G = (X.T @ X).toarray()
    idx = np.diag_indices_from(G)
    G[idx] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    B[idx] = 0.0
    return B


def recommend(X, B, users, k=K):
    S = X[users] @ B
    S[X[users].toarray() > 0] = -np.inf          # mask seen items
    top = np.argpartition(-S, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(S, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def ndcg_at_k(recs, users, gt, k=K):
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    scores = []
    for u, r in zip(users, recs):
        g = gt[u]
        dcg = (np.isin(r, list(g)) * disc).sum()
        idcg = disc[: min(len(g), k)].sum()
        scores.append(dcg / idcg)
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/RS/data")
    ap.add_argument("--out", default="submission_ease.csv")
    ap.add_argument("--lams", default="50,100,200,300,500,800,1000,1500,2000,3000")
    ap.add_argument("--min_rating", type=float, default=0.0,
                    help="val ground truth uses only ratings >= this (0 = all)")
    args = ap.parse_args()
    d = os.path.expanduser(args.data)

    tr = pd.read_csv(os.path.join(d, "train.csv"))
    te = pd.read_csv(os.path.join(d, "test.csv"))
    n_users = tr.userId.max() + 1
    n_items = 9724  # MovieLens-small id range (0..9723)

    # ---------- validation ----------
    inner, val = temporal_split(tr)
    val_gt = val[val.rating >= args.min_rating]
    gt = val_gt.groupby("userId")["movieId"].apply(set).to_dict()
    vusers = np.array(sorted(gt))
    X_in = to_csr(inner, n_users, n_items)
    print(f"inner {len(inner)}, val {len(val_gt)} ({len(vusers)} users)")

    best = (None, -1)
    for lam in map(float, args.lams.split(",")):
        t = time.time()
        B = ease_fit(X_in, lam)
        score = ndcg_at_k(recommend(X_in, B, vusers), vusers, gt)
        print(f"lam={lam:7.0f}  NDCG@20={score:.5f}  ({time.time()-t:.1f}s)")
        if score > best[1]:
            best = (lam, score)
    print(f"best lam={best[0]}  val NDCG@20={best[1]:.5f}")

    # ---------- full train -> submission ----------
    X = to_csr(tr, n_users, n_items)
    B = ease_fit(X, best[0])
    tusers = te.userId.values
    recs = recommend(X, B, tusers)
    sub = pd.DataFrame({
        "id": te.id.values,
        "recommendations": [" ".join(map(str, r)) for r in recs],
    })
    sub.to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
