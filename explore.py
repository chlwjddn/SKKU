"""
Sequence-signal diagnostics on the validation split.
Put next to ease_v2.py and run:  python explore2.py --data ~/RS/data
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
from ease_v2 import temporal_split, dense, ease_fit, ndcg, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--thr", type=float, default=4.0)
args = ap.parse_args()

tr = pd.read_csv(os.path.join(os.path.expanduser(args.data), "train.csv"))
n_users = int(tr.userId.max()) + 1
inner, val = temporal_split(tr)
inner = inner.sort_values(["userId", "time_idx"]).reset_index(drop=True)
inner["pos"] = inner.groupby("userId").cumcount()
inner["n"] = inner.groupby("userId")["movieId"].transform("size")
inner["rel"] = inner.pos / (inner.n - 1).clip(lower=1)
hi = inner[inner.rating >= args.thr].reset_index(drop=True)

gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
seen = dense(inner, n_users, np.ones(len(inner))) > 0


def ev(S):
    S = S[vusers].clone()
    S[seen[vusers]] = -float("inf")
    return ndcg(torch.topk(S, K, dim=1).indices, vusers, gt_mat)


def zrow(S):
    S = S.clone()
    S[seen] = 0
    return (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-8)


# ---------------- A. order statistics ----------------
print("=== A. order stats ===")
d = tr.sort_values(["userId", "time_idx"]).groupby("userId")["movieId"].diff().dropna()
print(f"consecutive movieId increases (train): {(d > 0).mean():.1%}   (random ~50%)")
last_id = inner.groupby("userId")["movieId"].last()
L = last_id.reindex(gt_df.userId).values
print(f"val GT items with id > user's last inner id: {(gt_df.movieId.values > L).mean():.1%}"
      f"   (random expectation {((N_ITEMS - 1 - L) / N_ITEMS).mean():.1%})")
vs = val.sort_values(["userId", "time_idx"]).groupby("userId").head(5)
L5 = last_id.reindex(vs.userId).values
print(f"first 5 val items (any rating) with id > last inner id: {(vs.movieId.values > L5).mean():.1%}")

# ---------------- base EASE (best so far) ----------------
X = dense(hi, n_users, np.ones(len(hi)))
Xw = dense(hi, n_users, np.exp(-1.0 * (1 - hi.rel.values)))
S_ease = Xw @ ease_fit(X, X, 100)
print(f"\nEASE (alpha=0, lam=100, score beta=1): {ev(S_ease):.5f}")

# ---------------- B. transition model ----------------
print("\n=== B. transition model (b follows a within w steps, high-rated sequence) ===")
it = torch.as_tensor(hi.movieId.values, device=dev)
us = torch.as_tensor(hi.userId.values, device=dev)
best_T = (-1, None)
for w in [3, 10, 30]:
    cnt = torch.zeros(N_ITEMS * N_ITEMS, device=dev)
    for k in range(1, w + 1):
        same = us[:-k] == us[k:]
        idx = it[:-k][same] * N_ITEMS + it[k:][same]
        cnt += torch.bincount(idx, minlength=N_ITEMS * N_ITEMS).float()
    T = cnt.view(N_ITEMS, N_ITEMS)
    T = T / (T.sum(1, keepdim=True) + 1.0)
    line = f"w={w:2d}:"
    for m in [5, 20, 50, 100000]:
        last = hi[hi.groupby("userId").cumcount(ascending=False) < m]
        S = dense(last, n_users, np.ones(len(last))) @ T
        s = ev(S)
        line += f"  last{m if m < 100000 else 'ALL'}={s:.5f}"
        if s > best_T[0]:
            best_T = (s, S, f"w={w} m={m}")
    print(line)
    del cnt, T
print(f"best transition: {best_T[2]} -> {best_T[0]:.5f}")
Pt = zrow(best_T[1]); Pe = zrow(S_ease)
print("blend EASE + g*transition:", "  ".join(f"g={g}: {ev(Pe + g * Pt):.5f}" for g in [0.1, 0.2, 0.3, 0.5, 1.0]))

# ---------------- C. temporal EASE (past -> future) ----------------
print("\n=== C. temporal EASE: learn B s.t. past_high @ B ~ future_high ===")


def past_future(frac):
    past = inner[inner.pos < np.floor(inner.n * frac)]
    fut = inner[inner.pos >= np.floor(inner.n * frac)]
    past, fut = past[past.rating >= args.thr], fut[fut.rating >= args.thr]
    return dense(past, n_users, np.ones(len(past))), dense(fut, n_users, np.ones(len(fut)))


best_C = (-1, None)
for cuts in [[0.8], [0.5, 0.6, 0.7, 0.8], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]:
    Xs, Ys = zip(*[past_future(f) for f in cuts])
    Xs, Ys = torch.cat(Xs), torch.cat(Ys)
    G = Xs.T @ Xs
    XtY = Xs.T @ Ys
    line = f"cuts={cuts}:"
    for lam in [20, 50, 100, 200, 500]:
        Gl = G.clone(); Gl.diagonal().add_(lam)
        B = torch.linalg.solve(Gl, XtY)
        S = X @ B
        s = ev(S)
        line += f"  lam={lam}: {s:.5f}"
        if s > best_C[0]:
            best_C = (s, S, f"cuts={cuts} lam={lam}")
    print(line)
print(f"best temporal EASE: {best_C[2]} -> {best_C[0]:.5f}")
Pc = zrow(best_C[1])
print("blend EASE + g*temporalEASE:", "  ".join(f"g={g}: {ev(Pe + g * Pc):.5f}" for g in [0.3, 0.5, 1.0, 2.0]))
print("blend all three (g_c=1, g_t=0.3):", f"{ev(Pe + Pc + 0.3 * Pt):.5f}")