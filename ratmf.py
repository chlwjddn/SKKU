"""
Biased MF rating prediction (all 80,621 ratings), rank by predicted rating.
  r_hat = mu + b_u + b_i + p_u . q_i    (mu = global mean, fixed)
  MSE on ALL ratings + L2. Prediction: top-20 by r_hat among unseen items.
Usage: python ratmf.py --data ~/RS/data --tag ratmf
"""
import argparse, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ease_v2 import temporal_split, dense, ndcg, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--reg", type=float, default=1e-2, help="L2 on embeddings and biases (per-sample)")
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--eval_every", type=int, default=20)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no_final", action="store_true")
ap.add_argument("--tag", default="ratmf")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)


def train(hist, epochs, val=None):
    u = torch.as_tensor(hist.userId.values, device=dev)
    i = torch.as_tensor(hist.movieId.values, device=dev)
    r = torch.as_tensor(hist.rating.values.astype(np.float32), device=dev)
    mu = r.mean()
    seen = dense(hist, n_users, np.ones(len(hist))) > 0
    P = nn.Parameter(torch.randn(n_users, args.dim, device=dev) * 0.01)
    Q = nn.Parameter(torch.randn(N_ITEMS, args.dim, device=dev) * 0.01)
    bu = nn.Parameter(torch.zeros(n_users, device=dev))
    bi = nn.Parameter(torch.zeros(N_ITEMS, device=dev))
    opt = torch.optim.Adam([P, Q, bu, bi], lr=args.lr)
    best, best_ep, best_state, bad = -1, 0, None, 0
    print(f"ratings {len(r)}  mu {mu:.3f}", flush=True)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(r), device=dev)
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s: s + args.batch]
            ub, ib, rb = u[b], i[b], r[b]
            pred = mu + bu[ub] + bi[ib] + (P[ub] * Q[ib]).sum(1)
            mse = ((pred - rb) ** 2).mean()
            reg = args.reg * (P[ub].pow(2).sum(1) + Q[ib].pow(2).sum(1) + bu[ub] ** 2 + bi[ib] ** 2).mean()
            loss = mse + reg
            opt.zero_grad(); loss.backward(); opt.step()
            tot += mse.item() * len(b)
        if val is not None and ep % args.eval_every == 0:
            gt_mat, vusers = val
            with torch.no_grad():
                S = mu + bu[vusers, None] + bi[None, :] + P[vusers] @ Q.T
            S[seen[vusers]] = -float("inf")
            sc = ndcg(torch.topk(S, K, 1).indices, vusers, gt_mat)
            rmse = (tot / len(r)) ** 0.5
            flag = ""
            if sc > best:
                best, best_ep, bad, flag = sc, ep, 0, " *"
                best_state = [t.detach().clone() for t in (P, Q, bu, bi)]
            else:
                bad += 1
            print(f"ep {ep:4d}  train RMSE {rmse:.4f}  val NDCG@20 {sc:.5f}{flag}", flush=True)
            if bad >= args.patience:
                break
    if best_state is not None:
        P, Q, bu, bi = best_state
    return (P, Q, bu, bi, mu), seen, best_ep, best


def full_scores(params):
    P, Q, bu, bi, mu = params
    with torch.no_grad():
        return mu + bu[:, None] + bi[None, :] + P @ Q.T


d = os.path.expanduser(args.data)
tr = pd.read_csv(os.path.join(d, "train.csv")); te = pd.read_csv(os.path.join(d, "test.csv"))
n_users = int(tr.userId.max()) + 1
os.makedirs("scores", exist_ok=True)
print("device:", dev, "|", vars(args), flush=True)

inner, val = temporal_split(tr)
gt_df = val[val.rating >= 4.0]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
t = time.time()
params, _, best_ep, best = train(inner, args.epochs, val=(gt_mat, vusers))
print(f"best epoch {best_ep}  val NDCG@20 {best:.5f}  ({time.time()-t:.0f}s)", flush=True)
torch.save(full_scores(params).cpu(), f"scores/{args.tag}_val.pt")
if args.no_final:
    raise SystemExit

params, seen, _, _ = train(tr, best_ep)
S = full_scores(params)
torch.save(S.cpu(), f"scores/{args.tag}_test.pt")
tusers = torch.as_tensor(te.userId.values, device=dev)
St = S[tusers].clone(); St[seen[tusers]] = -float("inf")
recs = torch.topk(St, K, 1).indices.cpu().numpy()
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(f"submission_{args.tag}.csv", index=False)
print(f"saved submission_{args.tag}.csv  (val {best:.5f}, epochs {best_ep})")
