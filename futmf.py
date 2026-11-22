"""
Future-window softmax MF (train objective = the evaluation task).
- No user ID embeddings. User repr at a cut = normalized weighted mean of item embeddings
  of their high-rated history BEFORE the cut (weights: rating map, optional recency).
- Training samples: for each cut c and each user, targets = high-rated items in
  [c, c + fw * c) of their relative timeline. Loss = cosine/temp softmax over all items,
  other future targets of the same (user, cut) masked from the denominator, 5-star upweighted.
- Inference: same formula with the full history (cut = 1.0). Seen items masked.
Put next to ease_v2.py.
Usage: python futmf.py --data ~/RS/data --tag futmf
"""
import argparse, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ease_v2 import temporal_split, dense, ndcg, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--thr", type=float, default=4.0)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--temp", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--eval_every", type=int, default=20)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--cuts", default="0.3,0.4,0.5,0.6,0.7,0.8")
ap.add_argument("--fw", type=float, default=0.25, help="target window length relative to cut")
ap.add_argument("--rating_w", default="4:0.7,5:1", help="weights for targets AND history mean")
ap.add_argument("--hist_time_beta", type=float, default=0.0, help="recency weight inside the history mean")
ap.add_argument("--tgt_time_beta", type=float, default=0.0, help="recency weight on target loss (rel within full history)")
ap.add_argument("--include_recon", action="store_true", help="also add plain reconstruction samples (cut=1, targets=all positives)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no_final", action="store_true")
ap.add_argument("--tag", default="futmf")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)


def map_w(spec, r):
    kv = sorted((float(a), float(b)) for a, b in (p.split(":") for p in spec.split(",")))
    keys, vals = np.array([k for k, _ in kv]), np.array([v for _, v in kv])
    return vals[np.clip(np.searchsorted(keys, r, side="right") - 1, 0, len(keys) - 1)]


def prep(hist):
    """Returns per-(user,cut) history matrices + training triplets."""
    h = hist.copy()
    n = h.groupby("userId")["time_idx"].transform("size")
    h["rel"] = (h.groupby("userId")["time_idx"].rank(method="first") - 1) / (n - 1).clip(lower=1)
    hi = h[h.rating >= args.thr].copy()
    hi["w"] = map_w(args.rating_w, hi.rating.values)
    cuts = [float(c) for c in args.cuts.split(",")]
    H, tgt_u, tgt_i, tgt_w, grp = [], [], [], [], []   # grp indexes rows of H
    gid = 0
    for c in cuts:
        past = hi[hi.rel < c]
        wp = past.w.values * (np.exp(-args.hist_time_beta * (1 - past.rel.values / c)) if args.hist_time_beta else 1.0)
        Hc = dense(past, n_users, wp)
        fut = hi[(hi.rel >= c) & (hi.rel < min(1.0, c * (1 + args.fw)))]
        ok_users = np.intersect1d(past.userId.unique(), fut.userId.unique())
        if len(ok_users) == 0:
            continue
        remap = -np.ones(n_users, dtype=np.int64); remap[ok_users] = np.arange(len(ok_users)) + gid
        H.append(Hc[ok_users])
        f = fut[fut.userId.isin(ok_users)]
        tgt_u.append(remap[f.userId.values]); tgt_i.append(f.movieId.values)
        w = f.w.values * (np.exp(-args.tgt_time_beta * (1 - f.rel.values)) if args.tgt_time_beta else 1.0)
        tgt_w.append(w)
        gid += len(ok_users)
    if args.include_recon:
        wp = hi.w.values
        Hc = dense(hi, n_users, wp)
        ok_users = hi.userId.unique()
        remap = -np.ones(n_users, dtype=np.int64); remap[ok_users] = np.arange(len(ok_users)) + gid
        H.append(Hc[ok_users])
        tgt_u.append(remap[hi.userId.values]); tgt_i.append(hi.movieId.values); tgt_w.append(wp)
    H = torch.cat(H)                                            # (G, N_ITEMS) weighted history rows
    tu = torch.as_tensor(np.concatenate(tgt_u), device=dev)
    ti = torch.as_tensor(np.concatenate(tgt_i), device=dev)
    tw = torch.as_tensor(np.concatenate(tgt_w).astype(np.float32), device=dev)
    # positives mask per group (for denominator masking)
    pm = torch.zeros(H.shape[0], N_ITEMS, dtype=torch.bool, device=dev)
    pm[tu, ti] = True
    print(f"  groups {H.shape[0]}  targets {len(ti)}", flush=True)
    return H, tu, ti, tw, pm


def full_hist(hist):
    h = hist.copy()
    n = h.groupby("userId")["time_idx"].transform("size")
    h["rel"] = (h.groupby("userId")["time_idx"].rank(method="first") - 1) / (n - 1).clip(lower=1)
    hi = h[h.rating >= args.thr].copy()
    w = map_w(args.rating_w, hi.rating.values) * (np.exp(-args.hist_time_beta * (1 - hi.rel.values)) if args.hist_time_beta else 1.0)
    return dense(hi, n_users, w), dense(h, n_users, np.ones(len(h))) > 0


def user_vec(E, Hrows):
    v = Hrows @ E
    return F.normalize(v, dim=1)


def scores(E, Hrows):
    return user_vec(E, Hrows) @ F.normalize(E, dim=1).T


def train(hist, epochs, val=None):
    H, tu, ti, tw, pm = prep(hist)
    Hfull, seen = full_hist(hist)
    E = nn.Parameter(torch.randn(N_ITEMS, args.dim, device=dev) * 0.1)
    opt = torch.optim.Adam([E], lr=args.lr)
    best, best_ep, best_E, bad = -1, 0, None, 0
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(ti), device=dev)
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s: s + args.batch]
            u, i, w = tu[b], ti[b], tw[b]
            En = F.normalize(E, dim=1)
            logits = user_vec(E, H[u]) @ En.T / args.temp
            m = pm[u].clone(); m[torch.arange(len(u), device=dev), i] = False
            logits = logits.masked_fill(m, float("-inf"))
            loss = (F.cross_entropy(logits, i, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if val is not None and ep % args.eval_every == 0:
            gt_mat, vusers = val
            with torch.no_grad():
                S = scores(E, Hfull[vusers])
            S[seen[vusers]] = -float("inf")
            sc = ndcg(torch.topk(S, K, 1).indices, vusers, gt_mat)
            flag = ""
            if sc > best:
                best, best_ep, bad, flag = sc, ep, 0, " *"
                best_E = E.detach().clone()
            else:
                bad += 1
            print(f"ep {ep:4d}  loss {tot:.3f}  val NDCG@20 {sc:.5f}{flag}", flush=True)
            if bad >= args.patience:
                break
    if best_E is not None:
        E = nn.Parameter(best_E)
    return E, Hfull, seen, best_ep, best


d = os.path.expanduser(args.data)
tr = pd.read_csv(os.path.join(d, "train.csv")); te = pd.read_csv(os.path.join(d, "test.csv"))
n_users = int(tr.userId.max()) + 1
os.makedirs("scores", exist_ok=True)
print("device:", dev, "|", vars(args), flush=True)

inner, val = temporal_split(tr)
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
t = time.time()
E, Hfull, _, best_ep, best = train(inner, args.epochs, val=(gt_mat, vusers))
print(f"best epoch {best_ep}  val NDCG@20 {best:.5f}  ({time.time()-t:.0f}s)", flush=True)
with torch.no_grad():
    torch.save(scores(E, Hfull).cpu(), f"scores/{args.tag}_val.pt")
if args.no_final:
    raise SystemExit

E, Hfull, seen, _, _ = train(tr, best_ep)
with torch.no_grad():
    S = scores(E, Hfull)
torch.save(S.cpu(), f"scores/{args.tag}_test.pt")
tusers = torch.as_tensor(te.userId.values, device=dev)
St = S[tusers].clone(); St[seen[tusers]] = -float("inf")
recs = torch.topk(St, K, 1).indices.cpu().numpy()
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(f"submission_{args.tag}.csv", index=False)
print(f"saved submission_{args.tag}.csv  (val {best:.5f}, epochs {best_ep})")
