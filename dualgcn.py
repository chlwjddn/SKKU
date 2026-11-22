"""
Dual-graph contrast model.
  A+ : edges r >= 4, weight = rating_w x exp(-beta (1 - rel))  -> LightGCN -> Z+
  A- : edges r <= 3, weight = (4 - r)/3 x exp(-beta (1 - rel)) -> LightGCN -> Z-  (separate embeddings)
  score(u, i) = cos(z+_u, z+_i)/temp - delta * cos(z-_u, z-_i)/temp
  loss = SOTA softmax CE on the combined score (pos-masked, rating-time weighted)
  -> to rank a positive high, the model must place it NEAR the user in the positive channel
     and FAR in the negative channel: the two graphs are contrasted directly in the score.
Put next to ease_v2.py.
Usage: python dualgcn.py --data ~/RS/data --tag dual
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
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--temp", type=float, default=0.1)
ap.add_argument("--delta", type=float, default=0.3, help="weight of the negative-channel subtraction")
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--epochs", type=int, default=1000)
ap.add_argument("--eval_every", type=int, default=20)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--rating_w", default="4:0.7,5:1")
ap.add_argument("--time_beta", type=float, default=1.25)
ap.add_argument("--neg_max", type=float, default=3.0, help="ratings <= this form the negative graph")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no_final", action="store_true")
ap.add_argument("--tag", default="dual")
args = ap.parse_args()
THR = 4.0


def map_w(spec, r):
    kv = sorted((float(a), float(b)) for a, b in (p.split(":") for p in spec.split(",")))
    keys, vals = np.array([k for k, _ in kv]), np.array([v for _, v in kv])
    return vals[np.clip(np.searchsorted(keys, r, side="right") - 1, 0, len(keys) - 1)]


def rel_of(df):
    n = df.groupby("userId")["time_idx"].transform("size")
    return ((df.groupby("userId")["time_idx"].rank(method="first") - 1) / (n - 1).clip(lower=1)).values


def build_adj_w(u, i, w, nU):
    n = nU + N_ITEMS
    r = np.concatenate([u, i + nU]); c = np.concatenate([i + nU, u]); ww = np.concatenate([w, w]).astype(np.float32)
    deg = np.bincount(r, weights=ww, minlength=n).astype(np.float32)
    dinv = np.zeros_like(deg); nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    v = dinv[r] * ww * dinv[c]
    return torch.sparse_coo_tensor(torch.tensor(np.vstack([r, c])), torch.tensor(v), (n, n)).coalesce().to(dev)


class Channel(nn.Module):
    def __init__(self, nU, A):
        super().__init__()
        self.nU, self.A = nU, A
        self.emb = nn.Embedding(nU + N_ITEMS, args.dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self):
        e = self.emb.weight; outs = [e]
        for _ in range(args.layers):
            e = torch.sparse.mm(self.A, e); outs.append(e)
        E = torch.stack(outs).mean(0)
        return F.normalize(E[:self.nU], dim=1), F.normalize(E[self.nU:], dim=1)


class Dual(nn.Module):
    def __init__(self, nU, Ap, An):
        super().__init__()
        self.pos = Channel(nU, Ap)
        self.neg = Channel(nU, An)

    def all_scores(self):
        Up, Ip = self.pos(); Un, In = self.neg()
        return (Up @ Ip.T - args.delta * (Un @ In.T)) / args.temp


def train(hist, epochs, val=None):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    h = hist.copy(); h["rel"] = rel_of(h)
    pos = h[h.rating >= THR]
    neg = h[h.rating <= args.neg_max]
    wp = map_w(args.rating_w, pos.rating.values) * np.exp(-args.time_beta * (1 - pos.rel.values))
    wn = ((THR - neg.rating.values) / 3.0) * np.exp(-args.time_beta * (1 - neg.rel.values))
    Ap = build_adj_w(pos.userId.values, pos.movieId.values, wp, n_users)
    An = build_adj_w(neg.userId.values, neg.movieId.values, wn, n_users)
    model = Dual(n_users, Ap, An).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pu = torch.as_tensor(pos.userId.values, device=dev); pi = torch.as_tensor(pos.movieId.values, device=dev)
    pw = torch.as_tensor(wp.astype(np.float32), device=dev)
    pos_mask = dense(pos, n_users, np.ones(len(pos))) > 0
    print(f"pos edges {len(pos)}  neg edges {len(neg)}", flush=True)
    best, best_ep, best_S, bad = -1, 0, None, 0
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(pu), device=dev)
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s: s + args.batch]
            u, i, w = pu[b], pi[b], pw[b]
            Up, Ip = model.pos(); Unn, Inn = model.neg()
            logits = (Up[u] @ Ip.T - args.delta * (Unn[u] @ Inn.T)) / args.temp
            m = pos_mask[u].clone(); m[torch.arange(len(u), device=dev), i] = False
            logits = logits.masked_fill(m, float("-inf"))
            loss = (F.cross_entropy(logits, i, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if val is not None and ep % args.eval_every == 0:
            gt_mat, vusers, seen = val
            with torch.no_grad():
                S = model.all_scores()[vusers]
            S[seen[vusers]] = -float("inf")
            sc = ndcg(torch.topk(S, K, 1).indices, vusers, gt_mat)
            flag = ""
            if sc > best:
                best, best_ep, bad, flag = sc, ep, 0, " *"
                with torch.no_grad():
                    best_S = model.all_scores().cpu()
            else:
                bad += 1
            print(f"ep {ep:4d}  loss {tot:.3f}  val NDCG@20 {sc:.5f}{flag}", flush=True)
            if bad >= args.patience:
                break
    if best_S is None:
        with torch.no_grad():
            best_S = model.all_scores().cpu()
    return best_S, best_ep, best


d = os.path.expanduser(args.data)
tr = pd.read_csv(os.path.join(d, "train.csv")); te = pd.read_csv(os.path.join(d, "test.csv"))
n_users = int(tr.userId.max()) + 1
os.makedirs("scores", exist_ok=True)
print("device:", dev, "|", vars(args), flush=True)

inner, val = temporal_split(tr)
gt_df = val[val.rating >= THR]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
seen_in = dense(inner, n_users, np.ones(len(inner))) > 0
t = time.time()
S_val, best_ep, best = train(inner, args.epochs, val=(gt_mat, vusers, seen_in))
print(f"best epoch {best_ep}  val NDCG@20 {best:.5f}  ({time.time()-t:.0f}s)", flush=True)
torch.save(S_val, f"scores/{args.tag}_val.pt")
if args.no_final:
    raise SystemExit

S_test, _, _ = train(tr, best_ep)
torch.save(S_test, f"scores/{args.tag}_test.pt")
seen_full = dense(tr, n_users, np.ones(len(tr))) > 0
tusers = torch.as_tensor(te.userId.values, device=dev)
St = S_test.to(dev)[tusers]; St[seen_full[tusers]] = -float("inf")
recs = torch.topk(St, K, 1).indices.cpu().numpy()
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(f"submission_{args.tag}.csv", index=False)
print(f"saved submission_{args.tag}.csv  (val {best:.5f}, epochs {best_ep})")
