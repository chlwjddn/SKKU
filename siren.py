"""
SiReN-style sign-aware heterogeneous graph model (Seo et al., NeurIPS 2021), adapted:
  - A_pos (r >= 4) + A_neu (r in [3, 4), weight gamma_neu): LightGCN propagation -> Z_P
    edge weight = rating_w x exp(-beta (1 - rel))   (rating AND time on edges)
  - negative relation (r <= neg_max): NO propagation (homophily breaks); MLP path -> Z_N
  - node-wise attention combine: z = a_P Z_P + a_N Z_N  (softmax over two scalars per node)
  - loss = SOTA softmax CE (cosine/temp, pos-masked, rating-time weighted)
           + lambda_neg * mean( softplus( (cos(z_u, z_i_neg) - margin_neg) / temp ) )  (push low-rated down)
Put next to ease_v2.py.
Usage: python siren.py --data ~/RS/data --tag siren
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
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--epochs", type=int, default=1000)
ap.add_argument("--eval_every", type=int, default=20)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--rating_w", default="4:0.7,5:1")
ap.add_argument("--time_beta", type=float, default=1.25)
ap.add_argument("--gamma_neu", type=float, default=0.3, help="neutral (3..4) edge weight scale")
ap.add_argument("--neg_max", type=float, default=2.5, help="ratings <= this are negative edges")
ap.add_argument("--lambda_neg", type=float, default=0.3, help="weight of the negative push-down loss")
ap.add_argument("--margin_neg", type=float, default=0.0, help="push cos below this margin")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no_final", action="store_true")
ap.add_argument("--tag", default="siren")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
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


class SiReN(nn.Module):
    def __init__(self, nU, A):
        super().__init__()
        self.nU, self.A = nU, A
        d = args.dim
        self.emb = nn.Embedding(nU + N_ITEMS, d)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.mlp_n = nn.Sequential(nn.Linear(d, d), nn.LeakyReLU(), nn.Linear(d, d))   # negative path
        self.attn = nn.Linear(d, 2)                                                    # per-node [a_P, a_N]

    def forward(self):
        e = self.emb.weight
        outs = [e]
        x = e
        for _ in range(args.layers):
            x = torch.sparse.mm(self.A, x)
            outs.append(x)
        Zp = torch.stack(outs).mean(0)
        Zn = self.mlp_n(e)
        a = torch.softmax(self.attn(e), dim=1)                                         # (N, 2)
        Z = a[:, :1] * Zp + a[:, 1:] * Zn
        return Z[:self.nU], Z[self.nU:]

    @torch.no_grad()
    def scores(self):
        U, I = self.forward()
        return F.normalize(U, dim=1) @ F.normalize(I, dim=1).T


def train(hist, epochs, val=None):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    h = hist.copy(); h["rel"] = rel_of(h)
    tw = np.exp(-args.time_beta * (1 - h.rel.values))
    pos = h[h.rating >= THR]
    neu = h[(h.rating >= 3) & (h.rating < THR)]
    negd = h[h.rating <= args.neg_max]
    # heterogeneous weighted graph: positive + scaled neutral
    gu = np.concatenate([pos.userId.values, neu.userId.values])
    gi = np.concatenate([pos.movieId.values, neu.movieId.values])
    gw = np.concatenate([map_w(args.rating_w, pos.rating.values) * np.exp(-args.time_beta * (1 - pos.rel.values)),
                         args.gamma_neu * np.exp(-args.time_beta * (1 - neu.rel.values))])
    model = SiReN(n_users, build_adj_w(gu, gi, gw, n_users)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pu = torch.as_tensor(pos.userId.values, device=dev); pi = torch.as_tensor(pos.movieId.values, device=dev)
    pw = torch.as_tensor((map_w(args.rating_w, pos.rating.values)
                          * np.exp(-args.time_beta * (1 - pos.rel.values))).astype(np.float32), device=dev)
    nu = torch.as_tensor(negd.userId.values, device=dev); ni = torch.as_tensor(negd.movieId.values, device=dev)
    pos_mask = dense(pos, n_users, np.ones(len(pos))) > 0
    print(f"edges pos {len(pos)} neu {len(neu)} | neg pairs {len(negd)}", flush=True)
    best, best_ep, best_S, bad = -1, 0, None, 0
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(pu), device=dev)
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s: s + args.batch]
            u, i, w = pu[b], pi[b], pw[b]
            U, I = model.forward()
            Un, In = F.normalize(U, dim=1), F.normalize(I, dim=1)
            logits = Un[u] @ In.T / args.temp
            m = pos_mask[u].clone(); m[torch.arange(len(u), device=dev), i] = False
            logits = logits.masked_fill(m, float("-inf"))
            ce = (F.cross_entropy(logits, i, reduction="none") * w).sum() / w.sum()
            # sign loss: push user's low-rated items down
            nb = torch.randint(0, len(nu), (min(len(nu), len(b)),), device=dev)
            cneg = (Un[nu[nb]] * In[ni[nb]]).sum(1)
            sign = F.softplus((cneg - args.margin_neg) / args.temp).mean()
            loss = ce + args.lambda_neg * sign
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if val is not None and ep % args.eval_every == 0:
            gt_mat, vusers, seen = val
            S = model.scores()[vusers]
            S[seen[vusers]] = -float("inf")
            sc = ndcg(torch.topk(S, K, 1).indices, vusers, gt_mat)
            flag = ""
            if sc > best:
                best, best_ep, bad, flag = sc, ep, 0, " *"
                best_S = model.scores().cpu()
            else:
                bad += 1
            print(f"ep {ep:4d}  loss {tot:.3f}  val NDCG@20 {sc:.5f}{flag}", flush=True)
            if bad >= args.patience:
                break
    if best_S is None:
        best_S = model.scores().cpu()
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
