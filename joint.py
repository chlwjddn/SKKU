"""
Joint training: LightGCN (softmax CE) + BERT4Rec-style masked item prediction (MLM),
sharing the same item embedding table. Recommendation uses LightGCN scores.

  L = L_lgcn + lambda_mlm * L_mlm
  - L_lgcn: current SOTA LightGCN loss (cosine/temp softmax, other positives masked,
            rating weight x recency weight)
  - L_mlm : user's high-rated sequence, mask_prob of tokens replaced by [MASK];
            bidirectional transformer predicts them over all items (cosine/temp),
            the user's other positives are removed from the denominator.
Put next to ease_v2.py.
Usage: python joint.py --data ~/RS/data --lambda_mlm 0.5 --tag joint
"""
import argparse, math, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ease_v2 import temporal_split, dense, ndcg, N_ITEMS, dev, K

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--thr", type=float, default=4.0)
# LightGCN part (SOTA defaults)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--temp", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--rating_w", default="4:0.7,5:1")
ap.add_argument("--time_beta", type=float, default=1.25)
# MLM part
ap.add_argument("--lambda_mlm", type=float, default=0.5)
ap.add_argument("--mask_prob", type=float, default=0.2)
ap.add_argument("--maxlen", type=int, default=200)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--heads", type=int, default=2)
ap.add_argument("--drop", type=float, default=0.2)
ap.add_argument("--seq_batch", type=int, default=64, help="user sequences per step for MLM")
ap.add_argument("--mlm_temp", type=float, default=None, help="default: same as --temp")
# pretrain -> fine-tune
ap.add_argument("--pretrain_epochs", type=int, default=0, help="MLM-only epochs before LightGCN training")
ap.add_argument("--pre_steps", type=int, default=20, help="MLM steps per pretrain epoch")
ap.add_argument("--pre_lr", type=float, default=1e-3)
ap.add_argument("--anchor", type=float, default=0.0,
                help="fine-tune: anchor * mean ||E_item - E_pretrained||^2")
# training
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--eval_every", type=int, default=20)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no_final", action="store_true")
ap.add_argument("--tag", default="joint")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
L = args.maxlen
MT = args.temp if args.mlm_temp is None else args.mlm_temp


def map_w(spec, r):
    kv = sorted((float(a), float(b)) for a, b in (p.split(":") for p in spec.split(",")))
    keys, vals = np.array([k for k, _ in kv]), np.array([v for _, v in kv])
    return vals[np.clip(np.searchsorted(keys, r, side="right") - 1, 0, len(keys) - 1)].astype(np.float32)


def norm_adj(df, n_users):
    N = n_users + N_ITEMS
    u = df.userId.values; i = df.movieId.values + n_users
    rows = np.concatenate([u, i]); cols = np.concatenate([i, u])
    deg = np.bincount(rows, minlength=N).astype(np.float32)
    d = np.zeros_like(deg); d[deg > 0] = deg[deg > 0] ** -0.5
    idx = torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.as_tensor(d[rows] * d[cols]), (N, N)).coalesce().to(dev)


class Joint(nn.Module):
    def __init__(self, n_users):
        super().__init__()
        d = args.dim
        self.n_users = n_users
        self.emb = nn.Embedding(n_users + N_ITEMS, d)          # shared: users + items
        nn.init.normal_(self.emb.weight, std=0.1)
        self.mask_vec = nn.Parameter(torch.randn(d) * 0.1)
        self.pos = nn.Embedding(L, d)
        layer = nn.TransformerEncoderLayer(d, args.heads, 4 * d, args.drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, args.blocks, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)

    def propagate(self, A):
        e = self.emb.weight; out = [e]
        for _ in range(args.layers):
            e = torch.sparse.mm(A, e); out.append(e)
        e = torch.stack(out).mean(0)
        return e[: self.n_users], e[self.n_users:]

    def mlm_hidden(self, seq, masked):
        """seq: (B, L) item ids + 1 (0 = pad); masked: (B, L) bool."""
        E0 = self.emb.weight[self.n_users:]
        h = torch.zeros(*seq.shape, args.dim, device=dev)
        real = seq > 0
        h[real] = E0[seq[real] - 1]
        h[masked] = self.mask_vec
        h = h + self.pos.weight[None]
        h = self.enc(h, src_key_padding_mask=~real)
        return self.ln(h)


def build_seqs(df):
    df = df.sort_values(["userId", "time_idx"])
    df = df[df.rating >= args.thr]
    seqs = np.zeros((n_users, L), dtype=np.int64)
    for u, g in df.groupby("userId"):
        s = g.movieId.values[-L:] + 1
        seqs[u, -len(s):] = s
    return torch.as_tensor(seqs, device=dev)


def mlm_loss(model, seqs, seq_users, pos_mask):
    su = seq_users[torch.randint(0, len(seq_users), (args.seq_batch,), device=dev)]
    sq = seqs[su]
    real = sq > 0
    masked = real & (torch.rand(sq.shape, device=dev) < args.mask_prob)
    none = masked.sum(1) == 0
    if none.any():
        last = real.long().cumsum(1).argmax(1)
        masked[none, last[none]] = True
    h = model.mlm_hidden(sq, masked)[masked]
    tgt = sq[masked] - 1
    rows = su[:, None].expand_as(sq)[masked]
    E0n = F.normalize(model.emb.weight[model.n_users:], dim=1)
    ml = F.normalize(h, dim=1) @ E0n.T / MT
    mm = pos_mask[rows].clone(); mm[torch.arange(len(tgt), device=dev), tgt] = False
    return F.cross_entropy(ml.masked_fill(mm, float("-inf")), tgt)


def scores(model, A):
    model.eval()
    with torch.no_grad():
        U, I = model.propagate(A)
        return F.normalize(U, dim=1) @ F.normalize(I, dim=1).T


def train(hist, epochs, val=None):
    hist = hist.copy()
    n = hist.groupby("userId")["time_idx"].transform("size")
    hist["rel"] = (hist.groupby("userId")["time_idx"].rank(method="first") - 1) / (n - 1).clip(lower=1)
    pos = hist[hist.rating >= args.thr]
    A = norm_adj(pos, n_users)
    pu = torch.as_tensor(pos.userId.values, device=dev)
    pi = torch.as_tensor(pos.movieId.values, device=dev)
    pw = torch.as_tensor(map_w(args.rating_w, pos.rating.values)
                         * np.exp(-args.time_beta * (1 - pos.rel.values)).astype(np.float32), device=dev)
    pos_mask = dense(pos, n_users, np.ones(len(pos))) > 0
    seen = dense(hist, n_users, np.ones(len(hist))) > 0
    seqs = build_seqs(hist)
    seq_users = torch.nonzero((seqs > 0).sum(1) >= 2).squeeze(1)
    model = Joint(n_users).to(dev)
    E_pre = None
    if args.pretrain_epochs > 0:
        popt = torch.optim.Adam(model.parameters(), lr=args.pre_lr)
        for pe in range(1, args.pretrain_epochs + 1):
            model.train(); tm = 0.0
            for _ in range(args.pre_steps):
                lm = mlm_loss(model, seqs, seq_users, pos_mask)
                popt.zero_grad(); lm.backward(); popt.step(); tm += lm.item()
            if pe % 20 == 0 or pe == args.pretrain_epochs:
                print(f"pretrain {pe:4d}  mlm {tm / args.pre_steps:.4f}", flush=True)
        E_pre = model.emb.weight[n_users:].detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_ep, best_state, bad = -1, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(pu), device=dev)
        tot_g = tot_m = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s: s + args.batch]
            u, i, w = pu[b], pi[b], pw[b]
            # ---- LightGCN loss ----
            U, I = model.propagate(A)
            In = F.normalize(I, dim=1)
            logits = F.normalize(U[u], dim=1) @ In.T / args.temp
            m = pos_mask[u].clone(); m[torch.arange(len(u), device=dev), i] = False
            logits = logits.masked_fill(m, float("-inf"))
            l_g = (F.cross_entropy(logits, i, reduction="none") * w).sum() / w.sum()
            loss = l_g
            # ---- MLM loss (joint) ----
            if args.lambda_mlm > 0:
                l_m = mlm_loss(model, seqs, seq_users, pos_mask)
                loss = loss + args.lambda_mlm * l_m
                tot_m += l_m.item()
            # ---- anchor to pretrained item embeddings ----
            if E_pre is not None and args.anchor > 0:
                loss = loss + args.anchor * (model.emb.weight[n_users:] - E_pre).pow(2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot_g += l_g.item()
        if val is not None and ep % args.eval_every == 0:
            gt_mat, vusers = val
            S = scores(model, A)[vusers]
            S[seen[vusers]] = -float("inf")
            sc = ndcg(torch.topk(S, K, 1).indices, vusers, gt_mat)
            flag = ""
            if sc > best:
                best, best_ep, bad, flag = sc, ep, 0, " *"
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"ep {ep:4d}  lgcn {tot_g:.3f}  mlm {tot_m:.3f}  val NDCG@20 {sc:.5f}{flag}", flush=True)
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, A, seen, best_ep, best


d = os.path.expanduser(args.data)
tr = pd.read_csv(os.path.join(d, "train.csv"))
te = pd.read_csv(os.path.join(d, "test.csv"))
n_users = int(tr.userId.max()) + 1
os.makedirs("scores", exist_ok=True)
print("device:", dev, "|", vars(args), flush=True)

inner, val = temporal_split(tr)
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
t = time.time()
model, A, _, best_ep, best = train(inner, args.epochs, val=(gt_mat, vusers))
print(f"best epoch {best_ep}  val NDCG@20 {best:.5f}  ({time.time()-t:.0f}s)", flush=True)
torch.save(scores(model, A).cpu(), f"scores/{args.tag}_val.pt")
if args.no_final:
    raise SystemExit

model, A, seen, _, _ = train(tr, best_ep)
S = scores(model, A)
torch.save(S.cpu(), f"scores/{args.tag}_test.pt")
tusers = torch.as_tensor(te.userId.values, device=dev)
St = S[tusers].clone(); St[seen[tusers]] = -float("inf")
recs = torch.topk(St, K, 1).indices.cpu().numpy()
out = f"submission_{args.tag}.csv"
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(out, index=False)
print(f"saved {out}  (val {best:.5f}, epochs {best_ep})")