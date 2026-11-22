"""
FMLP-Rec (Zhou et al., WWW 2022) adapted to the HW2 setup.
  - Encoder: filter blocks = FFT along the sequence -> learnable complex filter -> iFFT
             + FFN, pre-LN, dropout, residual (no attention)
  - Input tokens: item emb + rating emb (10 bins) + position emb (all ratings in history)
  - Objective: future-window, same as evaluation: (history before cut c) -> high-rated in [c, c(1+fw))
               user vector = last real position; cosine/temp softmax, other targets masked, 5-star weighted
  - Inference: full history -> top-20 (seen masked)
Put next to ease_v2.py.
Usage: python fmlp.py --data ~/RS/data --tag fmlp
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
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--drop", type=float, default=0.3)
ap.add_argument("--maxlen", type=int, default=200)
ap.add_argument("--temp", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--epochs", type=int, default=600)
ap.add_argument("--eval_every", type=int, default=5)
ap.add_argument("--patience", type=int, default=12)
ap.add_argument("--cuts", default="0.3,0.4,0.5,0.6,0.7,0.75,0.8")
ap.add_argument("--fw", type=float, default=0.25)
ap.add_argument("--rating_w", default="4:0.7,5:1")
ap.add_argument("--input", choices=["all", "high"], default="all")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no_final", action="store_true")
ap.add_argument("--tag", default="fmlp")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
L = args.maxlen


def map_w(spec, r):
    kv = sorted((float(a), float(b)) for a, b in (p.split(":") for p in spec.split(",")))
    keys, vals = np.array([k for k, _ in kv]), np.array([v for _, v in kv])
    return vals[np.clip(np.searchsorted(keys, r, side="right") - 1, 0, len(keys) - 1)]


class FilterBlock(nn.Module):
    """FMLP core: FFT -> learnable complex filter -> iFFT, then FFN; pre-LN residuals."""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(L // 2 + 1, dim, 2) * 0.02)   # complex filter
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(args.drop), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(args.drop)

    def forward(self, x):                                   # (B, L, d)
        h = self.ln1(x)
        f = torch.fft.rfft(h, dim=1, norm="ortho")
        f = f * torch.view_as_complex(self.w)
        h = torch.fft.irfft(f, n=L, dim=1, norm="ortho")
        x = x + self.drop(h)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class FMLP(nn.Module):
    def __init__(self):
        super().__init__()
        d = args.dim
        self.item = nn.Embedding(N_ITEMS + 1, d, padding_idx=0)
        self.rate = nn.Embedding(11, d, padding_idx=0)
        self.pos = nn.Embedding(L, d)
        self.blocks = nn.ModuleList(FilterBlock(d) for _ in range(args.blocks))
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(args.drop)
        nn.init.normal_(self.item.weight, std=0.02); nn.init.normal_(self.rate.weight, std=0.02)

    def user_vec(self, items, rates):
        real = items > 0
        h = self.item(items) * math.sqrt(args.dim) + self.rate(rates) + self.pos.weight[None]
        h = self.drop(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h) * real[..., None]
        idx = real.long().cumsum(1).argmax(1)
        v = h[torch.arange(len(h), device=items.device), idx]
        return F.normalize(v, dim=1)


def build_samples(hist):
    h = hist.sort_values(["userId", "time_idx"]).copy()
    n = h.groupby("userId")["time_idx"].transform("size")
    h["rel"] = h.groupby("userId").cumcount() / (n - 1).clip(lower=1)
    h["rb"] = (h.rating * 2).astype(int).clip(1, 10)
    src = h if args.input == "all" else h[h.rating >= args.thr]
    hi = h[h.rating >= args.thr]
    cuts = [float(c) for c in args.cuts.split(",")]
    I, R, tg, ti, tw = [], [], [], [], []
    gid = 0
    for c in cuts:
        fut = hi[(hi.rel >= c) & (hi.rel < min(1.0, c * (1 + args.fw)))]
        users = fut.userId.unique()
        past = src[(src.rel < c) & src.userId.isin(users)]
        users = np.intersect1d(users, past.userId.unique())
        if len(users) == 0:
            continue
        rows = {}
        for u, g in past[past.userId.isin(users)].groupby("userId"):
            iv = g.movieId.values[-L:] + 1; rv = g.rb.values[-L:]
            a = np.zeros(L, dtype=np.int64); b = np.zeros(L, dtype=np.int64)
            a[-len(iv):] = iv; b[-len(rv):] = rv
            rows[u] = gid; I.append(a); R.append(b); gid += 1
        f = fut[fut.userId.isin(users)]
        tg.append(np.array([rows[u] for u in f.userId.values]))
        ti.append(f.movieId.values)
        tw.append(map_w(args.rating_w, f.rating.values))
    I = torch.as_tensor(np.stack(I), device=dev); R = torch.as_tensor(np.stack(R), device=dev)
    tg = torch.as_tensor(np.concatenate(tg), device=dev)
    ti = torch.as_tensor(np.concatenate(ti), device=dev)
    tw = torch.as_tensor(np.concatenate(tw).astype(np.float32), device=dev)
    pm = torch.zeros(len(I), N_ITEMS, dtype=torch.bool, device=dev); pm[tg, ti] = True
    print(f"  groups {len(I)}  targets {len(ti)}", flush=True)
    return I, R, tg, ti, tw, pm


def full_inputs(hist):
    h = hist.sort_values(["userId", "time_idx"]).copy()
    h["rb"] = (h.rating * 2).astype(int).clip(1, 10)
    src = h if args.input == "all" else h[h.rating >= args.thr]
    I = np.zeros((n_users, L), dtype=np.int64); R = np.zeros((n_users, L), dtype=np.int64)
    for u, g in src.groupby("userId"):
        iv = g.movieId.values[-L:] + 1; rv = g.rb.values[-L:]
        I[u, -len(iv):] = iv; R[u, -len(rv):] = rv
    seen = dense(h, n_users, np.ones(len(h))) > 0
    return torch.as_tensor(I, device=dev), torch.as_tensor(R, device=dev), seen


@torch.no_grad()
def all_scores(model, I, R):
    model.eval()
    En = F.normalize(model.item.weight[1:], dim=1)
    out = []
    for s in range(0, len(I), 256):
        out.append(model.user_vec(I[s:s+256], R[s:s+256]) @ En.T)
    return torch.cat(out)


def train(hist, epochs, val=None):
    I, R, tg, ti, tw, pm = build_samples(hist)
    If, Rf, seen = full_inputs(hist)
    model = FMLP().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_ep, best_state, bad = -1, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(ti), device=dev)
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s: s + args.batch]
            g, i, w = tg[b], ti[b], tw[b]
            ug, inv = torch.unique(g, return_inverse=True)
            V = model.user_vec(I[ug], R[ug])[inv]
            En = F.normalize(model.item.weight[1:], dim=1)
            logits = V @ En.T / args.temp
            m = pm[g].clone(); m[torch.arange(len(g), device=dev), i] = False
            logits = logits.masked_fill(m, float("-inf"))
            loss = (F.cross_entropy(logits, i, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if val is not None and ep % args.eval_every == 0:
            gt_mat, vusers = val
            S = all_scores(model, If, Rf)[vusers]
            S[seen[vusers]] = -float("inf")
            sc = ndcg(torch.topk(S, K, 1).indices, vusers, gt_mat)
            flag = ""
            if sc > best:
                best, best_ep, bad, flag = sc, ep, 0, " *"
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"ep {ep:4d}  loss {tot:.3f}  val NDCG@20 {sc:.5f}{flag}", flush=True)
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, If, Rf, seen, best_ep, best


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
model, If, Rf, _, best_ep, best = train(inner, args.epochs, val=(gt_mat, vusers))
print(f"best epoch {best_ep}  val NDCG@20 {best:.5f}  ({time.time()-t:.0f}s)", flush=True)
torch.save(all_scores(model, If, Rf).cpu(), f"scores/{args.tag}_val.pt")
if args.no_final:
    raise SystemExit

model, If, Rf, seen, _, _ = train(tr, best_ep)
S = all_scores(model, If, Rf)
torch.save(S.cpu(), f"scores/{args.tag}_test.pt")
tusers = torch.as_tensor(te.userId.values, device=dev)
St = S[tusers].clone(); St[seen[tusers]] = -float("inf")
recs = torch.topk(St, K, 1).indices.cpu().numpy()
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(f"submission_{args.tag}.csv", index=False)
print(f"saved submission_{args.tag}.csv  (val {best:.5f}, epochs {best_ep})")
