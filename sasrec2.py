"""
SASRec v2 for skku-esw-4024-recommender-system
- Input sequence: high-rated items (default) or all rated items (--input all)
- Each position predicts the next future_k items; targets must be rating >= thr
- Full softmax over all items; the other future targets of the same position are
  removed from the denominator (no positive pushes another positive)
- Target weight = rating weight x exp(-time_beta * (1 - rel)), rel = position in user's history
- Optional cosine / temperature scoring (--temp > 0)
- Early stopping on validation NDCG@20, then retrain on full train for best #epochs
Put next to ease_v2.py.
Usage: python sasrec2.py --data ~/RS/data --tag sas2
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
ap.add_argument("--input", choices=["high", "all"], default="high")
ap.add_argument("--maxlen", type=int, default=200)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--heads", type=int, default=2)
ap.add_argument("--drop", type=float, default=0.3)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--eval_every", type=int, default=5)
ap.add_argument("--patience", type=int, default=10, help="in evals")
ap.add_argument("--future_k", type=int, default=10)
ap.add_argument("--no_mask_pos", action="store_true")
ap.add_argument("--rating_w", default="4:0.7,5:1")
ap.add_argument("--time_beta", type=float, default=1.25)
ap.add_argument("--temp", type=float, default=0.0, help="0 = dot product, >0 = cosine / temp")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no_final", action="store_true")
ap.add_argument("--tag", default="sas2")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
L, KF = args.maxlen, args.future_k


def map_w(spec, r):
    kv = sorted((float(a), float(b)) for a, b in (p.split(":") for p in spec.split(",")))
    keys, vals = np.array([k for k, _ in kv]), np.array([v for _, v in kv])
    return vals[np.clip(np.searchsorted(keys, r, side="right") - 1, 0, len(keys) - 1)]


def build_seqs(df):
    """per user: items(+1), target weight (0 if not a valid target)."""
    df = df.sort_values(["userId", "time_idx"]).copy()
    n = df.groupby("userId")["movieId"].transform("size")
    df["rel"] = df.groupby("userId").cumcount() / (n - 1).clip(lower=1)
    if args.input == "high":
        df = df[df.rating >= args.thr]
    w = np.where(df.rating.values >= args.thr,
                 map_w(args.rating_w, df.rating.values) * np.exp(-args.time_beta * (1 - df.rel.values)), 0.0)
    df["w"] = w
    seqs = {}
    for u, g in df.groupby("userId"):
        seqs[u] = (g.movieId.values + 1, g.w.values.astype(np.float32))
    return seqs


def train_tensors(seqs):
    W = L + KF
    items, ws = [], []
    for s, w in seqs.values():
        for end in range(len(s), 1, -L):
            st = max(0, end - W)
            items.append(s[st:end]); ws.append(w[st:end])
    I = np.zeros((len(items), W), dtype=np.int64); Wt = np.zeros((len(items), W), dtype=np.float32)
    for r, (a, b) in enumerate(zip(items, ws)):
        I[r, -len(a):] = a; Wt[r, -len(b):] = b
    return torch.as_tensor(I, device=dev), torch.as_tensor(Wt, device=dev)


def last_input(seqs, users):
    arr = np.zeros((len(users), L), dtype=np.int64)
    for r, u in enumerate(users):
        s = seqs[u][0][-L:]
        arr[r, -len(s):] = s
    return torch.as_tensor(arr, device=dev)


class SASRec(nn.Module):
    def __init__(self):
        super().__init__()
        d = args.dim
        self.item = nn.Embedding(N_ITEMS + 1, d, padding_idx=0)
        self.pos = nn.Embedding(L, d)
        layer = nn.TransformerEncoderLayer(d, args.heads, 4 * d, args.drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, args.blocks, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(args.drop)
        self.register_buffer("causal", torch.triu(torch.ones(L, L, dtype=torch.bool), 1))
        nn.init.normal_(self.item.weight, std=0.02)

    def forward(self, x):
        h = self.item(x) * math.sqrt(args.dim) + self.pos.weight[None]
        h = self.ln(self.enc(self.drop(h), mask=self.causal))
        E = self.item.weight[1:]
        if args.temp > 0:
            return F.normalize(h, dim=-1) @ F.normalize(E, dim=-1).T / args.temp
        return h @ E.T  # (B, L, N_ITEMS)


def user_scores(model, seqs, users):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(users), 256):
            out.append(model(last_input(seqs, users[s: s + 256]))[:, -1])
    return torch.cat(out)


def batch_loss(model, I, Wt):
    x = I[:, :L]
    logits = model(x)                                   # (B, L, N)
    B = x.shape[0]
    tgt = torch.stack([I[:, k: L + k] for k in range(1, KF + 1)], -1)    # (B, L, KF), 0 = pad
    tw = torch.stack([Wt[:, k: L + k] for k in range(1, KF + 1)], -1)   # weight, 0 = invalid
    valid = (tgt > 0) & (tw > 0) & (x > 0)[..., None]
    ti = (tgt - 1).clamp(min=0)
    if not args.no_mask_pos:
        mask = torch.zeros_like(logits, dtype=torch.bool)
        bi = torch.arange(B, device=dev)[:, None, None].expand_as(ti)
        li = torch.arange(L, device=dev)[None, :, None].expand_as(ti)
        mask[bi[valid], li[valid], ti[valid]] = True
        lse = torch.logsumexp(logits.masked_fill(mask, float("-inf")), -1)   # (B, L)
    else:
        lse = torch.logsumexp(logits, -1)
    st = logits.gather(-1, ti)                                            # (B, L, KF)
    if not args.no_mask_pos:
        nll = torch.logaddexp(lse[..., None], st) - st
    else:
        nll = lse[..., None] - st
    w = tw * valid
    return (nll * w).sum() / w.sum().clamp(min=1e-8)


def train(hist, epochs, val=None):
    seqs = build_seqs(hist)
    I, Wt = train_tensors(seqs)
    model = SASRec().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    seen = dense(hist, n_users, np.ones(len(hist))) > 0
    best, best_ep, best_state, bad = -1, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(I), device=dev)
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s: s + args.batch]
            loss = batch_loss(model, I[b], Wt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if val is not None and ep % args.eval_every == 0:
            gt_mat, vusers = val
            S = user_scores(model, seqs, vusers.cpu().numpy())
            S[seen[vusers]] = -float("inf")
            sc = ndcg(torch.topk(S, K, 1).indices, vusers, gt_mat)
            flag = ""
            if sc > best:
                best, best_ep, bad, flag = sc, ep, 0, " *"
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"ep {ep:4d}  loss {tot:.2f}  val NDCG@20 {sc:.5f}{flag}", flush=True)
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, seqs, seen, best_ep, best


d = os.path.expanduser(args.data)
tr = pd.read_csv(os.path.join(d, "train.csv"))
te = pd.read_csv(os.path.join(d, "test.csv"))
n_users = int(tr.userId.max()) + 1
all_users = np.arange(n_users)
os.makedirs("scores", exist_ok=True)
print("device:", dev, "|", vars(args), flush=True)

inner, val = temporal_split(tr)
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
t = time.time()
model, seqs, _, best_ep, best = train(inner, args.epochs, val=(gt_mat, vusers))
print(f"best epoch {best_ep}  val NDCG@20 {best:.5f}  ({time.time()-t:.0f}s)", flush=True)
torch.save(user_scores(model, seqs, all_users).cpu(), f"scores/{args.tag}_val.pt")
if args.no_final:
    raise SystemExit

model, seqs, seen, _, _ = train(tr, best_ep)
S = user_scores(model, seqs, all_users)
torch.save(S.cpu(), f"scores/{args.tag}_test.pt")
tusers = torch.as_tensor(te.userId.values, device=dev)
St = S[tusers].clone()
St[seen[tusers]] = -float("inf")
recs = torch.topk(St, K, 1).indices.cpu().numpy()
out = f"submission_{args.tag}.csv"
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(out, index=False)
print(f"saved {out}  (val {best:.5f}, epochs {best_ep})")