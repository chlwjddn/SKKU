"""
SASRec for skku-esw-4024-recommender-system
- Sequence = user's high-rated (>= thr) items in time order (falls back to all items if none)
- Loss: full softmax cross-entropy over all items; each position predicts the next
  future_k items (test is a future *set*, not just the next item)
- Early stopping on validation NDCG@20, retrain on full train for best #epochs
- Saves scores/{tag}_{val,test}.pt + submission
Put next to ease_v2.py.
Usage: python sasrec.py --data ~/RS/data --future_k 1 --tag sas_k1
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
ap.add_argument("--maxlen", type=int, default=200)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--heads", type=int, default=2)
ap.add_argument("--drop", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--eval_every", type=int, default=5)
ap.add_argument("--patience", type=int, default=8, help="in evals")
ap.add_argument("--future_k", type=int, default=1)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tag", default="sas")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
L, KF = args.maxlen, args.future_k


def build_seqs(df):
    df = df.sort_values(["userId", "time_idx"])
    seqs = {}
    for u, g in df.groupby("userId"):
        s = g.movieId.values[g.rating.values >= args.thr]
        if len(s) == 0:
            s = g.movieId.values
        seqs[u] = s + 1  # 0 = padding
    return seqs


def train_tensor(seqs):
    W = L + KF
    rows = []
    for s in seqs.values():
        for end in range(len(s), 1, -L):
            rows.append(s[max(0, end - W): end])
    arr = np.zeros((len(rows), W), dtype=np.int64)
    for r, w in enumerate(rows):
        arr[r, -len(w):] = w
    return torch.as_tensor(arr, device=dev)


def last_input(seqs, users):
    arr = np.zeros((len(users), L), dtype=np.int64)
    for r, u in enumerate(users):
        s = seqs[u][-L:]
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
        h = self.enc(self.drop(h), mask=self.causal)
        return self.ln(h) @ self.item.weight[1:].T  # (B, L, N_ITEMS); column j = movieId j


def user_scores(model, seqs, users):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(users), 256):
            out.append(model(last_input(seqs, users[s: s + 256]))[:, -1])
    return torch.cat(out)


def train(hist, epochs, val=None):
    seqs = build_seqs(hist)
    data = train_tensor(seqs)
    model = SASRec().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    seen = dense(hist, n_users, np.ones(len(hist))) > 0
    best, best_ep, best_state, bad = -1, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(data), device=dev)
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = data[perm[s: s + args.batch]]
            logits = model(b[:, :L]).reshape(-1, N_ITEMS)
            loss = 0.0
            for k in range(1, KF + 1):
                tgt = (b[:, k: L + k] - 1).reshape(-1)  # -1 = padding -> ignored
                loss = loss + F.cross_entropy(logits, tgt, ignore_index=-1)
            loss = loss / KF
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
print("device:", dev, "|", vars(args))

inner, val = temporal_split(tr)
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
t = time.time()
model, seqs, _, best_ep, best = train(inner, args.epochs, val=(gt_mat, vusers))
print(f"best epoch {best_ep}  val NDCG@20 {best:.5f}  ({time.time()-t:.0f}s)")
torch.save(user_scores(model, seqs, all_users).cpu(), f"scores/{args.tag}_val.pt")

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