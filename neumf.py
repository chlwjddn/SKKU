"""
NeuMF (He et al., WWW 2017) on the HW2 setup.
- Positives: rating >= 4. Pointwise BCE with `neg` uniform negatives per positive (as in the paper).
- GMF: p_u * q_i (dim = factors);  MLP: [p'_u ; q'_i] (2 x mlp_dim) -> mlp layers
- Prediction: sigmoid(h^T [GMF ; MLP])
- Evaluation: full-item ranking NDCG@20 on the temporal validation split (seen items masked),
  NOT the paper's 1-positive-vs-99-negatives protocol.
- Early stopping on val, retrain on full train for best #epochs, save scores + submission.
Put next to ease_v2.py.
Usage: python neumf.py --data ~/RS/data --tag neumf
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
ap.add_argument("--factors", type=int, default=32, help="GMF dim / last MLP layer dim")
ap.add_argument("--mlp_dim", type=int, default=64, help="MLP embedding dim per side")
ap.add_argument("--neg", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--drop", type=float, default=0.0)
ap.add_argument("--batch", type=int, default=2048)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--eval_every", type=int, default=2)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no_final", action="store_true")
ap.add_argument("--tag", default="neumf")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)


class NeuMF(nn.Module):
    def __init__(self, n_users):
        super().__init__()
        f, m = args.factors, args.mlp_dim
        self.gu, self.gi = nn.Embedding(n_users, f), nn.Embedding(N_ITEMS, f)
        self.mu, self.mi = nn.Embedding(n_users, m), nn.Embedding(N_ITEMS, m)
        dims = [2 * m, m, f]                       # e.g. 128 -> 64 -> 32
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(args.drop)]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(2 * f, 1)
        for e in [self.gu, self.gi, self.mu, self.mi]:
            nn.init.normal_(e.weight, std=0.01)

    def forward(self, u, i):
        g = self.gu(u) * self.gi(i)
        h = self.mlp(torch.cat([self.mu(u), self.mi(i)], -1))
        return self.out(torch.cat([g, h], -1)).squeeze(-1)

    @torch.no_grad()
    def score_users(self, users, chunk=32):
        """scores (len(users), N_ITEMS)."""
        items = torch.arange(N_ITEMS, device=dev)
        out = []
        for s in range(0, len(users), chunk):
            u = users[s: s + chunk]
            uu = u[:, None].expand(-1, N_ITEMS).reshape(-1)
            ii = items[None, :].expand(len(u), -1).reshape(-1)
            out.append(self.forward(uu, ii).view(len(u), N_ITEMS))
        return torch.cat(out)


def train(hist, epochs, val=None):
    pos = hist[hist.rating >= args.thr]
    pu = torch.as_tensor(pos.userId.values, device=dev); pi = torch.as_tensor(pos.movieId.values, device=dev)
    seen = dense(hist, n_users, np.ones(len(hist))) > 0
    model = NeuMF(n_users).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_ep, best_state, bad = -1, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        # positives + `neg` uniform negatives per positive
        u = pu.repeat(1 + args.neg)
        i = torch.cat([pi, torch.randint(0, N_ITEMS, (len(pi) * args.neg,), device=dev)])
        y = torch.cat([torch.ones(len(pi), device=dev), torch.zeros(len(pi) * args.neg, device=dev)])
        perm = torch.randperm(len(u), device=dev)
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s: s + args.batch]
            loss = F.binary_cross_entropy_with_logits(model(u[b], i[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if val is not None and ep % args.eval_every == 0:
            model.eval()
            gt_mat, vusers = val
            S = model.score_users(vusers)
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
    model.eval()
    return model, seen, best_ep, best


d = os.path.expanduser(args.data)
tr = pd.read_csv(os.path.join(d, "train.csv")); te = pd.read_csv(os.path.join(d, "test.csv"))
n_users = int(tr.userId.max()) + 1
all_users = torch.arange(n_users, device=dev)
os.makedirs("scores", exist_ok=True)
print("device:", dev, "|", vars(args), flush=True)

inner, val = temporal_split(tr)
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
t = time.time()
model, _, best_ep, best = train(inner, args.epochs, val=(gt_mat, vusers))
print(f"best epoch {best_ep}  val NDCG@20 {best:.5f}  ({time.time()-t:.0f}s)", flush=True)
torch.save(model.score_users(all_users).cpu(), f"scores/{args.tag}_val.pt")
if args.no_final:
    raise SystemExit

model, seen, _, _ = train(tr, best_ep)
S = model.score_users(all_users)
torch.save(S.cpu(), f"scores/{args.tag}_test.pt")
tusers = torch.as_tensor(te.userId.values, device=dev)
St = S[tusers].clone(); St[seen[tusers]] = -float("inf")
recs = torch.topk(St, K, 1).indices.cpu().numpy()
out = f"submission_{args.tag}.csv"
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(out, index=False)
print(f"saved {out}  (val {best:.5f}, epochs {best_ep})")
