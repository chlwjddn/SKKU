"""
Mult-VAE / Mult-DAE (Liang et al., WWW 2018) on the HW2 setup.
- Input : user's high-rated (>= 4) items as a binary vector, L2-normalised, input dropout
- Model : N -> hidden -> latent (mu, logvar) -> hidden -> N, tanh
- Loss  : multinomial log-likelihood  -sum_i t_i log softmax(logits)_i  + beta * KL  (KL annealed)
          t_i = 1 for positives, or (with --weighted) rating weight x exp(-time_beta (1 - rel))
- Eval  : full-item ranking NDCG@20 on the temporal validation split, seen items masked
Put next to ease_v2.py.
Usage: python multvae.py --data ~/RS/data --tag mvae
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
ap.add_argument("--model", choices=["vae", "dae"], default="vae")
ap.add_argument("--hidden", type=int, default=600)
ap.add_argument("--latent", type=int, default=200)
ap.add_argument("--drop", type=float, default=0.5)
ap.add_argument("--beta_max", type=float, default=0.2)
ap.add_argument("--anneal_epochs", type=int, default=300)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--eval_every", type=int, default=5)
ap.add_argument("--patience", type=int, default=20)
ap.add_argument("--weighted", action="store_true", help="target weights: rating x recency")
ap.add_argument("--input", choices=["high", "two_channel", "signed"], default="high",
                help="high: >=4 binary | two_channel: [>=4 ; <4] binary | signed: rating - user mean (all ratings)")
ap.add_argument("--future_cuts", default="",
                help='train on (history before cut -> high items in the next window), e.g. "0.4,0.5,0.6,0.7,0.8"')
ap.add_argument("--future_w", type=float, default=0.25, help="target window length relative to cut (0.25 = like val/test)")
ap.add_argument("--rating_w", default="4:0.7,5:1")
ap.add_argument("--tta", type=int, default=0, help="test-time augmentation: extra passes with input dropout, averaged")
ap.add_argument("--tta_drop", type=float, default=0.1)
ap.add_argument("--time_beta", type=float, default=1.25)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no_final", action="store_true")
ap.add_argument("--tag", default="mvae")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)


def map_w(spec, r):
    kv = sorted((float(a), float(b)) for a, b in (p.split(":") for p in spec.split(",")))
    keys, vals = np.array([k for k, _ in kv]), np.array([v for _, v in kv])
    return vals[np.clip(np.searchsorted(keys, r, side="right") - 1, 0, len(keys) - 1)]


class MultVAE(nn.Module):
    def __init__(self):
        super().__init__()
        h, z = args.hidden, args.latent
        self.enc = nn.Linear(N_ITEMS * (2 if args.input == "two_channel" else 1), h)
        self.mu, self.lv = nn.Linear(h, z), nn.Linear(h, z)
        self.dec1, self.dec2 = nn.Linear(z, h), nn.Linear(h, N_ITEMS)
        self.drop = nn.Dropout(args.drop)
        for m in [self.enc, self.mu, self.lv, self.dec1, self.dec2]:
            nn.init.xavier_uniform_(m.weight); nn.init.normal_(m.bias, std=0.001)

    def forward(self, x):
        x = self.drop(F.normalize(x, dim=1))
        h = torch.tanh(self.enc(x))
        mu = self.mu(h)
        if args.model == "vae":
            lv = self.lv(h)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
            kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()
        else:
            z, kl = mu, torch.zeros((), device=dev)
        return self.dec2(torch.tanh(self.dec1(z))), kl


def build_input(h):
    hi = h[h.rating >= args.thr]
    Xhi = dense(hi, n_users, np.ones(len(hi)))
    if args.input == "high":
        return Xhi
    if args.input == "two_channel":
        lo = h[h.rating < args.thr]
        return torch.cat([Xhi, dense(lo, n_users, np.ones(len(lo)))], 1)
    c = (h.rating - h.groupby("userId").rating.transform("mean")).values
    return dense(h, n_users, c)


def matrices(hist):
    h = hist.copy()
    n = h.groupby("userId")["time_idx"].transform("size")
    h["rel"] = (h.groupby("userId")["time_idx"].rank(method="first") - 1) / (n - 1).clip(lower=1)
    hi = h[h.rating >= args.thr]
    Xhi = dense(hi, n_users, np.ones(len(hi)))
    Xin = build_input(h)
    if args.weighted:
        w = map_w(args.rating_w, hi.rating.values) * np.exp(-args.time_beta * (1 - hi.rel.values))
        T = dense(hi, n_users, w)
    else:
        T = Xhi
    seen = dense(h, n_users, np.ones(len(h))) > 0
    samples = None
    if args.future_cuts:
        Xs, Ts = [], []
        for c in map(float, args.future_cuts.split(",")):
            past = h[h.rel < c]
            fut = h[(h.rel >= c) & (h.rel < min(1.0, c * (1 + args.future_w))) & (h.rating >= args.thr)]
            Xs.append(build_input(past))
            wf = map_w(args.rating_w, fut.rating.values) if args.weighted else np.ones(len(fut))
            Ts.append(dense(fut, n_users, wf))
        Xs, Ts = torch.cat(Xs), torch.cat(Ts)
        keep = (Ts.sum(1) > 0) & (Xs.abs().sum(1) > 0)
        samples = (Xs[keep], Ts[keep])
        print(f"  future-window samples: {int(keep.sum())}", flush=True)
    return Xin, T, seen, samples


@torch.no_grad()
def scores(model, Xin):
    model.eval()
    if args.tta > 0:
        out = model(Xin)[0]
        g = torch.Generator(device=Xin.device).manual_seed(args.seed)
        for _ in range(args.tta):
            keep = (torch.rand(Xin.shape, generator=g, device=Xin.device) >= args.tta_drop).float()
            Xa = Xin * keep
            out = out + model(F.normalize(Xa, dim=1) if args.input != "signed" else Xa)[0]
        return out / (args.tta + 1)
    return model(Xin)[0]


def train(hist, epochs, val=None):
    Xin, T, seen, samples = matrices(hist)
    if samples is not None:
        TX, TT = samples
    else:
        users = torch.nonzero(T.sum(1) > 0).squeeze(1)
        TX, TT = Xin[users], T[users]
    model = MultVAE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_ep, best_state, bad = -1, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        beta = args.beta_max * min(1.0, ep / max(args.anneal_epochs, 1))
        perm = torch.randperm(len(TX), device=dev)
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s: s + args.batch]
            logits, kl = model(TX[b])
            nll = -(F.log_softmax(logits, 1) * TT[b]).sum(1).mean()
            loss = nll + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += nll.item()
        if val is not None and ep % args.eval_every == 0:
            gt_mat, vusers = val
            S = scores(model, Xin)[vusers]
            S[seen[vusers]] = -float("inf")
            sc = ndcg(torch.topk(S, K, 1).indices, vusers, gt_mat)
            flag = ""
            if sc > best:
                best, best_ep, bad, flag = sc, ep, 0, " *"
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"ep {ep:4d}  nll {tot:.2f}  beta {beta:.3f}  val NDCG@20 {sc:.5f}{flag}", flush=True)
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, Xin, seen, best_ep, best


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
model, Xin, _, best_ep, best = train(inner, args.epochs, val=(gt_mat, vusers))
print(f"best epoch {best_ep}  val NDCG@20 {best:.5f}  ({time.time()-t:.0f}s)", flush=True)
torch.save(scores(model, Xin).cpu(), f"scores/{args.tag}_val.pt")
if args.no_final:
    raise SystemExit

model, Xin, seen, _, _ = train(tr, best_ep)
S = scores(model, Xin)
torch.save(S.cpu(), f"scores/{args.tag}_test.pt")
tusers = torch.as_tensor(te.userId.values, device=dev)
St = S[tusers].clone(); St[seen[tusers]] = -float("inf")
recs = torch.topk(St, K, 1).indices.cpu().numpy()
out = f"submission_{args.tag}.csv"
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(out, index=False)
print(f"saved {out}  (val {best:.5f}, epochs {best_ep})")
