"""
SimGCL-style CL on LightGCN propagation, MAIN LOSS = AlphaRec-style InfoNCE
  (cosine(u, i)/temp, 1 positive vs n_neg sampled negatives; scores = cosine).
  - Two extra propagations with random noise injected at every layer:
      e' = e + eps * sign(e) * (u / |u|)        (u ~ Uniform, per layer)
  - InfoNCE between the two views (users and items in the batch), temperature tau_cl
  - Total loss = BPR + lambda_cl * CL + reg
실행:
  CUDA_VISIBLE_DEVICES=$D0 python simgcl.py --data ~/RS/data --tag simgcl
동작: (1) val 분할 학습 -> best epoch  (2) train 전체 재학습 -> test 점수 + 제출 파일
"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

N_ITEMS = 9724
K_EVAL = 20
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def temporal_split(df, frac=0.8):
    df = df.sort_values(["userId", "time_idx"])
    rank = df.groupby("userId").cumcount()
    n = df.groupby("userId")["userId"].transform("size")
    is_tr = rank < np.floor(n * frac)
    return df[is_tr], df[~is_tr]


def seen_mask(df, nU):
    m = torch.zeros(nU, N_ITEMS, dtype=torch.bool, device=dev)
    m[torch.tensor(df.userId.values, device=dev), torch.tensor(df.movieId.values, device=dev)] = True
    return m


def build_adj(u, i, nU):
    n = nU + N_ITEMS
    r = np.concatenate([u, i + nU]); c = np.concatenate([i + nU, u])
    deg = np.bincount(r, minlength=n).astype(np.float32)
    dinv = np.zeros_like(deg); nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    v = dinv[r] * dinv[c]
    return torch.sparse_coo_tensor(torch.tensor(np.vstack([r, c])), torch.tensor(v), (n, n)).coalesce().to(dev)


class SimGCL(nn.Module):
    def __init__(self, nU, dim, layers, A, eps):
        super().__init__()
        self.nU, self.L, self.A, self.eps = nU, layers, A, eps
        self.emb = nn.Embedding(nU + N_ITEMS, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self, noise=False):
        e = self.emb.weight; outs = [e]                   # AlphaRec/LightGCN: e0 INCLUDED
        for _ in range(self.L):
            e = torch.sparse.mm(self.A, e)
            if noise:
                u = F.normalize(torch.rand_like(e) - 0.5, dim=1)
                e = e + self.eps * torch.sign(e) * u
            outs.append(e)
        E = torch.stack(outs).mean(0)
        return E[:self.nU], E[self.nU:]

    @torch.no_grad()
    def scores(self):
        U, I = self.propagate()
        return F.normalize(U, dim=1) @ F.normalize(I, dim=1).T


def sample_neg(u, seen):
    j = torch.randint(0, N_ITEMS, (len(u),), device=dev)
    bad = seen[u, j]
    while bad.any():
        j[bad] = torch.randint(0, N_ITEMS, (int(bad.sum()),), device=dev)
        bad = seen[u, j]
    return j


def sample_dns(u, seen, U, I, m):
    cand = torch.randint(0, N_ITEMS, (len(u), m), device=dev)
    s = (U[u].unsqueeze(1) * I[cand]).sum(-1)
    s = s.masked_fill(seen[u.unsqueeze(1), cand], -float("inf"))
    j = cand[torch.arange(len(u), device=dev), s.argmax(1)]
    allbad = torch.isinf(s).all(1)
    if allbad.any():
        j[allbad] = sample_neg(u[allbad], seen)
    return j


def info_nce(a, b, tau):
    """CL: raw dot, batch-sum denominator, both views trainable.
    Same value as exp/sum/log but computed via logsumexp -> cannot overflow."""
    logits = a @ b.T / tau                 # diag = positive pairs
    return -(logits.diagonal() - torch.logsumexp(logits, dim=1)).mean()


def make_ndcg(gt_df, seen, nU):
    gt = {u: set(g.movieId) for u, g in gt_df.groupby("userId")}
    users = torch.tensor(sorted(gt), device=dev)
    disc = 1.0 / np.log2(np.arange(2, K_EVAL + 2))

    def fn(S):
        S = S.masked_fill(seen, -float("inf"))
        top = S[users].topk(K_EVAL, dim=1).indices.cpu().numpy()
        tot = 0.0
        for uid, rec in zip(users.tolist(), top):
            g = gt[uid]
            hit = np.array([x in g for x in rec], dtype=float)
            tot += (hit * disc).sum() / disc[:min(len(g), K_EVAL)].sum()
        return tot / len(users)
    return fn


def train(tr_df, seen, nU, a, max_epochs, eval_fn=None):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    pos = tr_df[tr_df.rating >= a.min_rating]
    g = pos
    model = SimGCL(nU, a.dim, a.layers, build_adj(g.userId.values, g.movieId.values, nU), a.eps).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    eu = torch.tensor(pos.userId.values, device=dev); ei = torch.tensor(pos.movieId.values, device=dev)
    cnt = np.bincount(pos.movieId.values, minlength=N_ITEMS).astype(np.float32)
    logq = torch.tensor(np.log((cnt + 1.0) / (cnt.sum() + N_ITEMS)), device=dev)   # smoothed popularity log-prob
    best_nd, best_ep, best_S, bad = -1.0, max_epochs, None, 0
    for ep in range(1, max_epochs + 1):
        t0, perm, tot = time.time(), torch.randperm(len(eu), device=dev), 0.0
        for b in range(0, len(eu), a.batch):
            idx = perm[b:b + a.batch]
            u, i = eu[idx], ei[idx]
            U, I = model.propagate()
            Un, In = F.normalize(U, dim=1), F.normalize(I, dim=1)
            neg = torch.randint(0, N_ITEMS, (len(u), a.n_neg), device=dev)
            cand = torch.cat([i[:, None], neg], 1)                      # (B, 1+n)
            lg = (Un[u][:, None, :] * In[cand]).sum(-1) / a.temp_main
            if a.logq > 0:
                lg = lg - a.logq * logq[cand]              # sampled-softmax popularity correction
            lg[:, 1:] = lg[:, 1:].masked_fill(seen[u[:, None], neg], float("-inf"))
            main = F.cross_entropy(lg, torch.zeros(len(u), dtype=torch.long, device=dev))
            j = neg[:, 0]                                                # for reg bookkeeping
            U1, I1 = model.propagate(noise=True)
            U2, I2 = model.propagate(noise=True)
            uu = torch.unique(u); ii = torch.unique(torch.cat([i, j]))
            cl = info_nce(U1[uu], U2[uu], a.tau_cl) + info_nce(I1[ii], I2[ii], a.tau_cl)
            e0 = model.emb.weight
            reg = (e0[u].pow(2).sum() + e0[nU + i].pow(2).sum() + e0[nU + j].pow(2).sum()) / 2 / len(u)
            loss = main + a.lambda_cl * cl + a.reg * reg
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if eval_fn is not None and ep % a.eval_every == 0:
            S = model.scores(); nd = eval_fn(S)
            print(f"ep {ep:4d} loss {tot:.3f} ndcg@20 {nd:.4f} ({time.time()-t0:.1f}s)", flush=True)
            if nd > best_nd:
                best_nd, best_ep, best_S, bad = nd, ep, S.cpu(), 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
    if eval_fn is None:
        best_S = model.scores().cpu()
    return best_ep, best_nd, best_S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--tag", default="simgcl")
    p.add_argument("--dim", type=int, default=256); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=2048); p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--eval_every", type=int, default=10); p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min_rating", type=float, default=4.0)
    p.add_argument("--dns_m", type=int, default=32)
    p.add_argument("--n_neg", type=int, default=256, help="sampled negatives for the AlphaRec-style main loss")
    p.add_argument("--temp_main", type=float, default=0.15, help="main-loss cosine temperature")
    p.add_argument("--logq", type=float, default=1.0, help="logQ correction strength (0 = off)")
    p.add_argument("--eps", type=float, default=0.1, help="noise magnitude")
    p.add_argument("--lambda_cl", type=float, default=0.1, help="CL loss weight")
    p.add_argument("--tau_cl", type=float, default=0.2, help="CL temperature (cosine scale)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs("scores", exist_ok=True)

    full = pd.read_csv(f"{a.data}/train.csv")
    nU = int(full.userId.max()) + 1

    tr, ho = temporal_split(full)
    seen_tr = seen_mask(tr, nU)
    eval_fn = make_ndcg(ho[ho.rating >= 4], seen_tr, nU)
    best_ep, best_nd, S_val = train(tr, seen_tr, nU, a, a.epochs, eval_fn)
    print(f"[val] best ep {best_ep}  ndcg@20 {best_nd:.4f}")
    torch.save(S_val, f"scores/{a.tag}_val.pt")

    seen_full = seen_mask(full, nU)
    _, _, S_test = train(full, seen_full, nU, a, best_ep)
    torch.save(S_test, f"scores/{a.tag}_test.pt")

    test = pd.read_csv(f"{a.data}/test.csv")
    ucol = "userId" if "userId" in test.columns else "id"
    uids = test[ucol].values
    S = S_test.to(dev).masked_fill(seen_full, -float("inf"))
    top = S[torch.tensor(uids, device=dev)].topk(K_EVAL, dim=1).indices.cpu().numpy()
    pd.DataFrame({"id": uids, "recommendations": [" ".join(map(str, r)) for r in top]}) \
        .to_csv(f"submission_{a.tag}.csv", index=False)
    print(f"saved scores/{a.tag}_{{val,test}}.pt, submission_{a.tag}.csv")


if __name__ == "__main__":
    main()