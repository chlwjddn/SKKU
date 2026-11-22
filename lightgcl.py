"""
LightGCL (Cai et al., ICLR 2023) on the HW2 setup.
  - Bipartite normalized R (users x items); layer-wise propagation:
      E_u^(l+1) = R E_i^(l),  E_i^(l+1) = R^T E_u^(l)        (local view)
      G_u^(l+1) = (U_q S V_q^T) E_i^(l),  symmetric for items (SVD global view, rank q)
  - Loss = BPR (sampled negative) + lambda1 * layer-wise InfoNCE(G, E) + reg
  - Final embedding = sum over layers; score = dot product
실행: CUDA_VISIBLE_DEVICES=$D0 python lightgcl.py --data ~/RS/data --tag lgcl
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


def sample_neg(u, seen):
    j = torch.randint(0, N_ITEMS, (len(u),), device=dev)
    bad = seen[u, j]
    while bad.any():
        j[bad] = torch.randint(0, N_ITEMS, (int(bad.sum()),), device=dev)
        bad = seen[u, j]
    return j


def info_nce(a, b, tau):
    a = F.normalize(a, dim=1); b = F.normalize(b, dim=1)
    logits = a @ b.T / tau
    return -(logits.diagonal() - torch.logsumexp(logits, dim=1)).mean()


class LightGCL(nn.Module):
    def __init__(self, nU, R, q, layers):
        super().__init__()
        self.nU, self.L = nU, layers
        self.R = R                                           # (nU, N_ITEMS) normalized, dense
        U, S, V = torch.svd_lowrank(R, q=q)
        self.US = (U * S[None, :])                           # (nU, q)
        self.V = V                                           # (N_ITEMS, q)
        self.eu = nn.Parameter(torch.randn(nU, args.dim) * 0.1)
        self.ei = nn.Parameter(torch.randn(N_ITEMS, args.dim) * 0.1)

    def forward(self):
        Eu, Ei = [self.eu], [self.ei]
        Gu, Gi = [], []
        for l in range(self.L):
            Gu.append(self.US @ (self.V.T @ Ei[l]))          # SVD view
            Gi.append(self.V @ (self.US.T @ Eu[l]))
            Eu.append(self.R @ Ei[l])                        # local view
            Ei.append(self.R.T @ Eu[l])
        U = torch.stack(Eu).sum(0); I = torch.stack(Ei).sum(0)
        Ug = torch.stack(Gu).sum(0); Ig = torch.stack(Gi).sum(0)
        return U, I, Ug, Ig

    @torch.no_grad()
    def scores(self):
        U, I, _, _ = self.forward()
        return U @ I.T


def train(tr_df, seen, nU, a, max_epochs, eval_fn=None):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    pos = tr_df[tr_df.rating >= a.min_rating]
    R = torch.zeros(nU, N_ITEMS, device=dev)
    R[torch.tensor(pos.userId.values, device=dev), torch.tensor(pos.movieId.values, device=dev)] = 1.0
    du = R.sum(1, keepdim=True).clamp(min=1).pow(-0.5); di = R.sum(0, keepdim=True).clamp(min=1).pow(-0.5)
    Rn = du * R * di
    model = LightGCL(nU, Rn, a.q, a.layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    eu = torch.tensor(pos.userId.values, device=dev); ei = torch.tensor(pos.movieId.values, device=dev)
    best_nd, best_ep, best_S, bad = -1.0, max_epochs, None, 0
    for ep in range(1, max_epochs + 1):
        t0, perm, tot = time.time(), torch.randperm(len(eu), device=dev), 0.0
        for b in range(0, len(eu), a.batch):
            idx = perm[b:b + a.batch]
            u, i = eu[idx], ei[idx]
            U, I, Ug, Ig = model()
            j = sample_neg(u, seen)
            bpr = -F.logsigmoid((U[u] * I[i]).sum(1) - (U[u] * I[j]).sum(1)).mean()
            uu = torch.unique(u); ii = torch.unique(torch.cat([i, j]))
            cl = info_nce(Ug[uu], U[uu], a.tau) + info_nce(Ig[ii], I[ii], a.tau)
            reg = (model.eu[u].pow(2).sum() + model.ei[i].pow(2).sum() + model.ei[j].pow(2).sum()) / len(u)
            loss = bpr + a.lambda1 * cl + a.reg * reg
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
    global args
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--tag", default="lgcl")
    p.add_argument("--dim", type=int, default=64); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--q", type=int, default=5, help="SVD rank")
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=2048); p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--eval_every", type=int, default=10); p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_rating", type=float, default=4.0)
    p.add_argument("--lambda1", type=float, default=0.01, help="CL weight")
    p.add_argument("--tau", type=float, default=0.2, help="CL temperature")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    a = args
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
