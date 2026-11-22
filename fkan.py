"""
FourierKAN-GCF (Xu et al., 2024) on the HW2 setup.
  - Propagation: E^(l+1) = drop( FourierKAN( A_hat @ E^(l) ) ),  final = mean over layers (incl. E0)
  - FourierKAN layer: y_j = sum_i sum_k a_jik cos(k x_i) + b_jik sin(k x_i)   (gridsize harmonics)
  - Loss: BPR (uniform sampled negative) + L2 on E0
실행: CUDA_VISIBLE_DEVICES=$D0 python fkan.py --data ~/RS/data --tag fkan
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


class FourierKAN(nn.Module):
    """y_j = sum_i sum_k a[j,i,k] cos(k x_i) + b[j,i,k] sin(k x_i); small init -> near-linear start."""
    def __init__(self, dim, grid):
        super().__init__()
        self.grid = grid
        self.coef = nn.Parameter(torch.randn(2, dim, dim, grid) / (np.sqrt(dim) * grid))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):                                   # (N, d)
        k = torch.arange(1, self.grid + 1, device=x.device).float()
        xk = x[..., None] * k                               # (N, d, g)
        c = torch.cos(xk); s = torch.sin(xk)
        y = torch.einsum("nig,oig->no", c, self.coef[0]) + torch.einsum("nig,oig->no", s, self.coef[1])
        return y + self.bias


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


class FKANGCF(nn.Module):
    def __init__(self, nU, A, layers, dim, grid, mdrop):
        super().__init__()
        self.nU, self.A, self.L = nU, A, layers
        self.emb = nn.Embedding(nU + N_ITEMS, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.kan = nn.ModuleList(FourierKAN(dim, grid) for _ in range(layers))
        self.mdrop = mdrop

    def propagate(self, train=True):
        e = self.emb.weight; outs = [e]
        for l in range(self.L):
            e = torch.sparse.mm(self.A, e)
            e = self.kan[l](e)
            if train and self.mdrop > 0:
                e = F.dropout(e, self.mdrop)
            outs.append(e)
        E = torch.stack(outs).mean(0)
        return E[:self.nU], E[self.nU:]

    @torch.no_grad()
    def scores(self):
        U, I = self.propagate(train=False)
        return U @ I.T


def train(tr_df, seen, nU, a, max_epochs, eval_fn=None):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    pos = tr_df[tr_df.rating >= a.min_rating]
    model = FKANGCF(nU, build_adj(pos.userId.values, pos.movieId.values, nU),
                    a.layers, a.dim, a.grid, a.mdrop).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    eu = torch.tensor(pos.userId.values, device=dev); ei = torch.tensor(pos.movieId.values, device=dev)
    best_nd, best_ep, best_S, bad = -1.0, max_epochs, None, 0
    for ep in range(1, max_epochs + 1):
        t0, perm, tot = time.time(), torch.randperm(len(eu), device=dev), 0.0
        for b in range(0, len(eu), a.batch):
            idx = perm[b:b + a.batch]
            u, i = eu[idx], ei[idx]
            U, I = model.propagate(train=True)
            j = sample_neg(u, seen)
            bpr = -F.logsigmoid((U[u] * I[i]).sum(1) - (U[u] * I[j]).sum(1)).mean()
            e0 = model.emb.weight
            reg = (e0[u].pow(2).sum() + e0[nU + i].pow(2).sum() + e0[nU + j].pow(2).sum()) / 2 / len(u)
            loss = bpr + a.reg * reg
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
    p.add_argument("--data", required=True); p.add_argument("--tag", default="fkan")
    p.add_argument("--dim", type=int, default=64); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--grid", type=int, default=3, help="Fourier harmonics per input dim")
    p.add_argument("--mdrop", type=float, default=0.1, help="message dropout")
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=2048); p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--eval_every", type=int, default=10); p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_rating", type=float, default=4.0)
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
