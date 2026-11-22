"""
LightGCN — 평점 예측(rating prediction) 버전. 가장 기본 LightGCN에서 task만 변경
  - 예측: r̂ = μ + <e_u, e_i>   (--bias 시 μ + b_u + b_i + <e_u, e_i>)
  - 손실: MSE + E0에 L2 규제 (BPR 대신)
  - 그래프: train의 모든 평점 상호작용 (평점 무관, 가중치 없는 엣지)
  - 평가: val(유저별 뒤 20%) 전체 평점의 RMSE / MAE  ← early stopping 기준
          + 참고용 NDCG@20 (예측 평점으로 전체 아이템 랭킹, 정답 평점≥4)
  - 나머지는 basic과 동일: dim 64, 3 layers, Adam lr 1e-3, reg 1e-4, batch 2048

실행:
  CUDA_VISIBLE_DEVICES=$D0 python lightgcn_rating.py --data ~/RS/data --tag lgcn_rating
"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn

N_ITEMS = 9724
K_EVAL = 20
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def temporal_split(df, frac=0.8):
    df = df.sort_values(["userId", "time_idx"])
    rank = df.groupby("userId").cumcount()
    n = df.groupby("userId")["userId"].transform("size")
    m = rank < np.floor(n * frac)
    return df[m], df[~m]


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
    A = torch.sparse_coo_tensor(torch.tensor(np.vstack([r, c])), torch.tensor(dinv[r] * dinv[c]), (n, n))
    return A.coalesce().to(dev)


class LightGCNRating(nn.Module):
    def __init__(self, nU, dim, layers, A, mu, bias):
        super().__init__()
        self.nU, self.L, self.A, self.mu, self.bias = nU, layers, A, mu, bias
        self.emb = nn.Embedding(nU + N_ITEMS, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        if bias:
            self.bu = nn.Parameter(torch.zeros(nU)); self.bi = nn.Parameter(torch.zeros(N_ITEMS))

    def propagate(self):
        e = self.emb.weight; outs = [e]
        for _ in range(self.L):
            e = torch.sparse.mm(self.A, e); outs.append(e)
        E = torch.stack(outs).mean(0)
        return E[:self.nU], E[self.nU:]

    def predict(self, U, I, u, i):
        r = self.mu + (U[u] * I[i]).sum(1)
        if self.bias:
            r = r + self.bu[u] + self.bi[i]
        return r

    @torch.no_grad()
    def full_scores(self):
        U, I = self.propagate()
        S = self.mu + U @ I.T
        if self.bias:
            S = S + self.bu[:, None] + self.bi[None, :]
        return S


def make_eval(ho, seen, nU):
    hu = torch.tensor(ho.userId.values, device=dev); hi = torch.tensor(ho.movieId.values, device=dev)
    hr = torch.tensor(ho.rating.values, dtype=torch.float32, device=dev)
    gt = {u: set(g.movieId) for u, g in ho[ho.rating >= 4].groupby("userId")}
    users = torch.tensor(sorted(gt), device=dev)
    disc = 1.0 / np.log2(np.arange(2, K_EVAL + 2))

    def fn(S):
        pred = S[hu, hi].clamp(0.5, 5.0)
        rmse = torch.sqrt(((pred - hr) ** 2).mean()).item()
        mae = (pred - hr).abs().mean().item()
        top = S.masked_fill(seen, -float("inf"))[users].topk(K_EVAL, 1).indices.cpu().numpy()
        nd = 0.0
        for uid, rec in zip(users.tolist(), top):
            g = gt[uid]; hit = np.array([x in g for x in rec], dtype=float)
            nd += (hit * disc).sum() / disc[:min(len(g), K_EVAL)].sum()
        return rmse, mae, nd / len(users)
    return fn


def train(tr_df, nU, a, max_epochs, eval_fn=None):
    u_np, i_np = tr_df.userId.values, tr_df.movieId.values
    mu = float(tr_df.rating.mean())
    model = LightGCNRating(nU, a.dim, a.layers, build_adj(u_np, i_np, nU), mu, a.bias).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    eu, ei = torch.tensor(u_np, device=dev), torch.tensor(i_np, device=dev)
    er = torch.tensor(tr_df.rating.values, dtype=torch.float32, device=dev)
    best = (float("inf"), max_epochs, None, None); bad = 0

    for ep in range(1, max_epochs + 1):
        t0, perm, tot = time.time(), torch.randperm(len(eu), device=dev), 0.0
        for b in range(0, len(eu), a.batch):
            idx = perm[b:b + a.batch]
            u, i, r = eu[idx], ei[idx], er[idx]
            U, I = model.propagate()
            mse = ((model.predict(U, I, u, i) - r) ** 2).mean()
            e0 = model.emb.weight
            reg = (e0[u].pow(2).sum() + e0[nU + i].pow(2).sum()) / 2 / len(u)
            loss = mse + a.reg * reg
            opt.zero_grad(); loss.backward(); opt.step()
            tot += mse.item() * len(idx)

        if eval_fn is not None and ep % a.eval_every == 0:
            S = model.full_scores(); rmse, mae, nd = eval_fn(S)
            print(f"ep {ep:4d} train_rmse {np.sqrt(tot/len(eu)):.4f}  val_rmse {rmse:.4f}  "
                  f"val_mae {mae:.4f}  ndcg@20 {nd:.4f} ({time.time()-t0:.1f}s)", flush=True)
            if rmse < best[0]:
                best, bad = (rmse, ep, (mae, nd), S.cpu()), 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
    if eval_fn is None:
        return max_epochs, None, None, model.full_scores().cpu()
    return best[1], best[0], best[2], best[3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--tag", default="lgcn_rating")
    p.add_argument("--dim", type=int, default=64); p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=2048); p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=10); p.add_argument("--patience", type=int, default=10)
    p.add_argument("--bias", action="store_true", help="유저·아이템 편향 추가")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs("scores", exist_ok=True)

    full = pd.read_csv(f"{a.data}/train.csv")
    nU = int(full.userId.max()) + 1

    tr, ho = temporal_split(full)
    eval_fn = make_eval(ho, seen_mask(tr, nU), nU)
    best_ep, rmse, (mae, nd), S_val = train(tr, nU, a, a.epochs, eval_fn)
    print(f"[val] best ep {best_ep}  rmse {rmse:.4f}  mae {mae:.4f}  ndcg@20 {nd:.4f}")
    torch.save(S_val, f"scores/{a.tag}_val.pt")

    seen_full = seen_mask(full, nU)
    _, _, _, S_test = train(full, nU, a, best_ep)
    torch.save(S_test, f"scores/{a.tag}_test.pt")
    test = pd.read_csv(f"{a.data}/test.csv")
    ucol = "userId" if "userId" in test.columns else "id"
    uids = test[ucol].values
    top = S_test.to(dev).masked_fill(seen_full, -float("inf"))[torch.tensor(uids, device=dev)] \
        .topk(K_EVAL, 1).indices.cpu().numpy()
    pd.DataFrame({"id": uids, "recommendations": [" ".join(map(str, r)) for r in top]}) \
        .to_csv(f"submission_{a.tag}.csv", index=False)
    print(f"saved scores/{a.tag}_{{val,test}}.pt, submission_{a.tag}.csv")


if __name__ == "__main__":
    main()