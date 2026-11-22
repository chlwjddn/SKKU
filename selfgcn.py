"""
Self-iterative graph augmentation on REAL LightGCN (BPR + DNS + dot + L2 — the config
where graph propagation actually matters).

round 0: train on original graph G0 (ratings >= 4) -> scores S0
round t: top-k unseen items per user from S_{t-1} become predicted edges
         G_t = G0 + predicted edges (weight gamma < 1)   [always from G0, no accumulation]
         re-train from scratch on G_t -> S_t, measure val
stop   : when val drops; keep the best round. BPR positives stay ORIGINAL only.

실행: CUDA_VISIBLE_DEVICES=$D0 python selfgcn.py --data ~/RS/data --tag selfgcn
동작: (1) val 분할에서 라운드 반복 -> best round/epoch  (2) train 전체로 같은 스케줄 재학습 -> test 점수 + 제출
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
    return df[rank < np.floor(n * frac)], df[~(rank < np.floor(n * frac))]


def seen_mask(df, nU):
    m = torch.zeros(nU, N_ITEMS, dtype=torch.bool, device=dev)
    m[torch.tensor(df.userId.values, device=dev), torch.tensor(df.movieId.values, device=dev)] = True
    return m


def build_adj_w(u, i, w, nU):
    """weighted symmetric-normalised adjacency (weighted degree)."""
    n = nU + N_ITEMS
    r = np.concatenate([u, i + nU]); c = np.concatenate([i + nU, u]); ww = np.concatenate([w, w]).astype(np.float32)
    deg = np.bincount(r, weights=ww, minlength=n).astype(np.float32)
    dinv = np.zeros_like(deg); nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    v = dinv[r] * ww * dinv[c]
    return torch.sparse_coo_tensor(torch.tensor(np.vstack([r, c])), torch.tensor(v), (n, n)).coalesce().to(dev)


class LightGCN(nn.Module):
    def __init__(self, nU, dim, layers, A):
        super().__init__()
        self.nU, self.L, self.A = nU, layers, A
        self.emb = nn.Embedding(nU + N_ITEMS, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self):
        e = self.emb.weight; outs = [e]
        for _ in range(self.L):
            e = torch.sparse.mm(self.A, e); outs.append(e)
        E = torch.stack(outs).mean(0)
        return E[:self.nU], E[self.nU:]

    @torch.no_grad()
    def scores(self):
        U, I = self.propagate()
        return U @ I.T


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


def train_once(pos_u, pos_i, A, seen, nU, a, epochs, eval_fn=None):
    """One LightGCN training run on graph A. Returns best (ndcg, epoch, scores)."""
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = LightGCN(nU, a.dim, a.layers, A).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    eu = torch.tensor(pos_u, device=dev); ei = torch.tensor(pos_i, device=dev)
    best_nd, best_ep, best_S, bad = -1.0, epochs, None, 0
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(eu), device=dev)
        for b in range(0, len(eu), a.batch):
            idx = perm[b:b + a.batch]
            u, i = eu[idx], ei[idx]
            U, I = model.propagate()
            j = sample_dns(u, seen, U.detach(), I.detach(), a.dns_m)
            pu, pi, pj = U[u], I[i], I[j]
            loss = -F.logsigmoid((pu * pi).sum(1) - (pu * pj).sum(1)).mean()
            e0 = model.emb.weight
            reg = (e0[u].pow(2).sum() + e0[nU + i].pow(2).sum() + e0[nU + j].pow(2).sum()) / 2 / len(u)
            loss = loss + a.reg * reg
            opt.zero_grad(); loss.backward(); opt.step()
        if eval_fn is not None and ep % a.eval_every == 0:
            nd = eval_fn(model.scores())
            if nd > best_nd:
                best_nd, best_ep, best_S, bad = nd, ep, model.scores(), 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
    if eval_fn is None:
        best_S = model.scores()
    return best_nd, best_ep, best_S


def self_iterate(tr_df, seen, nU, a, eval_fn, round_epochs=None):
    """Returns per-round results. round_epochs: fixed epoch list for final retrain."""
    g = tr_df[tr_df.rating >= a.min_rating]
    u0, i0 = g.userId.values, g.movieId.values
    w0 = np.ones(len(u0), dtype=np.float32)
    pos = tr_df[tr_df.rating >= a.min_rating]
    results = []           # (round, ndcg, epoch, S)
    S_prev = None
    for t in range(a.rounds + 1):
        if t == 0:
            A = build_adj_w(u0, i0, w0, nU)
            n_add = 0
        else:
            Sm = S_prev.masked_fill(seen, -float("inf"))
            topk = torch.topk(Sm, a.self_k, dim=1).indices                 # (nU, k)
            au = np.repeat(np.arange(nU), a.self_k); ai = topk.cpu().numpy().ravel()
            keep = np.isin(au, np.unique(u0))                               # only users present in graph
            au, ai = au[keep], ai[keep]
            A = build_adj_w(np.concatenate([u0, au]), np.concatenate([i0, ai]),
                            np.concatenate([w0, np.full(len(au), a.self_gamma, dtype=np.float32)]), nU)
            n_add = len(au)
        ep_budget = a.epochs if round_epochs is None else round_epochs[t]
        ev = eval_fn if round_epochs is None else None
        nd, ep, S = train_once(pos.userId.values, pos.movieId.values, A, seen, nU, a, ep_budget, ev)
        S_prev = S.to(dev)
        results.append((t, nd, ep, S))
        if round_epochs is None:
            print(f"round {t}: +{n_add} edges  best ep {ep}  val ndcg {nd:.5f}", flush=True)
            if t >= 1 and nd <= results[t - 1][1]:
                break
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--tag", default="selfgcn")
    p.add_argument("--dim", type=int, default=64); p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=2048); p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--eval_every", type=int, default=10); p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min_rating", type=float, default=4.0)
    p.add_argument("--dns_m", type=int, default=32)
    p.add_argument("--rounds", type=int, default=3, help="max self-iteration rounds")
    p.add_argument("--self_k", type=int, default=10, help="predicted edges per user per round")
    p.add_argument("--self_gamma", type=float, default=0.5, help="weight of predicted edges")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs("scores", exist_ok=True)

    full = pd.read_csv(f"{a.data}/train.csv")
    nU = int(full.userId.max()) + 1

    tr, ho = temporal_split(full)
    seen_tr = seen_mask(tr, nU)
    eval_fn = make_ndcg(ho[ho.rating >= 4], seen_tr, nU)
    res = self_iterate(tr, seen_tr, nU, a, eval_fn)
    best = max(res, key=lambda x: x[1])
    print(f"[val] best round {best[0]}  ep {best[2]}  ndcg@20 {best[1]:.5f}")
    torch.save(best[3].cpu(), f"scores/{a.tag}_val.pt")

    # final: replay the same schedule (rounds up to best, each with its best epoch) on full train
    sched = [r[2] for r in res[:best[0] + 1]]
    seen_full = seen_mask(full, nU)
    res_f = self_iterate(full, seen_full, nU, a, None, round_epochs=sched)
    S_test = res_f[best[0]][3]
    torch.save(S_test.cpu(), f"scores/{a.tag}_test.pt")

    test = pd.read_csv(f"{a.data}/test.csv")
    ucol = "userId" if "userId" in test.columns else "id"
    uids = test[ucol].values
    S = S_test.masked_fill(seen_full, -float("inf"))
    top = S[torch.tensor(uids, device=dev)].topk(K_EVAL, dim=1).indices.cpu().numpy()
    pd.DataFrame({"id": uids, "recommendations": [" ".join(map(str, r)) for r in top]}) \
        .to_csv(f"submission_{a.tag}.csv", index=False)
    print(f"saved scores/{a.tag}_{{val,test}}.pt, submission_{a.tag}.csv")


if __name__ == "__main__":
    main()
