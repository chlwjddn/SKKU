"""
Vanilla LightGCN (He et al., SIGIR 2020) — 가장 기본 설정
  - BPR loss, 내적 점수, E0에 L2 규제
  - 대칭 정규화 인접행렬 D^-1/2 A D^-1/2, 층 평균 (alpha_k = 1/(K+1))
  - 오답 1개 균등 샘플링 (본 영화 제외)
  - 논문 기본값: dim 64, 3 layers, lr 1e-3, reg 1e-4, batch 2048

실행:
  CUDA_VISIBLE_DEVICES=$D0 python lightgcn_basic.py --data ~/RS/data --tag lgcn_basic
동작: (1) val 분할로 학습 → best epoch 결정  (2) train 전체로 그 epoch만큼 재학습 → test 점수 + 제출 파일
"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

N_ITEMS = 9724
K_EVAL = 20
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- data ----------------
def temporal_split(df, frac=0.8):
    """유저별 time_idx 순 앞 80% = train, 뒤 20% = holdout."""
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
    A = torch.sparse_coo_tensor(torch.tensor(np.vstack([r, c])), torch.tensor(v), (n, n))
    return A.coalesce().to(dev)


def adj_from_edges(u, i, nU):
    """GPU 텐서 엣지로 대칭 정규화 인접행렬 생성 (엣지 dropout 후 재정규화용)"""
    n = nU + N_ITEMS
    r = torch.cat([u, i + nU]); c = torch.cat([i + nU, u])
    deg = torch.bincount(r, minlength=n).float()
    dinv = deg.pow(-0.5); dinv[deg == 0] = 0
    return torch.sparse_coo_tensor(torch.stack([r, c]), dinv[r] * dinv[c], (n, n)).coalesce()


# ---------------- model ----------------
class LightGCN(nn.Module):
    def __init__(self, nU, dim, layers, A):
        super().__init__()
        self.nU, self.L, self.A = nU, layers, A
        self.emb = nn.Embedding(nU + N_ITEMS, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self, A=None):
        A = self.A if A is None else A
        e = self.emb.weight; outs = [e]
        for _ in range(self.L):
            e = torch.sparse.mm(A, e); outs.append(e)
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
    """DNS: 균등 후보 m개 중 현재 점수 최고를 오답으로 (본 영화 제외)."""
    cand = torch.randint(0, N_ITEMS, (len(u), m), device=dev)
    s = (U[u].unsqueeze(1) * I[cand]).sum(-1)
    s = s.masked_fill(seen[u.unsqueeze(1), cand], -float("inf"))
    j = cand[torch.arange(len(u), device=dev), s.argmax(1)]
    allbad = torch.isinf(s).all(1)
    if allbad.any():
        j[allbad] = sample_neg(u[allbad], seen)
    return j


def rating_weight(r, spec):
    """spec 예: "3:0.5,4:0.8,5:1.0" — 평점 내림(floor) 기준. 3.5→3, 4.5→4. 없는 키는 1.0"""
    if not spec:
        return np.ones(len(r), dtype=np.float32)
    w = {int(k): float(v) for k, v in (x.split(":") for x in spec.split(","))}
    return np.array([w.get(int(np.floor(x)), 1.0) for x in r], dtype=np.float32)


# ---------------- eval ----------------
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


def make_curve(tr_df, ho_df, seen, n_sample=20000, seed=0):
    """train / val 정답을 같은 지표로 측정: 본 적 없는 영화 중 더 높은 점수 개수(c) 기준
       hit20 = c<20 비율, prank = c / (본 적 없는 영화 수) 평균 (낮을수록 잘 맞춤)"""
    trp = tr_df[tr_df.rating >= 4]
    trp = trp.sample(min(n_sample, len(trp)), random_state=seed)
    vap = ho_df[ho_df.rating >= 4]
    pairs = {k: (torch.tensor(d.userId.values, device=dev), torch.tensor(d.movieId.values, device=dev))
             for k, d in [("tr", trp), ("va", vap)]}
    n_unseen = (~seen).sum(1).float()
    rows = []

    def fn(S, model, ep, nd, loss):
        Sm = S.masked_fill(seen, -float("inf"))
        out = {}
        for k, (u, i) in pairs.items():
            sc = S[u, i]
            c = torch.cat([(Sm[u[b:b + 4096]] > sc[b:b + 4096, None]).sum(1)
                           for b in range(0, len(u), 4096)]).float()
            out[f"{k}_hit20"] = (c < 20).float().mean().item()
            out[f"{k}_prank"] = (c / n_unseen[u]).mean().item()
        out["e0_norm"] = model.emb.weight.norm(dim=1).mean().item()
        rows.append(dict(ep=ep, loss=loss, ndcg=nd, **out))
        return "  ".join(f"{k} {v:.4f}" for k, v in out.items())
    return fn, rows


# ---------------- train ----------------
def train(tr_df, seen, nU, a, max_epochs, eval_fn=None, curve_fn=None):
    d = tr_df.sort_values(["userId", "time_idx"])
    rk = d.groupby("userId").cumcount()
    n = d.groupby("userId")["userId"].transform("size")
    d = d.assign(rel=np.where(n > 1, rk / (n - 1).clip(lower=1), 1.0))   # 유저 내 위치: 가장 오래된 0 → 최근 1
    pos = d[d.rating >= a.min_rating]
    u_np, i_np = pos.userId.values, pos.movieId.values
    gmin = a.min_rating if a.graph_min is None else a.graph_min
    g = tr_df[tr_df.rating >= gmin]
    model = LightGCN(nU, a.dim, a.layers, build_adj(g.userId.values, g.movieId.values, nU)).to(dev)
    gu = torch.tensor(g.userId.values, device=dev); gi = torch.tensor(g.movieId.values, device=dev)
    w_np = rating_weight(pos.rating.values, a.rw) * np.exp(-a.tw * (1 - pos.rel.values)).astype(np.float32)
    ew = torch.tensor(w_np, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    eu, ei = torch.tensor(u_np, device=dev), torch.tensor(i_np, device=dev)
    best_nd, best_ep, best_S, bad = -1.0, max_epochs, None, 0

    for ep in range(1, max_epochs + 1):
        t0, perm, tot = time.time(), torch.randperm(len(eu), device=dev), 0.0
        for b in range(0, len(eu), a.batch):
            idx = perm[b:b + a.batch]
            u, i = eu[idx], ei[idx]
            if a.edge_drop > 0:   # 엣지 dropout: 매 배치 p 비율 제거 후 재정규화
                keep = torch.rand(len(gu), device=dev) >= a.edge_drop
                U, I = model.propagate(adj_from_edges(gu[keep], gi[keep], nU))
            else:
                U, I = model.propagate()
            if a.dns_m > 1:
                j = sample_dns(u, seen, U.detach(), I.detach(), a.dns_m)
            else:
                j = sample_neg(u, seen)
            pu, pi, pj = U[u], I[i], I[j]
            w = ew[idx]
            loss = (w * -F.logsigmoid((pu * pi).sum(1) - (pu * pj).sum(1))).sum() / w.sum()
            e0 = model.emb.weight
            reg = (e0[u].pow(2).sum() + e0[nU + i].pow(2).sum() + e0[nU + j].pow(2).sum()) / 2 / len(u)
            loss = loss + a.reg * reg
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()

        if eval_fn is not None and ep % a.eval_every == 0:
            S = model.scores(); nd = eval_fn(S)
            extra = ("  " + curve_fn(S, model, ep, nd, tot)) if curve_fn else ""
            print(f"ep {ep:4d} loss {tot:.3f} ndcg@20 {nd:.4f}{extra} ({time.time()-t0:.1f}s)", flush=True)
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
    p.add_argument("--data", required=True); p.add_argument("--tag", default="lgcn_basic")
    p.add_argument("--dim", type=int, default=64); p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=2048); p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=10); p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_rating", type=float, default=4.0, help="그래프/학습 정답에 쓸 최소 평점 (0이면 전체)")
    p.add_argument("--dns_m", type=int, default=32, help="DNS 후보 수 (1 = 균등 샘플링 원본, 32 = 현재 고정값)")
    p.add_argument("--graph_min", type=float, default=None, help="그래프 엣지 최소 평점 (기본: min_rating과 동일)")
    p.add_argument("--rw", default="", help='긍정 평점별 가중치, 예 "3:0.5,4:0.8,5:1.0" (기본: 전부 1)')
    p.add_argument("--tw", type=float, default=0.0, help="최근성 가중 w = exp(-tw·(1-rel)), 0 = 끔")
    p.add_argument("--edge_drop", type=float, default=0.0, help="엣지 dropout 비율 (학습 시만, 0 = 끔)")
    p.add_argument("--curve", action="store_true", help="train/val 적합도 곡선 기록 (과적합 진단), test 재학습 생략")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs("scores", exist_ok=True)

    full = pd.read_csv(f"{a.data}/train.csv")
    nU = int(full.userId.max()) + 1

    # (1) validation
    tr, ho = temporal_split(full)
    seen_tr = seen_mask(tr, nU)
    eval_fn = make_ndcg(ho[ho.rating >= 4], seen_tr, nU)
    curve_fn, rows = make_curve(tr, ho, seen_tr, seed=a.seed) if a.curve else (None, None)
    best_ep, best_nd, S_val = train(tr, seen_tr, nU, a, a.epochs, eval_fn, curve_fn)
    print(f"[val] best ep {best_ep}  ndcg@20 {best_nd:.4f}")
    if a.curve:
        os.makedirs("logs", exist_ok=True)
        pd.DataFrame(rows).to_csv(f"logs/{a.tag}_curve.csv", index=False)
        print(f"saved logs/{a.tag}_curve.csv (curve 모드: test 재학습 생략)")
        return
    torch.save(S_val, f"scores/{a.tag}_val.pt")

    # (2) full retrain → test
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