"""
lgcn_basic 단독 실패 분석 — 저장된 val 점수만 사용 (학습 없음)
  python failure_basic.py --data ~/RS/data --tag lgcn_basic
  (pop = 인기도 baseline, 하한선 참고용)
[1] 전체: NDCG / Recall / 정답 순위 분포
[2] 정답 인기구간별 적중률
[3] 추천 성향 + 오답(FP) vs 적중(HIT) 인기도
[4] 유저 기록 길이 사분위별 NDCG
[5] NDCG=0 유저 특성
[6] 학습 신호: train 정답 vs DNS 오답 (--dns_m 학습값과 맞출 것) + FN 비율
"""
import argparse
import numpy as np, pandas as pd, torch

N_ITEMS, K = 9724, 20
BINS = [(0, 5, "cold≤5"), (6, 20, "6-20"), (21, 50, "21-50"), (51, 100, "51-100"), (101, 10**9, ">100")]


def temporal_split(df, frac=0.8):  # lightgcn_basic.py와 동일
    df = df.sort_values(["userId", "time_idx"])
    rank = df.groupby("userId").cumcount()
    n = df.groupby("userId")["userId"].transform("size")
    m = rank < np.floor(n * frac)
    return df[m], df[~m]


def bucket(c):
    for lo, hi, name in BINS:
        if lo <= c <= hi:
            return name


def evaluate(S_raw, seen, users, gt, disc):
    S = S_raw.masked_fill(seen, -float("inf"))[users]
    ranks = (S.argsort(1, descending=True).argsort(1) + 1).numpy()
    top = S.topk(K, 1).indices.numpy()
    nd, rc, pr = [], [], []
    for k, u in enumerate(users):
        r = ranks[k, gt[u]]; pr.append(r)
        hit = np.isin(top[k], gt[u]).astype(float)
        nd.append((hit * disc).sum() / disc[:min(len(gt[u]), K)].sum())
        rc.append(hit.sum() / len(gt[u]))
    return dict(nd=np.array(nd), rc=np.array(rc), pr=pr, top=top)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tag", default="lgcn_basic")
    p.add_argument("--dns_m", type=int, default=1, help="[6] 오답 후보 수 (학습 설정과 맞출 것)")
    p.add_argument("--n_neg", type=int, default=200, help="[6] 정답당 오답 수")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    full = pd.read_csv(f"{a.data}/train.csv")
    nU = int(full.userId.max()) + 1
    tr, ho = temporal_split(full)
    gt = {u: np.array(sorted(g.movieId)) for u, g in ho[ho.rating >= 4].groupby("userId")}
    users = np.array(sorted(gt))

    seen = torch.zeros(nU, N_ITEMS, dtype=torch.bool)
    seen[tr.userId.values, tr.movieId.values] = True
    pop = np.bincount(tr[tr.rating >= 4].movieId, minlength=N_ITEMS)
    top100 = set(np.argsort(-pop)[:100])
    hist_len = tr.groupby("userId").size().reindex(users).values
    disc = 1 / np.log2(np.arange(2, K + 2))

    S_basic = torch.load(f"scores/{a.tag}_val.pt").float()
    S_pop = torch.tensor(pop, dtype=torch.float).expand(nU, N_ITEMS).clone()
    R = {a.tag: evaluate(S_basic, seen, users, gt, disc), "pop": evaluate(S_pop, seen, users, gt, disc)}
    b = R[a.tag]

    # [1]
    print("\n[1] 전체")
    print(f"{'model':<14}{'NDCG':>8}{'Recall':>8}{'HIT≤20':>9}{'21-150':>9}{'>150':>9}")
    for t, r in R.items():
        allr = np.concatenate(r["pr"])
        print(f"{t:<14}{r['nd'].mean():8.4f}{r['rc'].mean():8.4f}"
              f"{(allr<=20).mean():9.3f}{((allr>20)&(allr<=150)).mean():9.3f}{(allr>150).mean():9.3f}")

    # [2]
    print("\n[2] 정답 인기구간별 적중률")
    allgt = np.concatenate([gt[u] for u in users])
    gb = np.array([bucket(pop[i]) for i in allgt])
    print(f"{'bucket':<10}{'share':>7}{a.tag:>14}{'pop':>10}{'median rank':>13}")
    allr_b, allr_p = np.concatenate(b["pr"]), np.concatenate(R["pop"]["pr"])
    for _, _, name in BINS:
        m = gb == name
        print(f"{name:<10}{m.mean():7.3f}{(allr_b[m]<=20).mean():14.3f}{(allr_p[m]<=20).mean():10.3f}"
              f"{np.median(allr_b[m]):13.0f}")

    # [3]
    print("\n[3] 추천 성향")
    top = b["top"]
    hit_pop, fp_pop = [], []
    for k, u in enumerate(users):
        h = np.isin(top[k], gt[u])
        hit_pop += list(pop[top[k][h]]); fp_pop += list(pop[top[k][~h]])
    flat = top.ravel()
    print(f"추천 평균인기 {pop[flat].mean():.1f} (중앙 {np.median(pop[flat]):.0f}), "
          f"Top100 비중 {np.mean([i in top100 for i in flat]):.3f}, 커버리지 {len(set(flat))}")
    print(f"HIT 아이템 중앙인기 {np.median(hit_pop):.0f} / FP 아이템 중앙인기 {np.median(fp_pop):.0f} "
          f"/ 정답 중앙인기 {np.median(pop[allgt]):.0f}")

    # [4]
    print("\n[4] 기록 길이 사분위별 NDCG")
    q = np.quantile(hist_len, [0.25, 0.5, 0.75])
    grp = np.digitize(hist_len, q)
    ngt = np.array([len(gt[u]) for u in users])
    print(f"{'group':<14}{a.tag:>12}{'pop':>9}{'avg #gt':>9}")
    for g in range(4):
        m = grp == g
        lab = f"Q{g+1}(≤{int(q[g])})" if g < 3 else f"Q4(>{int(q[2])})"
        print(f"{lab:<14}{b['nd'][m].mean():12.4f}{R['pop']['nd'][m].mean():9.4f}{ngt[m].mean():9.1f}")

    # [5]
    print("\n[5] NDCG=0 유저")
    z = b["nd"] == 0
    gt_medpop = np.array([np.median(pop[gt[u]]) for u in users])
    print(f"{z.sum()}명 / {len(users)} ({z.mean():.3f})")
    print(f"{'':<10}{'기록길이':>9}{'#정답':>8}{'정답 중앙인기':>14}")
    for lab, m in [("NDCG=0", z), ("나머지", ~z)]:
        print(f"{lab:<10}{np.median(hist_len[m]):9.0f}{np.median(ngt[m]):8.0f}{np.median(gt_medpop[m]):14.1f}")

    # [6]
    M = a.dns_m
    print(f"\n[6] 학습 신호 (train 정답 vs DNS 오답, 후보 M={M}, 정답당 {a.n_neg}개)")
    trp = tr[tr.rating >= 4]
    h4 = ho[ho.rating >= 4]
    fut = np.zeros((nU, N_ITEMS), dtype=bool); fut[h4.userId.values, h4.movieId.values] = True
    seen_np = seen.numpy(); Sn = S_basic.numpy()
    aucs, easy, fns = [], [], []
    for u, g in trp.groupby("userId"):
        pos = Sn[u, g.movieId.values]
        cand_pool = np.flatnonzero(~seen_np[u])
        cand = rng.choice(cand_pool, (a.n_neg, M))             # 본 영화 제외 후보
        neg_idx = cand[np.arange(a.n_neg), Sn[u, cand].argmax(1)]  # 후보 중 최고점 = DNS 오답
        diff = pos[:, None] - Sn[u, neg_idx][None, :]
        aucs.append((diff > 0).mean())
        easy.append((1 / (1 + np.exp(-diff)) > 0.99).mean())
        fns.append(fut[u, neg_idx].mean())
    print(f"AUC 평균 {np.mean(aucs):.4f},  σ>0.99 쌍 비율 {np.mean(easy):.3f},  "
          f"오답이 미래 정답(FN)인 비율 {np.mean(fns):.4f}")

if __name__ == "__main__":
    main()