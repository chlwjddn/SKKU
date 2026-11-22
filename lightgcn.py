"""
LightGCN (BPR loss) for skku-esw-4024-recommender-system
- Trains on high-rated (>= thr) interactions only (best setting found with EASE)
- Early stopping on validation NDCG@20, then retrains on full train for the best #epochs
- Saves submission + score matrices (val/test) for ensembling
Put next to ease_v2.py.
Usage: python lightgcn.py --data ~/RS/data --dim 64 --layers 3 --lr 1e-3 --reg 1e-4
"""
import argparse, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ease_v2 import temporal_split, dense, ndcg, N_ITEMS, dev, K

ap = argparse.ArgumentParser()  # defaults = current SOTA: ce, cos/0.1, d128, 3 layers, lr 5e-3, Adam, 4:0.7/5:1, time_beta 1.25
ap.add_argument("--data", default="~/RS/data")
ap.add_argument("--thr", type=float, default=4.0)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--reg", type=float, default=0.0)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--eval_every", type=int, default=20)
ap.add_argument("--patience", type=int, default=10, help="in evals")
ap.add_argument("--loss", choices=["bpr", "ce"], default="ce",
                help="bpr: 1 random negative | ce: full softmax over all items")
ap.add_argument("--temp", type=float, default=0.1, help="softmax temperature for --loss ce (cosine scores)")
ap.add_argument("--graph", choices=["high", "all"], default="high",
                help="edges of the propagation graph: rating>=graph_thr only, or all ratings")
ap.add_argument("--graph_thr", type=float, default=None, help="graph edge threshold (default: --thr)")
ap.add_argument("--pos_thr", type=float, default=None, help="training positive threshold (default: --thr)")
ap.add_argument("--pair_w", type=float, default=0.0,
                help="bpr only: weight of rating pairs (same user, higher rating > lower rating)")
ap.add_argument("--pair_only", action="store_true",
                help="bpr only: train on rating pairs only (no unseen-item negatives)")
ap.add_argument("--edge_w", default="",
                help='graph edge weight by rating, negative allowed, e.g. "0.5:-0.5,3:0.1,4:0.7,5:1"')
ap.add_argument("--low_neg", type=float, default=1.0,
                help="ce only: multiply the user's low-rated items' weight in the softmax denominator")
ap.add_argument("--un_z", type=float, default=None,
                help="user-normalised positives: also treat r >= un_min with user z-score >= un_z as positive")
ap.add_argument("--un_min", type=float, default=3.5, help="minimum raw rating for z-based positives")
ap.add_argument("--un_gamma", type=float, default=0.0,
                help="ce only: multiply positive weight by clip(1 + un_gamma * z, 0.2, 3)")
ap.add_argument("--time_beta", type=float, default=1.25,
                help="ce only: positive weight *= exp(-beta * (1 - rel)), rel = relative position in user history")
ap.add_argument("--edge_time_beta", type=float, default=0.0,
                help="graph edge weight *= exp(-beta * (1 - rel))")
ap.add_argument("--pos_pop_gamma", type=float, default=0.0,
                help="ce only: positive weight *= pop_i^(-gamma), pop_i = # training positives of item i")
ap.add_argument("--pp_user", action="store_true",
                help="scale pos_pop_gamma per user by nicheness in [0,1] (recent 10 positives' popularity rank)")
ap.add_argument("--user_pow", type=float, default=0.0,
                help="ce only: positive weight *= (# positives of the user)^(-user_pow)")
ap.add_argument("--dir_time_u", type=float, default=0.0,
                help="graph, user side (u <- i): weight exp(-b(1-rel_ui)), normalised to mean 1 per user")
ap.add_argument("--dir_time_i", type=float, default=0.0,
                help="graph, item side (i <- u): weight exp(-b(1-rank of rel among item's edges)), mean 1 per item")
ap.add_argument("--short_k", type=int, default=0,
                help="ce only: add g_u * mean(last k positives' item embeddings) to the user vector (per-user gate)")
ap.add_argument("--pop_aff", type=int, default=0,
                help="ce only: add <a_u, q_bin(i)> with item popularity-bin embeddings of this dim")
ap.add_argument("--pop_bins", type=int, default=10)
ap.add_argument("--skip_layer0", action="store_true",
                help="final embedding = mean of propagated layers 1..K only (forces use of graph connectivity)")
ap.add_argument("--no_user_emb", action="store_true",
                help="user ID embeddings fixed at 0: users are represented only through their connected items")
ap.add_argument("--opt", choices=["adam", "adamw"], default="adam")
ap.add_argument("--wd", type=float, default=0.0, help="weight decay (decoupled for adamw)")
ap.add_argument("--sched", choices=["none", "cosine", "linear"], default="none")
ap.add_argument("--warmup", type=int, default=0, help="linear warmup epochs (with --sched cosine)")
ap.add_argument("--n_neg", type=int, default=0, help="ce only: sampled softmax with K shared uniform negatives (0 = all items)")
ap.add_argument("--neg_mode", choices=["shared", "pair"], default="shared",
                help="n_neg sampling: one set shared by the batch, or K negatives per positive pair (AlphaRec-style)")
ap.add_argument("--pos_alpha", type=float, default=1.0,
                help="ce only: loss = -alpha * s_pos + logsumexp(all); alpha > 1 strengthens the positive pull")
ap.add_argument("--margin", type=float, default=0.0, help="ce only: subtract margin (cosine units) from the positive logit")
ap.add_argument("--xsim_lambda", type=float, default=0.0, help="XSimGCL: weight of cross-layer contrastive loss (0 = off)")
ap.add_argument("--xsim_eps", type=float, default=0.1, help="XSimGCL: noise magnitude")
ap.add_argument("--xsim_tau", type=float, default=0.2, help="XSimGCL: contrastive temperature")
ap.add_argument("--xsim_layer", type=int, default=1, help="XSimGCL: layer used as the contrastive view")
ap.add_argument("--edge_drop", type=float, default=0.0,
                help="drop this fraction of graph edges at every training step (renormalised); eval uses the full graph")
ap.add_argument("--tce_max", type=float, default=0.0,
                help="T-CE denoising: drop this max fraction of highest-loss positives per batch (ramped up)")
ap.add_argument("--tce_epochs", type=int, default=100, help="T-CE: epochs to ramp drop rate from 0 to tce_max")
ap.add_argument("--prune_q", type=float, default=0.0,
                help="similarity pruning: drop each user's lowest-similarity q fraction of positives (graph + targets)")
ap.add_argument("--prune_start", type=int, default=60, help="pruning: first epoch")
ap.add_argument("--prune_every", type=int, default=40, help="pruning: re-evaluate every N epochs")
ap.add_argument("--score", choices=["cos", "dot"], default="cos",
                help="ce only: cosine/temp (default) or raw dot product/temp (lets L2 act as a graph smoothness prior)")
ap.add_argument("--emb_drop", type=float, default=0.0, help="dropout on layer-0 embeddings during training")
ap.add_argument("--msg_drop", type=float, default=0.0, help="dropout on each propagated layer during training")
ap.add_argument("--sig_detach", action="store_true", help="prop sig: stop gradient through the edge weights")
ap.add_argument("--sig_degnorm", action="store_true", help="prop sig: multiply edge weights by 1/sqrt(d_u d_i)")
ap.add_argument("--sig_wdrop", type=float, default=0.0,
                help="prop sig: dropout on the edge-weight vector during training (no graph rebuild, rescaled by 1/(1-p))")
ap.add_argument("--sig_l2", action="store_true", help="prop sig: L2-normalise each propagated layer (off = raw sums)")
ap.add_argument("--prop", choices=["lightgcn", "mule", "sig"], default="lightgcn",
                help="mule: Xavier init, L2-normalise each propagated layer, final = x0 + sum_k x_k / k (MuLe's LightGCN)")
ap.add_argument("--ii_k", type=int, default=0,
                help="add ItemKNN item-item edges to the graph: top-k cosine neighbors per item")
ap.add_argument("--ii_w", type=float, default=0.3, help="weight scale of item-item edges")
ap.add_argument("--fn_topk", type=int, default=0,
                help="ce only: also exclude each user's current top-K unseen items from the softmax denominator (potential false negatives)")
ap.add_argument("--mlp", action="store_true",
                help="nonlinear residual towers: e' = LN(e + MLP(e)) for users and items before cosine scoring")
ap.add_argument("--mlp_drop", type=float, default=0.1)
ap.add_argument("--mlp_share", action="store_true", help="share the tower between users and items")
ap.add_argument("--diag", action="store_true", help="print layer / smoothing / softmax diagnostics at each eval")
ap.add_argument("--rating_w", default="4:0.7,5:1",
                help='ce only: per-positive loss weight by rating, e.g. "5:1,4:0.7,3:0.3" '
                     '(a rating gets the weight of the largest key <= it; empty = uniform)')
ap.add_argument("--no_mask_pos", action="store_true",
                help="ce only: keep the user's other positives in the softmax denominator (old, buggy behaviour)")
ap.add_argument("--no_final", action="store_true", help="skip full-train retrain (for sweeps)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tag", default="lgcn")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)


def map_w(spec, ratings):
    """'1:-0.5,4:0.7,5:1' -> weight of the largest key <= rating."""
    kv = sorted((float(a), float(b)) for a, b in (p.split(":") for p in spec.split(",")))
    keys, vals = np.array([k for k, _ in kv]), np.array([v for _, v in kv])
    return vals[np.clip(np.searchsorted(keys, ratings, side="right") - 1, 0, len(keys) - 1)].astype(np.float32)


def norm_adj_dir(df, n_users, wU, wI):
    """Directional, relative weights: user rows use wU, item rows use wI; degrees are plain edge counts."""
    N = n_users + N_ITEMS
    u = df.userId.values; i = df.movieId.values + n_users
    deg = np.bincount(np.concatenate([u, i]), minlength=N).astype(np.float32)
    d = np.zeros_like(deg); d[deg > 0] = deg[deg > 0] ** -0.5
    base = d[u] * d[i]
    rows = np.concatenate([u, i]); cols = np.concatenate([i, u])
    vals = np.concatenate([base * wU, base * wI]).astype(np.float32)
    idx = torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.as_tensor(vals), (N, N)).coalesce().to(dev)


def norm_adj(df, n_users, ew=None, extra=None):
    """ew: per-edge weights (may be negative). Degrees use |weight|, messages keep the sign.
    extra: optional (rows, cols, w) in full-node indexing, added symmetrically (item-item edges)."""
    N = n_users + N_ITEMS
    u = df.userId.values
    i = df.movieId.values + n_users
    w = np.ones(len(df), dtype=np.float32) if ew is None else ew
    rows = np.concatenate([u, i]); cols = np.concatenate([i, u]); w2 = np.concatenate([w, w])
    if extra is not None:
        er, ec, ew2 = extra
        rows = np.concatenate([rows, er, ec]); cols = np.concatenate([cols, ec, er])
        w2 = np.concatenate([w2, ew2, ew2])
    deg = np.bincount(rows, weights=np.abs(w2), minlength=N).astype(np.float32)
    d = np.zeros_like(deg); d[deg > 0] = deg[deg > 0] ** -0.5
    vals = d[rows] * d[cols] * w2
    idx = torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.as_tensor(vals), (N, N)).coalesce().to(dev)


class Tower(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(args.mlp_drop), nn.Linear(dim, dim))
        self.ln = nn.LayerNorm(dim)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)  # start = identity

    def forward(self, x):
        return self.ln(x + self.net(x))


class LightGCN(nn.Module):
    def __init__(self, n_users, dim, layers):
        super().__init__()
        self.n_users, self.layers = n_users, layers
        if args.mlp:
            self.tw_u = Tower(dim)
            self.tw_i = self.tw_u if args.mlp_share else Tower(dim)
        self.emb = nn.Embedding(n_users + N_ITEMS, dim)
        if args.prop == "mule":
            with torch.no_grad():
                wu = torch.empty(n_users, dim); wi = torch.empty(N_ITEMS, dim)
                nn.init.xavier_uniform_(wu); nn.init.xavier_uniform_(wi)
                self.emb.weight.copy_(torch.cat([wu, wi], 0))
        else:
            nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, A):
        e = self.emb.weight
        if args.no_user_emb:
            e = torch.cat([torch.zeros_like(e[: self.n_users]), e[self.n_users:]], 0)
        if args.emb_drop > 0 and self.training:
            e = F.dropout(e, args.emb_drop)
        if args.prop == "sig":
            out = [e]
            for k in range(self.layers):
                w = torch.sigmoid((e[self.eu] * e[self.ei]).sum(1))
                if args.sig_detach:
                    w = w.detach()
                if args.sig_degnorm:
                    w = w * self.dnorm
                if args.sig_wdrop > 0 and self.training:
                    w = F.dropout(w, args.sig_wdrop)
                new = torch.zeros_like(e)
                new = new.index_add(0, self.eu, w[:, None] * e[self.ei])
                new = new.index_add(0, self.ei, w[:, None] * e[self.eu])
                e = F.normalize(new, dim=1) if args.sig_l2 else new
                out.append(e)
            self.last_layers = out
            if args.skip_layer0 and self.layers > 0:
                out = out[1:]
            e = torch.stack(out).mean(0)
            return e[: self.n_users], e[self.n_users:]
        if args.prop == "mule":
            out = [e]
            for k in range(self.layers):
                e = F.normalize(torch.sparse.mm(A, e), dim=1)
                out.append(e / (k + 1))
            self.last_layers = out
            e = torch.stack(out).sum(0)
            return e[: self.n_users], e[self.n_users:]
        out = [e]
        xsim = args.xsim_lambda > 0
        for k in range(self.layers):
            e = torch.sparse.mm(A, e)
            if args.msg_drop > 0 and self.training:
                e = F.dropout(e, args.msg_drop)
            if xsim and self.training:
                e = e + torch.sign(e) * F.normalize(torch.rand_like(e), dim=1) * args.xsim_eps
            out.append(e)
            if xsim and k + 1 == args.xsim_layer:
                self.cl_view = e
        self.last_layers = out
        if xsim and self.layers > 0:  # XSimGCL: final = mean of layers 1..L
            e = torch.stack(out[1:]).mean(0)
            return e[: self.n_users], e[self.n_users:]
        if args.skip_layer0 and self.layers > 0:
            out = out[1:]
        e = torch.stack(out).mean(0)
        return e[: self.n_users], e[self.n_users:]


def logits_for(model, U, I, uidx):
    """cosine logits / temp (+ extras) for users uidx against all items."""
    if args.mlp:
        U, I = model.tw_u(U), model.tw_i(I)
    Uu = U[uidx]
    if args.short_k > 0:
        li, lm = model.last_idx[uidx], model.last_mask[uidx]               # (B, k)
        r = (I[li.clamp(min=0)] * lm[..., None]).sum(1) / lm.sum(1, keepdim=True).clamp(min=1)
        Uu = Uu + torch.sigmoid(model.gate[uidx])[:, None] * r
    if args.score == "dot":
        lg = Uu @ I.T / args.temp
    else:
        lg = F.normalize(Uu, dim=1) @ F.normalize(I, dim=1).T / args.temp
    if args.pop_aff > 0:
        lg = lg + model.pop_a[uidx] @ model.pop_q[model.item_bin].T
    return lg


def scores(model, A):
    model.eval()
    with torch.no_grad():
        U, I = model(A)
        if args.loss == "ce":
            return logits_for(model, U, I, torch.arange(U.shape[0], device=dev)) * args.temp
        return U @ I.T


def train(df_all, n_users, epochs, val=None):
    """df_all: all interactions (for seen mask). Trains on rating >= thr. Returns model, best_epoch, best_score."""
    g_thr = args.thr if args.graph_thr is None else args.graph_thr
    p_thr = args.thr if args.pos_thr is None else args.pos_thr
    mu = df_all.groupby("userId").rating.transform("mean")
    sd = df_all.groupby("userId").rating.transform("std").fillna(1.0).clip(lower=0.5)
    z_all = ((df_all.rating - mu) / sd).values
    nn_ = df_all.groupby("userId")["time_idx"].transform("size")
    rel_all = ((df_all.groupby("userId")["time_idx"].rank(method="first") - 1) / (nn_ - 1).clip(lower=1)).values
    pmask = (df_all.rating >= p_thr).values
    if args.un_z is not None:
        pmask |= (df_all.rating.values >= args.un_min) & (z_all >= args.un_z)
    pos = df_all[pmask]
    z_pos = z_all[pmask]
    if args.un_z is not None and args.graph == "high":
        gdf = pos  # graph follows the (expanded) positives
    else:
        gdf = df_all[df_all.rating >= g_thr] if args.graph == "high" else df_all
    ew = map_w(args.edge_w, gdf.rating.values) if args.edge_w else np.ones(len(gdf), dtype=np.float32)
    if args.edge_time_beta:
        ew = ew * np.exp(-args.edge_time_beta * (1 - rel_all[df_all.index.get_indexer(gdf.index)])).astype(np.float32)
    if args.dir_time_u or args.dir_time_i:
        rel_g = rel_all[df_all.index.get_indexer(gdf.index)]
        g = pd.DataFrame({"u": gdf.userId.values, "i": gdf.movieId.values, "rel": rel_g})
        wU = np.exp(-args.dir_time_u * (1 - g.rel.values))
        wU = wU / g.assign(w=wU).groupby("u").w.transform("mean").values
        ri = g.groupby("i").rel.rank(pct=True).values
        wI = np.exp(-args.dir_time_i * (1 - ri))
        wI = wI / g.assign(w=wI).groupby("i").w.transform("mean").values
        A = norm_adj_dir(gdf, n_users, wU * ew, wI * ew)
    else:
        extra = None
        if args.ii_k > 0:
            Xb = dense(gdf, n_users, np.ones(len(gdf)))
            C = (Xb.T @ Xb)
            dg = torch.sqrt(torch.diag(C)).clamp(min=1e-8)
            sim = C / (dg[:, None] * dg[None, :] + 10.0)
            sim.fill_diagonal_(0)
            tv, tidx = torch.topk(sim, args.ii_k, dim=1)
            er = np.repeat(np.arange(N_ITEMS), args.ii_k) + n_users
            ec = tidx.cpu().numpy().ravel() + n_users
            ew2 = (args.ii_w * tv.cpu().numpy().ravel()).astype(np.float32)
            keep = ew2 > 0
            extra = (er[keep], ec[keep], ew2[keep])
            print(f"  item-item edges added: {int(keep.sum())}", flush=True)
        A = norm_adj(gdf, n_users, ew, extra=extra)
    if args.edge_drop > 0 and (args.dir_time_u or args.dir_time_i):
        raise SystemExit("--edge_drop is not supported together with --dir_time_*")
    # per-user index of all rated items (for rating pairs)
    srt = df_all.sort_values("userId")
    cnt_all = np.bincount(srt.userId.values, minlength=n_users)
    off_t = torch.as_tensor(np.concatenate([[0], np.cumsum(cnt_all)[:-1]]), device=dev)
    cnt_t = torch.as_tensor(cnt_all, device=dev)
    ai = torch.as_tensor(srt.movieId.values.copy(), device=dev)
    ar = torch.as_tensor(srt.rating.values.copy(), device=dev)
    pu = torch.as_tensor(pos.userId.values, device=dev)
    pi = torch.as_tensor(pos.movieId.values, device=dev)
    pos_mask = dense(pos, n_users, np.ones(len(pos))) > 0  # user's training positives (kept full)
    pu_all, pi_all = pu.clone(), pi.clone()
    low = df_all[~pmask]
    low_bias = (dense(low, n_users, np.ones(len(low))) > 0).float() * float(np.log(args.low_neg)) \
        if args.low_neg != 1.0 else None
    if args.rating_w:
        pw = torch.as_tensor(map_w(args.rating_w, pos.rating.values), device=dev)
    else:
        pw = torch.ones(len(pos), device=dev)
    if args.pos_pop_gamma:
        ipop = np.bincount(pos.movieId.values, minlength=N_ITEMS).astype(np.float32)
        g = np.full(len(pos), args.pos_pop_gamma, dtype=np.float32)
        if args.pp_user:
            pr = pd.DataFrame({"u": pos.userId.values, "lp": np.log1p(ipop[pos.movieId.values]), "rel": rel_all[pmask]})
            recent = pr.sort_values(["u", "rel"]).groupby("u").tail(10).groupby("u").lp.mean()
            niche = (1.0 - recent.rank(pct=True)).reindex(np.arange(n_users)).fillna(0.5).values  # 1 = most niche
            g = g * niche[pos.userId.values].astype(np.float32)
        pw = pw * torch.as_tensor(ipop[pos.movieId.values] ** (-g), dtype=torch.float32, device=dev)
    if args.user_pow:
        ucnt = np.bincount(pos.userId.values, minlength=n_users).astype(np.float32)
        pw = pw * torch.as_tensor(ucnt[pos.userId.values] ** (-args.user_pow), device=dev)
    if args.time_beta:
        pw = pw * torch.as_tensor(np.exp(-args.time_beta * (1 - rel_all[pmask])), dtype=torch.float32, device=dev)
    if args.un_gamma != 0:
        pw = pw * torch.as_tensor(np.clip(1 + args.un_gamma * z_pos, 0.2, 3.0), dtype=torch.float32, device=dev)
    print(f"positives {len(pos)} (raw>={p_thr}: {(df_all.rating >= p_thr).sum()}) | graph edges {len(gdf)}", flush=True)
    pw_all = None  # filled after pw is built
    model = LightGCN(n_users, args.dim, args.layers).to(dev)
    if args.prop == "sig":
        model.register_buffer("eu", torch.as_tensor(gdf.userId.values, device=dev))
        model.register_buffer("ei", torch.as_tensor(gdf.movieId.values + n_users, device=dev))
        deg = torch.bincount(torch.cat([model.eu, model.ei]), minlength=n_users + N_ITEMS).float()
        model.register_buffer("dnorm", (deg[model.eu] * deg[model.ei]).rsqrt())
        if args.edge_drop > 0 or args.prune_q > 0:
            raise SystemExit("--prop sig does not support --edge_drop / --prune_q")
    if args.short_k > 0:
        pr = pd.DataFrame({"u": pos.userId.values, "i": pos.movieId.values, "rel": rel_all[pmask]})
        pr = pr.sort_values(["u", "rel"]).groupby("u").tail(args.short_k)
        li = np.full((n_users, args.short_k), -1, dtype=np.int64)
        for uu, g in pr.groupby("u"):
            v = g.i.values; li[uu, -len(v):] = v
        model.register_buffer("last_idx", torch.as_tensor(li, device=dev))
        model.register_buffer("last_mask", (model.last_idx >= 0).float())
        model.gate = nn.Parameter(torch.full((n_users,), -2.0, device=dev))
    if args.pop_aff > 0:
        ipop = np.bincount(pos.movieId.values, minlength=N_ITEMS)
        lp = np.log1p(ipop[ipop > 0])
        edges = np.quantile(lp, np.linspace(0, 1, args.pop_bins)[1:-1])
        ib = np.where(ipop > 0, np.digitize(np.log1p(ipop), edges) + 1, 0)
        model.register_buffer("item_bin", torch.as_tensor(ib, device=dev))
        model.pop_q = nn.Parameter(torch.randn(args.pop_bins + 1, args.pop_aff, device=dev) * 0.1)
        model.pop_a = nn.Parameter(torch.zeros(n_users, args.pop_aff, device=dev))
    if args.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = None
    if args.sched in ("cosine", "linear"):
        T = args.epochs  # same schedule for validation run and final retrain
        def lr_lambda(e):
            if e < args.warmup:
                return (e + 1) / max(args.warmup, 1)
            prog = min(1.0, (e - args.warmup) / max(T - args.warmup, 1))
            if args.sched == "linear":
                return max(0.0, 1.0 - prog)
            return 0.5 * (1 + np.cos(np.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    seen = dense(df_all, n_users, np.ones(len(df_all))) > 0
    best, best_ep, best_state, bad = -1, 0, None, 0
    last_batch = None
    pw_all = pw.clone()
    for ep in range(1, epochs + 1):
        if args.prune_q > 0 and ep >= args.prune_start and (ep - args.prune_start) % args.prune_every == 0:
            model.eval()
            with torch.no_grad():
                Uf, If = model(A)
                sim = (F.normalize(Uf[pu_all], dim=1) * F.normalize(If[pi_all], dim=1)).sum(1).cpu().numpy()
            dfp = pd.DataFrame({"u": pu_all.cpu().numpy(), "s": sim})
            r = dfp.groupby("u").s.rank(pct=True, method="first").values
            keep = torch.as_tensor(r > args.prune_q, device=dev)
            pu, pi, pw = pu_all[keep], pi_all[keep], pw_all[keep]
            if args.graph == "high" and not (args.dir_time_u or args.dir_time_i) and args.edge_w == "" \
                    and args.graph_thr in (None, args.pos_thr, args.thr):
                kp = keep.cpu().numpy()
                gdf, ew = pos[kp], (ew[kp] if len(ew) == len(kp) else ew)
                A = norm_adj(gdf, n_users, ew)
            print(f"   [prune] ep {ep}: kept {int(keep.sum())}/{len(keep)} positives", flush=True)
        model.train()
        perm = torch.randperm(len(pu), device=dev)
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s: s + args.batch]
            u, i = pu[b], pi[b]
            if args.edge_drop > 0:
                keep = np.random.rand(len(gdf)) >= args.edge_drop
                A_step = norm_adj(gdf[keep], n_users, ew[keep])
            else:
                A_step = A
            U, I = model(A_step)
            e0 = model.emb.weight
            if args.loss == "bpr":
                loss = 0.0
                if not args.pair_only:
                    j = torch.randint(0, N_ITEMS, (len(b),), device=dev)
                    x = (U[u] * I[i]).sum(1) - (U[u] * I[j]).sum(1)
                    reg = (e0[u].pow(2).sum() + e0[n_users + i].pow(2).sum() + e0[n_users + j].pow(2).sum()) / len(b)
                    loss = -F.logsigmoid(x).mean() + args.reg * reg
                if args.pair_only or args.pair_w > 0:
                    k = cnt_t[u]
                    a = off_t[u] + (torch.rand(len(b), device=dev) * k).long()
                    c = off_t[u] + (torch.rand(len(b), device=dev) * k).long()
                    ra, rc = ar[a], ar[c]
                    ok = ra != rc
                    hi_i = torch.where(ra > rc, ai[a], ai[c])[ok]
                    lo_i = torch.where(ra > rc, ai[c], ai[a])[ok]
                    uu = u[ok]
                    if len(uu) > 0:
                        xp = (U[uu] * I[hi_i]).sum(1) - (U[uu] * I[lo_i]).sum(1)
                        regp = (e0[uu].pow(2).sum() + e0[n_users + hi_i].pow(2).sum()
                                + e0[n_users + lo_i].pow(2).sum()) / len(uu)
                        w = 1.0 if args.pair_only else args.pair_w
                        loss = loss + w * (-F.logsigmoid(xp).mean() + args.reg * regp)
            else:
                logits = logits_for(model, U, I, u)
                if low_bias is not None:
                    logits = logits + low_bias[u]  # user's low-rated items weigh more in the denominator
                if not args.no_mask_pos:
                    m = pos_mask[u].clone()
                    m[torch.arange(len(u), device=dev), i] = False  # keep the target itself
                    if args.fn_topk > 0:
                        cand = logits.detach().masked_fill(seen[u], float("-inf"))
                        m = m.scatter(1, torch.topk(cand, args.fn_topk, dim=1).indices, True)
                        m[torch.arange(len(u), device=dev), i] = False  # never mask the positive itself
                    logits = logits.masked_fill(m, float("-inf"))
                reg = (e0[u].pow(2).sum() + e0[n_users + i].pow(2).sum()) / len(b)
                w = pw[b]
                if args.margin > 0:
                    logits = logits.scatter_add(1, i[:, None],
                                                torch.full((len(u), 1), -args.margin / args.temp, device=dev))
                if args.n_neg > 0 and args.neg_mode == "pair":
                    neg = torch.randint(0, N_ITEMS, (len(u), args.n_neg), device=dev)
                    ln = logits.gather(1, neg).masked_fill(pos_mask[u].gather(1, neg), float("-inf"))
                    lp = logits.gather(1, i[:, None])
                    ce = F.cross_entropy(torch.cat([lp, ln], 1), torch.zeros(len(u), dtype=torch.long, device=dev),
                                         reduction="none")
                elif args.n_neg > 0:
                    neg = torch.randint(0, N_ITEMS, (args.n_neg,), device=dev)
                    ln = logits[:, neg].masked_fill(pos_mask[u][:, neg], float("-inf"))
                    lp = logits.gather(1, i[:, None])
                    ce = F.cross_entropy(torch.cat([lp, ln], 1), torch.zeros(len(u), dtype=torch.long, device=dev),
                                         reduction="none")
                else:
                    ce = F.cross_entropy(logits, i, reduction="none")
                if args.tce_max > 0:
                    rate = args.tce_max * min(1.0, ep / max(args.tce_epochs, 1))
                    n_keep = max(1, int(len(ce) * (1 - rate)))
                    idx = torch.argsort(ce.detach())[:n_keep]
                    ce, w = ce[idx], w[idx]
                    logits, i = logits[idx], i[idx]
                if args.pos_alpha != 1.0:
                    ce = ce - (args.pos_alpha - 1.0) * logits.gather(1, i[:, None]).squeeze(1)
                loss = (ce * w).sum() / w.sum() + args.reg * reg
                if args.xsim_lambda > 0:
                    def infonce(a, b):
                        a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
                        return F.cross_entropy(a @ b.T / args.xsim_tau, torch.arange(len(a), device=dev))
                    uu, ii = torch.unique(u), torch.unique(i)
                    V = model.cl_view
                    l_cl = infonce(U[uu], V[uu]) + infonce(I[ii], V[n_users + ii])
                    loss = loss + args.xsim_lambda * l_cl
                last_batch = (logits.detach(), i)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if sched is not None:
            sched.step()
        if val is not None and ep % args.eval_every == 0:
            gt_mat, vusers = val
            S = scores(model, A)[vusers]
            S[seen[vusers]] = -float("inf")
            sc = ndcg(torch.topk(S, K, 1).indices, vusers, gt_mat)
            flag = ""
            if sc > best:
                best, best_ep, bad, flag = sc, ep, 0, " *"
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"ep {ep:4d}  loss {tot:.3f}  val NDCG@20 {sc:.5f}{flag}", flush=True)
            if args.diag:
                with torch.no_grad():
                    model(A); Ls = model.last_layers
                    un = [l[:n_users].norm(dim=1).mean().item() for l in Ls]
                    inn = [l[n_users:].norm(dim=1).mean().item() for l in Ls]
                    fin = torch.stack(Ls[1:] if (args.skip_layer0 and len(Ls) > 1) else Ls).mean(0)
                    c0u = F.cosine_similarity(fin[:n_users], Ls[0][:n_users], dim=1).mean().item()
                    c0i = F.cosine_similarity(fin[n_users:], Ls[0][n_users:], dim=1).mean().item()
                    act = torch.nonzero(pos_mask.any(0)).squeeze(1)
                    sel = act[torch.randperm(len(act), device=dev)[:2000]] + n_users
                    def msim(E):
                        En = F.normalize(E[sel], dim=1); M = En @ En.T
                        return ((M.sum() - M.diagonal().sum()) / (len(sel) * (len(sel) - 1))).item()
                    sims = [msim(l) for l in Ls] + [msim(fin)]
                    has_sm = last_batch is not None
                    lg, ti = last_batch if has_sm else (torch.zeros(1, 2, device=dev), torch.zeros(1, dtype=torch.long, device=dev))
                    lg = lg.clone()
                    p = torch.softmax(lg, 1)
                    ppos = p.gather(1, ti[:, None]).squeeze(1)
                    rank = (lg > lg.gather(1, ti[:, None])).sum(1).float() + 1
                    q = p.clone(); q.scatter_(1, ti[:, None], 0); q = q / q.sum(1, keepdim=True).clamp(min=1e-12)
                    H = -(q * torch.log(q.clamp(min=1e-12))).sum(1)
                    qs = torch.sort(q, 1, descending=True).values
                print(f"   [layers] user norm {['%.3f' % x for x in un]} | item norm {['%.3f' % x for x in inn]}"
                      f" | cos(final, layer0) user {c0u:.3f} item {c0i:.3f}")
                print(f"   [smooth] mean item-item cos per layer {['%.3f' % x for x in sims[:-1]]} final {sims[-1]:.3f}")
                if has_sm: print(f"   [softmax] p(pos) mean {ppos.mean():.4f} | pos rank median {rank.median():.0f}"
                      f" | effective #neg {torch.exp(H).median():.0f} | top10 neg mass {qs[:, :10].sum(1).mean():.2f}"
                      f" | top100 {qs[:, :100].sum(1).mean():.2f}", flush=True)
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, A, seen, best_ep, best


d = os.path.expanduser(args.data)
tr = pd.read_csv(os.path.join(d, "train.csv"))
te = pd.read_csv(os.path.join(d, "test.csv"))
n_users = int(tr.userId.max()) + 1
os.makedirs("scores", exist_ok=True)
print("device:", dev, "|", vars(args))

# ---------- validation ----------
inner, val = temporal_split(tr)
gt_df = val[val.rating >= args.thr]
gt_mat = dense(gt_df, n_users, np.ones(len(gt_df))) > 0
vusers = torch.as_tensor(np.sort(gt_df.userId.unique()), device=dev)
t = time.time()
model, A, _, best_ep, best = train(inner, n_users, args.epochs, val=(gt_mat, vusers))
print(f"best epoch {best_ep}  val NDCG@20 {best:.5f}  ({time.time()-t:.0f}s)")
torch.save(scores(model, A).cpu(), f"scores/{args.tag}_val.pt")
if args.no_final:
    raise SystemExit

# ---------- full train with best #epochs -> submission ----------
model, A, seen, _, _ = train(tr, n_users, best_ep)
S = scores(model, A)
torch.save(S.cpu(), f"scores/{args.tag}_test.pt")
tusers = torch.as_tensor(te.userId.values, device=dev)
St = S[tusers].clone()
St[seen[tusers]] = -float("inf")
recs = torch.topk(St, K, 1).indices.cpu().numpy()
out = f"submission_{args.tag}.csv"
pd.DataFrame({"id": te.id.values,
              "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(out, index=False)
print(f"saved {out}  (val {best:.5f}, epochs {best_ep})")
