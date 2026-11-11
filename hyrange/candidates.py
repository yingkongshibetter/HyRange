"""Candidate hyperedges, their filters and the rank test (Section IV of the experiments).

Input.  The co-occurrence matrix A^T A of the incidence A: pair counts W_uv = #{e : u, v in e} for u != v and the
diagonal d_u = #{e : u in e}.  A hyperedge e adds zeta(e) = 1 on each of its pairs and 1 on the diagonal of each
member.  vol = sum_u d_u = sum_e |e| is the same in every world; inside a residual component K, summing the diagonal
rows of Z x = R_K gives sum_e |e| x_e = vol_K for every x >= 0 (fractional too), which is the identity
engine.rig_bound needs.  So rig_bound, the integer program, verify_world and the rank test of engine.py apply with
wden = 1, L = 1 and C.vol = sum of the residual diagonal of K.

build_comp_wp builds one residual component: candidate cliques (cap 300,000 candidates / 60 s) by enum_cliques_wp,
with the size window and the pair-core filter below; the 0/1 matrix with pair rows then diagonal rows; the
multiplicity bound ub_e = min of the pair and diagonal entries of e; the rank test (full column rank modulo a prime
=> one world, solved and checked exactly).

Pair-core filter (kfilter=True).  For a residual pair (u, v) with R = W_uv, rule E (called P in the code) proves that
every one of the R hyperedges through u, v contains the core K(u, v) computed by forcing.WDForcer.pair_core, and that
the memberships outside K summed over these R copies are at most min(slack, budget),
slack = min_{x in K} [r_x - R(|K|-1) - (d_x - R)], budget = sum_{w in X} min(R, min_{x in K} W_xw).  Both statements
hold in every world of the residual (the residual worlds are exactly H - F), so a hyperedge e of any world satisfies,
for every pair {u, v} in e: K(u, v) is a subset of e and |e| <= |K(u, v)| + min(slack, budget).  Cliques violating
either are not candidates.  The forcing uses the same quantities, so nothing new is assumed.

Size window (sizewin=True).  Let r_u = sum_v W_uv = sum_{f ni u} (|f| - 1) on the residual and S_u the closed
support of u, s_u = |S_u|.  A hyperedge e of k members through u leaves d_u - 1 other hyperedges f through u, each
with 1 <= |f| - 1 <= s_u - 1 (f lies in S_u).  Hence
      d_u - 1 <= r_u - (k - 1) <= (d_u - 1)(s_u - 1),
i.e. max(2, r_u - (d_u - 1)(s_u - 1) + 1) <= k <= r_u - d_u + 2.  Every integer in that range is a sum of exactly
d_u - 1 integers in [1, s_u - 1], so this is exactly the node test "r_u - (k-1) is a sum of d_u - 1 terms (k'-1),
2 <= k' <= s_u".  It holds in every world, so no hyperedge of any world is pruned.  A pair (u, v) admits
k in [max(lo_u, lo_v), min(c_uv, hi_u, hi_v)], c_uv = |S_u cap S_v|.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from hyrange import engine as EN
from hyrange import forcing as WC


# ------------------------------------------------------------------ candidate enumeration with W+ filters
class _CoreShim:
    """Minimal object for forcing.WDForcer.pair_core (which reads only .adj and .deg) on the residual."""

    def __init__(self, adj, deg):
        self.adj = adj
        self.deg = deg


def make_kinfo(nodes, loc, adj, deg, r_loc, deadline=None):
    """kinfo(i, j) for local ids i < j: (bitmask of the pair core K(u, v) in local ids, |K| + min(slack, budget)).
    K(u, v) and the bound are those of the W+ pair rule P (the appendix of the experiments, section P), computed by the
    unchanged forcing.WDForcer.pair_core on the residual: every hyperedge through u, v contains K, and the memberships
    outside K summed over the R_uv copies through u, v are at most min(slack, budget), so one copy has at most
    |K| + min(slack, budget) members."""
    shim = _CoreShim(adj, deg)
    pc = WC.WDForcer.pair_core

    def kinfo(i, j):
        if deadline is not None and time.time() > deadline:
            raise EN.CapExceeded("time")
        u, v = int(nodes[i]), int(nodes[j])
        K, X, R = pc(shim, u, v)
        kk = len(K)
        slack = min(r_loc[loc[x]] - R * (kk - 1) - (deg[x] - R) for x in K)
        assert slack >= 0, ("negative slack on the residual", u, v)
        ub = 0
        for w in X:
            aw = adj[w]
            ub += min([R] + [aw[x] for x in K])
            if ub >= slack:
                break
        m = 0
        for x in K:
            m |= 1 << loc[x]
        return m, kk + min(slack, ub)
    return kinfo


def enum_cliques_wp(nl, badj, clo, lo_n, hi_n, kinfo=None, cap=EN.CAP, deadline=None):
    """engine.enum_cliques specialised to W+ (same DFS: vertices are added in increasing local id, a clique is output
    when its size lies in the admissible window, CapExceeded('cap' | 'time')), with per-pair data computed lazily:
      size window of a pair (i, j): [max(lo_i, lo_j), min(c_ij, hi_i, hi_j)] (node window, see the module docstring);
      if kinfo is given, also |e| <= |K(i, j)| + min(slack, budget) and K(i, j) contained in e (pair rule P).
    A partial clique cur + v is pruned when a node required by a pair core is neither in cur + v nor among the later
    candidates nc (every extension adds only nodes of nc), or when the size window is empty; both conditions hold for
    every hyperedge of every world, so no world hyperedge is lost."""
    hi_all = max(x.bit_count() for x in clo) if nl else 0
    out = []
    visits = [0]
    memo = {}

    def pinfo(u, v):
        key = (u, v) if u < v else (v, u)
        r = memo.get(key)
        if r is None:
            a = max(lo_n[u], lo_n[v])
            b = min((clo[u] & clo[v]).bit_count(), hi_n[u], hi_n[v])
            if kinfo is not None:
                km, hk = kinfo(*key)
                b = min(b, hk)
            else:
                km = 0
            r = memo[key] = (a, b, km)
        return r

    def rec(cur, curmask, cand, Lneed, H, Kreq):
        k = len(cur)
        visits[0] += 1
        if deadline is not None and (visits[0] & 4095) == 0 and time.time() > deadline:
            raise EN.CapExceeded("time")
        if k >= 2 and Lneed <= k <= H and (Kreq & ~curmask) == 0:
            out.append(tuple(cur))
            if len(out) > cap:
                raise EN.CapExceeded("cap")
        if k >= H:
            return
        c = cand
        while c:
            b = c & -c
            v = b.bit_length() - 1
            c ^= b
            nc = c & badj[v]
            L2, H2, K2 = Lneed, H, Kreq
            for u in cur:
                a_, b_, km = pinfo(u, v)
                if a_ > L2:
                    L2 = a_
                if b_ < H2:
                    H2 = b_
                K2 |= km
            if L2 > H2:
                continue
            nm = curmask | b
            if (K2 & ~nm) & ~nc:
                continue
            if k + 1 + nc.bit_count() < max(2, L2):
                continue
            cur.append(v)
            rec(cur, nm, nc, L2, H2, K2)
            cur.pop()

    for v in range(nl):
        if badj[v]:
            rec([v], 1 << v, badj[v] & ~((1 << (v + 1)) - 1), 0, hi_all, 0)
    return out


# ------------------------------------------------------------------ label-free preparation
def build_comp_wp(nodes, adj, deg, cap=EN.CAP, enum_s=EN.ENUM_S, rank_max=EN.RANK_MAX, enumerate_=True,
                  sizewin=True, kfilter=True, legacy_enum=False):
    """One residual component of W+ (local ids 0..nl-1; rows = pairs (i<j), then diagonals (i, i)).
    adj, deg: the residual (global ids).  legacy_enum=True uses engine.enum_cliques with dict need/hmax (node window
    only; kept to cross-check enum_cliques_wp)."""
    C = EN.Comp()
    C.nodes = np.asarray(nodes, dtype=np.int64)
    nl = len(C.nodes)
    loc = {int(u): i for i, u in enumerate(C.nodes)}
    pairs, Rint = [], []
    badj = [0] * nl
    for i, u in enumerate(C.nodes):
        for v, w in adj[int(u)].items():
            j = loc[v]                                   # KeyError would mean a pair leaving the component
            badj[i] |= 1 << j
            if i < j:
                pairs.append((i, j))
                Rint.append(int(w))
    npairs = len(pairs)
    dg = [int(deg[int(u)]) for u in C.nodes]
    assert all(x > 0 for x in dg), "residual node without degree"
    C.rows = pairs + [(i, i) for i in range(nl)]
    C.Rint = Rint + dg
    C.vol = sum(dg)                                      # sum_e |e| x_e in every world of K
    C.deg = dg
    C.r = [int(sum(adj[int(u)].values())) for u in C.nodes]
    clo = [badj[i] | (1 << i) for i in range(nl)]
    C.ssz = [x.bit_count() for x in clo]
    C.cuv = np.array([(clo[i] & clo[j]).bit_count() for (i, j) in pairs], dtype=np.int64)
    C.npairs = npairs
    C.sizewin = bool(sizewin)
    for i in range(nl):
        assert C.r[i] >= dg[i], ("r_u < d_u on the residual", int(C.nodes[i]))
    if not enumerate_:
        C.enum = "skipped"
        return C
    C.kfilter = bool(kfilter) and not legacy_enum
    if sizewin:
        lo_n = [max(2, C.r[i] - (dg[i] - 1) * (C.ssz[i] - 1) + 1) for i in range(nl)]
        hi_n = [C.r[i] - dg[i] + 2 for i in range(nl)]
    else:
        lo_n = [2] * nl
        hi_n = list(C.ssz)
    t0 = time.time()
    C.t_filter = 0.0
    try:
        if legacy_enum:
            need, hmax = {}, {}
            for t, (i, j) in enumerate(pairs):
                a = max(lo_n[i], lo_n[j])
                b = min(int(C.cuv[t]), hi_n[i], hi_n[j])
                need[(i, j)] = need[(j, i)] = a
                hmax[(i, j)] = hmax[(j, i)] = b
            cands = EN.enum_cliques(nl, badj, set(range(2, max(C.ssz) + 1)), need=need, cap=cap,
                                    deadline=t0 + enum_s, hmax=hmax, allow=None)
        else:
            kinfo = make_kinfo(C.nodes, loc, adj, deg, C.r, deadline=t0 + enum_s) if C.kfilter else None
            cands = enum_cliques_wp(nl, badj, clo, lo_n, hi_n, kinfo=kinfo, cap=cap, deadline=t0 + enum_s)
    except EN.CapExceeded as ex:
        C.enum = str(ex)
        C.t_enum = time.time() - t0
        return C
    C.t_enum = time.time() - t0
    rowid = {p: t for t, p in enumerate(pairs)}
    ri, ci = [], []
    for c, e in enumerate(cands):
        for a in range(len(e)):
            for b_ in range(a + 1, len(e)):
                ri.append(rowid[(e[a], e[b_])])
                ci.append(c)
            ri.append(npairs + e[a])
            ci.append(c)
    C.cands = cands
    C.k = np.array([len(e) for e in cands], dtype=np.int64)
    C.wden = np.ones(len(cands), dtype=np.int64)
    C.Zb = sp.csc_matrix((np.ones(len(ri), dtype=np.int64), (np.array(ri, dtype=np.int64), np.array(ci, dtype=np.int64))),
                         shape=(len(C.rows), len(cands)))
    if len(cands):
        Ri = np.array(C.Rint, dtype=np.int64)
        C.ub = np.minimum.reduceat(Ri[C.Zb.indices], C.Zb.indptr[:-1]).astype(np.float64)   # min pair / diagonal entry
    else:
        C.ub = np.zeros(0)
    nc = len(cands)
    if 0 < nc <= rank_max and nc <= len(C.rows) and nc * len(C.rows) <= 8_000_000:
        if EN.full_column_rank_mod_p(C.Zb):
            x = np.linalg.lstsq(C.Zb.toarray().astype(np.float64), np.array(C.Rint, dtype=np.float64), rcond=None)[0]
            xi = np.rint(x).astype(np.int64)
            if (xi >= 0).all() and EN.verify_world(C, xi, 1):
                C.unique = True
                C.xu = xi
            else:
                C.rank_full_no_world = True
    return C


def check_residual_wp(F, true_edges, n):
    """Independent check: W+ of (true multiset - forced multiset) equals the residual kept by the forcer."""
    from collections import Counter
    rem = Counter(tuple(sorted(e)) for e in true_edges)
    for e, m in F.forced.items():
        rem[e] -= m
        if rem[e] < 0:
            return False
    adj2, deg2 = WC.build_wdeg([e for e, m in rem.items() for _ in range(m)], n)
    return deg2 == list(F.deg) and all(adj2[u] == F.adj[u] for u in range(n))
