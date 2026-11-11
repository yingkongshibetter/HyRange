"""Exact certification in the upper-bound mode, with degree bounds lo_u <= d_u <= hi_u (paragraph "Degree bounds" of
Section IV; Table II, column Upper; the certified shares at t = 5 in paragraph Exports with a document threshold).

Model.  Input: pair counts W (exact) and per node an interval lo_u <= d_u <= hi_u on the number d_u of multi-member
hyperedges through u.  A world = multiset of node sets of size >= 2 with pair counts exactly W and every degree in its
interval.  Tightening: l_u = max(lo_u, max_v W_uv), U_u = min(hi_u, r_u), r_u = sum_v W_uv (d_u always lies in
[max_v W_uv, r_u]); empty interval = no world.  Special cases: lo = hi = d (exact mode), lo = 0, hi = D (upper-bound
mode), lo = 0, hi = r (pair counts alone).
Slack columns.  Per node u with U_u > l_u one column s_u (size 1) with a 1 in the degree row of u only, bounds
0 <= s_u <= U_u - l_u; degree row: sum_{c ni u} x_c + s_u = U_u.  Worlds <-> integer points (x, s) >= 0 of
[Z | S] (x, s) = (W, U), s <= U - l, one to one.  Slack columns are never certified or counted.
Pipeline (per input):
  1. forcing: forcing_upper.WDForcer(singleton=True), rules D1, A, B, P (as upper.py), subclassed so that the
     lower bound used by B and P is l_u = max(ell_u, lo_u) (residual lo drops by the forced copies; with lo = 0 the
     code path is identical to upper.py);
  2. components of the residual pair support (engine._components); isolated nodes carry only slack;
  3. per component: candidates = cliques of the support (candidates.enum_cliques_wp) with the filters
       size window max(2, r_u - (U_u - 1)(s_u - 1) + 1) <= |e| <= r_u - l_u + 2 (s_u = closed support size; proof:
       the other d_u - 1 hyperedges through u add between 1 and s_u - 1 each to r_u - (|e| - 1), and l_u <= d_u <= U_u),
       pair core of rule P (core K(u,v) contained in e, |e| <= |K| + min(slack, budget)) computed by forcing_upper's
       pair_core with upper bounds U and slack min_x [r_x - R(|K|-1) - max(0, l_x - R)],
     cap 300,000 / 60 s (engine.CAP, ENUM_S); rank test of [Z | S] (full column rank -> one world);
  4. one integer world (HiGHS MILP, random objective numpy default_rng(0), 60 s), checked exactly twice (matrix
     product in int64 and an independent recount of pair counts and degrees from the hyperedges); if the MILP finds
     none, an exact backtracking search (LP-Farkas pruned) for 60 s;
  5. exact certification of every candidate of that world (adapted copy of certify.certify_component):
     exhaustive enumeration (backtracking with per-column caps, <= 5,000 columns, 30 s), LP dual bound with the slack
     caps (exact rational evaluation: x_e >= y.R + sum_slack min(0, g_s) cap_s - vol * max_e' ((-g_e')/|e'|)^+ with
     g = phi - Z^T y, vol = sum_u U_u), refutation search (plain 2 s, then LP-Farkas pruned 60 s,
     certify.LPFarkas).  The IP-min step of certify.py (solver-based witnesses) is not used.
     lb_e proved, ub_e from verified witness worlds; copies of the found world: certified (<= lb), ambiguous
     (> ub, provably not certain), undecided (between).
Each run has a wall budget (--wall, 300 s; DBLP 600 s); components reached after it are not treated (reported).
Ground truth is used only after the run: false certified copies, and the residual truth must be a world of its
component made of candidates (checks the filters).

validate(N) checks the code against brute-force enumeration on small random inputs (used by tests/test_soundness.py).

usage: python -I -B -X utf8 experiments/upper_exact.py job bench <ds> <mode> [--wall s]    (six benchmarks, lo = 0, hi = d)
       python -I -B -X utf8 experiments/upper_exact.py job export <src> 5 [--wall s]       (exports at t = 5, lo = 0, hi = D)
       python -I -B -X utf8 experiments/upper_exact.py queue bench_pres bench_red export_t5 [--procs 8]
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from fractions import Fraction  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402
from hyrange import engine as EN  # noqa: E402
from hyrange import candidates as WP  # noqa: E402
from hyrange import upper as UP  # noqa: E402

OUT = CFG.RESULTS / "upper_exact"
JOBS = OUT / "jobs"
LOGS = CFG.LOGS / "upper_exact"
UPPER_RULES = ("D1", "A", "B", "P")
FORCE_TL = 900.0
CAP = EN.CAP
ENUM_S = EN.ENUM_S
RANK_MAX = EN.RANK_MAX
MILP_TL = 60.0
MILP_GUARD = EN.MILP_MAXCOLS
BUDGETS = dict(enum_max_cols=5000, enum_s=30.0, enum_max_worlds=1_000_000, lp=True, lp_s=60.0, dfs=True,
               dfs_s=2.0, dfslp_s=60.0, dfs_comp_s=900.0)
BENCH6 = ["crime", "hosts", "directors", "foursquare", "dblp", "topcs"]


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


_W2 = [None]


def W2():
    if _W2[0] is None:
        _W2[0] = UP.load_forcing_upper()
    return _W2[0]


class Infeasible(AssertionError):
    pass


def interval_forcer(adj, hi, lo, rules=UPPER_RULES, time_limit=FORCE_TL):
    base = W2().WDForcer

    class IntervalForcer(base):
        def __init__(self):
            super().__init__(adj, hi, rules=rules, time_limit=time_limit, copy_adj=True, singleton=True)
            self.lo = [int(x) for x in lo]
            self.vol_by_rule = {}
            for u in range(self.n):
                if self.lo[u] > self.lb[u]:
                    self.lb[u] = self.lo[u]
                if self.lb[u] > self.ub[u] or (self.lo[u] > 0 and not self.adj[u]):
                    raise Infeasible(("empty degree interval", u))

        def subtract(self, e, m, rule, touched):
            super().subtract(e, m, rule, touched)
            if m > 0:
                self.vol_by_rule[rule] = self.vol_by_rule.get(rule, 0) + m * len(e)
                for x in e:
                    self.lo[x] -= m
                    if self.lo[x] > self.lb[x]:
                        self.lb[x] = self.lo[x]
                    if self.lb[x] > self.ub[x]:
                        raise Infeasible(("empty degree interval after forcing", x, rule))
    return IntervalForcer()


# ------------------------------------------------------------------ component with slack columns
class _Shim:
    def __init__(self, adj, U):
        self.adj = adj
        self.deg = None
        self._U = U

    def upper(self, w):
        return self._U[w]


def build_comp_iv(nodes, adj, hi, lo, cap=CAP, enum_s=ENUM_S, rank_max=RANK_MAX, sizewin=True, kfilter=True,
                  enumerate_=True):
    C = EN.Comp()
    C.nodes = np.asarray(sorted(int(u) for u in nodes), dtype=np.int64)
    nl = len(C.nodes)
    loc = {int(u): i for i, u in enumerate(C.nodes)}
    pairs, Wp = [], []
    badj = [0] * nl
    for i, u in enumerate(C.nodes):
        for v, w in adj[int(u)].items():
            j = loc[v]
            badj[i] |= 1 << j
            if i < j:
                pairs.append((i, j))
                Wp.append(int(w))
    npairs = len(pairs)
    r = [int(sum(adj[int(u)].values())) for u in C.nodes]
    ell = [int(max(adj[int(u)].values())) for u in C.nodes]
    l_ = [max(int(lo[int(u)]), ell[i]) for i, u in enumerate(C.nodes)]
    U = [min(int(hi[int(u)]), r[i]) for i, u in enumerate(C.nodes)]
    for i in range(nl):
        if l_[i] > U[i]:
            raise Infeasible(("empty interval in component", int(C.nodes[i])))
    C.pairs, C.Wp, C.npairs, C.l, C.U, C.r = pairs, Wp, npairs, l_, U, r
    C.rows = pairs + [(i, i) for i in range(nl)]
    C.Rint = Wp + U
    C.vol = int(sum(U))
    clo = [badj[i] | (1 << i) for i in range(nl)]
    C.ssz = [x.bit_count() for x in clo]
    C.ncand = None
    C.slack_nodes = [i for i in range(nl) if U[i] > l_[i]]
    if not enumerate_:
        C.enum = "skipped"
        return C
    if sizewin:
        lo_n = [max(2, r[i] - (U[i] - 1) * (C.ssz[i] - 1) + 1) for i in range(nl)]
        hi_n = [r[i] - l_[i] + 2 for i in range(nl)]
    else:
        lo_n = [2] * nl
        hi_n = list(C.ssz)
    t0 = time.time()
    kinfo = None
    if kfilter:
        Ud = {int(u): U[i] for i, u in enumerate(C.nodes)}
        ld = {int(u): l_[i] for i, u in enumerate(C.nodes)}
        rd = {int(u): r[i] for i, u in enumerate(C.nodes)}
        shim = _Shim(adj, Ud)
        pc = W2().WDForcer.pair_core
        deadline = t0 + enum_s

        def kinfo(i, j):
            if time.time() > deadline:
                raise EN.CapExceeded("time")
            u, v = int(C.nodes[i]), int(C.nodes[j])
            K, X, R = pc(shim, u, v)
            kk = len(K)
            slack = min(rd[x] - R * (kk - 1) - max(0, ld[x] - R) for x in K)
            if slack < 0:
                raise Infeasible(("negative slack in the pair core", u, v))
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
    try:
        cands = WP.enum_cliques_wp(nl, badj, clo, lo_n, hi_n, kinfo=kinfo, cap=cap, deadline=t0 + enum_s)
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
    nc = len(cands)
    for t, i in enumerate(C.slack_nodes):
        ri.append(npairs + i)
        ci.append(nc + t)
    ncols = nc + len(C.slack_nodes)
    C.cands = cands
    C.ncand = nc
    C.ncols = ncols
    C.k = np.array([len(e) for e in cands] + [1] * len(C.slack_nodes), dtype=np.int64)
    C.wden = np.ones(ncols, dtype=np.int64)
    C.Zb = sp.csc_matrix((np.ones(len(ri), dtype=np.int64), (np.array(ri, dtype=np.int64),
                                                              np.array(ci, dtype=np.int64))),
                         shape=(len(C.rows), ncols))
    Ri = np.array(C.Rint, dtype=np.int64)
    colcap = np.zeros(ncols, dtype=np.int64)
    if nc:
        colcap[:nc] = np.minimum.reduceat(Ri[C.Zb.indices], C.Zb.indptr[:-1])[:nc]
    for t, i in enumerate(C.slack_nodes):
        colcap[nc + t] = U[i] - l_[i]
    C.colcap = colcap
    C.ub = colcap.astype(np.float64)
    C.unique = False
    C.xu = None
    C.rank_full_no_world = False
    if 0 < ncols <= rank_max and ncols <= len(C.rows) and ncols * len(C.rows) <= 8_000_000:
        if EN.full_column_rank_mod_p(C.Zb):
            x = np.linalg.lstsq(C.Zb.toarray().astype(np.float64), np.array(C.Rint, dtype=np.float64), rcond=None)[0]
            xi = np.rint(x).astype(np.int64)
            if world_ok(C, xi):
                C.unique = True
                C.xu = xi
            else:
                C.rank_full_no_world = True
    return C


def world_ok(C, x):
    """Exact check twice: int64 matrix product, and an independent recount from the hyperedges."""
    x = np.asarray(x, dtype=np.int64)
    if len(x) != C.ncols or (x < 0).any() or (x > C.colcap).any():
        return False
    if not np.array_equal(C.Zb.tocsc().astype(np.int64) @ x, np.asarray(C.Rint, dtype=np.int64)):
        return False
    nl = len(C.nodes)
    pc = Counter()
    dg = [0] * nl
    for j in np.nonzero(x[:C.ncand])[0]:
        e = C.cands[int(j)]
        m = int(x[j])
        for a in e:
            dg[a] += m
        for a, b in combinations(e, 2):
            pc[(a, b)] += m
    if dict(pc) != {p: w for p, w in zip(C.pairs, C.Wp)}:
        return False
    sl = {i: C.ncand + t for t, i in enumerate(C.slack_nodes)}
    for i in range(nl):
        if not (C.l[i] <= dg[i] <= C.U[i]):
            return False
        s = int(x[sl[i]]) if i in sl else 0
        if dg[i] + s != C.U[i]:
            return False
    return True


# ------------------------------------------------------------------ exact backtracking with column caps
class Budget(Exception):
    pass


class Found(Exception):
    pass


class WorldSearchCap:
    """certify.WorldSearch with a per-column cap (slack columns: U - l)."""

    def __init__(self, C):
        Zb = C.Zb.tocsc()
        self.nc = Zb.shape[1]
        self.nr = Zb.shape[0]
        self.crow = [Zb.indices[Zb.indptr[c]:Zb.indptr[c + 1]].tolist() for c in range(self.nc)]
        Zr = Zb.tocsr()
        self.rcand = [Zr.indices[Zr.indptr[r]:Zr.indptr[r + 1]].tolist() for r in range(self.nr)]
        self.R0 = [int(v) for v in C.Rint]
        self.cc = [int(v) for v in C.colcap]

    def search(self, on_world, target=None, cap=None, time_s=10.0, prune=None):
        crow, rcand, cc = self.crow, self.rcand, self.cc
        self.pruned = 0
        R = list(self.R0)
        nc = self.nc
        alive = [1] * nc
        cnt = [len(rc) for rc in rcand]
        x = [0] * nc
        pos = set(r for r in range(self.nr) if R[r] > 0)
        undo_log = []
        nodes = [0]
        deadline = time.perf_counter() + time_s

        def kill(c):
            alive[c] = 0
            undo_log.append((0, c, 0))
            ok = True
            for r in crow[c]:
                cnt[r] -= 1
                if cnt[r] == 0 and R[r] > 0:
                    ok = False
            return ok

        def assign(c, v):
            undo_log.append((1, c, v))
            x[c] = v
            zero = []
            if v:
                for r in crow[c]:
                    R[r] -= v
                    if R[r] == 0:
                        pos.discard(r)
                        zero.append(r)
            ok = kill(c)
            for r in zero:
                for c2 in rcand[r]:
                    if alive[c2]:
                        if not kill(c2):
                            ok = False
            return ok

        def undo(mark):
            while len(undo_log) > mark:
                t, c, v = undo_log.pop()
                if t == 0:
                    alive[c] = 1
                    for r in crow[c]:
                        cnt[r] += 1
                else:
                    x[c] = 0
                    if v:
                        for r in crow[c]:
                            if R[r] == 0:
                                pos.add(r)
                            R[r] += v

        def rec():
            nodes[0] += 1
            if (nodes[0] & 255) == 0 and time.perf_counter() > deadline:
                raise Budget()
            if not pos:
                on_world(x)
                return
            if prune is not None and prune(alive, R, pos):
                self.pruned += 1
                return
            r = min(pos, key=cnt.__getitem__)
            m = cnt[r]
            if m == 0:
                return
            Rr = R[r]
            c = -1
            best = -1
            for c2 in rcand[r]:
                if alive[c2] and len(crow[c2]) > best:
                    best = len(crow[c2])
                    c = c2
            ub = min(min(R[r2] for r2 in crow[c]), cc[c])
            if c == target:
                ub = min(ub, cap)
            if m == 1:
                if ub < Rr:
                    return
                vals = (Rr,)
            else:
                vals = range(min(ub, Rr), -1, -1)
            for v in vals:
                mark = len(undo_log)
                if assign(c, v):
                    rec()
                undo(mark)

        if any(cnt[r] == 0 for r in pos):
            return "complete", 0
        out = ["complete"]

        def root():
            if target is None:
                rec()
                return
            ubt = min(min(R[r2] for r2 in crow[target]), cap, cc[target])
            for v in range(ubt, -1, -1):
                mark = len(undo_log)
                if assign(target, v):
                    rec()
                undo(mark)

        def runner():
            try:
                root()
            except Budget:
                out[0] = "budget"
            except Found:
                out[0] = "found"
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 4 * nc + 10000))
        threading.stack_size(255 * 1024 * 1024)
        th = threading.Thread(target=runner)
        th.start()
        th.join()
        threading.stack_size(0)
        return out[0], nodes[0]


# ------------------------------------------------------------------ LP lower bound with slack caps
class LPMinIV:
    def __init__(self, C):
        from scipy.optimize._highspy import _core as hc
        A = sp.csc_matrix(C.Zb, dtype=np.float64)
        nr, nc = A.shape
        lp = hc.HighsLp()
        lp.num_col_ = nc
        lp.num_row_ = nr
        lp.col_cost_ = np.zeros(nc)
        lp.col_lower_ = np.zeros(nc)
        up = np.full(nc, hc.kHighsInf)
        up[C.ncand:] = C.colcap[C.ncand:].astype(np.float64)
        lp.col_upper_ = up
        b = np.array([float(v) for v in C.Rint])
        lp.row_lower_ = b
        lp.row_upper_ = b
        m = lp.a_matrix_
        m.format_ = hc.MatrixFormat.kColwise
        m.start_ = A.indptr.astype(np.int32)
        m.index_ = A.indices.astype(np.int32)
        m.value_ = A.data
        m.num_col_ = nc
        m.num_row_ = nr
        lp.a_matrix_ = m
        lp.sense_ = hc.ObjSense.kMinimize
        h = hc._Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("presolve", "off")
        h.setOptionValue("primal_feasibility_tolerance", 1e-9)
        h.setOptionValue("dual_feasibility_tolerance", 1e-9)
        h.passModel(lp)
        self.h = h
        self.nc = nc
        self.cur = None
        self.C = C
        self.Zt = C.Zb.T.tocsr().astype(np.int64)
        nnz = np.diff(C.Zb.indptr)
        self.mrow = int(nnz.max()) if len(nnz) else 1

    def lower_bound(self, j, tl):
        h = self.h
        if self.cur is None:
            h.changeColsCost(self.nc, np.arange(self.nc, dtype=np.int32), np.zeros(self.nc))
        else:
            h.changeColCost(int(self.cur), 0.0)
        h.changeColCost(int(j), 1.0)
        self.cur = j
        h.setOptionValue("time_limit", float(h.getRunTime() + tl))
        h.run()
        st = h.modelStatusToString(h.getModelStatus())
        sol = h.getSolution()
        if st != "Optimal" or not sol.dual_valid:
            return None, None, st
        y = np.array(sol.row_dual)
        lb = max(self.rig(j, y), self.rig(j, -y))
        return lb, float(np.array(sol.col_value)[j]), st

    def rig(self, j, y):
        """Exact: for every x >= 0 with Z x = R and s <= cap,  x_j >= Y.R/2^K + sum_slack min(0, g_s) cap_s
        - vol * max_{hyperedge e} ((-g_e)/|e|)^+,  g = phi - Z^T Y / 2^K (phi = indicator of j)."""
        C = self.C
        ymax = float(np.abs(y).max()) if len(y) else 0.0
        K = 40
        while K > 0 and (ymax + 1.0) * 2.0 ** K * self.mrow >= 2.0 ** 61:
            K -= 1
        scale = 2 ** K
        Y = np.rint(y * scale).astype(np.int64)
        S = self.Zt @ Y
        nc = C.ncand
        ph = np.zeros(nc)
        ph[j] = 1.0
        zy = S[:nc].astype(np.float64) / scale
        t = (zy - ph) / C.k[:nc]
        tol = 1e-9 * (1.0 + (float(np.abs(zy).max()) if nc else 0.0))
        viol = Fraction(0)
        if nc:
            tmax = float(t.max())
            if tmax > -tol:
                for e in np.nonzero(t >= tmax - tol)[0]:
                    te = (Fraction(int(S[e]), scale) - (1 if e == j else 0)) / int(C.k[e])
                    if te > viol:
                        viol = te
        slack_term = Fraction(0)
        for t_, i in enumerate(C.slack_nodes):
            col = nc + t_
            g = -Fraction(int(S[col]), scale)
            if g < 0:
                slack_term += g * int(C.colcap[col])
        yR = Fraction(sum(int(a) * int(b) for a, b in zip(C.Rint, Y)), scale)
        return yR + slack_term - C.vol * viol


def find_world(C, rng, tl=MILP_TL):
    if C.unique and C.xu is not None:
        return C.xu, "pinned"
    A = sp.csc_matrix(C.Zb, dtype=np.float64)
    b = np.array([float(v) for v in C.Rint])
    c = rng.random(A.shape[1])
    if A.shape[1] > MILP_GUARD:
        from hyrange import pipeline as MO
        r = MO.hsolve_guarded(c, A, b, C.ub, tl)
    else:
        r = EN.hsolve(c, A, b, ub=C.ub, integer=True, time_limit=tl)
    if "x" not in r:
        return None, "nosol:" + str(r.get("status"))
    xi = np.rint(r["x"]).astype(np.int64)
    if not world_ok(C, xi):
        return None, "unverified"
    return xi, "found"


def certify_iv(C, x0, B, deadline, keep=None):
    """Adapted certify.certify_component: targets = hyperedge columns of the found world x0."""
    from hyrange import certify as CX
    rec = dict(times={})
    assert world_ok(C, x0)
    nc = C.ncand
    I = [int(j) for j in np.nonzero(x0[:nc])[0]]
    ub = {j: int(x0[j]) for j in I}
    lb = {j: 0 for j in I}
    how = {}

    def undecided():
        return [j for j in I if lb[j] < ub[j]]

    def take(xw):
        for j in I:
            if int(xw[j]) < ub[j]:
                ub[j] = int(xw[j])
        if keep is not None:
            keep.append(np.array(xw, dtype=np.int64))
    WS = WorldSearchCap(C)
    t0 = time.perf_counter()
    rec["enum"] = dict(attempted=False)
    if C.ncols <= B["enum_max_cols"] and undecided():
        mn = {j: ub[j] for j in I}
        nw = [0]

        def on_world(x):
            nw[0] += 1
            new = False
            for j in I:
                if x[j] < mn[j]:
                    mn[j] = x[j]
                    new = True
            if new:
                assert world_ok(C, x), "enumerated world fails the exact check"
                if keep is not None:
                    keep.append(np.array(x, dtype=np.int64))
            if nw[0] >= B["enum_max_worlds"]:
                raise Budget()
            if all(mn[j] == 0 for j in I):
                raise Found()
        st, nodes = WS.search(on_world, time_s=max(0.1, min(B["enum_s"], deadline - time.perf_counter())))
        rec["enum"] = dict(attempted=True, status=st, worlds=nw[0], nodes=nodes)
        for j in I:
            ub[j] = min(ub[j], mn[j])
        if st == "complete":
            for j in I:
                lb[j] = ub[j]
                how[j] = "enum"
        elif st == "found":
            for j in I:
                how[j] = "witness_enum"
    rec["times"]["enum"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if B["lp"] and undecided():
        LP = LPMinIV(C)
        for j in undecided():
            if time.perf_counter() > deadline:
                break
            lbf, val, st = LP.lower_bound(j, B["lp_s"])
            if lbf is not None:
                v = max(0, math.ceil(lbf))
                assert v <= ub[j], ("rigorous LP bound above a witnessed world", j, v, ub[j])
                if v > lb[j]:
                    lb[j] = v
                    if lb[j] == ub[j]:
                        how[j] = "lp"
    rec["times"]["lp"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    FK = None
    if B["dfs"]:
        for j in I:
            use_lp = False
            while lb[j] < ub[j]:
                if time.perf_counter() - t0 > B["dfs_comp_s"] or time.perf_counter() > deadline:
                    break
                cap = ub[j] - 1
                got = [None]

                def on_world(x, got=got):
                    got[0] = list(x)
                    raise Found()
                tl = min(B["dfslp_s"] if use_lp else B["dfs_s"], max(0.1, deadline - time.perf_counter()))
                if use_lp:
                    if FK is None:
                        FK = CX.LPFarkas(C)
                    st, nodes = WS.search(on_world, target=j, cap=cap, time_s=tl, prune=FK)
                else:
                    st, nodes = WS.search(on_world, target=j, cap=cap, time_s=tl)
                if st == "complete":
                    lb[j] = ub[j]
                    how[j] = "dfs_lp" if use_lp else "dfs"
                elif st == "found":
                    xw = np.asarray(got[0], dtype=np.int64)
                    assert world_ok(C, xw) and xw[j] <= cap
                    take(xw)
                    if lb[j] >= ub[j]:
                        how[j] = "witness_dfs"
                elif not use_lp:
                    use_lp = True
                else:
                    break
    rec["times"]["dfs"] = time.perf_counter() - t0
    for j in I:
        if lb[j] == ub[j] and j not in how:
            how[j] = "witness"
    rec["cands"] = [dict(j=j, x=int(x0[j]), lb=int(lb[j]), ub=int(ub[j]), how=how.get(j, "undecided")) for j in I]
    rec["all_decided"] = all(lb[j] == ub[j] for j in I)
    rec["how_counts"] = dict(Counter(c["how"] for c in rec["cands"]))
    return rec


# ------------------------------------------------------------------ pipeline
def pipeline(adj, lo, hi, n, rules=UPPER_RULES, B=None, wall=None, keep=False, rank_max=RANK_MAX, force_tl=FORCE_TL,
             seed=0, logf=None, sizewin=True, kfilter=True):
    B = dict(BUDGETS, **(B or {}))
    t_all = time.perf_counter()
    deadline = t_all + (wall if wall is not None else 1e9)
    out = dict(stage="start")
    F = interval_forcer(adj, hi, lo, rules=rules, time_limit=force_tl)
    F.run()
    forced = Counter({tuple(sorted(int(v) for v in e)): int(m) for e, m in F.forced.items() if m > 0})
    out.update(t_force_s=time.perf_counter() - t_all, force_complete=bool(F.complete),
               forced_copies=int(sum(forced.values())), forced_vol=int(sum(len(e) * m for e, m in forced.items())),
               copies_by_rule=dict(F.counts), vol_by_rule=dict(F.vol_by_rule))
    for u in range(n):
        if not F.adj[u] and F.lo[u] > 0:
            raise Infeasible(("isolated node with positive lower bound", u))
    comps_nodes = sorted(EN._components(F.adj, n), key=len)
    rng = np.random.default_rng(seed)
    cert = Counter(forced)
    crecs = []
    comps = []
    for ci, cn in enumerate(comps_nodes):
        rem = deadline - time.perf_counter()
        if rem <= 0:
            C = build_comp_iv(cn, F.adj, F.deg, F.lo, enumerate_=False)
            crecs.append(dict(index=ci, nodes=len(C.nodes), vol=C.vol, status="wall_budget"))
            comps.append((C, None, None))
            continue
        C = build_comp_iv(cn, F.adj, F.deg, F.lo, cap=CAP, enum_s=min(ENUM_S, rem), rank_max=rank_max,
                          sizewin=sizewin, kfilter=kfilter)
        cr = dict(index=ci, nodes=len(C.nodes), vol=C.vol, enum=C.enum, ncand=C.ncand,
                  slack_cols=len(C.slack_nodes))
        if C.cands is None:
            cr["status"] = "over_cap:" + C.enum
            crecs.append(cr)
            comps.append((C, None, None))
            continue
        if C.ncand == 0 or C.rank_full_no_world:
            cr["status"] = "no_world:" + ("no_candidates" if C.ncand == 0 else "rank_full_no_world")
            crecs.append(cr)
            comps.append((C, None, None))
            continue
        x, st = find_world(C, rng, min(MILP_TL, max(1.0, deadline - time.perf_counter())))
        if x is None:
            # exact fallback: backtracking for a world (LP-Farkas pruned), within the remaining wall time
            from hyrange import certify as CX
            got = [None]

            def on_world(xx, got=got):
                got[0] = list(xx)
                raise Found()
            s2, _ = WorldSearchCap(C).search(on_world, time_s=min(60.0, max(0.1, deadline - time.perf_counter())),
                                             prune=CX.LPFarkas(C))
            if s2 == "found" and world_ok(C, got[0]):
                x, st = np.asarray(got[0], dtype=np.int64), st + "|dfs_found"
        cr["world"] = st
        if x is None:
            cr["status"] = "no_world:" + st
            crecs.append(cr)
            comps.append((C, None, None))
            continue
        wit = [np.array(x)] if keep else None
        if st == "pinned":
            cands = [dict(j=int(j), x=int(x[j]), lb=int(x[j]), ub=int(x[j]), how="rank") for j in
                     np.nonzero(x[:C.ncand])[0]]
            crec = dict(cands=cands, all_decided=True, how_counts={"rank": len(cands)})
            cr["status"] = "pinned"
        else:
            tc = time.perf_counter()
            crec = certify_iv(C, x, B, deadline, keep=wit)
            cr["t_cert_s"] = time.perf_counter() - tc
            cr["status"] = "decided" if crec["all_decided"] else "partial"
            cr["how_counts"] = crec["how_counts"]
            cr["enum_cert"] = crec["enum"]
        cr["copies_world"] = int(sum(c["x"] for c in crec["cands"]))
        cr["certified"] = int(sum(c["lb"] for c in crec["cands"]))
        cr["ambiguous"] = int(sum(c["x"] - c["ub"] for c in crec["cands"]))
        cr["undecided"] = int(sum(c["ub"] - c["lb"] for c in crec["cands"]))
        cr["cert_vol"] = int(sum(c["lb"] * len(C.cands[c["j"]]) for c in crec["cands"]))
        cr["undecided_vol"] = int(sum((c["ub"] - c["lb"]) * len(C.cands[c["j"]]) for c in crec["cands"]))
        cr["world_vol"] = int(sum(c["x"] * len(C.cands[c["j"]]) for c in crec["cands"]))
        for c in crec["cands"]:
            if c["lb"] > 0:
                cert[tuple(sorted(int(C.nodes[i]) for i in C.cands[c["j"]]))] += c["lb"]
        if cr["status"] == "partial":
            cr["undecided_cands"] = [dict(members=[int(C.nodes[i]) for i in C.cands[c["j"]]], x=c["x"], lb=c["lb"],
                                          ub=c["ub"]) for c in crec["cands"] if c["lb"] < c["ub"]][:50]
        crecs.append(cr)
        comps.append((C, x, wit))
        if logf and (len(C.nodes) > 50 or cr.get("t_cert_s", 0) > 5):
            logf(f"  comp {ci}: {len(C.nodes)} nodes, {C.ncand} cands, {cr['status']}, cert {cr['certified']} "
                 f"undec {cr['undecided']}")
    st = Counter(c["status"].split(":")[0] for c in crecs)
    skipped = [c for c in crecs if c["status"].split(":")[0] in ("over_cap", "no_world", "wall_budget")]
    lc = max(crecs, key=lambda c: c["nodes"]) if crecs else None
    out.update(
        stage="done", comps=len(crecs), comp_status=dict(st),
        largest_comp=dict(nodes=lc["nodes"], ncand=lc.get("ncand"), status=lc["status"]) if lc else None,
        skipped_comps=len(skipped),
        skipped=[dict(nodes=c["nodes"], vol=c["vol"], status=c["status"], ncand=c.get("ncand")) for c in skipped][:30],
        skipped_vol_U=int(sum(c["vol"] for c in skipped)),
        copies_certified_resid=int(sum(c.get("certified", 0) for c in crecs)),
        copies_ambiguous=int(sum(c.get("ambiguous", 0) for c in crecs)),
        copies_undecided=int(sum(c.get("undecided", 0) for c in crecs)),
        undecided_vol=int(sum(c.get("undecided_vol", 0) for c in crecs)),
        cert_copies=int(sum(cert.values())), cert_vol=int(sum(len(e) * m for e, m in cert.items())),
        fcert_proved=(not skipped and all(c["status"] in ("pinned", "decided") for c in crecs)),
        comp_records=crecs, t_total_s=time.perf_counter() - t_all)
    # label-free upper bound on the volume of F^cert: certified + undecided (ub - lb) + residual U of skipped comps
    out["fcert_vol_upper"] = out["cert_vol"] + out["undecided_vol"] + out["skipped_vol_U"]
    return out, forced, cert, (F, comps_nodes, comps)


def truth_checks(out, cert, truth, extra):
    """Truth used only here: false certified copies; residual truth is a world of each enumerated component."""
    F, comps_nodes, comps = extra
    false = int(sum(max(0, m - truth.get(e, 0)) for e, m in cert.items()))
    resid = Counter(truth)
    resid.subtract(Counter({tuple(sorted(int(v) for v in e)): int(m) for e, m in F.forced.items()}))
    neg = sum(1 for m in resid.values() if m < 0)
    comp_of = {}
    for ci, cn in enumerate(comps_nodes):
        for u in cn:
            comp_of[int(u)] = ci
    per = [Counter() for _ in comps_nodes]
    outside = 0
    for e, m in resid.items():
        if m <= 0:
            continue
        cs = {comp_of.get(u) for u in e}
        if len(cs) != 1 or None in cs:
            outside += 1
            continue
        per[cs.pop()][e] += m
    notcand = notworld = 0
    for ci, (C, x, wit) in enumerate(comps):
        if C.cands is None or C.ncand is None:
            continue
        idx = {tuple(int(C.nodes[i]) for i in c): j for j, c in enumerate(C.cands)}
        xt = np.zeros(C.ncols, dtype=np.int64)
        ok = True
        for e, m in per[ci].items():
            j = idx.get(e)
            if j is None:
                notcand += 1
                ok = False
                continue
            xt[j] = m
        if ok:
            loc = {int(u): i for i, u in enumerate(C.nodes)}
            dg = Counter()
            for e, m in per[ci].items():
                for u in e:
                    dg[loc[u]] += m
            for t, i in enumerate(C.slack_nodes):
                xt[C.ncand + t] = C.U[i] - dg[i]
            if not world_ok(C, xt):
                notworld += 1
    out["truth"] = dict(false_certified_copies=false, forced_above_truth=neg, truth_hyperedges_outside_residual=outside,
                        truth_hyperedges_not_candidates=notcand, truth_vector_not_a_world=notworld,
                        ok=(false == 0 and neg == 0 and outside == 0 and notcand == 0 and notworld == 0))
    return out["truth"]


# ------------------------------------------------------------------ validation
class BruteCap(Exception):
    pass


def brute_worlds(n, Wd, lo, hi, max_worlds=50_000, max_steps=2_000_000):
    """All worlds (pair counts exactly Wd, lo <= degree <= hi), independent of the certification code: pick the
    smallest positive pair p and choose, in nondecreasing index order, the R_p node sets containing p (each world is
    reached exactly once)."""
    subsets = [s for k in range(2, n + 1) for s in combinations(range(n), k)]
    sp_ = {s: list(combinations(s, 2)) for s in subsets}
    res = {p: w for p, w in Wd.items() if w > 0}
    deg = [0] * n
    cur = []
    out = []
    steps = [0]

    def rec():
        steps[0] += 1
        if steps[0] > max_steps:
            raise BruteCap()
        pos = sorted(p for p, w in res.items() if w > 0)
        if not pos:
            if all(lo[u] <= deg[u] <= hi[u] for u in range(n)):
                out.append(tuple(sorted(cur)))
                if len(out) > max_worlds:
                    raise BruteCap()
            return
        p = pos[0]
        cand = [s for s in subsets if p[0] in s and p[1] in s and all(res.get(q, 0) > 0 for q in sp_[s])]
        pick(res[p], 0, cand)

    def pick(left, start, cand):
        if left == 0:
            rec()
            return
        for i in range(start, len(cand)):
            s = cand[i]
            if any(res.get(q, 0) < 1 for q in sp_[s]) or any(deg[x] + 1 > hi[x] for x in s):
                continue
            for q in sp_[s]:
                res[q] -= 1
            for x in s:
                deg[x] += 1
            cur.append(s)
            pick(left - 1, i, cand)
            cur.pop()
            for q in sp_[s]:
                res[q] += 1
            for x in s:
                deg[x] -= 1
    rec()
    assert len(out) == len(set(out)), "brute force produced a duplicate world"
    return out


def gen_instance(rng, kind, p_single):
    n = rng.randint(5, 8)
    H = []
    for _ in range(rng.randint(2, 7)):
        if H and rng.random() < 0.25:
            H.append(rng.choice(H))
        else:
            H.append(tuple(sorted(rng.sample(range(n), rng.randint(2, 4)))))
    d = [0] * n
    W = Counter()
    for e in H:
        for u in e:
            d[u] += 1
        for p in combinations(e, 2):
            W[p] += 1
    r = [sum(w for p, w in W.items() if u in p) for u in range(n)]
    single = [rng.randint(1, 3) if rng.random() < p_single else 0 for _ in range(n)]
    if kind == "upper":
        lo, hi = [0] * n, [d[u] + single[u] for u in range(n)]
    elif kind == "exact":
        lo, hi = list(d), list(d)
    elif kind == "pairs":
        lo, hi = [0] * n, list(r)
    else:
        lo = [max(0, d[u] - rng.randint(0, 3)) for u in range(n)]
        hi = [d[u] + rng.randint(0, 3) + single[u] for u in range(n)]
    return n, H, W, d, lo, hi, single


def validate(N):
    rng = random.Random(20261110)
    configs = {
        "full": dict(rules=UPPER_RULES, B={}),
        "noforce": dict(rules=(), B={}),
        "noforce_noenum_norank": dict(rules=(), B=dict(enum_max_cols=-1), rank_max=0),
        "noforce_lp_only": dict(rules=(), B=dict(enum_max_cols=-1, dfs=False), rank_max=0),
        "noforce_dfs_only": dict(rules=(), B=dict(enum_max_cols=-1, lp=False), rank_max=0),
    }
    kinds = [("upper", 0.0), ("upper", 0.2), ("upper", 0.5), ("interval", 0.2), ("exact", 0.0), ("pairs", 0.0)]
    st = Counter()
    fails = []
    per_cfg = {c: Counter() for c in configs}
    t0 = time.perf_counter()
    outp = OUT / "validation.json"
    it = 0

    def dump(complete):
        res = dict(script="upper_exact.py validate", seed=20261110, generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                   complete=complete, instance_generator=dict(
                       n="uniform 5..8", groups="2..7 multi-member groups, size uniform 2..4, each with prob. 0.25 a "
                       "repeat of an earlier group", kinds="cycled: upper p=0, upper p=0.2, upper p=0.5 (one-member "
                       "groups 1..3 per node with prob. p added to hi), interval (lo = d - U{0..3}, hi = d + U{0..3} + "
                       "singles, p=0.2), exact (lo = hi = d), pairs-only (lo = 0, hi = r)"),
                   configs={k: dict(rules=list(v["rules"]), budgets=v["B"], rank_max=v.get("rank_max", RANK_MAX))
                            for k, v in configs.items()},
                   stats=dict(st), per_config={k: dict(v) for k, v in per_cfg.items()}, n_failures=len(fails),
                   failures=fails[:30], seconds=time.perf_counter() - t0)
        MB.save_json(res, outp)
    while st["instances"] < N:
        it += 1
        kind, ps = kinds[it % len(kinds)]
        n, H, W, d, lo, hi, single = gen_instance(rng, kind, ps)
        try:
            worlds = brute_worlds(n, dict(W), lo, hi)
        except BruteCap:
            st["skipped_brute_cap"] += 1
            continue
        truth = Counter(H)
        wset = set(worlds)
        assert tuple(sorted(H)) in wset, "hidden hypergraph is not a brute-force world"
        common = None
        for w in worlds:
            c = Counter(w)
            common = c if common is None else (common & c)
        fcert = common
        st["instances"] += 1
        st["kind:" + kind + f":p{ps}"] += 1
        st["worlds"] += len(worlds)
        st["multi_world"] += len(worlds) > 1
        st["with_one_member_groups"] += any(single) and kind in ("upper", "interval")
        st["with_repeats"] += len(set(H)) < len(H)
        st["fcert_has_ge3"] += any(len(e) >= 3 for e in fcert)
        adj = [dict() for _ in range(n)]
        for (u, v), w in W.items():
            adj[u][v] = w
            adj[v][u] = w
        for cname, cfg in configs.items():
            pc = per_cfg[cname]
            try:
                out, forced, cert, extra = pipeline(adj, lo, hi, n, rules=cfg["rules"], B=cfg["B"], keep=True,
                                                    rank_max=cfg.get("rank_max", RANK_MAX), force_tl=60)
            except Exception as ex:
                fails.append(dict(it=it, cfg=cname, why="exception " + repr(ex)[:300], H=H, lo=lo, hi=hi))
                continue
            pc["runs"] += 1
            bad = {e: m for e, m in cert.items() if m > fcert.get(e, 0)}
            if bad:
                pc["soundness_violations"] += 1
                fails.append(dict(it=it, cfg=cname, why="certified above F^cert",
                                  bad=[[list(e), m] for e, m in bad.items()], H=H, lo=lo, hi=hi))
            if any(len(e) >= 3 for e in cert) and kind == "pairs":
                pc["pairs_mode_ge3_certified"] += 1
                fails.append(dict(it=it, cfg=cname, why="pairs-only mode certified a >=3 hyperedge"))
            if out["fcert_proved"]:
                pc["fcert_claimed"] += 1
                if Counter({e: m for e, m in cert.items() if m}) != Counter({e: m for e, m in fcert.items() if m}):
                    pc["equality_violations"] += 1
                    fails.append(dict(it=it, cfg=cname, why="no undecided copy but certified != F^cert", H=H, lo=lo,
                                      hi=hi, cert=[[list(e), m] for e, m in cert.items()],
                                      fcert=[[list(e), m] for e, m in fcert.items()]))
                else:
                    pc["fcert_equal"] += 1
            pc["undecided_copies"] += out["copies_undecided"]
            pc["skipped_comps"] += out["skipped_comps"]
            F, comps_nodes, comps = extra
            base = Counter(forced)
            found = [(C, x, wit) for C, x, wit in comps if x is not None]
            for C, x, wit in found:
                base.update({tuple(sorted(int(C.nodes[i]) for i in C.cands[j])): int(x[j])
                             for j in np.nonzero(x[:C.ncand])[0]})
            if found or not comps:
                key0 = tuple(sorted(e for e, m in base.items() for _ in range(max(0, m))))
                pc["found_world_checked"] += 1
                if not comps or len(found) == len(comps):
                    if key0 not in wset:
                        pc["found_world_not_world"] += 1
                        fails.append(dict(it=it, cfg=cname, why="found world is not a world"))
            for C, x, wit in found:
                for xw in (wit or []):
                    pc["witnesses_checked"] += 1
                    full = Counter(base)
                    full.subtract({tuple(sorted(int(C.nodes[i]) for i in C.cands[j])): int(x[j])
                                   for j in np.nonzero(x[:C.ncand])[0]})
                    full.update({tuple(sorted(int(C.nodes[i]) for i in C.cands[j])): int(xw[j])
                                 for j in np.nonzero(xw[:C.ncand])[0]})
                    key = tuple(sorted(e for e, m in full.items() for _ in range(max(0, m))))
                    if len(found) == len(comps) and key not in wset:
                        pc["witness_not_world"] += 1
                        fails.append(dict(it=it, cfg=cname, why="witness is not a world"))
            cand_sets = set()
            for C, x, wit in comps:
                if C.cands is not None:
                    for c in C.cands:
                        cand_sets.add(tuple(sorted(int(C.nodes[i]) for i in c)))
            miss = 0
            for w in worlds:
                rest = Counter(w)
                rest.subtract(forced)
                for e, m in rest.items():
                    if m > 0 and e not in cand_sets:
                        miss += 1
            if miss:
                pc["candidate_misses"] += 1
                fails.append(dict(it=it, cfg=cname, why="world hyperedge not a candidate", miss=miss, H=H, lo=lo,
                                  hi=hi))
            tc = truth_checks(out, cert, truth, extra)
            if not tc["ok"]:
                pc["truth_check_fail"] += 1
                fails.append(dict(it=it, cfg=cname, why="truth check", tc=tc))
            pc["certified_ge3_copies"] += sum(m for e, m in cert.items() if len(e) >= 3)
            pc["fcert_copies_total"] += sum(fcert.values())
            pc["certified_copies_total"] += sum(cert.values())
        if st["instances"] % 20 == 0:
            dump(False)
            log(f"validate {st['instances']}: failures {len(fails)}")
    dump(True)
    log(f"validation done: {dict(st)}; failures {len(fails)}")
    for k, v in per_cfg.items():
        log(f"  {k}: {dict(v)}")


# ------------------------------------------------------------------ jobs
def jkey(kind, a, b):
    return f"{kind}_{a}_{b}"


def ref_upper_bench(ds, mode):
    p = CFG.RESULTS / "upper_forcing" / "jobs" / f"{ds}_{mode}.json"
    return json.load(open(p, encoding="utf-8"))["forced_vol_share"] if p.exists() else None


def job(kind, a, b, wall=300.0):
    from hyrange.pairs_only import certified_pairs
    from hyrange import forcing as WC
    outp = JOBS / (jkey(kind, a, b) + ".json")
    t_all = time.perf_counter()
    res = dict(kind=kind, a=a, b=b, stage="start", wall_budget_s=wall)
    if kind == "bench":
        ds, mode = a, b
        lines, _ = MB.target(ds, mode)
        nodes = sorted(set(v for e in lines for v in e))
        loc = {v: i for i, v in enumerate(nodes)}
        n = len(nodes)
        lines_loc = [tuple(loc[v] for v in e) for e in lines]
        adj, d = WC.build_wdeg(lines_loc, n)
        truth = Counter(lines_loc)
        vol = int(sum(d))
        lo, hi = [0] * n, list(d)
        res["ref_forced_share_upper_forcing"] = ref_upper_bench(ds, mode)
        res.update(ds=ds, name=MB.DISP[ds], mode=mode, n=n, vol=vol)
    elif kind == "export":
        from hyrange import exports as EXP
        src, t = a, int(b)
        docs, info = EXP.source_docs(src)
        S, trace, adj, d, D, facts = EXP.build_export(docs, t)
        n = len(S)
        truth = Counter(trace)
        vol = facts["trace_vol"]
        lo, hi = [0] * n, list(D)
        rp = CFG.RESULTS / "exports" / "jobs" / f"{src}_t{t:02d}_upper.json"
        ref = json.load(open(rp, encoding="utf-8")) if rp.exists() else {}
        res.update(src=src, name=EXP.DISP[src], t=t, n=n, vol=vol, ref_forced_share_exports=ref.get("cert_vol_share"),
                   ref_w_only_share_exports=ref.get("w_only_vol_share"))
    else:
        raise SystemExit(kind)
    wo = certified_pairs({u: x for u, x in enumerate(adj)})
    wo_c = Counter({tuple(p): cc for p, cc in wo.items()})
    res["w_only_vol_share"] = 2 * sum(wo.values()) / vol
    res["w_only_false"] = int(sum(max(0, m - truth.get(e, 0)) for e, m in wo_c.items()))
    MB.save_json(res, outp)
    log(f"{outp.stem}: n {n}, vol {vol}")
    try:
        out, forced, cert, extra = pipeline(adj, lo, hi, n, wall=wall, logf=log)
    except Infeasible as ex:
        res.update(stage="infeasible", exception=repr(ex)[:300])
        MB.save_json(res, outp)
        return
    out["forced_vol_share"] = out["forced_vol"] / vol
    out["cert_vol_share"] = out["cert_vol"] / vol
    res.update(out)
    MB.save_json(res, outp)
    tc = truth_checks(res, cert, truth, extra)
    res["incomplete"] = not res["fcert_proved"]
    res["t_job_s"] = time.perf_counter() - t_all
    res["stage"] = "done"
    MB.save_json(res, outp)
    log(f"{outp.stem}: forced {res['forced_vol_share']:.4f} cert {res['cert_vol_share']:.4f} proved "
        f"{res['fcert_proved']} undec {res['copies_undecided']} skipped {res['skipped_comps']} truth {tc}")


def job_list(name):
    if name == "bench_pres":
        return [("bench", ds, "preserved") for ds in BENCH6]
    if name == "bench_red":
        return [("bench", ds, "reduced") for ds in BENCH6]
    if name == "export_t5":
        return [("export", s, "5") for s in ["dblp", "openalex_db", "topcs", "openalex_cs_size3plus",
                                                    "openalex_bio"]]
    raise SystemExit(name)


def queue(names, procs):
    LOGS.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    pending = [j for nm in names for j in job_list(nm)]
    running = []
    while pending or running:
        for rj in list(running):
            p_, fh, k, t0, kill = rj
            rc = p_.poll()
            if rc is None and time.time() - t0 > kill:
                p_.kill()
                p_.wait()
                rc = "killed"
            if rc is not None:
                fh.close()
                running.remove(rj)
                jp = JOBS / (k + ".json")
                rec = json.load(open(jp, encoding="utf-8")) if jp.exists() else dict(key=k)
                if rec.get("stage") != "done":
                    rec.update(killed=(rc == "killed"), returncode=str(rc), stage_reached=rec.get("stage"),
                               stage="failed" if rc != "killed" else "timeout", incomplete=True)
                    MB.save_json(rec, jp)
                log(f"end {k} rc {rc} ({time.time() - t0:.0f}s)")
        while pending and len(running) < procs:
            spec = pending.pop(0)
            k = jkey(*spec)
            jp = JOBS / (k + ".json")
            if jp.exists() and json.load(open(jp, encoding="utf-8")).get("stage") == "done":
                continue
            wall = 600.0 if spec[1] == "dblp" else 300.0
            fh = open(LOGS / f"{k}.log", "w", encoding="utf-8")
            cmd = [*CFG.module_cmd("hyrange.upper_exact"), "job", spec[0], spec[1], spec[2], "--wall", str(wall)]
            p_ = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE))
            running.append((p_, fh, k, time.time(), wall + 240.0))
            log(f"start {k}")
        time.sleep(0.5)
    log("queue done")


def main():
    cmd = sys.argv[1]
    if cmd == "validate":
        validate(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    elif cmd == "job":
        args = sys.argv[2:]
        wall = 300.0
        if "--wall" in args:
            i = args.index("--wall")
            wall = float(args[i + 1])
            args = args[:i] + args[i + 2:]
        job(args[0], args[1], args[2], wall=wall)
    elif cmd == "queue":
        procs = 8
        args = sys.argv[2:]
        if "--procs" in args:
            i = args.index("--procs")
            procs = int(args[i + 1])
            args = args[:i] + args[i + 2:]
        queue(args, procs)
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main()
