"""Shared numerical core: HiGHS calls with one thread, clique enumeration, the rank test modulo a prime, residual
components, the exact world check and the rigorous dual bound.

A component of the residual input is held in a Comp: its local nodes, its rows (the residual pairs), the residual
pair counts Rint, its volume (sum_e |e| x_e, the same in every world), its candidate hyperedges cands with sizes k,
the 0/1 row-candidate matrix Zb and the multiplicity bounds ub.  For the co-occurrence input every candidate has
weight one, so wden = 1 and L = 1 throughout.
"""
from __future__ import annotations

import time
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

CAP = 300_000           # candidate cap per component
ENUM_S = 60.0           # time limit of the candidate listing of one component (s)
MILP_S = 60.0
MILP_MAXCOLS = 30_000   # HiGHS did not honor its time limit on a 106,762-column MILP; larger MILPs run in a child
                        # process that is killed after the limit (pipeline.hsolve_guarded)
RANK_MAX = 2000
PRIME = 2147483629

# ------------------------------------------------------------------ HiGHS (vendored by scipy) with one thread
def hsolve(c, A, b, ub=None, integer=False, maximize=False, time_limit=60.0):
    from scipy.optimize._highspy import _core as hc
    A = sp.csc_matrix(A, dtype=np.float64)
    nr, nc = A.shape
    lp = hc.HighsLp()
    lp.num_col_ = nc
    lp.num_row_ = nr
    lp.col_cost_ = np.asarray(c, dtype=np.float64)
    lp.col_lower_ = np.zeros(nc)
    lp.col_upper_ = np.full(nc, hc.kHighsInf) if ub is None else np.asarray(ub, dtype=np.float64)
    lp.row_lower_ = np.asarray(b, dtype=np.float64)
    lp.row_upper_ = np.asarray(b, dtype=np.float64)
    m = lp.a_matrix_
    m.format_ = hc.MatrixFormat.kColwise
    m.start_ = A.indptr.astype(np.int32)
    m.index_ = A.indices.astype(np.int32)
    m.value_ = A.data
    m.num_col_ = nc
    m.num_row_ = nr
    lp.a_matrix_ = m
    lp.sense_ = hc.ObjSense.kMaximize if maximize else hc.ObjSense.kMinimize
    if integer:
        lp.integrality_ = [hc.HighsVarType.kInteger] * nc
    h = hc._Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("primal_feasibility_tolerance", 1e-9)
    h.setOptionValue("dual_feasibility_tolerance", 1e-9)
    if integer:
        h.setOptionValue("mip_rel_gap", 0.0)
        h.setOptionValue("mip_abs_gap", 0.0)
        h.setOptionValue("mip_feasibility_tolerance", 1e-7)
    h.passModel(lp)
    t0 = time.perf_counter()
    h.run()
    t = time.perf_counter() - t0
    st = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    sol = h.getSolution()
    out = dict(status=st, t=t, obj=float(info.objective_function_value))
    if sol.value_valid:
        out["x"] = np.array(sol.col_value)
    if not integer and sol.dual_valid:
        out["y"] = np.array(sol.row_dual)
    if integer:
        out["bound"] = float(info.mip_dual_bound)
        out["gap"] = float(info.mip_gap)
    return out


# ------------------------------------------------------------------ cliques
class CapExceeded(Exception):
    pass


def enum_cliques(n, adj, sizes, need=None, cap=2_000_000, deadline=None, hmax=None, allow=None):
    """All cliques e (sorted tuples of local ids) of the bitset graph adj with |e| in sizes, |e| >= the largest
    need[(u, v)] and |e| <= the smallest hmax[(u, v)] over its pairs, and |e| in allow[(u, v)] for every pair whose
    allow entry is not None (sizes admitted by the degree filter).  Raises CapExceeded('cap' | 'time')."""
    lo, hi = min(sizes), max(sizes)
    out = []
    visits = [0]

    def ok_sizes(cur, k):
        for a in range(len(cur)):
            for b_ in range(a + 1, len(cur)):
                s_ = allow.get((cur[a], cur[b_]))
                if s_ is not None and k not in s_:
                    return False
        return True

    def rec(cur, cand, Lneed, H):
        k = len(cur)
        visits[0] += 1
        if deadline is not None and (visits[0] & 4095) == 0 and time.time() > deadline:
            raise CapExceeded("time")
        if k in sizes and Lneed <= k <= H and (allow is None or k < 2 or ok_sizes(cur, k)):
            out.append(tuple(cur))
            if len(out) > cap:
                raise CapExceeded("cap")
        if k >= H:
            return
        c = cand
        while c:
            b = c & -c
            v = b.bit_length() - 1
            c ^= b
            nc = c & adj[v]
            L2 = Lneed
            H2 = H
            if need is not None:
                for u in cur:
                    t = need[(u, v)]
                    if t > L2:
                        L2 = t
                    if hmax is not None:
                        t = hmax[(u, v)]
                        if t < H2:
                            H2 = t
                if L2 > H2:
                    continue
            if k + 1 + nc.bit_count() < max(lo, L2):
                continue
            cur.append(v)
            rec(cur, nc, L2, H2)
            cur.pop()

    for v in range(n):
        if adj[v]:
            rec([v], adj[v] & ~((1 << (v + 1)) - 1), 0, hi)
    return out



def full_column_rank_mod_p(Zb, p=PRIME):
    """True if the integer matrix has full column rank modulo p (which implies full column rank over Q)."""
    A = np.asarray(Zb.toarray(), dtype=np.int64) % p
    m, n = A.shape
    if n > m:
        return False
    r = 0
    for c in range(n):
        nz = np.nonzero(A[r:, c])[0]
        if len(nz) == 0:
            return False
        i = r + nz[0]
        if i != r:
            A[[r, i]] = A[[i, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        below = r + 1 + np.nonzero(A[r + 1:, c])[0]
        if len(below):
            A[below] = (A[below] - (np.outer(A[below, c], A[r]) % p)) % p
        r += 1
    return True


# ------------------------------------------------------------------ residual components
class Comp:
    """One residual component (label-free).  Local node ids 0..nl-1; rows = pairs (i<j) then, for P, diagonals."""

    def __init__(self):
        self.nodes = None      # np.int64 global ids
        self.rows = None       # list of (i, j); (i, i) for diagonal rows
        self.Rint = None       # list of Python ints (scaled by L)
        self.vol = 0           # sum_e |e| x_e, the same in every world
        self.deg = None        # residual degrees (ints)
        self.ssz = None        # closed support sizes
        self.cuv = None        # np.int64, |S_i cap S_j| per pair row (pairs only)
        self.enum = "ok"       # ok | cap | time
        self.t_enum = 0.0
        self.cands = None      # list of tuples (local)
        self.k = None
        self.wden = None       # omega(k) = 1 / wden
        self.Zb = None         # csc rows x cands, 0/1 int64
        self.ub = None
        self.unique = False
        self.xu = None


def _components(res_adj, n):
    rr, cc = [], []
    for u in range(n):
        for v in res_adj[u]:
            if v != u:
                rr.append(u)
                cc.append(v)
    rr = np.array(rr, dtype=np.int64)
    cc = np.array(cc, dtype=np.int64)
    A = sp.coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, n)).tocsr()
    lab = connected_components(A, directed=False)[1]
    act = np.unique(rr)
    order = np.argsort(lab[act], kind="stable")
    act = act[order]
    labs = lab[act]
    cut = np.nonzero(np.diff(labs))[0] + 1
    return np.split(act, cut) if len(act) else []



def verify_world(C, x, L) -> bool:
    """Exact check that the integer multiset x on the candidates reproduces the component's residual."""
    acc = [Fraction(0)] * len(C.rows)
    Zb = C.Zb
    for c in np.nonzero(x)[0]:
        w = Fraction(int(x[c]), int(C.wden[c]))
        for t in Zb.indices[Zb.indptr[c]:Zb.indptr[c + 1]]:
            acc[t] += w
    return all(a * L == R for a, R in zip(acc, C.Rint))


# ------------------------------------------------------------------ rigorous dual bounds
def rig_bound(C, L, num, den, y, sense):
    """Exact rational bound on max (sense='max') or min ('min') of phi^T x over all x >= 0 on the candidates
    with Z x = R_K, from an arbitrary dual vector y (rounded to a dyadic grid first)."""
    y = np.asarray(y, dtype=np.float64)
    nnz_col = np.diff(C.Zb.indptr)
    mrow = int(nnz_col.max()) if len(nnz_col) else 1
    ymax = float(np.abs(y).max()) if len(y) else 0.0
    K = 40
    while K > 0 and (ymax + 1.0) * 2.0 ** K * mrow >= 2.0 ** 61:
        K -= 1
    scale = 2 ** K
    Y = np.rint(y * scale).astype(np.int64)
    Zt = getattr(C, "_Zt", None)
    if Zt is None:
        Zt = C._Zt = C.Zb.T.tocsr()
    S = Zt @ Y                                               # exact int64 sums over the rows of each candidate
    zy = S.astype(np.float64) / (C.wden.astype(np.float64) * scale)
    ph = num.astype(np.float64) / den
    t = (ph - zy) / C.k if sense == "max" else (zy - ph) / C.k
    tol = 1e-9 * (1.0 + float(np.abs(ph).max()) + float(np.abs(zy).max()))
    tmax = float(t.max()) if len(t) else -1.0
    viol = Fraction(0)
    if tmax > -tol:
        for e in np.nonzero(t >= tmax - tol)[0]:
            te = (Fraction(int(num[e]), int(den[e])) - Fraction(int(S[e]), int(C.wden[e]) * scale)) / int(C.k[e])
            if sense == "min":
                te = -te
            if te > viol:
                viol = te
    yR = Fraction(sum(int(a) * int(b) for a, b in zip(C.Rint, Y)), L * scale)
    return yR + C.vol * viol if sense == "max" else yR - C.vol * viol


def best_rig(C, L, num, den, y, sense):
    """y and -y are both valid dual vectors; keep the tighter bound (guards against sign conventions)."""
    b1 = rig_bound(C, L, num, den, y, sense)
    b2 = rig_bound(C, L, num, den, -np.asarray(y), sense)
    return min(b1, b2) if sense == "max" else max(b1, b2)
