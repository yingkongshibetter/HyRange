"""Local certification (the levels of the experiments): L+ at nodes of any degree, and its level-B generalisation
(paragraph Local certification).

Setting.  Input (W, d) of a hidden multiset of node sets of size >= 2; a world reproduces W and d exactly.  The forcer
pipeline.WDForcerTL (rules D1, A, B, P, L) is run to its fixpoint; L+ then works on its residual input (the residual
worlds are exactly {H - F : H a world}, so a lower bound proved on the residual is a number of copies every world of
the original input contains beyond F).

Level-B system (B = a node set of the residual; B = {u} is the node level "L+").  Unknowns x_c >= 0 integer, one per
candidate clique c that contains at least one node of B.  Candidates: cliques of the residual support with the
filters of the exact-mode candidate listing (candidates.enum_cliques_wp): node size window max(2, r_x - (d_x - 1)(s_x - 1)
+ 1) <= |c| <= r_x - d_x + 2 for every member x (s_x = closed support size in the residual), pair window
|c| <= |S_x cap S_y cap V_loc| (V_loc = the closed neighbourhood of the root node, every listed clique lies in it),
pair core of rule P (K(x, y) contained in c, |c| <= |K| + min(slack, budget), forcing.WDForcer.pair_core on the
residual).  Multiplicity bound ub_c = min of the pair counts inside c and the degrees of its members.  A clique through
a node of B is listed once, at its first node of B in the order of B (rooted DFS restricted to the closed
neighbourhood of that node minus the earlier nodes of B).  If the listing exceeds CAP cliques or ENUM_S seconds the
system is SKIPPED (the list is never truncated).
Constraints (each holds for the restriction of every world to the hyperedges that touch B):
  (E1) every pair {u, v} with u in B, W_uv > 0:  sum_{c ni u, v} x_c = W_uv        (every such hyperedge touches B)
  (E2) every u in B:                              sum_{c ni u} x_c = d_u
  (I1) pair {v, w} outside B, W_vw > 0:           sum_{c ni v, w} x_c <= W_vw
  (I2) node v outside B:                          sum_{c ni v} x_c <= d_v, and
                                                  sum_{c ni v} (|c| - 2) x_c <= r_v - d_v
       (the d_v - sum_{c ni v} x_c hyperedges through v that do not touch B add >= 1 each to r_v:
        sum_{c ni v} (|c| - 1) x_c + d_v - sum_{c ni v} x_c <= r_v).  For B = {u} this is the rule-L bound
        sum_{T ni v} |T| x_T <= r_v - (d_v - W_uv), since sum_{T ni v} x_T = W_uv.
  (T)  triple cut, t = {a, b, c} with a node of B and all three pairs in the support:
       lo_t <= sum_{c' contains t} x_c' <= hi_t,  lo_t = max(0, W_ab + W_ac - d_a, W_ab + W_bc - d_b, W_ac + W_bc - d_c)
       (at a: the hyperedges through a, b and those through a, c are both among the d_a through a, so at least
       W_ab + W_ac - d_a contain all three; every hyperedge containing t contains a node of B, so it is an unknown),
       hi_t = min(W_ab, W_ac, W_bc) (implied by E1 / I1 and x >= 0).  For B = {u} these are the bounds lo_vw, hi_vw of
       rule L.
Proofs (never the solver's word):
  (c) full enumeration of the integer points (exact backtracking LSearch, below) when it completes: lambda_c = min.
  (a) LP dual bound: any vector y over the rows gives, for every real point,
        x_j >= [ sum_eq y_i b_i + sum_le min(0, y_i) h_i + sum_ge (y_i lo_i if y_i > 0 else y_i hi_i)
                 + sum_c min(0, g_c) ub_c ] ,  g = e_j - A^T y
      (le rows have nonnegative coefficients, so 0 <= a_i x; 0 <= x_c <= ub_c).  y = the HiGHS row duals (both signs
      tried), rounded to a dyadic grid y = Y / 2^K, the bound evaluated in exact integers; lambda = ceil.
  (b) refutation: an exhaustive LSearch for an integer point with x_j <= lambda - 1 that ends without one proves
      x_j >= lambda (a search stopped by its budget proves nothing; a point found is checked exactly and is a witness).
  Only candidates of a verified integer point x0 (HiGHS MILP, then checked exactly) can have lambda > 0.
LSearch: certify.WorldSearch (branch on the positive equality row with the fewest alive columns, values
min(caps), ..., 0; a row reaching 0 kills its columns) with per-column caps ub_c, the <= rows as monotone loads (a
value is capped by the remaining room of its <= rows; a saturated <= row kills its columns; loads only grow along a
branch, so no point is lost) and the >= rows checked at the leaves.  Every candidate touches B, so every column lies
in an equality row and the leaves are fully determined points; every integer point is reached exactly once.

Forced copies: lambda_c copies of every candidate c (simultaneously valid: each is a lower bound in every world, and
distinct candidates are distinct node sets), subtracted with WDForcer.subtract (consistency checks), then the old
rules rerun.  Ground truth is used only to check soundness afterwards.

usage: python -I -B -X utf8 experiments/local_cert.py m4 <ds> <preserved|reduced> <exact|upper>
       python -I -B -X utf8 experiments/local_cert.py m4all [--workers 6]        (the 24 benchmark inputs)
       python -I -B -X utf8 experiments/local_cert.py sample <sid> [--levels node|all] [--wall s] [--tag t]
       python -I -B -X utf8 experiments/local_cert.py drive [<sid> ...] [--workers 6] [--levels node|all] [--tag t]
              (default: the 13 samples of dense_samples.LOCAL_CERT_SAMPLES; the experiments's runs use
               --levels node --wall 900 --tag _node and --levels all --wall 1800 --tag _lv)
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402
from hyrange import engine as EN  # noqa: E402
from hyrange import forcing as WC  # noqa: E402
from hyrange import candidates as WP  # noqa: E402
from hyrange import pipeline as MO  # noqa: E402

OUTD = CFG.RESULTS / "local_cert"

CAP = 20_000          # candidate cliques per system (over it: system skipped, never truncated)
ENUM_S = 3.0          # listing time per system
SYS_S = 10.0          # wall budget per system (listing + MILP + bounds + searches)
MILP_S = 5.0          # MILP for the integer point x0
LP_S = 2.0            # one LP min x_j
DFS_S = 0.5           # one refutation search
FULL_COLS = 400       # (c) full enumeration attempted up to this many columns
FULL_S = 2.0          # (c) budget
FULL_POINTS = 200_000
LEVELS = [("edge", 0), ("e2", 2), ("e4", 4), ("e8", 8)]
LEVEL_S = 300.0       # wall budget per level stage (edge, e2, e4, e8) of one input


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


class Skip(Exception):
    pass


class Budget(Exception):
    pass


class Found(Exception):
    pass


# ------------------------------------------------------------------ degree mode of the forcer
# Exact mode (pipeline.WDForcerTL): l_u = U_u = d_u.  Upper-bound mode (forcing_upper WDForcer(singleton=True) via
# upper_exact.interval_forcer): the multi-member degree lies in [l_u, U_u], l_u = F.lb[u] (= max(lo_u, max_v W_uv)),
# U_u = F.ub[u] (= min(D_u, r_u)).  Every statement used below holds for any d_u in [l_u, U_u].
def is_interval(F):
    return getattr(F, "singleton", False)


def lower(F, u):
    return F.lb[u] if is_interval(F) else F.deg[u]


def upper(F, u):
    return F.ub[u] if is_interval(F) else F.deg[u]


class _UShim:
    """adj + upper() for forcing_upper.WDForcer.pair_core (interval mode)."""

    def __init__(self, F):
        self.adj = F.adj
        self.deg = F.deg
        self.F = F

    def upper(self, w):
        return self.F.ub[w]


# ------------------------------------------------------------------ candidate listing (cliques touching B)
def enum_touching(F, B, cap=CAP, deadline=None):
    """All candidate cliques (sorted global tuples) of the residual of F that contain a node of B, with the sound
    filters of candidates.enum_cliques_wp.  Raises Skip('cap' | 'time')."""
    adj, deg, r = F.adj, F.deg, F.r
    iv = is_interval(F)
    if iv:
        from hyrange import upper_exact as RU
        shim = _UShim(F)
        pc = RU.W2().WDForcer.pair_core
    else:
        shim = WP._CoreShim(adj, deg)
        pc = WC.WDForcer.pair_core
    kmemo = {}
    nwin = {}

    def win(x):
        # size window of candidates (exact) / upper_exact (interval): max(2, r - (U - 1)(s - 1) + 1) <= |e| <= r - l + 2
        w = nwin.get(x)
        if w is None:
            lx, ux, rx = lower(F, x), upper(F, x), r[x]
            w = nwin[x] = (max(2, rx - (ux - 1) * len(adj[x]) + 1), rx - lx + 2)
        return w

    def kinfo(x, y):
        key = (x, y) if x < y else (y, x)
        v = kmemo.get(key)
        if v is None:
            K, X, R = pc(shim, key[0], key[1])
            kk = len(K)
            slack = min(r[z] - R * (kk - 1) - max(0, lower(F, z) - R) for z in K)
            if slack < 0:
                raise WC.Inconsistent(("negative slack", key))
            ub = 0
            for w in X:
                aw = adj[w]
                ub += min([R] + [aw[z] for z in K])
                if ub >= slack:
                    break
            v = kmemo[key] = (K, kk + min(slack, ub))
        return v

    out = []
    excl = set()
    visits = [0]
    for b in B:
        if not adj[b]:
            excl.add(b)
            continue
        L = [b] + sorted(v for v in adj[b] if v not in excl)
        m = len(L)
        lid = {v: i for i, v in enumerate(L)}
        badj = [0] * m
        for i, v in enumerate(L):
            av = adj[v]
            mk = 0
            for w in av:
                j = lid.get(w)
                if j is not None:
                    mk |= 1 << j
            badj[i] = mk
        clo = [badj[i] | (1 << i) for i in range(m)]
        lo_n = [win(v)[0] for v in L]
        hi_n = [win(v)[1] for v in L]
        hi_all = max(x.bit_count() for x in clo)
        memo = {}

        def pinfo(i, j):
            key = (i, j) if i < j else (j, i)
            res = memo.get(key)
            if res is None:
                a = max(lo_n[i], lo_n[j])
                bb = min((clo[i] & clo[j]).bit_count(), hi_n[i], hi_n[j])
                K, hk = kinfo(L[i], L[j])
                km = 0
                for z in K:
                    t = lid.get(z)
                    if t is None:          # core node outside the allowed set: no listed clique has this pair
                        bb = -1
                        break
                    km |= 1 << t
                bb = min(bb, hk)
                res = memo[key] = (a, bb, km)
            return res

        def rec(cur, curmask, cand, Lneed, H, Kreq):
            k = len(cur)
            visits[0] += 1
            if deadline is not None and (visits[0] & 1023) == 0 and time.perf_counter() > deadline:
                raise Skip("time")
            if k >= 2 and Lneed <= k <= H and (Kreq & ~curmask) == 0:
                out.append(tuple(sorted(L[i] for i in cur)))
                if len(out) > cap:
                    raise Skip("cap")
            if k >= H:
                return
            c = cand
            while c:
                bt = c & -c
                v = bt.bit_length() - 1
                c ^= bt
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
                nm = curmask | bt
                if (K2 & ~nm) & ~nc:
                    continue
                if k + 1 + nc.bit_count() < max(2, L2):
                    continue
                cur.append(v)
                rec(cur, nm, nc, L2, H2, K2)
                cur.pop()

        rec([0], 1, badj[0], 0, hi_all, 0)
        excl.add(b)
    return out


# ------------------------------------------------------------------ the level-B system
def _hsolve_rows(c, A, rl, ru, ub, integer, tl):
    from scipy.optimize._highspy import _core as hc
    A = sp.csc_matrix(A, dtype=np.float64)
    nr, nc = A.shape
    lp = hc.HighsLp()
    lp.num_col_ = nc
    lp.num_row_ = nr
    lp.col_cost_ = np.asarray(c, dtype=np.float64)
    lp.col_lower_ = np.zeros(nc)
    lp.col_upper_ = np.asarray(ub, dtype=np.float64)
    lp.row_lower_ = np.asarray(rl, dtype=np.float64)
    lp.row_upper_ = np.asarray(ru, dtype=np.float64)
    mm = lp.a_matrix_
    mm.format_ = hc.MatrixFormat.kColwise
    mm.start_ = A.indptr.astype(np.int32)
    mm.index_ = A.indices.astype(np.int32)
    mm.value_ = A.data
    mm.num_col_ = nc
    mm.num_row_ = nr
    lp.a_matrix_ = mm
    lp.sense_ = hc.ObjSense.kMinimize
    if integer:
        lp.integrality_ = [hc.HighsVarType.kInteger] * nc
    h = hc._Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(tl))
    h.setOptionValue("primal_feasibility_tolerance", 1e-9)
    h.setOptionValue("dual_feasibility_tolerance", 1e-9)
    if integer:
        h.setOptionValue("mip_feasibility_tolerance", 1e-7)
    h.passModel(lp)
    return h


class LevelSystem:
    """Rows: eq (E1, E2) | le (I1, I2a, I2b) | ge (triple cuts).  Integer data in Python ints."""

    def __init__(self, F, B, cands):
        adj, deg, r = F.adj, F.deg, F.r
        self.B = list(B)
        Bs = set(B)
        self.cands = cands
        nreal = self.nreal = len(cands)
        lo_ = {}
        up_ = {}

        def L_(x):
            v = lo_.get(x)
            if v is None:
                v = lo_[x] = lower(F, x)
            return v

        def U_(x):
            v = up_.get(x)
            if v is None:
                v = up_[x] = upper(F, x)
            return v
        self.ub = [min(min(U_(x) for x in c), min(adj[x][y] for x, y in combinations(c, 2))) for c in cands]
        eq, eqb = {}, []
        slack_rows = []                 # (eq row, cap U - l) of interval degree rows
        for b in self.B:
            if not adj[b]:
                continue
            eq[("d", b)] = len(eqb)
            eqb.append(U_(b))
            if U_(b) > L_(b):
                slack_rows.append((len(eqb) - 1, U_(b) - L_(b)))
            for v, w in adj[b].items():
                key = ("p", min(b, v), max(b, v))
                if key not in eq:
                    eq[key] = len(eqb)
                    eqb.append(w)
        nc = self.nc = nreal + len(slack_rows)
        le, leh = {}, []
        crow = [[] for _ in range(nc)]
        cle = [[] for _ in range(nc)]
        colmask = {}
        for t, (i, capv) in enumerate(slack_rows):
            crow[nreal + t].append(i)
            self.ub.append(capv)

        def le_row(key, h):
            i = le.get(key)
            if i is None:
                i = le[key] = len(leh)
                leh.append(h)
            return i
        for j, c in enumerate(cands):
            k = len(c)
            bit = 1 << j
            # (slack columns nreal.. have only their degree row)
            for x in c:
                colmask[x] = colmask.get(x, 0) | bit
                if x in Bs:
                    crow[j].append(eq[("d", x)])
                else:
                    cle[j].append((le_row(("n", x), U_(x)), 1))
                    if k > 2:
                        cle[j].append((le_row(("s", x), r[x] - L_(x)), k - 2))
            for x, y in combinations(c, 2):
                if x in Bs or y in Bs:
                    crow[j].append(eq[("p", x, y)])
                else:
                    cle[j].append((le_row(("p", x, y), adj[x][y]), 1))
        # triple cuts
        cuts = {}
        for b in self.B:
            ab = adj[b]
            if not ab:
                continue
            db = U_(b)
            N = list(ab)
            for i in range(len(N)):
                v = N[i]
                av = adj[v]
                wbv = ab[v]
                for jj in range(i + 1, len(N)):
                    w = N[jj]
                    wvw = av.get(w, 0)
                    if wvw == 0:
                        if wbv + ab[w] - db > 0:
                            raise WC.Inconsistent(("cut on a non-support pair", b, v, w))
                        continue
                    wbw = ab[w]
                    lo = max(wbv + wbw - db, wbv + wvw - U_(v), wbw + wvw - U_(w))
                    if lo > 0:
                        key = tuple(sorted((b, v, w)))
                        if key not in cuts:
                            cuts[key] = (lo, min(wbv, wbw, wvw))
        self.ge_keys = list(cuts)
        gelo = [cuts[t][0] for t in self.ge_keys]
        gehi = [cuts[t][1] for t in self.ge_keys]
        cge = [[] for _ in range(nc)]
        for t_i, t in enumerate(self.ge_keys):
            m = colmask.get(t[0], 0) & colmask.get(t[1], 0) & colmask.get(t[2], 0)
            if m == 0:
                raise AssertionError(("cut with lo > 0 and no candidate: filters unsound?", t, cuts[t]))
            for j in _bits(m):
                cge[j].append(t_i)
        self.eq_keys = [k for k, _ in sorted(eq.items(), key=lambda t: t[1])]
        self.le_keys = [k for k, _ in sorted(le.items(), key=lambda t: t[1])]
        self.eqb, self.leh, self.gelo, self.gehi = eqb, leh, gelo, gehi
        self.crow, self.cle, self.cge = crow, cle, cge
        self.neq, self.nle, self.nge = len(eqb), len(leh), len(gelo)
        cnt = [0] * self.neq
        for j in range(nc):
            for i in crow[j]:
                cnt[i] += 1
        for i in range(self.neq):
            if cnt[i] == 0:
                raise AssertionError(("equality row without a candidate: filters unsound?", self.eq_keys[i]))
        # sparse matrix (rows: eq, le, ge), integer coefficients
        ri, ci, vv = [], [], []
        for j in range(nc):
            for i in crow[j]:
                ri.append(i)
                ci.append(j)
                vv.append(1)
            for i, a in cle[j]:
                ri.append(self.neq + i)
                ci.append(j)
                vv.append(a)
            for t in cge[j]:
                ri.append(self.neq + self.nle + t)
                ci.append(j)
                vv.append(1)
        self.nrows = self.neq + self.nle + self.nge
        self.A = sp.csc_matrix((np.array(vv, dtype=np.int64), (np.array(ri, dtype=np.int64),
                                                               np.array(ci, dtype=np.int64))), shape=(self.nrows, nc))
        self.At = self.A.T.tocsr()
        self.colw = int(np.abs(self.A).sum(axis=0).max()) if nc else 1
        self.rl = np.array([float(b) for b in eqb] + [0.0] * self.nle + [float(x) for x in gelo])
        self.ru = np.array([float(b) for b in eqb] + [float(h) for h in leh] + [float(x) for x in gehi])
        self._lp = None
        self._lpcur = None

    # ---- exact check of an integer point
    def is_point(self, x):
        x = [int(v) for v in x]
        if any(v < 0 or v > u for v, u in zip(x, self.ub)):
            return False
        R = [0] * self.neq
        Ld = [0] * self.nle
        G = [0] * self.nge
        for j, v in enumerate(x):
            if v:
                for i in self.crow[j]:
                    R[i] += v
                for i, a in self.cle[j]:
                    Ld[i] += a * v
                for t in self.cge[j]:
                    G[t] += v
        return (R == list(self.eqb) and all(a <= h for a, h in zip(Ld, self.leh))
                and all(lo <= g <= hi for g, lo, hi in zip(G, self.gelo, self.gehi)))

    # ---- integer point by MILP (checked exactly)
    def milp_point(self, tl=MILP_S):
        h = _hsolve_rows(np.zeros(self.nc), self.A, self.rl, self.ru, self.ub, True, tl)
        h.run()
        sol = h.getSolution()
        st = h.modelStatusToString(h.getModelStatus())
        if not sol.value_valid:
            return None, st
        x = np.rint(np.array(sol.col_value)).astype(np.int64)
        if not self.is_point(x):
            return None, st + "|unverified"
        return x, st

    # ---- LP min x_j with exact dual bound
    def lp_bound(self, j, tl=LP_S):
        """(lambda (int, proved), lp value float or None, status)."""
        if self._lp is None:
            self._lp = _hsolve_rows(np.zeros(self.nc), self.A, self.rl, self.ru, self.ub, False, tl)
            self._lp.setOptionValue("presolve", "off")
        h = self._lp
        if self._lpcur is not None:
            h.changeColCost(int(self._lpcur), 0.0)
        h.changeColCost(int(j), 1.0)
        self._lpcur = j
        h.setOptionValue("time_limit", float(h.getRunTime() + tl))
        h.run()
        st = h.modelStatusToString(h.getModelStatus())
        sol = h.getSolution()
        if st != "Optimal" or not sol.dual_valid:
            return 0, None, st
        y = np.array(sol.row_dual)
        val = float(np.array(sol.col_value)[j])
        lam = max(self.rig(j, y), self.rig(j, -y))
        return max(0, lam), val, st

    def rig(self, j, y):
        """ceil of the exact Lagrangian lower bound on x_j from y (rounded to Y / 2^K)."""
        ymax = float(np.abs(y).max()) if len(y) else 0.0
        K = 40
        while K > 0 and (ymax + 1.0) * 2.0 ** K * (self.colw + 1) >= 2.0 ** 60:
            K -= 1
        scale = 1 << K
        Y = np.rint(y * scale).astype(np.int64)
        S = self.At @ Y                                     # exact int64: (A^T Y)_c
        Yl = [int(v) for v in Y]
        tot = 0
        for i in range(self.neq):
            tot += Yl[i] * self.eqb[i]
        o = self.neq
        for i in range(self.nle):
            if Yl[o + i] < 0:
                tot += Yl[o + i] * self.leh[i]
        o += self.nle
        for t in range(self.nge):
            yy = Yl[o + t]
            tot += yy * (self.gelo[t] if yy > 0 else self.gehi[t])
        G = -S
        G[j] += scale
        Gl = G.tolist()
        for c in np.nonzero(G < 0)[0].tolist():
            tot += Gl[c] * self.ub[c]
        return -((-tot) // scale)


def _bits(m):
    """Indices of the set bits of a Python int."""
    out = []
    base = 0
    while m:
        chunk = m & ((1 << 4096) - 1)
        while chunk:
            b = chunk & -chunk
            out.append(base + b.bit_length() - 1)
            chunk ^= b
        m >>= 4096
        base += 4096
    return out


# ------------------------------------------------------------------ exact backtracking over the integer points
class LSearch:
    def __init__(self, S):
        self.S = S
        nc = self.nc = S.nc
        self.crow = S.crow
        self.cle = S.cle
        self.cge = S.cge
        self.rcand = [[] for _ in range(S.neq)]
        self.lecols = [[] for _ in range(S.nle)]
        for j in range(nc):
            for i in S.crow[j]:
                self.rcand[i].append(j)
            for i, a in S.cle[j]:
                self.lecols[i].append(j)
        self.nrows_c = [len(S.crow[j]) for j in range(nc)]

    def search(self, on_point, target=None, tcap=None, time_s=1.0, max_nodes=None):
        """DFS over all integer points.  Returns ('complete' | 'found' | 'budget', nodes)."""
        S = self.S
        crow, cle, cge, rcand, lecols = self.crow, self.cle, self.cge, self.rcand, self.lecols
        H, LO, cap = S.leh, S.gelo, S.ub
        nc = self.nc
        R = list(S.eqb)
        alive = [1] * nc
        cnt = [len(rc) for rc in rcand]
        x = [0] * nc
        load = [0] * S.nle
        gl = [0] * S.nge
        deficit = [sum(1 for t in range(S.nge) if LO[t] > 0)]
        pos = set(i for i in range(S.neq) if R[i] > 0)
        ulog = []
        nodes = [0]
        deadline = time.perf_counter() + time_s
        nrc = self.nrows_c

        def kill(c):
            alive[c] = 0
            ulog.append((0, c, 0))
            ok = True
            for i in crow[c]:
                cnt[i] -= 1
                if cnt[i] == 0 and R[i] > 0:
                    ok = False
            return ok

        def assign(c, v):
            ulog.append((1, c, v))
            x[c] = v
            ok = True
            zero, sat = [], []
            if v:
                for i in crow[c]:
                    R[i] -= v
                    if R[i] == 0:
                        pos.discard(i)
                        zero.append(i)
                    elif R[i] < 0:
                        ok = False
                for i, a in cle[c]:
                    load[i] += a * v
                    if load[i] > H[i]:
                        ok = False
                    elif load[i] == H[i]:
                        sat.append(i)
                for t in cge[c]:
                    b0 = gl[t]
                    gl[t] = b0 + v
                    if b0 < LO[t] <= b0 + v:
                        deficit[0] -= 1
            if not kill(c):
                ok = False
            for i in zero:
                for c2 in rcand[i]:
                    if alive[c2] and not kill(c2):
                        ok = False
            for i in sat:
                for c2 in lecols[i]:
                    if alive[c2] and not kill(c2):
                        ok = False
            return ok

        def undo(mark):
            while len(ulog) > mark:
                t, c, v = ulog.pop()
                if t == 0:
                    alive[c] = 1
                    for i in crow[c]:
                        cnt[i] += 1
                else:
                    x[c] = 0
                    if v:
                        for i in crow[c]:
                            if R[i] == 0:
                                pos.add(i)
                            R[i] += v
                        for i, a in cle[c]:
                            load[i] -= a * v
                        for tt in cge[c]:
                            b0 = gl[tt]
                            gl[tt] = b0 - v
                            if b0 - v < LO[tt] <= b0:
                                deficit[0] += 1

        def colub(c):
            ub = min(min(R[i] for i in crow[c]), cap[c])
            for i, a in cle[c]:
                q = (H[i] - load[i]) // a
                if q < ub:
                    ub = q
            return ub

        def rec():
            nodes[0] += 1
            if (nodes[0] & 255) == 0 and time.perf_counter() > deadline:
                raise Budget()
            if max_nodes is not None and nodes[0] > max_nodes:
                raise Budget()
            if not pos:
                if deficit[0] == 0:
                    on_point(x)
                return
            r = min(pos, key=cnt.__getitem__)
            m = cnt[r]
            if m == 0:
                return
            Rr = R[r]
            c = -1
            best = -1
            for c2 in rcand[r]:
                if alive[c2] and nrc[c2] > best:
                    best = nrc[c2]
                    c = c2
            ub = colub(c)
            if c == target:
                ub = min(ub, tcap)
            if m == 1:
                if ub < Rr:
                    return
                vals = (Rr,)
            else:
                vals = range(min(ub, Rr), -1, -1)
            for v in vals:
                mark = len(ulog)
                if assign(c, v):
                    rec()
                undo(mark)

        if any(cnt[i] == 0 for i in pos):
            return "complete", 0
        out = ["complete"]

        def root():
            if target is None:
                rec()
                return
            ubt = min(colub(target), tcap)
            for v in range(ubt, -1, -1):
                mark = len(ulog)
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


# ------------------------------------------------------------------ bounds of one system
def solve_system(S, methods="abc", deadline=None, x0=None, budgets=None, track=None):
    """Proved lower bounds lam[j] for the candidates of a verified integer point x0.  Returns a record and lam
    (dict j -> int) and ub (dict j -> witnessed min), plus the open candidates (lam < ub)."""
    Bg = dict(milp_s=MILP_S, lp_s=LP_S, dfs_s=DFS_S, full_cols=FULL_COLS, full_s=FULL_S, full_points=FULL_POINTS)
    if budgets:
        Bg.update(budgets)
    rec = dict(ncols=S.nc, neq=S.neq, nle=S.nle, nge=S.nge, how=Counter())
    if deadline is None:
        deadline = time.perf_counter() + 1e9
    t0 = time.perf_counter()
    if x0 is None:
        x0, st = S.milp_point(min(Bg["milp_s"], max(0.5, deadline - time.perf_counter())))
        rec["milp"] = st
        rec["t_milp"] = time.perf_counter() - t0
        if x0 is None:
            rec["status"] = "no_point"
            if st == "Infeasible":
                raise AssertionError(("level system infeasible: the truth is a point", S.B, S.nc))
            return rec, {}, {}
    else:
        assert S.is_point(x0)
    I = [int(j) for j in np.nonzero(x0)[0] if j < S.nreal]      # slack columns are never certified
    if track is not None:
        I = [j for j in I if j in track] + [j for j in I if j not in track]
    ub = {j: int(x0[j]) for j in I}
    lam = {j: 0 for j in I}
    rec["_x0"] = dict(ub)

    def take(xw):
        for j in I:
            if xw[j] < ub[j]:
                ub[j] = int(xw[j])
    # (c) full enumeration
    if "c" in methods and S.nc <= Bg["full_cols"]:
        LS = LSearch(S)
        mn = dict(ub)
        npts = [0]

        def on_point(x):
            npts[0] += 1
            for j in I:
                if x[j] < mn[j]:
                    mn[j] = x[j]
            if npts[0] >= Bg["full_points"]:
                raise Budget()
            if all(mn[j] == 0 for j in I):
                raise Found()
        st, nodes = LS.search(on_point, time_s=min(Bg["full_s"], max(0.05, deadline - time.perf_counter())))
        rec["full"] = dict(status=st, points=npts[0], nodes=nodes)
        for j in I:
            ub[j] = min(ub[j], int(mn[j]))
        if st == "complete":
            for j in I:
                lam[j] = ub[j]
                if lam[j] > 0:
                    rec["how"]["enum"] += lam[j]
            rec["status"] = "ok"
            rec["t"] = time.perf_counter() - t0
            return rec, lam, ub
    # (a) LP dual bounds
    if "a" in methods:
        nlp = 0
        for j in I:
            if lam[j] >= ub[j]:
                continue
            if time.perf_counter() > deadline:
                rec["budget_hit"] = "lp"
                break
            v, val, st = S.lp_bound(j, min(Bg["lp_s"], max(0.05, deadline - time.perf_counter())))
            nlp += 1
            if v > ub[j]:
                raise AssertionError(("proved LP bound above a verified point", S.B, j, v, ub[j]))
            if v > lam[j]:
                rec["how"]["lp"] += v - lam[j]
                lam[j] = v
        rec["lp_calls"] = nlp
    # (b) refutation search
    if "b" in methods:
        LS = LSearch(S)
        nd = 0
        for j in I:
            while lam[j] < ub[j]:
                if time.perf_counter() > deadline:
                    rec["budget_hit"] = "dfs"
                    break
                capv = ub[j] - 1
                got = [None]

                def on_point(x, got=got):
                    got[0] = list(x)
                    raise Found()
                st, nodes = LS.search(on_point, target=j, tcap=capv,
                                      time_s=min(Bg["dfs_s"], max(0.05, deadline - time.perf_counter())))
                nd += 1
                if st == "complete":
                    rec["how"]["dfs"] += ub[j] - lam[j]
                    lam[j] = ub[j]
                elif st == "found":
                    xw = got[0]
                    assert S.is_point(xw) and xw[j] <= capv, "search point fails the exact check"
                    take(xw)
                else:
                    break
        rec["dfs_calls"] = nd
    rec["status"] = "ok"
    rec["t"] = time.perf_counter() - t0
    return rec, lam, ub


# ------------------------------------------------------------------ driver: forcer + L+ + levels
class Runner:
    def __init__(self, adj, deg, rules=MO.RULES, methods="abc", cap=CAP, enum_s=ENUM_S, sys_s=SYS_S,
                 force_tl=900.0, logf=None, budgets=None, upper_mode=False, lo=None):
        if upper_mode:
            # upper-bound mode: deg = the declared upper bounds D, lo = declared lower bounds (0 in the experiments)
            from hyrange import upper_exact as RU
            self.F = RU.interval_forcer(adj, deg, lo if lo is not None else [0] * len(deg), rules=rules,
                                        time_limit=force_tl)
        else:
            self.F = MO.WDForcerTL(adj, deg, rules=rules, time_limit=force_tl, copy_adj=True)
        self.rules = rules
        self.methods = methods
        self.cap, self.enum_s, self.sys_s = cap, enum_s, sys_s
        self.budgets = budgets
        self.logf = logf
        # forcing events: every subtraction (old rules or L+) marks its members and their neighbours (support before
        # the subtraction).  The level-{u} system reads only residual data of nodes within distance 2 of u, so it
        # is unchanged unless a node of N[u] was marked after it was last built.
        self.event = 1
        self.mark = [1] * self.F.n
        self.event_old = 0
        orig = self.F.subtract

        def sub(e, m, rule, touched, _orig=orig):
            if m > 0:
                self.event += 1
                ev = self.event
                mk = self.mark
                adj = self.F.adj
                for z in e:
                    mk[z] = ev
                    for x in adj[z]:
                        mk[x] = ev
            return _orig(e, m, rule, touched)
        self.F.subtract = sub
        self.solved = {}            # u -> event at which its level-{u} system was built
        self.enum_cache = {}        # u -> (event, count | 'cap' | 'time')
        self.node_res = {}          # u -> open candidates of its latest system
        self.sys_log = []           # per system records
        self.first_level = {}       # tuple -> Counter(tag -> copies) of L+ forced copies
        self.n_skip = Counter()
        self.n_run = Counter()
        self.node_ran = set()       # nodes whose level-{u} system was built and solved at least once
        self.node_skip = {}         # node -> reason of its last skip ('cap' | 'time')
        self.res_nodes_after_old = None

    def lg(self, msg):
        if self.logf:
            self.logf(msg)

    def stale(self, u, since):
        mk = self.mark
        if mk[u] > since:
            return True
        return any(mk[x] > since for x in self.F.adj[u])

    def run_old(self):
        if self.rules and self.event != self.event_old:
            self.F.run()
        self.event_old = self.event

    def forced_vol(self):
        return sum(len(e) * m for e, m in self.F.forced.items())

    def force(self, e, m, tag):
        touched = set()
        self.F.subtract(tuple(e), int(m), tag, touched)
        self.first_level.setdefault(tuple(e), Counter())[tag] += int(m)

    def system(self, B, tag, deadline_total, track=None):
        """Build, solve and force the level-B system.  Returns (forced copies, open set, record)."""
        t0 = time.perf_counter()
        dl = min(t0 + self.sys_s, deadline_total)
        rec = dict(tag=tag, B=len(B))
        try:
            cands = enum_touching(self.F, B, cap=self.cap, deadline=min(t0 + self.enum_s, dl))
        except Skip as ex:
            rec.update(status="skip_" + str(ex), t=time.perf_counter() - t0)
            self.n_skip[tag + ":" + str(ex)] += 1
            self.sys_log.append(rec)
            return 0, None, rec
        rec["ncols"] = len(cands)
        if not cands:
            raise AssertionError(("no candidate through a residual node", B))
        S = LevelSystem(self.F, B, cands)
        idx = {c: j for j, c in enumerate(cands)}
        tr = None if track is None else set(idx[c] for c in track if c in idx)
        r, lam, ub = solve_system(S, self.methods, deadline=dl, budgets=self.budgets, track=tr)
        rec.update(status=r["status"], t=time.perf_counter() - t0, how=dict(r["how"]), nsupp=len(lam),
                   full=r.get("full", {}).get("status"), budget_hit=r.get("budget_hit"))
        self.n_run[tag] += 1
        nf = 0
        for j, v in lam.items():
            if v > 0:
                self.force(cands[j], v, tag)
                nf += v
        rec["forced"] = nf
        # open = not proved up to the value of the level point x0 (a level point is not a world, so a witness with
        # fewer copies does not close a candidate for larger B)
        x0d = r.get("_x0", {})
        opn = set(cands[j] for j in lam if lam[j] < x0d[j])
        self.sys_log.append(rec)
        return nf, opn, rec

    def _enum_count(self, u, deadline):
        c = self.enum_cache.get(u)
        if c is not None and not self.stale(u, c[0]):
            return c[1]
        ev = self.event
        t0 = time.perf_counter()
        try:
            n = len(enum_touching(self.F, [u], cap=self.cap, deadline=min(t0 + self.enum_s, deadline)))
        except Skip as ex:
            n = str(ex)
        self.enum_cache[u] = (ev, n)
        return n

    def node_level(self, deadline, tag="node"):
        """Repeat (old rules to fixpoint; L+ at every residual node whose system changed) until nothing new."""
        F = self.F
        rounds = 0
        while time.perf_counter() < deadline:
            rounds += 1
            ev0 = self.event
            self.run_old()
            nodes = [u for u in range(F.n) if F.adj[u] and self.stale(u, self.solved.get(u, -1))]
            if not nodes:
                break
            cnts = {}
            for u in nodes:
                if time.perf_counter() > deadline:
                    break
                cnts[u] = self._enum_count(u, deadline)
            order = sorted((u for u in cnts if isinstance(cnts[u], int)), key=lambda u: (cnts[u], u))
            for u in cnts:
                if not isinstance(cnts[u], int):
                    self.n_skip[tag + ":" + cnts[u]] += 1
                    self.node_skip[u] = cnts[u]
                    self.solved[u] = self.enum_cache[u][0]     # retried only after a change near u
            for u in order:
                if time.perf_counter() > deadline:
                    break
                if not F.adj[u]:
                    continue
                ev = self.event
                nf, opn, rec = self.system([u], tag, deadline)
                rec["u"] = u
                if rec["status"].startswith("skip_"):
                    self.node_skip[u] = rec["status"][5:]
                else:
                    self.node_ran.add(u)
                if opn is not None:
                    self.node_res[u] = opn
                self.solved[u] = ev
            self.lg(f"  {tag} round {rounds}: forced vol {self.forced_vol()} events {ev0}->{self.event}, "
                    f"{len(order)} systems, {len(cnts) - len(order)} skipped")
            if self.event == ev0:
                break
        self.run_old()
        return rounds

    def open_targets(self):
        F = self.F
        out = set()
        for u, opn in self.node_res.items():
            for c in opn:
                if all(y in F.adj[x] for x, y in combinations(c, 2)):
                    out.add(c)
        return sorted(out, key=lambda c: (len(c), c))

    def grow(self, e, extra):
        F = self.F
        B = list(e)
        Bs = set(B)
        while len(B) < len(e) + extra:
            sc = Counter()
            for b in B:
                for v, w in F.adj[b].items():
                    if v not in Bs:
                        sc[v] += w
            if not sc:
                break
            v = min(sc, key=lambda z: (-sc[z], z))
            B.append(v)
            Bs.add(v)
        return B

    def level_pass(self, tag, extra, deadline):
        tg = self.open_targets()
        seenB = set()
        tried = set()
        nsys = 0
        for e in tg:
            if time.perf_counter() > deadline:
                break
            if e in tried:
                continue
            F = self.F
            if not all(y in F.adj[x] for x, y in combinations(e, 2)):
                continue
            B = self.grow(e, extra)
            key = frozenset(B)
            if key in seenB:
                continue
            seenB.add(key)
            nf, opn, rec = self.system(B, tag, deadline, track=[e])
            nsys += 1
            tried.add(e)
        return dict(targets=len(tg), systems=nsys)


def run_pipeline(adj, deg, levels="all", wall=600.0, rules=MO.RULES, methods="abc", logf=None, budgets=None,
                 level_s=LEVEL_S, sys_s=SYS_S, cap=CAP, upper_mode=False, lo=None, level_list=None):
    """old rules -> node level -> edge, e2, e4, e8 (each followed by the node-level cascade).  Snapshots."""
    t_start = time.perf_counter()
    deadline = t_start + wall
    Rn = Runner(adj, deg, rules=rules, methods=methods, logf=logf, budgets=budgets, sys_s=sys_s, cap=cap,
                upper_mode=upper_mode, lo=lo)
    snaps = []
    vol = sum(deg)

    def snap(stage, t0, extra=None):
        d = dict(stage=stage, forced_vol=Rn.forced_vol(), forced_share=Rn.forced_vol() / vol,
                 forced_copies=int(sum(Rn.F.forced.values())), t_s=time.perf_counter() - t0,
                 t_cum_s=time.perf_counter() - t_start, counts=dict(Rn.F.counts),
                 skips=dict(Rn.n_skip), runs=dict(Rn.n_run),
                 hit_wall=time.perf_counter() > deadline)
        if extra:
            d.update(extra)
        snaps.append(d)
        if logf:
            logf(f" stage {stage}: share {d['forced_share']:.4f} in {d['t_s']:.1f}s runs {dict(Rn.n_run)} "
                 f"skips {dict(Rn.n_skip)}")
    t0 = time.perf_counter()
    Rn.run_old()
    Rn.res_nodes_after_old = sum(1 for u in range(Rn.F.n) if Rn.F.adj[u])
    snap("old", t0, dict(force_complete=bool(Rn.F.complete), residual_nodes=Rn.res_nodes_after_old))
    if levels in ("node", "all", "edge"):
        t0 = time.perf_counter()
        rounds = Rn.node_level(deadline)
        snap("node", t0, dict(rounds=rounds))
    if levels in ("all", "edge"):
        for tag, extra in (LEVELS if levels == "all" else LEVELS[:1]):
            t0 = time.perf_counter()
            dl = min(deadline, t0 + level_s)
            info = Rn.level_pass(tag, extra, dl)
            Rn.node_level(min(deadline, time.perf_counter() + level_s), tag="node")
            snap(tag, t0, info)
    return Rn, snaps


# ------------------------------------------------------------------ helpers for runs
def residual_components(F):
    comps = EN._components(F.adj, F.n)
    sizes = sorted((len(c) for c in comps), reverse=True)
    return dict(count=len(sizes), largest=sizes[:5], nodes=int(sum(sizes)),
                vol=int(sum(F.deg[int(u)] for c in comps for u in c)))


def soundness(F, lines_loc, n):
    ver = WC.verify_forced(F.forced, lines_loc)
    rok = WP.check_residual_wp(F, lines_loc, n)
    return dict(verify_forced_ok=bool(ver["ok"]), invalid_copies=int(ver["invalid"]), residual_ok=bool(rok))


def soundness_iv(F, lines_loc, n, lo):
    """Upper-bound mode: forced <= truth, W(truth - forced) == residual W, lo_res <= deg(truth - forced) <= D_res."""
    true = Counter(tuple(sorted(e)) for e in lines_loc)
    invalid = sum(max(0, m - true.get(e, 0)) for e, m in F.forced.items())
    rem = Counter(true)
    for e, m in F.forced.items():
        rem[e] -= m
    ok = invalid == 0 and all(m >= 0 for m in rem.values())
    rok = False
    if ok:
        adj2, deg2 = WC.build_wdeg([e for e, m in rem.items() for _ in range(m)], n)
        rok = all(adj2[u] == F.adj[u] for u in range(n)) and all(F.lo[u] <= deg2[u] <= F.deg[u] for u in range(n))
    return dict(verify_forced_ok=bool(invalid == 0), invalid_copies=int(invalid), residual_ok=bool(rok))


def first_level_hist(Rn):
    """Volume and copies of the L+ forced copies by the stage tag that forced them (each copy counted once, at the
    stage that proved it; cascades of the old rules are in the stage snapshots)."""
    h = Counter()
    hv = Counter()
    for e, c in Rn.first_level.items():
        for tag, m in c.items():
            h[tag] += m
            hv[tag] += m * len(e)
    return dict(copies=dict(h), vol=dict(hv))


def sys_summary(Rn):
    out = {}
    for tag in sorted(set(r["tag"] for r in Rn.sys_log)):
        R = [r for r in Rn.sys_log if r["tag"] == tag]
        st = Counter(r["status"] for r in R)
        ok = [r for r in R if r["status"] == "ok"]
        out[tag] = dict(systems=len(R), status=dict(st), forced_copies=int(sum(r.get("forced", 0) for r in R)),
                        t_s=float(sum(r.get("t", 0.0) for r in R)),
                        ncols_max=int(max((r.get("ncols", 0) for r in ok), default=0)),
                        ncols_median=float(np.median([r["ncols"] for r in ok])) if ok else None,
                        full_complete=sum(1 for r in ok if r.get("full") == "complete"),
                        how=dict(sum((Counter(r.get("how", {})) for r in ok), Counter())))
    return out


# ------------------------------------------------------------------ density samples (rebuilt as dense_samples.py)
def density_sample(sid):
    from hyrange import dense_samples as RD
    spec = RD.load_json(RD.PLAN)[sid]
    seq = RD.sequence(spec["src"], spec["seed"])
    S = seq[:spec["k"]]
    nodes = sorted(set(v for e in S for v in e))
    loc = {v: i for i, v in enumerate(nodes)}
    lines_loc = [tuple(sorted(loc[v] for v in e)) for e in S]
    ref = RD.load_json(RD.SAMP / f"{sid}.json")
    return spec, lines_loc, len(nodes), ref


def run_input(name, lines_loc, n, levels, wall, outp, ref=None, extra=None):
    t_script = time.perf_counter()
    adj, deg = WC.build_wdeg(lines_loc, n)
    vol = int(sum(deg))
    res = dict(name=name, n=n, copies=len(lines_loc), vol=vol, mean_degree=vol / n, levels=levels, wall_s=wall,
               limits=dict(cap=CAP, enum_s=ENUM_S, sys_s=SYS_S, milp_s=MILP_S, lp_s=LP_S, dfs_s=DFS_S,
                           full_cols=FULL_COLS, full_s=FULL_S, level_s=LEVEL_S, l_call_s=MO.WDForcerTL.L_CALL_S),
               stage="start")
    if extra:
        res.update(extra)
    MB.save_json(res, outp)
    Rn, snaps = run_pipeline(adj, deg, levels=levels, wall=wall, logf=log)
    res["snapshots"] = snaps
    res["old_forced_share"] = snaps[0]["forced_share"]
    res["final_forced_share"] = snaps[-1]["forced_share"]
    res["incomplete_wall"] = any(s["hit_wall"] for s in snaps)
    if ref is not None:
        res["ref_forced_vol_share"] = ref
        res["old_reproduces_ref"] = abs(snaps[0]["forced_share"] - ref) < 1e-12
    res["first_level"] = first_level_hist(Rn)
    res["systems"] = sys_summary(Rn)
    res["residual_components_after"] = residual_components(Rn.F)
    res["node_counts"] = dict(residual_nodes_after_old=Rn.res_nodes_after_old, lplus_ran=len(Rn.node_ran),
                              skipped_never_ran=len(set(Rn.node_skip) - Rn.node_ran),
                              skipped_reasons=dict(Counter(v for u, v in Rn.node_skip.items()
                                                           if u not in Rn.node_ran)),
                              residual_nodes_end=sum(1 for u in range(Rn.F.n) if Rn.F.adj[u]))
    snd = soundness(Rn.F, lines_loc, n)
    res["soundness"] = snd
    res["forced_multiset"] = [[list(e), int(m)] for e, m in sorted(Rn.F.forced.items()) if m > 0]
    res["t_total_s"] = time.perf_counter() - t_script
    res["stage"] = "done"
    MB.save_json(res, outp)
    if not (snd["verify_forced_ok"] and snd["residual_ok"]):
        raise AssertionError(("UNSOUND", name, snd))
    log(f"{name}: old {res['old_forced_share']:.4f} -> {res['final_forced_share']:.4f} in {res['t_total_s']:.1f}s "
        f"sound {snd}")
    return res, Rn


def cmd_sample(sid, levels, wall, tag):
    spec, lines_loc, n, ref = density_sample(sid)
    outp = OUTD / "samples" / f"{sid}{tag}.json"
    extra = dict(sid=sid, src=spec["src"], level=spec["level"], seed=spec["seed"],
                 ref_certified_share=ref.get("certified_share"),
                 ref_all_decided=all(c.get("all_decided", False) for c in ref.get("cert_components", [])
                                     if c["kind"] == "treated")
                 and all(c["kind"] == "treated" for c in ref.get("cert_components", [])),
                 ref_exact_complete=bool(ref.get("exact_complete", False)))
    run_input(sid, lines_loc, n, levels, wall, outp, ref=ref["forced_vol_share"], extra=extra)


# ------------------------------------------------------------------ exact certification of a residual
def exact_residual_cert(F):
    """dense_samples / certify procedure on the residual components of F: candidates (build_comp_wp, cap 300,000,
    60 s), found worlds of seeds 0, 1, 2 (pipeline.world_of, MILP 60 s), certify_component.  Returns (Counter of
    certified residual copies, all decided, records)."""
    from hyrange import certify as CX
    cert = Counter()
    recs = []
    alld = True
    comps = EN._components(F.adj, F.n)
    for cn in sorted(comps, key=len):
        C = WP.build_comp_wp(cn, F.adj, F.deg, cap=EN.CAP, enum_s=EN.ENUM_S)
        if C.cands is None:
            recs.append(dict(nodes=len(cn), status="not_enumerated:" + C.enum))
            alld = False
            continue
        if C.unique and C.xu is not None:
            for j in np.nonzero(C.xu)[0]:
                cert[tuple(sorted(int(C.nodes[i]) for i in C.cands[j]))] += int(C.xu[j])
            recs.append(dict(nodes=len(cn), status="pinned"))
            continue
        worlds = []
        for seed in MB.SEEDS:
            x, st = MO.world_of(C, np.random.default_rng(seed), 60.0)
            if x is not None:
                worlds.append(x)
        if not worlds:
            recs.append(dict(nodes=len(cn), status="no_world"))
            alld = False
            continue
        r = CX.certify_component(C, worlds)
        for c in r["cands"]:
            if c["lb"] > 0:
                cert[tuple(sorted(int(C.nodes[i]) for i in C.cands[c["j"]]))] += c["lb"]
        alld = alld and r["all_decided"]
        recs.append(dict(nodes=len(cn), ncands=len(C.cands), method=r["method"], all_decided=r["all_decided"],
                         t=r["t_total_s"]))
    return cert, alld, recs


# ------------------------------------------------------------------ the six benchmarks, both settings, both modes (command m4)
def _refs_m4(ds, setting, mode):
    out = {}
    a = json.load(open(CFG.RESULTS / "bench" / "ablation_summary.json", encoding="utf-8"))
    for r in a["records"]:
        if r["ds"] == ds and r["mode"] == setting:
            out["two_core_outside_share"] = r["two_core_outside_share"]
    if mode == "exact":
        out["old_ref"] = json.load(open(MB.OUT / "ours" / f"{ds}_{setting}.json", encoding="utf-8"))["forced_vol_share"]
        c = json.load(open(CFG.RESULTS / "certification" / "cert_exact.json", encoding="utf-8"))
        for r in c["benchmark"]:
            if r["ds"] == ds and r["mode"] == setting:
                out["paper_certified"] = r["certified_vol_share_after"]
                out["paper_certified_src"] = "results/certification/cert_exact.json certified_vol_share_after"
    else:
        rp = CFG.RESULTS / "upper_forcing" / "jobs" / f"{ds}_{setting}.json"
        out["old_ref"] = json.load(open(rp, encoding="utf-8"))["forced_vol_share"] if rp.exists() else None
        u = json.load(open(CFG.RESULTS / "upper_exact" / "summary.json", encoding="utf-8"))["bench"]
        r = u.get(f"{ds}_{setting}", {})
        out["paper_certified"] = r.get("cert_vol_share")
        out["paper_certified_src"] = "results/upper_exact/summary.json bench.cert_vol_share"
    return out


def core_share(x, out):
    if x is None or out is None or out >= 1 - 1e-12:
        return None
    return (x - out) / (1 - out)


def cmd_m4(ds, setting, mode, tag=""):
    from hyrange import upper_exact as RU
    upper = mode == "upper"
    lines, truth = MB.target(ds, setting)
    nodes = sorted(set(v for e in lines for v in e))
    loc = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    lines_loc = [tuple(sorted(loc[v] for v in e)) for e in lines]
    adj, d = WC.build_wdeg(lines_loc, n)
    vol = int(sum(d))
    outp = OUTD / "m4" / f"{ds}_{setting}_{mode}{tag}.json"
    res = dict(ds=ds, setting=setting, mode=mode, n=n, copies=len(lines_loc), vol=vol, stage="start",
               rules=list(RU.UPPER_RULES if upper else MO.RULES),
               limits=dict(cap=CAP, enum_s=ENUM_S, sys_s=SYS_S, milp_s=MILP_S, lp_s=LP_S, dfs_s=DFS_S))
    res.update(_refs_m4(ds, setting, mode))
    MB.save_json(res, outp)
    lo = [0] * n if upper else None
    rules = RU.UPPER_RULES if upper else MO.RULES
    # old rules alone (for the residual before L+)
    R0 = Runner(adj, d, rules=rules, upper_mode=upper, lo=lo)
    t0 = time.perf_counter()
    R0.run_old()
    t_old = time.perf_counter() - t0
    old_copies = int(sum(R0.F.forced.values()))
    res["residual_before"] = dict(copies=len(lines_loc) - old_copies,
                                  vol=vol - R0.forced_vol(), components=residual_components(R0.F))
    del R0
    t1 = time.perf_counter()
    Rn, snaps = run_pipeline(adj, d, levels="edge", wall=1800.0, rules=rules, upper_mode=upper, lo=lo, logf=log)
    res["snapshots"] = snaps
    res["t_old_s"] = t_old
    res["t_lplus_node_s"] = snaps[1]["t_s"]
    res["t_lplus_edge_s"] = snaps[2]["t_s"] if len(snaps) > 2 else None
    res["old_share"] = snaps[0]["forced_share"]
    res["node_share"] = snaps[1]["forced_share"]
    res["edge_share"] = snaps[2]["forced_share"] if len(snaps) > 2 else None
    res["old_reproduces_ref"] = (res.get("old_ref") is not None and abs(res["old_share"] - res["old_ref"]) < 1e-12)
    snd = soundness_iv(Rn.F, lines_loc, n, lo) if upper else soundness(Rn.F, lines_loc, n)
    res["soundness"] = snd
    if not (snd["verify_forced_ok"] and snd["residual_ok"]):
        MB.save_json(res, outp)
        raise AssertionError(("UNSOUND", ds, setting, mode, snd))
    fc = int(sum(Rn.F.forced.values()))
    res["residual_after"] = dict(copies=len(lines_loc) - fc, vol=vol - Rn.forced_vol(),
                                 components=residual_components(Rn.F))
    res["first_level"] = first_level_hist(Rn)
    res["systems"] = sys_summary(Rn)
    res["node_counts"] = dict(lplus_ran=len(Rn.node_ran), skipped_never_ran=len(set(Rn.node_skip) - Rn.node_ran))
    MB.save_json(res, outp)
    # exact certification of the residual after L+ (paper procedure), for consistency with the experiments's share
    t2 = time.perf_counter()
    forced = Counter({tuple(sorted(e)): int(m) for e, m in Rn.F.forced.items() if m > 0})
    if upper:
        out, f2, cert2, extra = RU.pipeline([dict(a) for a in Rn.F.adj], list(Rn.F.lo), list(Rn.F.deg), n, wall=600.0)
        allc = forced + cert2
        res["after_lplus_exact"] = dict(fcert_proved=bool(out["fcert_proved"]), comps=out["comps"],
                                        comp_status=out["comp_status"], copies_undecided=out["copies_undecided"])
    else:
        cert2, alld, recs = exact_residual_cert(Rn.F)
        allc = forced + cert2
        res["after_lplus_exact"] = dict(all_decided=bool(alld), comps=len(recs),
                                        status=dict(Counter(r.get("method", r.get("status")) for r in recs)))
    tl = Counter(lines_loc)
    false = int(sum(max(0, m - tl.get(e, 0)) for e, m in allc.items()))
    assert false == 0, ("UNSOUND exact certification after L+", ds, setting, mode, false)
    res["after_lplus_exact"]["certified_share"] = sum(len(e) * m for e, m in allc.items()) / vol
    res["after_lplus_exact"]["false_copies"] = false
    res["after_lplus_exact"]["t_s"] = time.perf_counter() - t2
    res["after_lplus_exact"]["matches_paper"] = (res.get("paper_certified") is not None and
                                                 abs(res["after_lplus_exact"]["certified_share"]
                                                     - res["paper_certified"]) < 1e-12)
    out2 = res.get("two_core_outside_share")
    res["core"] = dict(old=core_share(res["old_share"], out2), node=core_share(res["node_share"], out2),
                       edge=core_share(res["edge_share"], out2), paper_certified=core_share(res.get("paper_certified"),
                                                                                            out2),
                       note="(share - outside) / (1 - outside), outside = two_core_outside_share "
                            "(results/bench/ablation_summary.json, exact mode), the experiments's formula")
    res["stage"] = "done"
    res["t_total_s"] = time.perf_counter() - t1 + t_old
    MB.save_json(res, outp)
    log(f"{ds} {setting} {mode}: old {res['old_share']:.5f} node {res['node_share']:.5f} edge "
        f"{res['edge_share']} paper cert {res.get('paper_certified')} after L+ exact "
        f"{res['after_lplus_exact']['certified_share']:.5f}; resid copies {res['residual_before']['copies']} -> "
        f"{res['residual_after']['copies']}")


# ------------------------------------------------------------------ driver over many jobs
def cmd_drive(jobs, workers, levels, wall_fn, tag):
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "logs").mkdir(parents=True, exist_ok=True)
    pend = list(jobs)
    running = {}
    failed = []
    while pend or running:
        while pend and len(running) < workers:
            kind, a = pend.pop(0)
            wall = wall_fn(kind, a)
            lf = open(OUTD / "logs" / f"{kind}_{a}{tag}.log", "w", encoding="utf-8")
            pr = subprocess.Popen([*CFG.module_cmd("hyrange.levels"), kind, a,
                                   "--levels", levels, "--wall", str(wall), "--tag", tag],
                                  stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE))
            running[(kind, a)] = (pr, lf, time.time(), wall)
        time.sleep(2.0)
        for key in list(running):
            pr, lf, t_l, wall = running[key]
            rc = pr.poll()
            if rc is not None:
                lf.close()
                del running[key]
                log(f"{'done' if rc == 0 else 'FAILED rc %d' % rc} {key} after {time.time() - t_l:.0f}s")
                if rc != 0:
                    failed.append(key)
            elif time.time() > t_l + wall + 900:
                CFG.kill_tree(pr.pid)
                pr.wait()
                lf.close()
                del running[key]
                failed.append(key)
                log(f"KILLED {key} (wall + 900 s)")
    log(f"drive finished; failed {failed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--levels", default="all")
    ap.add_argument("--wall", type=float, default=600.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    if a.what == "sample":
        cmd_sample(a.args[0], a.levels, a.wall, a.tag)
    elif a.what == "m4":
        cmd_m4(a.args[0], a.args[1], a.args[2], a.tag)
    elif a.what == "m4all":
        import concurrent.futures as cf
        jobs = [(ds, st, md) for md in ("exact", "upper") for st in ("preserved", "reduced")
                for ds in ["crime", "hosts", "directors", "foursquare", "topcs", "dblp"]]
        (OUTD / "logs").mkdir(parents=True, exist_ok=True)

        def runj(j):
            lf = open(OUTD / "logs" / f"m4_{j[0]}_{j[1]}_{j[2]}.log", "w", encoding="utf-8")
            rc = subprocess.run([*CFG.module_cmd("hyrange.levels"), "m4", *j],
                                stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE)).returncode
            lf.close()
            log(f"m4 {j} rc {rc}")
            return rc
        with cf.ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(runj, jobs))
    elif a.what == "drive":
        from hyrange import dense_samples as RD
        names = a.args or list(RD.LOCAL_CERT_SAMPLES)
        if len(names) == 1 and names[0].startswith("@"):
            names = [x for x in names[0][1:].split(",") if x]
        cmd_drive([("sample", nm) for nm in names], a.workers, a.levels, lambda kind, x: a.wall, a.tag)
    else:
        raise SystemExit("unknown " + a.what)


if __name__ == "__main__":
    main()
