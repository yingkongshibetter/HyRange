"""Exact certification of the residual components (Section IV of the experiments; Tables II, III and VI).

Setting.  A world of a release (W, d) is an integer multiset of hyperedges (cliques of the support with admissible
sizes) that reproduces W and d.  Worlds = F + (worlds of the residual), and the residual worlds factor over the
connected components of the residual support, so the certain multiplicity of a node set e is
cert(e) = F(e) + min over the worlds of e's component of x_e.  Inside a component K whose candidate enumeration
completed (candidates.build_comp_wp, sound filters: every hyperedge of every world is a candidate), the worlds of K are
exactly the vectors x in Z^n_{>=0} with Z x = R_K (pair rows and diagonal rows).  Every candidate e with
cert_K(e) >= 1 lies in every world, so it lies in the found world; it suffices to decide the candidates of the found
world(s).

Per treated component (enumerated, not pinned by the rank test), for every candidate e of the found worlds:
  ub(e)  = min of x_e over the verified witness worlds (the found worlds, worlds met by the enumeration, the IP
           argmin worlds, worlds met by the refutation search).  Each witness is checked in exact integer arithmetic
           (Z x == R_K, x >= 0).  ub(e) copies of e at most are certain; x_e - ub(e) copies are provably not certain.
  lb(e)  = a proved lower bound, from
    (enum) exhaustive enumeration of all worlds by exact integer backtracking (below), when it completes within its
           budget: then lb = ub = the exact minimum for every candidate;
    (lp)   the LP min of x_e (HiGHS) gives a dual vector y; engine.best_rig evaluates, in exact rational arithmetic,
           <R_K, y> - vol_K * max_e' ((zeta_e'^T y - [e' = e]) / |e'|)^+, a lower bound on x_e over every x >= 0 with
           Z x = R_K (it uses sum_e' |e'| x_e' = vol_K, implied by the diagonal rows); lb = ceil of it;
    (dfs)  exhaustive backtracking for a world with x_e <= ub(e) - 1: if the search completes without one, no world
           has fewer than ub(e) copies, so lb = ub(e).  A world found instead is a new witness (ub decreases).
    (dfs_lp) the same backtracking where a node is also discarded when the LP relaxation of the node is infeasible,
           proved by an exact Farkas-type certificate (LPFarkas: the solver proposes the dual vector, the check is in
           exact integers / rationals); used when the plain backtracking runs out of its budget.
  IP min = HiGHS MILP min x_e (zero gap, time limit), reported separately: solver-based, not a proof (its argmin is a
           checked witness, so it is an upper bound on cert(e) whatever its status).
A copy of e in the found world is CERTAIN (proved) for the first lb(e) copies, PROVABLY NOT CERTAIN beyond ub(e),
UNDECIDED in between.
Components with candidates but no found world (no_world_component, used by experiments/fewer_rules.py): (1) a
bounded MILP feasibility search (zero objective, guarded child process); (2) LP screening: candidates positive in
every one of three LP solutions with random objectives are the only ones the LP bound can certify (a feasible LP
point with x_e = 0 makes the LP min 0); the LP lower bound is evaluated for them, warm-started, within a budget.

Backtracking (WorldSearch).  State: residual R (exact ints), alive candidates (free, ub > 0), per row the number of
alive candidates.  Pick the positive row with the fewest alive candidates; branch on one of its alive candidates
c with values min(ub_c, R_r), ..., 0 (exactly R_r if it is the last alive candidate of the row); assigning v
subtracts v from c's rows and kills every alive candidate of a row that reaches 0.  A positive row with no alive
candidate is a dead end; no positive row left is a world.  Every world (x >= 0, Z x = R) is reached exactly once.

Inputs: the six sparse benchmarks in both settings (residuals rebuilt from results/bench/forced, found worlds =
ours_pred/<ds>_<mode>_s<k>.json minus F, split by component and checked as in experiments/hybrid.py), written by
pipeline.py (experiments/hyrange_bench.py), which must run first.  Ground truth is read only for the soundness assertion: every newly certified
copy is a true hyperedge with at least F(e) + lb(e) copies; a failure raises.

Budgets (fixed before any run, identical for all inputs): enumeration 30 s and 1,000,000 worlds per component,
attempted when the component has at most 5,000 candidates; LP 60 s per solve; IP min 30 s per candidate and 900 s per
component (guarded child above 30,000 columns); refutation search 2 s (plain) then 60 s (LP-pruned) per candidate
and step, 900 s per component;
no-world components: feasibility MILP 1,800 s, LP screening and bounds 1,800 s, IP min 120 s per candidate and 600 s in all.

usage: python -I -B -X utf8 experiments/exact_certification.py bench <ds> <mode>
Writes results/certification/cert_exact/bench_<ds>_<mode>.json; experiments/same_input_summary.py cert collects them
into results/certification/cert_exact.json.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from fractions import Fraction  # noqa: E402
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

CERT_DIR = CFG.RESULTS / "certification"
CE = CERT_DIR / "cert_exact"

ENUM_MAX_CANDS = 5000
ENUM_S = 30.0
ENUM_MAX_WORLDS = 1_000_000
LP_S = 60.0
IP_S = 30.0
IP_COMP_S = 900.0
DFS_S = 2.0
DFSLP_S = 60.0
DFS_COMP_S = 900.0
FIND_S = 1800.0
SCREEN_S = 1800.0
MILP_GUARD = MO.MILP_GUARD


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


# ------------------------------------------------------------------ exact backtracking over the worlds of a component
class Budget(Exception):
    pass


class Found(Exception):
    pass


class WorldSearch:
    def __init__(self, C):
        Zb = C.Zb.tocsc()
        self.nc = Zb.shape[1]
        self.nr = Zb.shape[0]
        self.crow = [Zb.indices[Zb.indptr[c]:Zb.indptr[c + 1]].tolist() for c in range(self.nc)]
        Zr = Zb.tocsr()
        self.rcand = [Zr.indices[Zr.indptr[r]:Zr.indptr[r + 1]].tolist() for r in range(self.nr)]
        self.R0 = [int(v) for v in C.Rint]

    def search(self, on_world, target=None, cap=None, time_s=10.0, max_nodes=None, prune=None):
        """DFS over all worlds.  on_world(x) is called on every world (x: list of ints, do not keep); it may raise
        Found to stop.  prune(alive, R, pos) -> True only if the node provably has no world (LPFarkas).  With a
        target, the target's value (<= cap) is branched on first.  Returns ('complete' | 'found' | 'budget', nodes)."""
        crow, rcand = self.crow, self.rcand
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
            if max_nodes is not None and nodes[0] > max_nodes:
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
            # branch on the alive candidate of row r with the most rows (most propagation)
            c = -1
            best = -1
            for c2 in rcand[r]:
                if alive[c2] and len(crow[c2]) > best:
                    best = len(crow[c2])
                    c = c2
            ub = min(R[r2] for r2 in crow[c])
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

        # initial dead rows (none in a component, but keep it general)
        if any(cnt[r] == 0 for r in pos):
            return "complete", 0
        out = ["complete"]

        def root():
            if target is None:
                rec()
                return
            # branch on the target first: x_target in {min(ub, cap), ..., 0}
            ubt = min(min(R[r2] for r2 in crow[target]), cap)
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


class LPFarkas:
    """Exact infeasibility certificate for the LP relaxation of a search node (alive columns, residual R on the
    positive rows).  Phase-1 LP min sum(s+ + s-) s.t. A x + s+ - s- = b, x, s >= 0 (always feasible); if its optimum is
    positive, its row duals y are rounded to Y = round(y 2^K) and checked exactly: for every x >= 0 with A x = b,
    Y^T b = (A^T Y)^T x <= max_e ((A^T Y)_e^+ / |e|) * sum_e |e| x_e = max_e (...) * vol', where vol' = sum of b over the
    node rows (every alive column has all its rows positive, so the node rows of b sum to sum_e |e| x_e).  If
    Y^T b > vol' * max_e ((A^T Y)_e^+ / |e|) (exact integers / Fractions), the node has no world.  The solver only
    proposes Y; the pruning decision is exact."""

    def __init__(self, C):
        self.Zr = sp.csr_matrix(C.Zb, dtype=np.int64)
        self.k = np.asarray(C.k, dtype=np.int64)
        self.isdiag = np.array([1 if i == j else 0 for (i, j) in C.rows], dtype=np.int64)
        self.calls = 0
        self.proved = 0
        self.t = 0.0

    def __call__(self, alive, R, pos):
        t0 = time.perf_counter()
        try:
            return self._infeasible(alive, R, pos)
        finally:
            self.t += time.perf_counter() - t0

    def _infeasible(self, alive, R, pos):
        self.calls += 1
        cols = np.flatnonzero(np.frombuffer(bytes(alive), dtype=np.uint8)) if isinstance(alive, bytearray) else \
            np.flatnonzero(np.asarray(alive, dtype=np.int8))
        rows = np.array(sorted(pos), dtype=np.int64)
        if len(cols) == 0:
            return len(rows) > 0
        A = self.Zr[rows][:, cols].tocsc()
        b = np.array([R[r] for r in rows], dtype=np.int64)
        m = len(rows)
        I = sp.identity(m, format="csc", dtype=np.float64)
        Af = sp.hstack([A.astype(np.float64), I, -I], format="csc")
        c = np.r_[np.zeros(len(cols)), np.ones(2 * m)]
        r = EN.hsolve(c, Af, b.astype(np.float64), time_limit=10.0)
        if r.get("status") != "Optimal" or "y" not in r or r["obj"] < 1e-7:
            return False
        kk = self.k[cols]
        vol = int(np.dot(b, self.isdiag[rows]))
        AT = A.T.tocsr()
        for sgn in (1.0, -1.0):
            y = sgn * np.asarray(r["y"], dtype=np.float64)
            ymax = float(np.abs(y).max()) if len(y) else 0.0
            K = 30
            while K > 0 and (ymax + 1.0) * 2.0 ** K * 400 >= 2.0 ** 61:
                K -= 1
            Y = np.rint(y * 2.0 ** K).astype(np.int64)
            S = AT @ Y                                           # exact int64
            yb = sum(int(a) * int(bb) for a, bb in zip(Y, b))
            pos_ = S > 0
            if pos_.any():
                ratio = S[pos_].astype(np.float64) / kk[pos_]
                top = np.nonzero(ratio >= ratio.max() * (1 - 1e-9) - 1e-9)[0]
                Sp, kp = S[pos_], kk[pos_]
                viol = max(Fraction(int(Sp[t]), int(kp[t])) for t in top)
            else:
                viol = Fraction(0)
            if Fraction(yb) > vol * viol:
                self.proved += 1
                return True
        return False


def is_world(C, x):
    """Exact integer check: x >= 0 and Z x == R_K."""
    x = np.asarray(x, dtype=np.int64)
    if (x < 0).any():
        return False
    Zb = C.Zb.tocsc().astype(np.int64)
    return bool(np.array_equal(Zb @ x, np.asarray([int(v) for v in C.Rint], dtype=np.int64)))


# ------------------------------------------------------------------ LP lower bound (warm-started HiGHS) + exact rig
class LPMin:
    def __init__(self, C):
        from scipy.optimize._highspy import _core as hc
        self.hc = hc
        A = sp.csc_matrix(C.Zb, dtype=np.float64)
        nr, nc = A.shape
        lp = hc.HighsLp()
        lp.num_col_ = nc
        lp.num_row_ = nr
        lp.col_cost_ = np.zeros(nc)
        lp.col_lower_ = np.zeros(nc)
        lp.col_upper_ = np.full(nc, hc.kHighsInf)
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

    def _run(self, tl):
        h = self.h
        h.setOptionValue("time_limit", float(h.getRunTime() + tl))
        t0 = time.perf_counter()
        h.run()
        t = time.perf_counter() - t0
        st = h.modelStatusToString(h.getModelStatus())
        sol = h.getSolution()
        return st, sol, t

    def solve_cost(self, c, tl):
        """LP min c^T x (c: float array over all columns).  Returns (status, x or None, t)."""
        self.h.changeColsCost(self.nc, np.arange(self.nc, dtype=np.int32), np.asarray(c, dtype=np.float64))
        self.cur = None
        st, sol, t = self._run(tl)
        return st, (np.array(sol.col_value) if sol.value_valid else None), t

    def lower_bound(self, j, tl=LP_S):
        """(lb as Fraction or None, lp value, status, seconds) for min x_j."""
        h = self.h
        if self.cur is None:
            h.changeColsCost(self.nc, np.arange(self.nc, dtype=np.int32), np.zeros(self.nc))
        else:
            h.changeColCost(int(self.cur), 0.0)
        h.changeColCost(int(j), 1.0)
        self.cur = j
        st, sol, t = self._run(tl)
        if st != "Optimal" or not sol.dual_valid:
            return None, None, st, t
        y = np.array(sol.row_dual)
        num = np.zeros(self.nc, dtype=np.int64)
        num[j] = 1
        den = np.ones(self.nc, dtype=np.int64)
        lb = EN.best_rig(self.C, 1, num, den, y, "min")
        return lb, float(np.array(sol.col_value)[j]), st, t


def ip_min(C, j, tl=IP_S):
    """HiGHS MILP min x_j (zero gap).  Returns dict(status, value (of a verified argmin) or None, bound, x, t)."""
    A = sp.csc_matrix(C.Zb, dtype=np.float64)
    b = np.array([float(v) for v in C.Rint])
    c = np.zeros(A.shape[1])
    c[j] = 1.0
    t0 = time.perf_counter()
    if A.shape[1] > MILP_GUARD:
        r = MO.hsolve_guarded(c, A, b, C.ub, tl)
    else:
        r = EN.hsolve(c, A, b, ub=C.ub, integer=True, time_limit=tl)
    t = time.perf_counter() - t0
    out = dict(status=r.get("status"), bound=r.get("bound"), t=t, value=None, x=None)
    if "x" in r:
        xi = np.rint(r["x"]).astype(np.int64)
        if is_world(C, xi):
            out["value"] = int(xi[j])
            out["x"] = xi
        else:
            out["status"] = str(out["status"]) + "|unverified"
    return out


# ------------------------------------------------------------------ one component
def certify_component(C, worlds, budgets=None):
    """worlds: list of int arrays over C.cands (the found worlds, one per seed; may be empty).  Returns a record with
    per-candidate lb (proved), ub (witnessed), ip results and the methods used."""
    B = dict(enum_max_cands=ENUM_MAX_CANDS, enum_s=ENUM_S, enum_max_worlds=ENUM_MAX_WORLDS, lp_s=LP_S, ip_s=IP_S,
             ip_comp_s=IP_COMP_S, dfs_s=DFS_S, dfslp_s=DFSLP_S, dfs_comp_s=DFS_COMP_S)
    if budgets:
        B.update(budgets)
    nc = len(C.cands)
    rec = dict(nodes=len(C.nodes), vol=int(C.vol), ncands=nc, nrows=len(C.rows), times={})
    for x in worlds:
        assert is_world(C, x), "found world does not reproduce the component"
    I = sorted(set(int(j) for x in worlds for j in np.nonzero(x)[0]))
    ub = {j: min(int(x[j]) for x in worlds) for j in I}
    lb = {j: 0 for j in I}
    how = {}
    for j in I:
        if ub[j] == 0:
            how[j] = "witness0"

    def undecided():
        return [j for j in I if lb[j] < ub[j]]

    def take_witness(xw):
        for j in I:
            v = int(xw[j])
            if v < ub[j]:
                ub[j] = v

    # ---- (enum) exhaustive enumeration
    WS = WorldSearch(C)
    t0 = time.perf_counter()
    rec["enum"] = dict(attempted=False)
    if nc <= B["enum_max_cands"] and undecided():
        mn = {j: ub[j] for j in I}
        nworlds = [0]
        target_set = list(I)

        def on_world(x):
            nworlds[0] += 1
            new = False
            for j in target_set:
                if x[j] < mn[j]:
                    mn[j] = x[j]
                    new = True
            if new:
                assert is_world(C, x), "enumerated world fails the exact check"
            if nworlds[0] >= B["enum_max_worlds"]:
                raise Budget()
            if all(mn[j] == 0 for j in target_set):
                raise Found()          # every candidate has a world without it: nothing is certain
        st, nodes = WS.search(on_world, time_s=B["enum_s"])
        rec["enum"] = dict(attempted=True, status=st, worlds=nworlds[0], nodes=nodes)
        for j in I:
            if mn[j] < ub[j]:
                ub[j] = mn[j]
        if st == "complete":
            # mn = the exact minimum over all worlds
            for j in I:
                lb[j] = ub[j]
                how.setdefault(j, "enum")
        elif st == "found":
            # the enumeration met, for every candidate, a world without it: nothing is certain (witnesses)
            for j in I:
                assert ub[j] == 0
                how.setdefault(j, "witness_enum")
    rec["times"]["enum"] = time.perf_counter() - t0

    # ---- IP min (solver-based, reported) for every candidate of the found worlds; its argmin is a witness
    t0 = time.perf_counter()
    ipr = {}
    for j in I:
        if ub[j] == 0:
            ipr[j] = dict(status="not_run_witness_world_with_0_copies", value=0)
            continue
        if time.perf_counter() - t0 > B["ip_comp_s"]:
            ipr[j] = dict(status="not_run_budget")
            continue
        r = ip_min(C, j, B["ip_s"])
        if r["x"] is not None:
            take_witness(r["x"])
        r.pop("x")
        ipr[j] = r
    rec["times"]["ip"] = time.perf_counter() - t0

    # ---- (lp) exact LP-dual lower bounds
    t0 = time.perf_counter()
    lpr = {}
    LP = None
    for j in I:
        if lb[j] >= ub[j] and how.get(j) != "enum":
            continue
        if LP is None:
            LP = LPMin(C)
        lbf, val, st, t = LP.lower_bound(j, B["lp_s"])
        lpr[j] = dict(status=st, lp_value=val, rig=(str(lbf) if lbf is not None else None), t=t)
        if lbf is not None:
            v = max(0, math.ceil(lbf))
            lpr[j]["lb"] = int(v)
            assert v <= ub[j], ("rigorous LP bound above a witnessed world", j, v, ub[j])
            if how.get(j) == "enum":
                assert v <= lb[j], ("LP bound above the enumerated minimum", j, v, lb[j])
            elif v > lb[j]:
                lb[j] = v
                if lb[j] == ub[j]:
                    how[j] = "lp"
    rec["times"]["lp"] = time.perf_counter() - t0

    # ---- (dfs) refutation: no world with x_j <= ub_j - 1.  First plain backtracking (DFS_S); if it runs out of
    # budget, backtracking with exact LP-infeasibility pruning (LPFarkas, DFSLP_S).
    t0 = time.perf_counter()
    dfs = {}
    FK = None
    for j in I:
        use_lp = False
        while lb[j] < ub[j]:
            if time.perf_counter() - t0 > B["dfs_comp_s"]:
                dfs.setdefault(j, []).append(dict(status="not_run_budget"))
                break
            cap = ub[j] - 1
            got = [None]

            def on_world(x, got=got):
                got[0] = list(x)
                raise Found()
            if use_lp:
                if FK is None:
                    FK = LPFarkas(C)
                p0, c0 = FK.proved, FK.calls
                st, nodes = WS.search(on_world, target=j, cap=cap, time_s=B["dfslp_s"], prune=FK)
                dfs.setdefault(j, []).append(dict(cap=cap, status=st, nodes=nodes, lp_pruning=True,
                                                  lp_calls=FK.calls - c0, lp_pruned=FK.proved - p0))
            else:
                st, nodes = WS.search(on_world, target=j, cap=cap, time_s=B["dfs_s"])
                dfs.setdefault(j, []).append(dict(cap=cap, status=st, nodes=nodes, lp_pruning=False))
            if st == "complete":
                lb[j] = ub[j]
                how[j] = "dfs_lp" if use_lp else "dfs"
            elif st == "found":
                xw = np.asarray(got[0], dtype=np.int64)
                assert is_world(C, xw) and xw[j] <= cap
                take_witness(xw)
                if lb[j] >= ub[j]:
                    how.setdefault(j, "witness_dfs")
            elif not use_lp:
                use_lp = True
            else:
                break
    rec["times"]["dfs"] = time.perf_counter() - t0

    # consistency of the solver with the proofs (reported; the proofs never use it)
    incons = []
    for j in I:
        r = ipr[j]
        # a verified IP world below a proved lower bound would refute the proof: stop
        assert not (r.get("value") is not None and r["value"] < lb[j]), ("verified IP world below the proved lb", j)
        if r.get("status") == "Optimal" and r.get("value") is not None and lb[j] == ub[j] and r["value"] != ub[j]:
            incons.append(dict(j=j, kind="solver_optimal_value_differs_from_exact_minimum", ip=r["value"],
                               exact=ub[j]))
    rec["solver_inconsistencies"] = incons
    for j in I:
        if lb[j] == ub[j] and j not in how:
            how[j] = "witness"
    rec["cands"] = [dict(j=j, k=int(len(C.cands[j])), x=[int(x[j]) for x in worlds], lb=int(lb[j]), ub=int(ub[j]),
                         decided=bool(lb[j] == ub[j]), how=how.get(j, "undecided"),
                         ip=ipr.get(j), lp=lpr.get(j), dfs=dfs.get(j)) for j in I]
    hows = Counter(c["how"] for c in rec["cands"])
    rec["how_counts"] = dict(hows)
    if rec["enum"].get("status") == "complete":
        rec["method"] = "enumeration"
    elif all(c["how"] in ("witness0", "witness_enum") for c in rec["cands"]):
        rec["method"] = "witness worlds (nothing certain)"
    elif all(c["decided"] for c in rec["cands"]):
        rec["method"] = "+".join(sorted(k for k in hows))
    else:
        rec["method"] = "partial:" + "+".join(sorted(k for k in hows))
    rec["all_decided"] = all(c["decided"] for c in rec["cands"])
    rec["t_total_s"] = sum(rec["times"].values())
    return rec


# ------------------------------------------------------------------ components without a found world
def no_world_component(C, find_s=FIND_S, screen_s=SCREEN_S, ip_s=120.0, ip_total_s=600.0):
    rec = dict(nodes=len(C.nodes), vol=int(C.vol), ncands=len(C.cands), nrows=len(C.rows), times={})
    # (1) bounded feasibility search for a world (zero objective: the first incumbent is optimal)
    A = sp.csc_matrix(C.Zb, dtype=np.float64)
    b = np.array([float(v) for v in C.Rint])
    t0 = time.perf_counter()
    r = MO.hsolve_guarded(np.zeros(A.shape[1]), A, b, C.ub, find_s)
    rec["times"]["find_world"] = time.perf_counter() - t0
    world = None
    if "x" in r:
        xi = np.rint(r["x"]).astype(np.int64)
        if is_world(C, xi):
            world = xi
    rec["find_world"] = dict(status=r.get("status"), found=world is not None, limit_s=find_s)
    log(f"   feasibility MILP: {r.get('status')}, world {'found' if world is not None else 'not found'} "
        f"in {rec['times']['find_world']:.0f}s")
    # (2) LP screening + exact LP lower bounds
    t0 = time.perf_counter()
    LP = LPMin(C)
    supp = None
    scr = []
    rng = np.random.default_rng(0)
    for i in range(3):
        st, x, t = LP.solve_cost(rng.random(A.shape[1]), max(1.0, screen_s - (time.perf_counter() - t0)))
        scr.append(dict(status=st, t=t))
        if x is None or st != "Optimal":
            supp = None
            break
        s = set(np.nonzero(x > 1e-7)[0].tolist())
        supp = s if supp is None else (supp & s)
    rec["screen"] = scr
    S = sorted(supp) if supp is not None else []
    if world is not None:
        S = [j for j in S if world[j] > 0]
    rec["screened_candidates"] = len(S)
    lbs = {}
    t_lp0 = time.perf_counter()
    nbud = 0
    for j in S:
        if time.perf_counter() - t0 > screen_s:
            nbud += 1
            continue
        lbf, val, st, t = LP.lower_bound(j, LP_S)
        v = max(0, math.ceil(lbf)) if lbf is not None else 0
        lbs[j] = dict(lb=int(v), status=st, lp_value=val, rig=(str(lbf) if lbf is not None else None), t=t)
        if world is not None:
            assert v <= world[j], ("rigorous LP bound above a found world", j, v, int(world[j]))
    rec["times"]["lp"] = time.perf_counter() - t_lp0
    rec["times"]["screen_total"] = time.perf_counter() - t0
    rec["lp_not_run_budget"] = nbud
    rec["lp_bounds"] = {int(j): d for j, d in lbs.items()}
    # (3) IP min (solver) for a few candidates with a positive bound, bounded
    t0 = time.perf_counter()
    ipr = {}
    if world is not None:
        for j in [j for j in S if lbs.get(j, {}).get("lb", 0) < world[j]]:
            if time.perf_counter() - t0 > ip_total_s:
                break
            r = ip_min(C, j, ip_s)
            r.pop("x")
            ipr[int(j)] = r
    rec["ip"] = ipr
    rec["times"]["ip"] = time.perf_counter() - t0
    rec["world_support"] = int((world > 0).sum()) if world is not None else None
    rec["world_copies"] = int(world.sum()) if world is not None else None
    rec["certified"] = [dict(j=int(j), k=int(len(C.cands[j])), lb=d["lb"],
                             x=(int(world[j]) if world is not None else None))
                        for j, d in sorted(lbs.items()) if d["lb"] > 0]
    rec["t_total_s"] = sum(v for k, v in rec["times"].items() if k != "screen_total") + 0.0
    return rec, world


# ------------------------------------------------------------------ report helpers
def comp_summary(C, crec, seeds_x, nodes_of):
    """Per-seed copy counts for one treated component.  seeds_x: list of world vectors (one per seed)."""
    out = []
    by_j = {c["j"]: c for c in crec["cands"]}
    for x in seeds_x:
        unc = cert = notc = und = 0
        for j in np.nonzero(x)[0]:
            c = by_j[int(j)]
            m = int(x[j])
            unc += m
            cert += c["lb"]
            notc += m - c["ub"]
            und += c["ub"] - c["lb"]
        out.append(dict(uncertain_copies=unc, certain=cert, not_certain=notc, undecided=und))
    return out


def finish_source(res, comps_treated, forced, vol, truth_mult, nodes_map, pinned_vol, seeds_unc):
    """Aggregate over treated components + soundness (truth used only here)."""
    cert_new = Counter()
    vol_lb = vol_ub = 0
    cop_lb = cop_ub = 0
    for C, crec in comps_treated:
        for c in crec["cands"]:
            e = tuple(sorted(int(nodes_map(C.nodes[i])) for i in C.cands[c["j"]]))
            if c["lb"] > 0:
                cert_new[e] += c["lb"]
            vol_lb += c["k"] * c["lb"]
            vol_ub += c["k"] * c["ub"]
            cop_lb += c["lb"]
            cop_ub += c["ub"]
    # SOUNDNESS: every newly certified copy is a true hyperedge with at least F(e) + lb(e) copies
    bad = {e: m for e, m in cert_new.items() if truth_mult.get(e, 0) < forced.get(e, 0) + m}
    res["soundness"] = dict(new_certified_copies=int(sum(cert_new.values())),
                            new_certified_distinct=len(cert_new), violations=len(bad))
    if bad:
        raise AssertionError(("UNSOUND new certified copies", len(bad), list(bad.items())[:5]))
    fvol = int(sum(len(e) * m for e, m in forced.items()))
    res["certified_vol_before"] = fvol + pinned_vol
    res["certified_vol_share_before"] = (fvol + pinned_vol) / vol
    res["certified_vol_after"] = fvol + pinned_vol + vol_lb
    res["certified_vol_share_after"] = (fvol + pinned_vol + vol_lb) / vol
    res["treated_new_certified_vol"] = vol_lb
    res["treated_Fcert_vol_upper"] = vol_ub
    res["treated_new_certified_copies"] = cop_lb
    res["treated_Fcert_copies_upper"] = cop_ub
    res["Fcert_share_reached_on_treated"] = dict(
        vol=(vol_lb / vol_ub if vol_ub else None), copies=(cop_lb / cop_ub if cop_ub else None),
        exact=all(c["decided"] for _, cr in comps_treated for c in cr["cands"]),
        note="lb / ub summed over the candidates of the found worlds; equals the share of F^cert when all are decided "
             "(then ub = cert), otherwise a lower bound on it")
    return res


# ------------------------------------------------------------------ benchmark settings
def bench(ds, mode):
    t_all = time.perf_counter()
    lines, truth = MB.target(ds, mode)
    line_mult = Counter(lines)
    nodes = sorted(set(v for e in lines for v in e))
    loc = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    adj, deg = WC.build_wdeg([tuple(loc[v] for v in e) for e in lines], n)
    vol = int(sum(deg))
    ou = json.load(open(MB.OUT / "ours" / f"{ds}_{mode}.json", encoding="utf-8"))
    F = MB.load_hyperedges(MB.OUT / "forced" / f"{ds}_{mode}_forced.json")
    assert all(line_mult.get(e, 0) >= m for e, m in F.items())
    Floc = Counter({tuple(loc[v] for v in e): m for e, m in F.items()})
    radj = [dict(a) for a in adj]
    rdeg = list(deg)
    for e, m in Floc.items():
        for x in e:
            rdeg[x] -= m
            for y in e:
                if y != x:
                    radj[x][y] -= m
    for x in range(n):
        for y in [y for y, w in radj[x].items() if w <= 0]:
            assert radj[x][y] == 0
            del radj[x][y]
    resid_file = [tuple(t) for t in json.load(open(MB.OUT / "forced" / f"{ds}_{mode}_residual.json",
                                                    encoding="utf-8"))]
    assert sorted((nodes[u], nodes[v], int(w)) for u in range(n) for v, w in radj[u].items() if u < v) == resid_file
    comps_nodes = sorted(EN._components(radj, n), key=len)
    assert len(comps_nodes) == ou["comps"]
    comp_of = {}
    for ci, c in enumerate(comps_nodes):
        for u in c:
            comp_of[int(u)] = ci
    res = dict(input=f"{ds}_{mode}", kind="benchmark", ds=ds, name=MB.DISP[ds], mode=mode, n=n, vol=vol,
               forced_vol=int(sum(len(e) * m for e, m in F.items())), comps=len(comps_nodes), components=[])
    # found worlds (reused from pipeline.py), split by component
    seeds_world = []
    for s in MB.SEEDS:
        P = MB.load_hyperedges(MB.OUT / "ours_pred" / f"{ds}_{mode}_s{s}.json")
        per = [Counter() for _ in comps_nodes]
        for e, m in P.items():
            f = F.get(e, 0)
            if m - f <= 0:
                continue
            cs = {comp_of.get(loc[v]) for v in e}
            assert len(cs) == 1 and None not in cs
            per[cs.pop()][tuple(loc[v] for v in e)] += m - f
        seeds_world.append(per)
    pinned_vol = 0
    treated = []
    seeds_unc = [0, 0, 0]
    for ci, cn in enumerate(comps_nodes):
        C = WP.build_comp_wp(cn, radj, rdeg, cap=EN.CAP, enum_s=EN.ENUM_S)
        crec = dict(index=ci, nodes=len(C.nodes), vol=int(C.vol), enum_status=C.enum,
                    ncands=(len(C.cands) if C.cands is not None else None))
        if C.cands is None:
            crec["treated"] = False
            crec["why"] = f"not enumerated ({C.enum})"
            res["components"].append(crec)
            continue
        if C.unique:
            pinned_vol += int(C.vol)
            crec["treated"] = False
            crec["why"] = "pinned by the rank test (already certified)"
            res["components"].append(crec)
            continue
        idx = {tuple(int(C.nodes[i]) for i in c): j for j, c in enumerate(C.cands)}
        xs = []
        for s, per in enumerate(seeds_world):
            if not per[ci]:
                continue
            x = np.zeros(len(C.cands), dtype=np.int64)
            for e, m in per[ci].items():
                x[idx[tuple(sorted(e))]] += m
            xs.append(x)
        if not xs:
            crec["treated"] = False
            crec["why"] = "no found world (handled by the no-world procedure)"
            crec["no_world"] = True
            res["components"].append(crec)
            continue
        assert len(xs) == 3
        t0 = time.perf_counter()
        r = certify_component(C, xs)
        r.update(crec)
        r["treated"] = True
        r["per_seed"] = comp_summary(C, r, xs, None)
        for s in range(3):
            seeds_unc[s] += r["per_seed"][s]["uncertain_copies"]
        res["components"].append(r)
        treated.append((C, r))
        log(f"{ds} {mode} comp {ci}: {len(C.nodes)} nodes, {len(C.cands)} cands, method {r['method']}, "
            f"{time.perf_counter() - t0:.1f}s")
    for s in range(3):
        assert seeds_unc[s] == ou["seeds"][s]["uncertain_copies"], ("uncertain copies differ from pipeline", s)
    res["treated_components"] = len(treated)
    res = finish_source(res, treated, Counter({tuple(loc[v] for v in e): m for e, m in F.items()}), vol,
                        Counter({tuple(loc[v] for v in e): m for e, m in line_mult.items()}), lambda u: int(u),
                        pinned_vol, seeds_unc)
    res["per_seed"] = per_seed_totals(res)
    res["t_script_s"] = time.perf_counter() - t_all
    res["done"] = True
    MB.save_json(res, CE / f"bench_{ds}_{mode}.json")


def per_seed_totals(res):
    out = []
    for s in range(3):
        tot = Counter()
        for c in res["components"]:
            if c.get("treated"):
                tot.update(c["per_seed"][s])
        out.append(dict(seed=s, **{k: int(v) for k, v in tot.items()}))
    return out


def main():
    CE.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.what == "bench":
        bench(a.args[0], a.args[1])
    else:
        raise SystemExit("unknown " + a.what)


if __name__ == "__main__":
    main()
