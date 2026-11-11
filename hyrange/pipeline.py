"""HyRange on one input of MARIOH's benchmark (forcing, components, rank test, one world per seed).  Input: the target's pair multiplicities W and node degrees d
(number of target hyperedges containing the node), nothing else: no source hypergraph, no training, no parameter
set on the target.

Pipeline, with the view built once and handed over as (W, d):
  1. (W, d) = A^T A of the target lines (bench: the multiset whose projection MARIOH reconstructs);
  2. forcing: forcing.WDForcer, rules D1, A, B, P, L, wall limit --force-tl (on a time-out the forced multiset so
     far is kept; it is sound, the forcing is just incomplete);
  3. connected components of the residual support; per component (smallest first) candidates.build_comp_wp: candidate
     cliques (cap 3e5 / 60 s per component), rank test (full column rank => one world, verified exactly); components
     reached after --enum-budget seconds from the start are not enumerated;
  4. per seed s in {0, 1, 2}: per component its unique world (pinned) or an integer world found by HiGHS with the
     random objective (positive weights, numpy default_rng(s)), MILP limit --milp-tl s, checked exactly
     (engine.verify_world); components over the cap, not enumerated, without a solution, or reached after
     --world-budget seconds get no hyperedges (reported as uncovered volume).
     MILPs with more than MILP_GUARD columns run in a child process that is killed at --milp-tl + 60 s (HiGHS has
     ignored its time limit on very large MILPs, engine.MILP_MAXCOLS).
  Copies are labelled CERTIFIED (forced, or in a pinned component) or UNCERTAIN.
Ground truth is read only after the reconstruction, to (a) assert soundness: every certified copy is a target line with at
least that multiplicity, the forced multiset passes forcing.verify_forced and the residual equals the W+ of
(target - forced); a failure raises and stops; (b) score.

usage: python -I -B -X utf8 experiments/hyrange_bench.py <ds> <preserved|reduced> [--force-tl 900] [--enum-budget 1200]
       [--world-budget 180] [--milp-tl 60]
Writes results/bench/ours/<ds>_<mode>.json (updated after each stage), forced/<ds>_<mode>_forced.json,
forced/<ds>_<mode>_residual.json ([[u, v, W_uv - W(F)_uv], ...], the input of the learned completion), and
ours_pred/<ds>_<mode>_s<seed>.json.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse  # noqa: E402
import queue  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
from hyrange import bench as MB  # noqa: E402
from hyrange import engine as EN  # noqa: E402
from hyrange import forcing as WC  # noqa: E402
from hyrange import candidates as WP  # noqa: E402

RULES = ("D1", "A", "B", "P", "L")
MILP_GUARD = EN.MILP_MAXCOLS


class WDForcerTL(WC.WDForcer):
    """forcing.WDForcer with wall budgets on rule L.  WDForcer checks time_limit only between nodes, and one local
    enumeration (local_enum) can run unbounded: its leaf budget does not count pruned internal branches (on eu,
    preserved, one call ran > 18 min).  Here local_enum is the same code with one added check: past min(call start +
    L_CALL_S, forcing deadline) it stops and returns None, exactly as over the leaf budget (rule L is then skipped at
    that node, which forces nothing, so soundness is unchanged); run() stops at the next node after the forcing
    deadline as before.  Identical to WDForcer whenever no rule-L call exceeds L_CALL_S and the forcing ends before
    its limit (l_call_stops = l_deadline_stops = 0 in the output)."""

    L_CALL_S = 10.0      # wall budget of one rule-L enumeration (next to WDForcer's leaf budget LMAX_LEAVES)

    def run(self, order=None, log=None):
        self._global_deadline = None if self.time_limit is None else time.perf_counter() + self.time_limit
        self.l_deadline_stops = 0
        self.l_call_stops = 0
        return super().run(order=order, log=log)

    def local_rule(self, u, touched):
        t = time.perf_counter() + self.L_CALL_S
        g = self._global_deadline
        self._deadline = t if g is None else min(t, g)
        self._deadline_is_call = g is None or t < g
        return super().local_rule(u, touched)

    def local_enum(self, u):
        """Copy of WDForcer.local_enum (forcing.py) plus the deadline check marked below."""
        adj, deg, r = self.adj, self.deg, self.r
        d = deg[u]
        au = adj[u]
        N = sorted(au, key=lambda v: (-au[v], v))
        nn = len(N)
        lo = [[0] * nn for _ in range(nn)]
        hi = [[0] * nn for _ in range(nn)]
        for i, v in enumerate(N):
            av = adj[v]
            for j in range(i):
                w = N[j]
                rvw = av.get(w, 0)
                hi[i][j] = hi[j][i] = min(rvw, au[v], au[w])
                lb = max(0, au[v] + rvw - deg[v], au[w] + rvw - deg[w], au[v] + au[w] - d)
                lo[i][j] = lo[j][i] = lb
                if lb > hi[i][j]:
                    return []
        cap = [r[v] - (deg[v] - au[v]) for v in N]
        slots = [0] * d
        sizes = [0] * d
        assign = [None] * nn
        out = set()
        leaves = [0]
        visits = [0]
        deadline = self._deadline

        def classes():
            seen = {}
            for s in range(d):
                seen.setdefault(slots[s], []).append(s)
            return list(seen.values())

        def choose(cls, R):
            res = []

            def rec(ci, left, cur):
                if ci == len(cls):
                    if left == 0:
                        res.append(tuple(cur))
                    return
                rem = sum(len(c) for c in cls[ci + 1:])
                for k in range(min(left, len(cls[ci])), -1, -1):
                    if left - k > rem:
                        break
                    rec(ci + 1, left - k, cur + cls[ci][:k])
            rec(0, R, [])
            return res

        def rec(i):
            if leaves[0] > self.lmax_leaves:
                return
            # ---- added: wall deadline of the forcing (checked every 1024 visits)
            visits[0] += 1
            if deadline is not None and (visits[0] & 1023) == 0 and time.perf_counter() > deadline:
                leaves[0] = self.lmax_leaves + 1
                if self._deadline_is_call:
                    self.l_call_stops += 1
                else:
                    self.l_deadline_stops += 1
                return
            # ----
            if i == nn:
                leaves[0] += 1
                if any(sizes[s] == 0 for s in range(d)):
                    return
                for j in range(nn):
                    if sum(sizes[s] for s in assign[j]) > cap[j]:
                        return
                key = tuple(sorted(tuple(sorted(N[b] for b in range(nn) if slots[s] >> b & 1)) for s in range(d)))
                out.add(key)
                return
            empty = sum(1 for s in range(d) if sizes[s] == 0)
            if empty > sum(au[N[j]] for j in range(i, nn)):
                return
            R = au[N[i]]
            for S in choose(classes(), R):
                ok = True
                Sset = set(S)
                for j in range(i):
                    c = sum(1 for s in assign[j] if s in Sset)
                    if c < lo[i][j] or c > hi[i][j]:
                        ok = False
                        break
                if not ok:
                    continue
                for s in S:
                    slots[s] |= 1 << i
                    sizes[s] += 1
                bad = False
                for j in range(i + 1):
                    sj = assign[j] if j < i else S
                    if sum(sizes[s] for s in sj) > cap[j]:
                        bad = True
                        break
                if not bad:
                    assign[i] = S
                    rec(i + 1)
                    assign[i] = None
                for s in S:
                    slots[s] &= ~(1 << i)
                    sizes[s] -= 1
                if leaves[0] > self.lmax_leaves:
                    return

        rec(0)
        if leaves[0] > self.lmax_leaves:
            return None
        return [tuple(frozenset(t) for t in key) for key in out]


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def _milp_child(q, c, A, b, ub, tl):
    r = EN.hsolve(c, A, b, ub=ub, integer=True, time_limit=tl)
    q.put(r)


def hsolve_guarded(c, A, b, ub, tl):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_milp_child, args=(q, c, A, b, ub, tl))
    p.start()
    try:
        r = q.get(timeout=tl + 60)
    except queue.Empty:
        r = dict(status="killed_after_limit")
    p.join(10)
    if p.is_alive():
        p.terminate()
        p.join()
    return r


def world_of(C, rng, tl):
    """The world of a component: its unique world if the rank test pins it, else a HiGHS world with the random
    objective of rng, checked exactly; with the child-process guard for very large MILPs."""
    if C.unique and C.xu is not None:
        return C.xu, "pinned"
    Zb = C.Zb.tocsc().astype(np.float64)
    w = 1.0 / np.asarray(C.wden, dtype=np.float64)
    A = Zb @ sp.diags(w)
    b = np.array([float(r) for r in C.Rint])
    c = rng.random(A.shape[1])
    if A.shape[1] > MILP_GUARD:
        r = hsolve_guarded(c, sp.csc_matrix(A), b, C.ub, tl)
    else:
        r = EN.hsolve(c, A, b, ub=C.ub, integer=True, time_limit=tl)
    if "x" not in r:
        return None, "nosol:" + r["status"]
    x = np.rint(r["x"]).astype(np.int64)
    if (x < 0).any() or not EN.verify_world(C, x, 1):
        return None, "unverified"
    return x, "found"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("mode", choices=["preserved", "reduced"])
    ap.add_argument("--force-tl", type=float, default=900.0)
    ap.add_argument("--enum-budget", type=float, default=1200.0)
    ap.add_argument("--world-budget", type=float, default=180.0)
    ap.add_argument("--milp-tl", type=float, default=60.0)
    ap.add_argument("--cap", type=int, default=EN.CAP)
    a = ap.parse_args()
    ds, mode = a.ds, a.mode
    outp = MB.OUT / "ours" / f"{ds}_{mode}.json"
    res = dict(ds=ds, mode=mode, stage="start", rules=list(RULES), force_tl=a.force_tl, enum_budget=a.enum_budget,
               world_budget=a.world_budget, milp_tl=a.milp_tl, cap=a.cap, enum_s_per_comp=EN.ENUM_S,
               rank_max=EN.RANK_MAX, milp_guard_cols=MILP_GUARD)

    # ---------------- input view (W, d); the target lines are not passed to HyRange
    lines, truth = MB.target(ds, mode)
    nodes = sorted(set(v for e in lines for v in e))
    loc = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    adj, deg = WC.build_wdeg([tuple(loc[v] for v in e) for e in lines], n)
    vol = int(sum(deg))
    res.update(n=n, vol=vol, input_pairs=int(sum(len(x) for x in adj) // 2),
               input_pair_weight=int(sum(sum(x.values()) for x in adj) // 2))
    log(f"{ds} {mode}: n {n}, lines {len(lines)}, vol {vol}, pairs {res['input_pairs']}")

    # ---------------- decoding
    t_start = time.perf_counter()
    F = WDForcerTL(adj, deg, rules=RULES, time_limit=a.force_tl, copy_adj=False).run(log=log)
    t_force = time.perf_counter() - t_start
    forced = Counter({tuple(sorted(int(v) for v in e)): int(m) for e, m in F.forced.items()})
    forced_vol = sum(len(e) * m for e, m in forced.items())
    res.update(stage="forced", t_force_s=t_force, force_complete=bool(F.complete), force_rounds=F.rounds,
               force_counts=dict(F.counts), l_runs=F.l_runs, l_skipped=F.l_skipped,
               l_deadline_stops=F.l_deadline_stops, l_call_stops=F.l_call_stops, l_call_s=F.L_CALL_S,
               forcer="pipeline.WDForcerTL (forcing.WDForcer + wall budgets in rule L)",
               forced_copies=int(sum(forced.values())), forced_distinct=len(forced), forced_vol=forced_vol,
               forced_vol_share=forced_vol / vol)
    log(f"forcing {t_force:.1f}s complete {F.complete}: {sum(forced.values())} copies, vol share "
        f"{forced_vol / vol:.4f}, {dict(F.counts)}")

    t1 = time.perf_counter()
    comp_nodes = EN._components(F.adj, n)
    comp_nodes.sort(key=len)
    comps = []
    for cn in comp_nodes:
        rem = t_start + a.enum_budget - time.perf_counter()
        if rem <= 0:
            C = WP.build_comp_wp(cn, F.adj, F.deg, enumerate_=False)
            C.enum = "budget"
        else:
            C = WP.build_comp_wp(cn, F.adj, F.deg, cap=a.cap, enum_s=min(EN.ENUM_S, rem))
        comps.append(C)
    t_prep = time.perf_counter() - t1
    assert forced_vol + sum(C.vol for C in comps) == vol, "forced + residual volume != vol"
    enum_status = Counter(C.enum for C in comps)
    res.update(stage="prepared", t_components_enum_rank_s=t_prep, comps=len(comps),
               largest_comp_nodes=max((len(C.nodes) for C in comps), default=0), enum_status=dict(enum_status),
               comps_not_enumerated=sum(1 for C in comps if C.cands is None),
               vol_not_enumerated=int(sum(C.vol for C in comps if C.cands is None)),
               cands_total=int(sum(len(C.cands) for C in comps if C.cands is not None)),
               cands_max=int(max((len(C.cands) for C in comps if C.cands is not None), default=0)),
               unique_comps=sum(1 for C in comps if C.unique), unique_vol=int(sum(C.vol for C in comps if C.unique)),
               rank_full_no_world=sum(1 for C in comps if getattr(C, "rank_full_no_world", False)))
    log(f"components {len(comps)} in {t_prep:.1f}s: enum {dict(enum_status)}, unique {res['unique_comps']}, "
        f"not enumerated vol {res['vol_not_enumerated']}")

    # ---------------- soundness of the forcing (truth used only to check)
    lines_loc = [tuple(loc[v] for v in e) for e in lines]
    ver = WC.verify_forced(F.forced, lines_loc)
    res["verify_forced"] = ver
    if not ver["ok"]:
        MB.save_json(res, outp)
        raise AssertionError(("UNSOUND forcing", ver))
    rok = WP.check_residual_wp(F, lines_loc, n)
    res["residual_equals_Wplus_of_truth_minus_forced"] = bool(rok)
    if not rok:
        MB.save_json(res, outp)
        raise AssertionError("UNSOUND: residual mismatch")
    del lines_loc

    # outputs for the variants: forced multiset and residual pair counts (original ids), from the view only
    MB.save_hyperedges(Counter({tuple(nodes[i] for i in e): m for e, m in forced.items()}),
                       MB.OUT / "forced" / f"{ds}_{mode}_forced.json")
    resid = sorted((nodes[u], nodes[v], int(w)) for u in range(n) for v, w in F.adj[u].items() if u < v)
    MB.save_json(resid, MB.OUT / "forced" / f"{ds}_{mode}_residual.json")
    MB.save_json(res, outp)

    # ---------------- worlds, three seeds
    line_mult = Counter(lines)
    res["seeds"] = []
    for seed in MB.SEEDS:
        rng = np.random.default_rng(seed)
        tw = time.perf_counter()
        cert = Counter(forced)
        unc = Counter()
        status = Counter()
        uncovered = 0
        for C in comps:
            if C.cands is None:
                status["not_enumerated:" + C.enum] += 1
                uncovered += int(C.vol)
                continue
            if not (C.unique and C.xu is not None) and time.perf_counter() - tw > a.world_budget:
                status["world_budget"] += 1
                uncovered += int(C.vol)
                continue
            x, st = world_of(C, rng, a.milp_tl)
            status[st] += 1
            if x is None:
                uncovered += int(C.vol)
                continue
            tgt = cert if st == "pinned" else unc
            for j in np.nonzero(x)[0]:
                tgt[tuple(sorted(int(C.nodes[i]) for i in C.cands[j]))] += int(x[j])
        t_world = time.perf_counter() - tw
        cert_o = Counter({tuple(nodes[i] for i in e): m for e, m in cert.items()})
        unc_o = Counter({tuple(nodes[i] for i in e): m for e, m in unc.items()})
        # SOUNDNESS: every certified copy is a target line with at least that multiplicity
        bad = {e: m for e, m in cert_o.items() if line_mult.get(e, 0) < m}
        if bad:
            res["seeds"].append(dict(seed=seed, UNSOUND=len(bad)))
            MB.save_json(res, outp)
            raise AssertionError(("UNSOUND certified copies", len(bad), list(bad.items())[:5]))
        out_h = cert_o + unc_o
        met = MB.metrics(out_h, truth)
        cvol = sum(len(e) * m for e, m in cert_o.items())
        uvol = sum(len(e) * m for e, m in unc_o.items())
        row = dict(seed=seed, t_world_s=t_world, status=dict(status), uncovered_vol=uncovered,
                   uncovered_vol_share=uncovered / vol,
                   certified_copies=int(sum(cert_o.values())), certified_distinct=len(cert_o),
                   certified_vol=cvol, certified_vol_share=cvol / vol,
                   certified_copy_share=sum(cert_o.values()) / len(lines),
                   certified_distinct_share=len(set(cert_o) & set(truth)) / len(truth),
                   certified_sound=True,
                   uncertain_copies=int(sum(unc_o.values())), uncertain_vol=uvol,
                   uncertain_true_copies=int(sum(min(m, line_mult.get(e, 0)) for e, m in unc_o.items())),
                   **met)
        res["seeds"].append(row)
        MB.save_hyperedges(out_h, MB.OUT / "ours_pred" / f"{ds}_{mode}_s{seed}.json")
        res["stage"] = f"seed{seed}"
        MB.save_json(res, outp)
        log(f"seed {seed}: {t_world:.1f}s {dict(status)}  J {met['jaccard']:.4f} MJ {met['multi_jaccard']:.4f} "
            f"rec {met['exact_recall']:.4f} prec {met['precision']:.4f} cert vol {cvol / vol:.4f}")
    res["stage"] = "done"
    res["t_total_s"] = time.perf_counter() - t_start
    MB.save_json(res, outp)
    log(f"done {res['t_total_s']:.1f}s")


if __name__ == "__main__":
    main()
