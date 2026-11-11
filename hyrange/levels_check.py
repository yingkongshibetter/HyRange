"""Validation of levels.py (L+ and the level-B systems) against the exact certain multiset F^cert on small random
inputs, in the exact and the upper-bound mode (library code; tests/test_soundness.py runs one() on a few seeds).

Instances: random multisets of node sets, 6 to 9 nodes, sizes 2 to 5, mean degree drawn uniformly in [3, 8], a
hyperedge repeats an earlier one with probability 0.3 (python random.Random(seed), seed = instance index).
Ground truth F^cert(e) = min over ALL worlds of the multiplicity of e, by code that shares nothing with the new
step: every clique (size >= 2) of the pair support is a column (no filter), certify.WorldSearch enumerates every
integer x >= 0 with Z x = (W, d) (complete backtracking), min tracked for the distinct hyperedges of the input
(F^cert lies inside every world, in particular inside the input); stopped early when all minima are 0.  Instances
whose enumeration does not finish within its budget are dropped (counted).  On the first instances with small
volume, F^cert is also recomputed with brute.worlds (pair-pivot brute force) and compared.
Checks per instance (any failure raises after saving):
  S1 old rules alone (pipeline.WDForcerTL) forced <= F^cert;
  S2 old rules + L+ node level + levels edge, e2, e4, e8 (methods a, b, c): forced <= F^cert;
  S3 L+ alone (no old rules), node level, methods abc / a / b / c separately: forced <= F^cert;
  S4 random B (2 per instance) on the input and on the residual after the old rules: forced + lambda <= F^cert;
  S5 B = all residual nodes, method c with a large budget: if the enumeration completes, forced + lambda == F^cert;
  S6 monotonicity B subset B' (random nested pair): LP min of every candidate of the level-B point and, when both
     enumerations complete, the exact minima, are not smaller at B' (exceptions are recorded, not raised).
"""
import json
import random
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402
from hyrange import forcing as WC  # noqa: E402
from hyrange import pipeline as MO  # noqa: E402
from hyrange import certify as CX  # noqa: E402
from hyrange import levels as LC  # noqa: E402

TRUTH_S = 20.0
BIG = dict(full_cols=10 ** 9, full_s=10.0, full_points=5_000_000)


def gen(seed):
    rng = random.Random(seed)
    n = rng.randint(6, 9)
    D = rng.uniform(3.0, 8.0)
    edges = []
    vol = 0
    while vol < D * n:
        if edges and rng.random() < 0.3:
            e = rng.choice(edges)
        else:
            k = rng.randint(2, 5)
            e = tuple(sorted(rng.sample(range(n), k)))
        edges.append(e)
        vol += len(e)
    return n, edges


def certain_truth(edges, n, time_s=TRUTH_S):
    adj, deg = WC.build_wdeg(edges, n)
    act = [u for u in range(n) if deg[u] > 0]
    pairs = [(u, v) for u in act for v in adj[u] if u < v]
    rows = pairs + [(u, u) for u in act]
    rid = {p: i for i, p in enumerate(rows)}
    cols = []
    for k in range(2, len(act) + 1):
        for s in combinations(act, k):
            if all(y in adj[x] for x, y in combinations(s, 2)):
                cols.append(s)
    ri, ci = [], []
    for j, s in enumerate(cols):
        for x, y in combinations(s, 2):
            ri.append(rid[(x, y)])
            ci.append(j)
        for x in s:
            ri.append(rid[(x, x)])
            ci.append(j)
    Zb = sp.csc_matrix((np.ones(len(ri), dtype=np.int64), (ri, ci)), shape=(len(rows), len(cols)))
    Rint = [adj[u][v] for (u, v) in pairs] + [deg[u] for u in act]
    C = SimpleNamespace(Zb=Zb, Rint=Rint)
    WS = CX.WorldSearch(C)
    truth = Counter(tuple(sorted(e)) for e in edges)
    cid = {s: j for j, s in enumerate(cols)}
    I = [cid[e] for e in truth]
    mn = {j: truth[cols[j]] for j in I}
    nw = [0]

    def on_world(x):
        nw[0] += 1
        for j in I:
            if x[j] < mn[j]:
                mn[j] = x[j]
        if all(mn[j] == 0 for j in I):
            raise CX.Found()
    st, nodes = WS.search(on_world, time_s=time_s)
    Fc = Counter({cols[j]: int(mn[j]) for j in I if mn[j] > 0})
    return Fc, st, nw[0], len(cols)


def brute_cert(edges, n):
    from hyrange import brute as T
    adj, deg = WC.build_wdeg(edges, n)
    ws = T.worlds(adj, deg, n, exact=True)
    c = None
    for w in ws:
        c = Counter(w) if c is None else (c & w)
    return (c or Counter()), len(ws)


def within(F, Fc):
    return all(Fc.get(tuple(sorted(e)), 0) >= m for e, m in F.items() if m > 0)


def lam_counter(S, lam):
    return Counter({S.cands[j]: v for j, v in lam.items() if v > 0})


def one(seed):
    n, edges = gen(seed)
    adj, deg = WC.build_wdeg(edges, n)
    vol = sum(deg)
    act = [u for u in range(n) if deg[u] > 0]
    rec = dict(seed=seed, n=len(act), copies=len(edges), vol=vol, mean_degree=vol / len(act),
               max_size=max(len(e) for e in edges), repeats=len(edges) - len(set(edges)))
    t0 = time.perf_counter()
    Fc, st, nw, ncol = certain_truth(edges, n)
    rec.update(truth_status=st, truth_worlds=nw, truth_cols=ncol, t_truth=time.perf_counter() - t0)
    if st == "budget":
        return rec
    rec["Fcert_copies"] = int(sum(Fc.values()))
    rec["Fcert_vol"] = int(sum(len(e) * m for e, m in Fc.items()))
    fails = []
    # cross-check of the ground truth with the pair-pivot brute force on small inputs
    if vol <= 26 and len(edges) <= 9:
        Fb, nwb = brute_cert(edges, n)
        rec["brute_worlds"] = nwb
        rec["brute_agrees"] = (Fb == Fc)
        if Fb != Fc:
            fails.append(("ground truth disagrees with brute brute force", dict(Fb), dict(Fc)))
    # S1 old rules
    F0 = MO.WDForcerTL(adj, deg, rules=MO.RULES, time_limit=60, copy_adj=True).run()
    old = Counter({tuple(sorted(e)): m for e, m in F0.forced.items() if m > 0})
    rec["S1_old_ok"] = within(old, Fc)
    rec["old_reaches"] = old == Fc
    rec["old_vol"] = int(sum(len(e) * m for e, m in old.items()))
    if not rec["S1_old_ok"]:
        fails.append(("S1 old rules unsound", dict(old)))
    # S2 old + L+ + levels
    Rn, snaps = LC.run_pipeline(adj, deg, levels="all", wall=120.0)
    full = Counter({tuple(sorted(e)): m for e, m in Rn.F.forced.items() if m > 0})
    rec["S2_ok"] = within(full, Fc)
    rec["S2_vol"] = int(sum(len(e) * m for e, m in full.items()))
    rec["S2_vol_by_stage"] = [s["forced_vol"] for s in snaps]
    rec["S2_node_reaches"] = snaps[1]["forced_vol"] == rec["Fcert_vol"]
    rec["S2_reaches"] = full == Fc
    if not rec["S2_ok"]:
        fails.append(("S2 forcer + L+ unsound", dict(full)))
    # S3 L+ alone, methods
    for meth in ("abc", "a", "b", "c"):
        Rm, sm = LC.run_pipeline(adj, deg, levels="node", wall=120.0, rules=(), methods=meth)
        fm = Counter({tuple(sorted(e)): m for e, m in Rm.F.forced.items() if m > 0})
        ok = within(fm, Fc)
        rec[f"S3_{meth}_ok"] = ok
        rec[f"S3_{meth}_vol"] = int(sum(len(e) * m for e, m in fm.items()))
        if not ok:
            fails.append((f"S3 L+ alone ({meth}) unsound", dict(fm)))
    # S4 random B on the input and on the residual after the old rules
    rng = random.Random(10_000 + seed)
    rec["S4"] = []
    for base in ("input", "residual"):
        R0 = LC.Runner(adj, deg, rules=(() if base == "input" else MO.RULES))
        R0.run_old()
        base_forced = Counter({tuple(sorted(e)): m for e, m in R0.F.forced.items() if m > 0})
        nodes = [u for u in range(n) if R0.F.deg[u] > 0]
        for _ in range(2):
            if not nodes:
                break
            B = rng.sample(nodes, rng.randint(1, len(nodes)))
            for meth in ("abc", "a", "b", "c"):
                try:
                    cands = LC.enum_touching(R0.F, B, cap=10 ** 6)
                except LC.Skip:
                    continue
                S = LC.LevelSystem(R0.F, B, cands)
                r, lam, ub = LC.solve_system(S, meth)
                tot = base_forced + lam_counter(S, lam)
                ok = within(tot, Fc)
                rec["S4"].append(dict(base=base, B=len(B), meth=meth, ncols=S.nc, ok=ok,
                                      lam_vol=int(sum(len(S.cands[j]) * v for j, v in lam.items()))))
                if not ok:
                    fails.append(("S4 random B unsound", base, B, meth, dict(lam_counter(S, lam))))
    # S5 B = all residual nodes, full enumeration
    R0 = LC.Runner(adj, deg, rules=MO.RULES)
    R0.run_old()
    base_forced = Counter({tuple(sorted(e)): m for e, m in R0.F.forced.items() if m > 0})
    nodes = [u for u in range(n) if R0.F.deg[u] > 0]
    if nodes:
        cands = LC.enum_touching(R0.F, nodes, cap=10 ** 6)
        S = LC.LevelSystem(R0.F, nodes, cands)
        r, lam, ub = LC.solve_system(S, "c", budgets=BIG)
        tot = base_forced + lam_counter(S, lam)
        rec["S5_full"] = r.get("full", {}).get("status")
        rec["S5_ok"] = within(tot, Fc)
        if r.get("full", {}).get("status") in ("complete", "found"):
            rec["S5_exact"] = (tot == Fc)
            if tot != Fc:
                fails.append(("S5 B = all does not reproduce F^cert", dict(tot), dict(Fc)))
        if not rec["S5_ok"]:
            fails.append(("S5 unsound", dict(tot)))
    else:
        rec["S5_full"] = "no_residual"
        rec["S5_exact"] = old == Fc
    # S6 monotonicity on the residual (B subset B')
    mono = []
    if len(nodes) >= 2:
        k1 = rng.randint(1, len(nodes) - 1)
        Bp = rng.sample(nodes, rng.randint(k1 + 1, len(nodes)))
        B = Bp[:k1]
        try:
            S1 = LC.LevelSystem(R0.F, B, LC.enum_touching(R0.F, B, cap=10 ** 6))
            S2 = LC.LevelSystem(R0.F, Bp, LC.enum_touching(R0.F, Bp, cap=10 ** 6))
            x0, _ = S1.milp_point()
            idx2 = {c: j for j, c in enumerate(S2.cands)}
            r1, lam1, _ = LC.solve_system(S1, "c", budgets=BIG)
            r2, lam2, _ = LC.solve_system(S2, "c", budgets=BIG)
            # exact minima (complete enumeration) over the support of the level-B point of S1
            c1 = r1.get("full", {}).get("status") == "complete"
            for j in np.nonzero(x0)[0]:
                c = S1.cands[j]
                assert c in idx2, "a level-B candidate missing at level B'"
                v1, val1, _ = S1.lp_bound(int(j))
                v2, val2, _ = S2.lp_bound(idx2[c])
                if val1 is not None and val2 is not None and val2 < val1 - 1e-6:
                    mono.append(dict(kind="lp", cand=list(c), lp_B=val1, lp_Bp=val2))
                if v2 < v1:
                    mono.append(dict(kind="lp_rig", cand=list(c), rig_B=v1, rig_Bp=v2))
            if c1 and r2.get("full", {}).get("status") == "complete":
                for j, v in lam1.items():
                    c = S1.cands[j]
                    j2 = idx2[c]
                    # lam2 only covers the support of S2's own point; a candidate outside it has min 0 at B'
                    v2 = lam2.get(j2, 0)
                    if v2 < v:
                        mono.append(dict(kind="exact", cand=list(c), min_B=v, min_Bp=v2))
            rec["S6_checked"] = True
        except LC.Skip:
            rec["S6_checked"] = False
    rec["S6_exceptions"] = mono
    rec["fails"] = [str(f)[:2000] for f in fails]
    rec["t_total"] = time.perf_counter() - t0
    return rec


def certain_truth_upper(edges, n, D, time_s=TRUTH_S):
    """Upper-bound mode ground truth: every clique of the support is a column, plus one slack column per node
    (degree row: sum_c x_c + s_u = U_u = min(D_u, r_u), 0 <= s_u <= U_u - max_v W_uv); every integer point is a world
    (pair counts W, multi-member degree <= D) and conversely.  Complete backtracking upper_exact.WorldSearchCap
    (not the new code); minima tracked for the distinct input hyperedges, early stop when all are 0."""
    from hyrange import upper_exact as RU
    adj, deg = WC.build_wdeg(edges, n)
    act = [u for u in range(n) if adj[u]]
    pairs = [(u, v) for u in act for v in adj[u] if u < v]
    rows = pairs + [(u, u) for u in act]
    rid = {p: i for i, p in enumerate(rows)}
    cols = []
    for k in range(2, len(act) + 1):
        for s_ in combinations(act, k):
            if all(y in adj[x] for x, y in combinations(s_, 2)):
                cols.append(s_)
    U = {u: min(D[u], sum(adj[u].values())) for u in act}
    L = {u: max(adj[u].values()) for u in act}
    ri, ci = [], []
    for j, s_ in enumerate(cols):
        for x, y in combinations(s_, 2):
            ri.append(rid[(x, y)])
            ci.append(j)
        for x in s_:
            ri.append(rid[(x, x)])
            ci.append(j)
    nc = len(cols)
    cap = [10 ** 9] * nc
    for t, u in enumerate(act):
        ri.append(rid[(u, u)])
        ci.append(nc + t)
        cap.append(U[u] - L[u])
    Zb = sp.csc_matrix((np.ones(len(ri), dtype=np.int64), (ri, ci)), shape=(len(rows), nc + len(act)))
    Rint = [adj[u][v] for (u, v) in pairs] + [U[u] for u in act]
    C = SimpleNamespace(Zb=Zb, Rint=Rint, colcap=np.array(cap, dtype=np.int64))
    WS = RU.WorldSearchCap(C)
    truth = Counter(tuple(sorted(e)) for e in edges)
    cid = {s_: j for j, s_ in enumerate(cols)}
    I = [cid[e] for e in truth]
    mn = {j: truth[cols[j]] for j in I}
    nw = [0]

    def on_world(x):
        nw[0] += 1
        for j in I:
            if x[j] < mn[j]:
                mn[j] = x[j]
        if all(mn[j] == 0 for j in I):
            raise RU.Found()
    st, nodes = WS.search(on_world, time_s=time_s)
    Fc = Counter({cols[j]: int(mn[j]) for j in I if mn[j] > 0})
    return Fc, st, nw[0]


def one_upper(seed):
    """Upper-bound mode: input W and declared upper bounds D (D = d, or d + 0..2 on half of the instances), lo = 0.
    Ground truth: all worlds by upper_exact.brute_worlds (pair-pivot enumeration, independent of levels)."""
    from hyrange import upper_exact as RU
    n, edges = gen(seed)
    rng = random.Random(20_000 + seed)
    adj, d = WC.build_wdeg(edges, n)
    loose = rng.random() < 0.5
    D = [x + (rng.randint(0, 2) if (loose and x > 0) else 0) for x in d]
    lo = [0] * n
    vol = sum(d)
    act = [u for u in range(n) if d[u] > 0]
    rec = dict(seed=seed, mode="upper", n=len(act), copies=len(edges), vol=vol, mean_degree=vol / len(act),
               loose=loose, repeats=len(edges) - len(set(edges)))
    t0 = time.perf_counter()
    Fc, st, nw = certain_truth_upper(edges, n, D)
    if st == "budget":
        rec.update(truth_status="budget", t_truth=time.perf_counter() - t0)
        return rec
    ws = [None] * nw
    # cross-check on small inputs with the pair-pivot brute force of upper_exact
    if vol <= 24 and len(edges) <= 8:
        Wd = {(u, v): w for u in range(n) for v, w in adj[u].items() if u < v}
        try:
            wb = RU.brute_worlds(n, Wd, lo, D, max_worlds=50_000, max_steps=300_000)
            Fb = None
            for w in wb:
                c = Counter(w)
                Fb = c if Fb is None else (Fb & c)
            Fb = Counter({e: m for e, m in (Fb or Counter()).items() if m > 0})
            rec["brute_agrees"] = Fb == Fc
            rec["brute_worlds"] = len(wb)
        except RU.BruteCap:
            rec["brute_agrees"] = None
    rec.update(truth_status=st, truth_worlds=len(ws), t_truth=time.perf_counter() - t0,
               Fcert_copies=int(sum(Fc.values())), Fcert_vol=int(sum(len(e) * m for e, m in Fc.items())))
    truth = Counter(tuple(sorted(e)) for e in edges)
    assert all(truth.get(e, 0) >= m for e, m in Fc.items())
    fails = []
    if rec.get("brute_agrees") is False:
        fails.append(("upper ground truth disagrees with brute_worlds",))
    F0 = RU.interval_forcer(adj, D, lo)
    F0.run()
    old = Counter({tuple(sorted(e)): m for e, m in F0.forced.items() if m > 0})
    rec["S1_old_ok"] = within(old, Fc)
    rec["old_reaches"] = old == Fc
    rec["old_vol"] = int(sum(len(e) * m for e, m in old.items()))
    if not rec["S1_old_ok"]:
        fails.append(("S1 old upper rules unsound", dict(old)))
    Rn, snaps = LC.run_pipeline(adj, D, levels="all", wall=90.0, rules=RU.UPPER_RULES, upper_mode=True, lo=lo,
                                level_s=20.0, sys_s=5.0)
    full = Counter({tuple(sorted(e)): m for e, m in Rn.F.forced.items() if m > 0})
    rec["S2_ok"] = within(full, Fc)
    rec["S2_vol"] = int(sum(len(e) * m for e, m in full.items()))
    rec["S2_vol_by_stage"] = [x["forced_vol"] for x in snaps]
    rec["S2_node_reaches"] = snaps[1]["forced_vol"] == rec["Fcert_vol"]
    rec["S2_reaches"] = full == Fc
    snd = LC.soundness_iv(Rn.F, edges, n, lo)
    rec["S2_residual_ok"] = snd["residual_ok"]
    if not rec["S2_ok"] or not snd["residual_ok"]:
        fails.append(("S2 upper forcer + L+ unsound", dict(full), snd))
    for meth in ("abc", "a", "b", "c"):
        Rm, sm = LC.run_pipeline(adj, D, levels="node", wall=30.0, rules=(), methods=meth, upper_mode=True, lo=lo,
                                 sys_s=5.0)
        fm = Counter({tuple(sorted(e)): m for e, m in Rm.F.forced.items() if m > 0})
        ok = within(fm, Fc)
        rec[f"S3_{meth}_ok"] = ok
        rec[f"S3_{meth}_vol"] = int(sum(len(e) * m for e, m in fm.items()))
        if not ok:
            fails.append((f"S3 upper L+ alone ({meth}) unsound", dict(fm)))
    rng2 = random.Random(30_000 + seed)
    rec["S4"] = []
    for base in ("input", "residual"):
        R0 = LC.Runner(adj, D, rules=(() if base == "input" else RU.UPPER_RULES), upper_mode=True, lo=lo)
        R0.run_old()
        bf = Counter({tuple(sorted(e)): m for e, m in R0.F.forced.items() if m > 0})
        nodes = [u for u in range(n) if R0.F.adj[u]]
        for _ in range(2):
            if not nodes:
                break
            B = rng2.sample(nodes, rng2.randint(1, len(nodes)))
            for meth in ("abc", "a", "b", "c"):
                cands = LC.enum_touching(R0.F, B, cap=10 ** 6)
                S = LC.LevelSystem(R0.F, B, cands)
                r, lam, ub = LC.solve_system(S, meth)
                ok = within(bf + lam_counter(S, lam), Fc)
                rec["S4"].append(dict(base=base, B=len(B), meth=meth, ncols=S.nc, ok=ok))
                if not ok:
                    fails.append(("S4 upper random B unsound", base, B, meth, dict(lam_counter(S, lam))))
    R0 = LC.Runner(adj, D, rules=RU.UPPER_RULES, upper_mode=True, lo=lo)
    R0.run_old()
    bf = Counter({tuple(sorted(e)): m for e, m in R0.F.forced.items() if m > 0})
    nodes = [u for u in range(n) if R0.F.adj[u]]
    if nodes:
        cands = LC.enum_touching(R0.F, nodes, cap=10 ** 6)
        S = LC.LevelSystem(R0.F, nodes, cands)
        r, lam, ub = LC.solve_system(S, "c", budgets=BIG)
        tot = bf + lam_counter(S, lam)
        rec["S5_full"] = r.get("full", {}).get("status")
        rec["S5_ok"] = within(tot, Fc)
        if rec["S5_full"] in ("complete", "found"):
            rec["S5_exact"] = tot == Fc
            if tot != Fc:
                fails.append(("S5 upper B = all does not reproduce F^cert", dict(tot), dict(Fc)))
        if not rec["S5_ok"]:
            fails.append(("S5 upper unsound", dict(tot)))
    else:
        rec["S5_full"] = "no_residual"
        rec["S5_exact"] = old == Fc
    rec["fails"] = [str(f)[:2000] for f in fails]
    rec["t_total"] = time.perf_counter() - t0
    return rec


def worker_upper(lo_, hi_, outp):
    out = []
    for seed in range(lo_, hi_):
        try:
            out.append(one_upper(seed))
        except Exception as ex:
            import traceback
            out.append(dict(seed=seed, mode="upper", fails=["EXCEPTION " + repr(ex)[:1000]],
                            tb=traceback.format_exc()[-3000:]))
        MB.save_json(out, outp)


def summarize_upper(recs):
    done = [r for r in recs if r.get("truth_status") in ("complete", "found") and "S1_old_ok" in r]
    fails = [r for r in recs if r.get("fails")]
    S = dict(instances_tried=len(recs), truth_budget_dropped=sum(1 for r in recs if r.get("truth_status") == "budget"),
             instances_checked=len(done), loose_bounds=sum(1 for r in done if r["loose"]),
             mean_degree_range=[min(r["mean_degree"] for r in done), max(r["mean_degree"] for r in done)] if done else None,
             nonempty_Fcert=sum(1 for r in done if r["Fcert_copies"] > 0),
             brute_upper_crosschecked=sum(1 for r in done if r.get("brute_agrees") is not None),
             brute_upper_agrees=sum(1 for r in done if r.get("brute_agrees") is True),
             S1_old_sound=sum(1 for r in done if r["S1_old_ok"]), S2_sound=sum(1 for r in done if r.get("S2_ok")),
             S2_residual_ok=sum(1 for r in done if r.get("S2_residual_ok")),
             S3_sound={m: sum(1 for r in done if r.get(f"S3_{m}_ok")) for m in ("abc", "a", "b", "c")},
             S4_systems=sum(len(r.get("S4", [])) for r in done),
             S4_sound=sum(1 for r in done for x in r.get("S4", []) if x["ok"]),
             S5_complete=sum(1 for r in done if "S5_exact" in r), S5_exact=sum(1 for r in done if r.get("S5_exact")),
             reach_Fcert=dict(old=sum(1 for r in done if r["old_reaches"]),
                              old_plus_node=sum(1 for r in done if r.get("S2_node_reaches")),
                              old_plus_all_levels=sum(1 for r in done if r.get("S2_reaches"))),
             vol=dict(Fcert=sum(r["Fcert_vol"] for r in done), old=sum(r["old_vol"] for r in done),
                      old_plus_node=sum(r["S2_vol_by_stage"][1] for r in done),
                      old_plus_all_levels=sum(r["S2_vol"] for r in done)),
             failures=len(fails), failure_records=fails[:20])
    return S


def worker(args):
    lo, hi, outp = args
    out = []
    for seed in range(lo, hi):
        try:
            out.append(one(seed))
        except Exception as ex:     # recorded as a failure (an exception inside the new code is a bug)
            import traceback
            out.append(dict(seed=seed, fails=["EXCEPTION " + repr(ex)[:1000]], tb=traceback.format_exc()[-3000:]))
        MB.save_json(out, outp)
    return out


def validate(N, procs=6):
    import subprocess
    OUT = LC.OUTD
    parts = []
    tmpd = OUT / "validation_parts"
    tmpd.mkdir(parents=True, exist_ok=True)
    per = (N + procs - 1) // procs
    prs = []
    for p in range(procs):
        lo, hi = p * per, min(N, (p + 1) * per)
        if lo >= hi:
            continue
        outp = tmpd / f"part{p}.json"
        parts.append(outp)
        prs.append(subprocess.Popen([*CFG.module_cmd("hyrange.levels_check"),
                                     str(lo), str(hi), str(outp)], cwd=str(HERE)))
    for pr in prs:
        pr.wait()
    recs = []
    for p in parts:
        recs += json.load(open(p, encoding="utf-8"))
    summarize(recs, N)


def summarize(recs, N):
    done = [r for r in recs if r.get("truth_status") in ("complete", "found") and "fails" in r
            and "S1_old_ok" in r]
    exc = [r for r in recs if any(str(f).startswith("EXCEPTION") for f in r.get("fails", []))]
    fails = [r for r in recs if r.get("fails")]
    S = dict(
        script="levels_check.py", generated=time.strftime("%Y-%m-%d %H:%M:%S"), instances_tried=len(recs),
        truth_budget_dropped=sum(1 for r in recs if r.get("truth_status") == "budget"),
        instances_checked=len(done),
        mean_degree_range=[min(r["mean_degree"] for r in done), max(r["mean_degree"] for r in done)] if done else None,
        dense_instances_md_ge_6=sum(1 for r in done if r["mean_degree"] >= 6),
        with_repeats=sum(1 for r in done if r["repeats"] > 0),
        nonempty_Fcert=sum(1 for r in done if r["Fcert_copies"] > 0),
        brute_crosschecked=sum(1 for r in done if "brute_agrees" in r),
        brute_agrees=sum(1 for r in done if r.get("brute_agrees")),
        S1_old_sound=sum(1 for r in done if r["S1_old_ok"]),
        S2_sound=sum(1 for r in done if r.get("S2_ok")),
        S3_sound={m: sum(1 for r in done if r.get(f"S3_{m}_ok")) for m in ("abc", "a", "b", "c")},
        S4_systems=sum(len(r.get("S4", [])) for r in done),
        S4_sound=sum(1 for r in done for x in r.get("S4", []) if x["ok"]),
        S5_complete=sum(1 for r in done if "S5_exact" in r),
        S5_exact=sum(1 for r in done if r.get("S5_exact")),
        S6_checked=sum(1 for r in done if r.get("S6_checked")),
        S6_exceptions=sum(len(r.get("S6_exceptions", [])) for r in done),
        reach_Fcert=dict(old=sum(1 for r in done if r["old_reaches"]),
                         old_plus_node=sum(1 for r in done if r.get("S2_node_reaches")),
                         old_plus_all_levels=sum(1 for r in done if r.get("S2_reaches"))),
        reach_Fcert_nonempty=dict(
            old=sum(1 for r in done if r["old_reaches"] and r["Fcert_copies"] > 0),
            old_plus_node=sum(1 for r in done if r.get("S2_node_reaches") and r["Fcert_copies"] > 0),
            old_plus_all_levels=sum(1 for r in done if r.get("S2_reaches") and r["Fcert_copies"] > 0)),
        vol=dict(Fcert=sum(r["Fcert_vol"] for r in done), old=sum(r["old_vol"] for r in done),
                 old_plus_node=sum(r["S2_vol_by_stage"][1] for r in done),
                 old_plus_all_levels=sum(r["S2_vol"] for r in done),
                 lplus_alone={m: sum(r.get(f"S3_{m}_vol", 0) for r in done) for m in ("abc", "a", "b", "c")}),
        failures=len(fails), exceptions=len(exc),
        failure_records=fails[:20])
    S["sound_all"] = (S["failures"] == 0 and S["S2_sound"] == len(done)
                      and all(v == len(done) for v in S["S3_sound"].values()) and S["S4_sound"] == S["S4_systems"])
    MB.save_json(dict(summary=S, records=recs), LC.OUTD / "validation.json")
    print(json.dumps({k: v for k, v in S.items() if k != "failure_records"}, indent=1))
    if fails:
        print("FAILURES:", json.dumps(fails[:3], indent=1)[:5000])


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "upper":
        worker_upper(int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]))
    elif len(sys.argv) == 2 and sys.argv[1] == "upper_summary":
        recs = []
        for p in sorted((LC.OUTD / "validation_parts").glob("upper*.json")):
            recs += json.load(open(p, encoding="utf-8"))
        S = summarize_upper(recs)
        MB.save_json(dict(summary=S, records=recs), LC.OUTD / "validation_upper.json")
        print(json.dumps({k: v for k, v in S.items() if k != "failure_records"}, indent=1))
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        validate(int(sys.argv[2]), procs=4)
    elif len(sys.argv) == 4:
        worker((int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])))
    elif len(sys.argv) == 2 and sys.argv[1] == "summary":
        recs = []
        for p in sorted((LC.OUTD / "validation_parts").glob("part*.json")):
            recs += json.load(open(p, encoding="utf-8"))
        summarize(recs, len(recs))
    else:
        seeds = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
        for s in seeds:
            r = one(s)
            print(json.dumps({k: v for k, v in r.items() if k not in ("S4",)}, indent=None)[:3000])
