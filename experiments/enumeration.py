"""Paragraph Enumeration: the random inputs and HyRange on them (the TwinSearch runs on the same inputs:
experiments/twinsearch/run_ts.py).

Input (random, fixed seeds): n nodes, m = round(mean_deg * n / 3) distinct hyperedges (sizes uniform on {2, 3, 4},
members uniform at random, a repeated hyperedge is redrawn), so the mean node degree is mean_deg.  View: pair counts
W and degrees d (forcing.build_wdeg).  A world is a multiset of node sets of size >= 2 with exactly this W and d.
Series deg2, deg3, deg6 (mean degree two, three, six), ten inputs per size.

HyRange: pipeline pipeline (WDForcerTL rules D1, A, B, P, L, 900 s; components; candidates.build_comp_wp with the
default cap / 60 s; rank test; found worlds by MO.world_of, seeds 0, 1, 2, MILP 60 s, world budget 180 s per seed)
then certify.certify_component with its default budgets on every component that is neither pinned nor over the cap.
Certified multiset = forced + pinned components + the proved lower bounds lb.  Labels on the seed-0 world: certified,
provably not certain (x - ub), undecided (ub - lb).  Asserted: every certified copy lies in the hidden hypergraph.

usage: python -I -B -X utf8 experiments/enumeration.py run <series> <n1,n2,...> [--inst 10]
       (the experiments's inputs: deg2 and deg3 with 6,8,10,12,15,20,25,30,40,60,100; deg6 with the same sizes)
writes results/enumeration/parts/<series>_n<n>.json after every instance.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402
from hyrange import engine as EN  # noqa: E402
from hyrange import forcing as WC  # noqa: E402
from hyrange import candidates as WP  # noqa: E402
from hyrange import pipeline as MO  # noqa: E402
from hyrange import certify as CX  # noqa: E402

OUTD = CFG.RESULTS / "enumeration"
SERIES = {"deg2": 2.0, "deg3": 3.0, "deg6": 6.0}
# HyRange limits: the defaults of pipeline.py / certify.py on the benchmarks
FORCE_TL, ENUM_BUDGET, WORLD_BUDGET, MILP_TL = 900.0, 1200.0, 180.0, 60.0


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def gen(n, mean_deg, seed):
    rnd = random.Random(seed)
    m = max(1, round(mean_deg * n / 3.0))
    S = set()
    while len(S) < m:
        k = rnd.randint(2, 4)
        S.add(tuple(sorted(rnd.sample(range(n), k))))
    return sorted(S)


def key(e):
    return "-".join(map(str, e))


def cm(C):
    return {key(e): int(m) for e, m in sorted(C.items()) if m > 0}


# ------------------------------------------------------------------ HyRange with exact certification
def run_B(adj, deg, n):
    t_start = time.perf_counter()
    F = MO.WDForcerTL(adj, deg, rules=MO.RULES, time_limit=FORCE_TL, copy_adj=True).run()
    forced = Counter({tuple(sorted(int(v) for v in e)): int(m) for e, m in F.forced.items() if m > 0})
    t_force = time.perf_counter() - t_start
    comps = []
    for cn in sorted(EN._components(F.adj, n), key=len):
        rem = t_start + ENUM_BUDGET - time.perf_counter()
        if rem <= 0:
            C = WP.build_comp_wp(cn, F.adj, F.deg, enumerate_=False)
            C.enum = "budget"
        else:
            C = WP.build_comp_wp(cn, F.adj, F.deg, cap=EN.CAP, enum_s=min(EN.ENUM_S, rem))
        comps.append(C)
    # found worlds, seeds 0, 1, 2 (as pipeline.py)
    found = [dict() for _ in MB.SEEDS]
    for s in MB.SEEDS:
        rng = np.random.default_rng(s)
        tw = time.perf_counter()
        for ci, C in enumerate(comps):
            if C.cands is None or (C.unique and C.xu is not None):
                continue
            if time.perf_counter() - tw > WORLD_BUDGET:
                continue
            x, st = MO.world_of(C, rng, MILP_TL)
            if x is not None:
                found[s][ci] = x
    cert = Counter(forced)
    lab = Counter(certified=sum(forced.values()), not_certain=0, undecided=0)
    methods = Counter()
    how = Counter()
    uncovered = 0
    for ci, C in enumerate(comps):
        if C.cands is None:
            methods["over_cap_or_time:" + str(C.enum)] += 1
            uncovered += 1
            continue
        if C.unique and C.xu is not None:
            methods["pinned_rank_test"] += 1
            for j in np.nonzero(C.xu)[0]:
                e = tuple(sorted(int(C.nodes[i]) for i in C.cands[j]))
                cert[e] += int(C.xu[j])
                lab["certified"] += int(C.xu[j])
            continue
        xs = [found[s][ci] for s in MB.SEEDS if ci in found[s]]
        if not xs:
            methods["no_found_world"] += 1
            uncovered += 1
            continue
        r = CX.certify_component(C, xs)
        methods[r["method"]] += 1
        how.update(r["how_counts"])
        x0 = xs[0]
        for c in r["cands"]:
            j = c["j"]
            e = tuple(sorted(int(C.nodes[i]) for i in C.cands[j]))
            if c["lb"] > 0:
                cert[e] += c["lb"]
            m0 = int(x0[j])
            lab["certified"] += min(c["lb"], m0)
            lab["not_certain"] += max(0, m0 - c["ub"])
            lab["undecided"] += max(0, min(c["ub"], m0) - c["lb"])
    t = time.perf_counter() - t_start
    rec = dict(t_s=t, t_force_s=t_force, force_complete=bool(F.complete), forced_copies=int(sum(forced.values())),
               force_counts=dict(F.counts), residual_components=len(comps),
               largest_residual_component=max((len(C.nodes) for C in comps), default=0),
               component_methods=dict(methods), cand_how=dict(how), uncovered_components=uncovered,
               labels_seed0=dict(lab), certified_copies=int(sum(cert.values())))
    return rec, cert


# ------------------------------------------------------------------ one instance
def le(A, B):
    return all(B.get(e, 0) >= m for e, m in A.items())


def instance(series, n, i):
    seed = 13_000_000 + 10_000 * int(SERIES[series]) + 100 * n + i
    H = gen(n, SERIES[series], seed)
    truth = Counter(H)
    adj, deg = WC.build_wdeg(H, n)
    rec = dict(series=series, n=n, i=i, seed=seed, m=len(H), mean_degree=sum(deg) / n,
               isolated_nodes=sum(1 for d in deg if d == 0), hyperedges=[key(e) for e in H])
    rB, certB = run_B(adj, deg, n)
    assert le(certB, truth), "UNSOUND: HyRange certified a copy that is not in the truth"
    rec["B"] = rB
    rec["B"]["certified"] = cm(certB)
    return rec


def run(series, sizes, ninst):
    for n in sizes:
        out = OUTD / "parts" / f"{series}_n{n}.json"
        recs = []
        if out.exists():          # resume: keep the finished instances of an interrupted run
            import json
            recs = json.load(open(out, encoding="utf-8"))["records"]
            assert [r["i"] for r in recs] == list(range(len(recs)))
        for i in range(len(recs), ninst):
            r = instance(series, n, i)
            recs.append(r)
            MB.save_json(dict(script="experiments/enumeration.py", series=series, n=n, records=recs), out)
            log(f"{series} n={n} i={i}: m {r['m']} | HyRange {r['B']['t_s']:.2f}s cert {r['B']['certified_copies']} "
                f"undecided {r['B']['labels_seed0']['undecided']} methods {r['B']['component_methods']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("series", nargs="?")
    ap.add_argument("sizes", nargs="?")
    ap.add_argument("--inst", type=int, default=10)
    a = ap.parse_args()
    if a.what == "run":
        run(a.series, [int(x) for x in a.sizes.split(",")], a.inst)


if __name__ == "__main__":
    main()
