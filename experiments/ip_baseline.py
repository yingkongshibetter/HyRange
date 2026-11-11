"""The integer program baseline (Table III, row IP; paragraph Same input) on the six benchmarks: one integer
program with degree rows.

No forcing and no decomposition into residual components:
  candidates = all admissible cliques of the whole support of (W, d): the admissibility tests of candidates.build_comp_wp
               (W+ size window and pair-core filter, both computed on the full view since nothing is forced), cap
               3,000,000 candidates in total and 600 s in total to build.  (The cliques are enumerated connected
               component by connected component of the support, which yields exactly the cliques of the whole
               support; the candidate set and the program are global.)
  program    = one integer program over all candidates: pair rows Z x = W and degree rows = d, 0 <= x_e <= ub_e
               (ub_e = min of e's pair and degree entries), objective a random vector (numpy default_rng(seed),
               seeds 0, 1, 2), solved by HiGHS (one thread, mip_rel_gap 0, time limit 600 s; above 30,000 columns in a
               child process killed at the limit + 60 s, as pipeline.py); the returned x is checked exactly (integer
               Z x == (W, d), x >= 0).
Output = that world (every copy unlabelled: nothing is certified).  Metrics of bench.metrics.
Ground truth is read only to score (and to assert that the truth is a world of the candidate system when it builds).

usage: python -I -B -X utf8 experiments/ip_baseline.py <ds> <mode>          -> results/certification/mono/<ds>_<mode>.json
       python -I -B -X utf8 experiments/ip_baseline.py jobs                  (appends the 12 command lines to results/certification/queue/jobs.jsonl)
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402

OUTD = CFG.RESULTS / "certification" / "mono"
CAP_TOTAL = 3_000_000
BUILD_S = 600.0
MILP_S = 600.0


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def run(ds, mode):
    from hyrange import engine as EN
    from hyrange import forcing as WC
    from hyrange import candidates as WP
    from hyrange import pipeline as MO
    outp = OUTD / f"{ds}_{mode}.json"
    lines, truth = MB.target(ds, mode)
    nodes = sorted(set(v for e in lines for v in e))
    loc = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    adj, deg = WC.build_wdeg([tuple(loc[v] for v in e) for e in lines], n)
    vol = int(sum(deg))
    res = dict(ds=ds, name=MB.DISP[ds], mode=mode, n=n, vol=vol, cap_total=CAP_TOTAL, build_s=BUILD_S, milp_s=MILP_S,
               pairs=int(sum(len(a) for a in adj) // 2))
    # ---- build
    t0 = time.perf_counter()
    comps = sorted(EN._components(adj, n), key=len)
    blocks = []
    total = 0
    built = True
    why = None
    for cn in comps:
        rem_t = t0 + BUILD_S - time.perf_counter()
        rem_c = CAP_TOTAL - total
        if rem_t <= 0:
            built, why = False, "time"
            break
        C = WP.build_comp_wp(cn, adj, deg, cap=rem_c, enum_s=rem_t, rank_max=0)
        if C.cands is None:
            built, why = False, C.enum
            total += 0
            break
        total += len(C.cands)
        blocks.append(C)
    t_build = time.perf_counter() - t0
    res.update(support_components=len(comps), built=built, build_fail=why, t_build_s=t_build,
               candidates=(total if built else None), candidates_before_failure=total,
               components_enumerated=len(blocks))
    log(f"{ds} {mode}: build {'ok' if built else 'FAILED (' + str(why) + ')'} in {t_build:.1f}s, "
        f"{total} candidates over {len(blocks)}/{len(comps)} support components")
    if not built:
        res["seeds"] = []
        res["note"] = "the monolithic program does not build within the cap / time limit: no world, no output"
        MB.save_json(res, outp)
        return
    # one global matrix (block diagonal over the support components; rows: pairs then degrees of each block)
    Z = sp.block_diag([C.Zb for C in blocks], format="csc").astype(np.float64)
    R = np.concatenate([np.asarray([int(v) for v in C.Rint], dtype=np.int64) for C in blocks])
    ub = np.concatenate([C.ub for C in blocks])
    cand_nodes = [tuple(int(C.nodes[i]) for i in c) for C in blocks for c in C.cands]
    res.update(rows=int(Z.shape[0]), cols=int(Z.shape[1]), nnz=int(Z.nnz))
    # the truth is a world of this candidate system (soundness of the admissibility tests; truth used only here)
    idx = {e: j for j, e in enumerate(cand_nodes)}
    xt = np.zeros(len(cand_nodes), dtype=np.int64)
    for e, m in Counter(lines).items():
        xt[idx[tuple(loc[v] for v in e)]] += m
    Zi = Z.astype(np.int64)
    res["truth_is_a_world"] = bool(np.array_equal(Zi @ xt, R))
    assert res["truth_is_a_world"]
    del xt, idx
    MB.save_json(res, outp)
    res["seeds"] = []
    for seed in MB.SEEDS:
        rng = np.random.default_rng(seed)
        c = rng.random(Z.shape[1])
        t1 = time.perf_counter()
        if Z.shape[1] > MO.MILP_GUARD:
            r = MO.hsolve_guarded(c, Z, R.astype(np.float64), ub, MILP_S)
        else:
            r = EN.hsolve(c, Z, R.astype(np.float64), ub=ub, integer=True, time_limit=MILP_S)
        t_solve = time.perf_counter() - t1
        row = dict(seed=seed, status=r.get("status"), t_solve_s=t_solve, t_total_s=t_build + t_solve)
        x = None
        if "x" in r:
            xi = np.rint(r["x"]).astype(np.int64)
            if (xi >= 0).all() and np.array_equal(Zi @ xi, R):
                x = xi
            else:
                row["status"] = str(row["status"]) + "|unverified"
        row["world"] = x is not None
        if x is not None:
            out_h = Counter()
            for j in np.nonzero(x)[0]:
                out_h[tuple(nodes[i] for i in cand_nodes[j])] += int(x[j])
            row.update(MB.metrics(out_h, truth))
        res["seeds"].append(row)
        MB.save_json(res, outp)
        log(f"seed {seed}: {row['status']} world {row['world']} {t_solve:.1f}s " +
            (f"J {row['jaccard']:.4f} MJ {row['multi_jaccard']:.4f}" if x is not None else ""))
    res["done"] = True
    MB.save_json(res, outp)


def jobs():
    q = CFG.RESULTS / "certification" / "queue" / "jobs.jsonl"
    order = ["crime", "directors", "hosts", "foursquare", "topcs", "dblp"]
    with open(q, "a", encoding="utf-8") as f:
        for ds in order:
            for mode in MB.MODES:
                cmd = [sys.executable, "-I", "-B", "-X", "utf8", str(HERE / "ip_baseline.py"), ds, mode]
                f.write(json.dumps(dict(name=f"mono_{ds}_{mode}", cmd=cmd, w=1, prio=40, group="mono",
                                        timeout=4 * 3600.0, done=str(OUTD / f"{ds}_{mode}.done"))) + "\n")


def main():
    if sys.argv[1] == "jobs":
        jobs()
        return
    OUTD.mkdir(parents=True, exist_ok=True)
    run(sys.argv[1], sys.argv[2])
    (OUTD / f"{sys.argv[1]}_{sys.argv[2]}.done").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    main()
