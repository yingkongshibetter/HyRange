"""The CP-SAT sentence of paragraph Same input: the monolithic integer program of experiments/ip_baseline.py solved by OR-Tools CP-SAT instead of HiGHS.

Same candidates, same program, same exact check and same scoring as experiments/ip_baseline.py; only the solve step differs.
experiments/ip_baseline.py has no importable build / check / score functions (everything sits inside its run()), so the build code is
copied verbatim from ip_baseline.run() into build() below; constants (CAP_TOTAL, BUILD_S, MILP_S) and log() are imported
from ip_baseline, and scoring uses bench.metrics as ip_baseline does.

  candidates = all admissible cliques of the whole support (candidates.build_comp_wp per support component, cap 3,000,000
               candidates in total, 600 s in total), exactly as experiments/ip_baseline.py.
  program    = pair rows Z x = W and degree rows = d (all coefficients 1), integer 0 <= x_e <= ub_e,
               minimise c . x with c = numpy default_rng(seed).random(#cols) (the HiGHS vector of experiments/ip_baseline.py), scaled
               to integers for CP-SAT as c_int = rint(c * 10^6) (CP-SAT needs integer objective coefficients).
  solver     = CP-SAT (ortools), max_time_in_seconds = 600, num_workers = 1 (setting a, one thread as every run).  Each solve
               runs in a child process (this script, "child" mode) that loads the model from a cache file, builds
               the CpModelProto, solves and writes x; the parent kills it 600 + 120 s after the solver starts (or
               LOAD_MAX_S after launch if loading never finishes), or when the memory of its process tree passes
               MEM_KILL_GB (24 GB) or the machine's available memory drops below AVAIL_KILL_GB (recorded with the
               peak).  The build itself runs in a separate process ("build" mode) that writes the model cache, so
               the parent holds nothing large while CP-SAT runs.
  check      = the returned x is checked exactly in integer arithmetic (x >= 0, x <= ub, Z x == (W, d) via np.add.at on
               int64) and scored with bench.metrics against the truth (truth used only for the soundness assertion
               that it is a world of the candidate system, and for scoring).

usage: <python with requirements-cpsat.txt> -I -B -X utf8 experiments/cpsat_baseline.py run <ds> <mode> <runs>
           runs = comma list of <setting>:<seed>, setting a = 1 worker: a:0,a:1,a:2
       inputs of the experiments: hosts, topcs (both settings) and dblp reduced
       (child mode is internal)
Writes results/cpsat/<ds>_<mode>_build.json and <ds>_<mode>_w<workers>_s<seed>.json, logs in .../logs/.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import gc  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
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
import ip_baseline as IPB  # noqa: E402

OUTD = CFG.RESULTS / "cpsat"
CACHE = OUTD / "cache"
LOGD = OUTD / "logs"
CAP_TOTAL = IPB.CAP_TOTAL
BUILD_S = IPB.BUILD_S
SOLVE_S = IPB.MILP_S
KILL_EXTRA_S = 120.0
LOAD_MAX_S = 1800.0      # wall cap on loading the model into CP-SAT, before the solver starts
OBJ_SCALE = 10 ** 6
MEM_KILL_GB = float(os.environ.get("HYRANGE_MEM_KILL_GB", "24.0"))
AVAIL_KILL_GB = float(os.environ.get("HYRANGE_AVAIL_KILL_GB", "1.0"))
WORKERS = {"a": 1}
log = IPB.log


def build(ds, mode):
    """Copy of the build part of ip_baseline.run() (same calls, same order, same caps)."""
    from hyrange import engine as EN
    from hyrange import forcing as WC
    from hyrange import candidates as WP
    lines, truth = MB.target(ds, mode)
    nodes = sorted(set(v for e in lines for v in e))
    loc = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    adj, deg = WC.build_wdeg([tuple(loc[v] for v in e) for e in lines], n)
    vol = int(sum(deg))
    res = dict(ds=ds, name=MB.DISP[ds], mode=mode, n=n, vol=vol, cap_total=CAP_TOTAL, build_s=BUILD_S,
               solve_s=SOLVE_S, pairs=int(sum(len(a) for a in adj) // 2))
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
        return res, None
    del adj, deg, comps
    # one global matrix, block diagonal over the support components (rows: pairs then degrees of each block), as
    # sp.block_diag in experiments/ip_baseline.py; all entries are 1, so only the pattern is kept (CSC indptr / int32 indices).
    nrows = sum(C.Zb.shape[0] for C in blocks)
    ncols = total
    ro = 0
    ind_parts, ptr_parts = [], [np.zeros(1, dtype=np.int64)]
    nnz = 0
    for C in blocks:
        Zb = C.Zb.tocsc()
        assert (Zb.data == 1).all()
        ind_parts.append((Zb.indices.astype(np.int64) + ro).astype(np.int32))
        ptr_parts.append(Zb.indptr[1:].astype(np.int64) + nnz)
        nnz += Zb.nnz
        ro += Zb.shape[0]
    indices = np.concatenate(ind_parts)
    indptr = np.concatenate(ptr_parts)
    del ind_parts, ptr_parts
    R = np.concatenate([np.asarray([int(v) for v in C.Rint], dtype=np.int64) for C in blocks])
    ub = np.rint(np.concatenate([C.ub for C in blocks])).astype(np.int64)
    cand_nodes = [tuple(int(C.nodes[i]) for i in c) for C in blocks for c in C.cands]
    del blocks
    gc.collect()
    res.update(rows=int(nrows), cols=int(ncols), nnz=int(nnz))
    M = dict(indptr=indptr, indices=indices, R=R, ub=ub, nrows=nrows, ncols=ncols)
    # the truth is a world of this candidate system (as experiments/ip_baseline.py; truth used only here and for scoring)
    idx = {e: j for j, e in enumerate(cand_nodes)}
    xt = np.zeros(ncols, dtype=np.int64)
    for e, m in Counter(lines).items():
        xt[idx[tuple(loc[v] for v in e)]] += m
    res["truth_is_a_world"] = exact_check(M, xt)
    assert res["truth_is_a_world"]
    del xt, idx
    M.update(cand_nodes=cand_nodes, nodes=nodes, truth=truth)
    return res, M


def exact_check(M, x):
    """Exact integer check of x: x >= 0, x <= ub and Z x == R (int64 accumulation over the nonzero columns)."""
    x = np.asarray(x, dtype=np.int64)
    if x.shape != (M["ncols"],) or (x < 0).any() or (x > M["ub"]).any():
        return False
    nzc = np.nonzero(x)[0]
    s, e = M["indptr"][nzc], M["indptr"][nzc + 1]
    lens = e - s
    pos = np.repeat(s - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(int(lens.sum()))
    rows = M["indices"][pos].astype(np.int64)
    Zx = np.zeros(M["nrows"], dtype=np.int64)
    np.add.at(Zx, rows, np.repeat(x[nzc], lens))
    return bool(np.array_equal(Zx, M["R"]))


def save_cache(M, path):
    """CSR pattern (rows of the program) for the child, plus R and ub."""
    import scipy.sparse as sp
    nnz = len(M["indices"])
    Zc = sp.csc_matrix((np.ones(nnz, dtype=np.int8), M["indices"], M["indptr"]), shape=(M["nrows"], M["ncols"]))
    Zr = Zc.tocsr()
    np.savez(path, indptr=Zr.indptr.astype(np.int64), indices=Zr.indices.astype(np.int32), R=M["R"], ub=M["ub"],
             shape=np.array([M["nrows"], M["ncols"]], dtype=np.int64))
    del Zc, Zr
    gc.collect()
    # what the parent needs for the exact check and the scoring (CSC pattern, candidates as node lists)
    cn = M["cand_nodes"]
    cand_ptr = np.zeros(len(cn) + 1, dtype=np.int64)
    cand_ptr[1:] = np.cumsum([len(e) for e in cn])
    cand_flat = np.fromiter((i for e in cn for i in e), dtype=np.int64, count=int(cand_ptr[-1]))
    np.savez(check_path(path), csc_indptr=M["indptr"], csc_indices=M["indices"], R=M["R"], ub=M["ub"],
             shape=np.array([M["nrows"], M["ncols"]], dtype=np.int64), cand_ptr=cand_ptr, cand_flat=cand_flat,
             nodes=np.asarray(M["nodes"], dtype=np.int64))


def check_path(path):
    return Path(str(path).replace(".npz", "_check.npz"))


def load_check(path):
    z = np.load(check_path(path))
    nrows, ncols = (int(v) for v in z["shape"])
    return dict(indptr=z["csc_indptr"], indices=z["csc_indices"], R=z["R"], ub=z["ub"], nrows=nrows, ncols=ncols,
                cand_ptr=z["cand_ptr"], cand_flat=z["cand_flat"], nodes=z["nodes"])


def child(cache, seed, workers, tl, outx, outj):
    from ortools.sat.python import cp_model
    t0 = time.perf_counter()
    z = np.load(cache)
    nrows, ncols = (int(v) for v in z["shape"])
    indptr = z["indptr"].tolist()
    indices = z["indices"]
    R = z["R"].tolist()
    ub = z["ub"].tolist()
    m = cp_model.CpModel()
    P = m.proto
    for u in ub:
        v = P.variables.add()
        v.domain.extend([0, u])
    for i in range(nrows):
        s, e = indptr[i], indptr[i + 1]
        lin = P.constraints.add().linear
        lin.vars.extend(indices[s:e].tolist())
        lin.coeffs.extend([1] * (e - s))
        lin.domain.extend([R[i], R[i]])
    c = np.random.default_rng(seed).random(ncols)
    ci = np.rint(c * OBJ_SCALE).astype(np.int64)
    P.objective.vars.extend(range(ncols))
    P.objective.coeffs.extend(ci.tolist())
    del z, indices, indptr, R, ub, c, ci
    gc.collect()
    t_load = time.perf_counter() - t0
    print(f"model loaded in {t_load:.1f}s ({nrows} rows, {ncols} cols)", flush=True)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = int(workers)
    solver.parameters.max_time_in_seconds = float(tl)
    solver.parameters.log_search_progress = True
    solver.parameters.log_to_stdout = True
    Path(str(outj).replace("_child.json", "_solving")).write_text("solving", encoding="utf-8")
    t1 = time.perf_counter()
    st = solver.solve(m)
    t_solve = time.perf_counter() - t1
    name = solver.status_name(st)
    out = dict(cp_status=name, t_load_s=t_load, t_solve_s=t_solve, solver_wall_s=solver.wall_time,
               objective=None, best_bound=None)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        x = np.asarray(list(solver.response_proto.solution), dtype=np.int64)
        np.save(outx, x)
        out["objective"] = float(solver.objective_value)
        out["best_bound"] = float(solver.best_objective_bound)
    Path(outj).write_text(json.dumps(out), encoding="utf-8")


def run_child(cache, seed, workers, tag):
    outx = CACHE / f"{tag}_x.npy"
    outj = CACHE / f"{tag}_child.json"
    mark = CACHE / f"{tag}_solving"
    for p in (outx, outj, mark):
        if p.exists():
            p.unlink()
    cmd = [sys.executable, "-I", "-B", "-X", "utf8", str(Path(__file__).resolve()), "child", str(cache), str(seed),
           str(workers), str(SOLVE_S), str(outx), str(outj)]
    r = monitor(cmd, LOGD / f"{tag}.log", LOAD_MAX_S, mark)
    r["child_wall_s"] = r.pop("wall_s")
    if not r["killed"] and outj.exists():
        r.update(json.loads(outj.read_text(encoding="utf-8")))
        if outx.exists():
            r["x"] = np.load(outx)
    return r


def monitor(cmd, logpath, wall_limit, mark=None):
    """Run cmd, poll the memory of its process tree every second, kill the tree past the memory thresholds or past
    wall_limit; once the marker file exists (the child is about to call the solver), the limit becomes the time the
    marker appeared + SOLVE_S + KILL_EXTRA_S."""
    import psutil
    logf = open(logpath, "w", encoding="utf-8")
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    pp = psutil.Process(p.pid)
    peak_rss = peak_priv = 0.0
    min_avail = 1e9
    killed = None
    t_mark = None
    tree = [pp]
    while p.poll() is None:
        if mark is not None and t_mark is None and mark.exists():
            t_mark = time.perf_counter()
            wall_limit = t_mark - t0 + SOLVE_S + KILL_EXTRA_S
        # on Windows the venv python.exe is a launcher that starts the base interpreter as its child: sum the tree
        try:
            tree = [pp] + pp.children(recursive=True)
        except psutil.Error:
            pass
        rss = priv = 0.0
        for q in tree:
            try:
                mi = q.memory_info()
                rss += mi.rss / 2 ** 30
                priv += getattr(mi, "private", mi.rss) / 2 ** 30
            except psutil.Error:
                pass
        peak_rss = max(peak_rss, rss)
        peak_priv = max(peak_priv, priv)
        avail = psutil.virtual_memory().available / 2 ** 30
        min_avail = min(min_avail, avail)
        el = time.perf_counter() - t0
        if max(peak_rss, peak_priv) > MEM_KILL_GB:
            killed = "out of memory (process > %.0f GB)" % MEM_KILL_GB
        elif avail < AVAIL_KILL_GB:
            killed = "out of memory (machine available < %.1f GB)" % AVAIL_KILL_GB
        elif el > wall_limit:
            killed = "killed" if t_mark is not None or mark is None else "killed while loading the model"
        if killed:
            for q in reversed(tree):
                try:
                    q.kill()
                except psutil.Error:
                    pass
            p.wait()
            break
        time.sleep(1.0)
    t_all = time.perf_counter() - t0
    logf.close()
    return dict(wall_s=t_all, peak_rss_gb=peak_rss, peak_private_gb=peak_priv, min_machine_available_gb=min_avail,
                returncode=p.returncode, killed=killed,
                t_solver_started_after_s=(None if t_mark is None else t_mark - t0))


def run(ds, mode, runs):
    """Build in a separate process (so that this parent stays small while CP-SAT runs), then one child per solve."""
    import ortools
    OUTD.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    LOGD.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{ds}_{mode}.npz"
    bj = OUTD / f"{ds}_{mode}_build.json"
    if not (cache.exists() and check_path(cache).exists() and bj.exists()):
        cmd = [sys.executable, "-I", "-B", "-X", "utf8", str(Path(__file__).resolve()), "build", ds, mode]
        mon = monitor(cmd, LOGD / f"{ds}_{mode}_build.log", BUILD_S + 3600.0)
        log(f"{ds} {mode}: build process {mon}")
        if not bj.exists():
            MB.save_json(dict(ds=ds, mode=mode, built=False, build_process=mon,
                              note="the build process ended before writing its result"), bj)
        res = json.loads(bj.read_text(encoding="utf-8"))
        res.update(build_process={k: v for k, v in mon.items() if k != "proc"})
        MB.save_json(res, bj)
    res = json.loads(bj.read_text(encoding="utf-8"))
    if not res.get("built") or not cache.exists():
        log(f"{ds} {mode}: no model (built={res.get('built')}), no solve")
        return
    truth = MB.target(ds, mode)[1]
    for item in runs:
        setting, seed = item.split(":")
        seed = int(seed)
        workers = WORKERS[setting]
        tag = f"{ds}_{mode}_w{workers}_s{seed}"
        log(f"{tag}: solving")
        r = run_child(cache, seed, workers, tag)
        row = dict(ds=ds, name=res["name"], mode=mode, setting=setting, workers=workers, seed=seed,
                   obj_scale=OBJ_SCALE, solver="CP-SAT", ortools=ortools.__version__,
                   candidates=res["candidates"], rows=res["rows"], cols=res["cols"], nnz=res["nnz"],
                   t_build_s=res["t_build_s"])
        x = r.pop("x", None)
        row.update(r)
        if r["killed"]:
            status = r["killed"]
        else:
            status = {"OPTIMAL": "optimal", "FEASIBLE": "feasible", "UNKNOWN": "no solution",
                      "INFEASIBLE": "infeasible", "MODEL_INVALID": "model invalid"}.get(r.get("cp_status"),
                                                                                         f"child error rc={r['returncode']}")
        row["status"] = status
        row["t_total_s"] = res["t_build_s"] + r["child_wall_s"]
        row["world"] = False
        row["exact_check"] = None
        if x is not None:
            M = load_check(cache)
            ok = exact_check(M, x)
            row["exact_check"] = ok
            if ok:
                row["world"] = True
                out_h = Counter()
                cp, cf, nd = M["cand_ptr"], M["cand_flat"], M["nodes"]
                for j in np.nonzero(x)[0]:
                    out_h[tuple(int(nd[i]) for i in cf[cp[j]:cp[j + 1]])] += int(x[j])
                row.update(MB.metrics(out_h, truth))
            del M
            gc.collect()
        MB.save_json(row, OUTD / f"{tag}.json")
        log(f"{tag}: {status} world {row['world']} load {r.get('t_load_s', float('nan')):.1f}s "
            f"solve {r.get('t_solve_s', float('nan')):.1f}s child {r['child_wall_s']:.1f}s peak {peak(r):.2f} GB "
            + (f"J {row['jaccard']:.4f} MJ {row['multi_jaccard']:.4f}" if row["world"] else ""))


def build_main(ds, mode):
    import ortools
    OUTD.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    res, M = build(ds, mode)
    res["ortools"] = ortools.__version__
    bj = OUTD / f"{ds}_{mode}_build.json"
    MB.save_json(res, bj)
    if M is None:
        return
    t = time.perf_counter()
    save_cache(M, CACHE / f"{ds}_{mode}.npz")
    res["t_cache_write_s"] = time.perf_counter() - t
    MB.save_json(res, bj)


def peak(r):
    return max(r["peak_rss_gb"], r["peak_private_gb"])


def main():
    if sys.argv[1] == "child":
        child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6], sys.argv[7])
        return
    if sys.argv[1] == "build":
        build_main(sys.argv[2], sys.argv[3])
        return
    run(sys.argv[2], sys.argv[3], sys.argv[4].split(","))


if __name__ == "__main__":
    main()
