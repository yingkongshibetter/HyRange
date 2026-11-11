"""Random samples of the dense benchmarks with known certain hyperedges F^cert: the inputs of the local certification
(paragraph Local certification: 13 samples of Enron, H.School and P.School with mean degree 6 to 32).

A sample of a dense MARIOH target (preserved setting, bench.target) keeps the first k copies of a uniformly random
permutation of the target's lines (numpy default_rng(1000 + seed)), with k chosen so that the sample's mean node degree
(vol / #nodes that occur in the sample) is closest to the level (ties: largest k).  The sample id <src>_L<level>_s<seed>
fixes the sample; LOCAL_CERT_SAMPLES lists the 13 samples of the experiments, on which exact certification decides every
copy, so that the certified share equals the share of F^cert.

Per sample (`one`), nothing tuned, every limit is the default of pipeline.py / certify.py:
  1. (W, d) of the sample lines (forcing.build_wdeg on local ids);
  2. forcing: pipeline.WDForcerTL, rules D1, A, B, P, L, time limit 900 s (pipeline default), 10 s per rule-L call;
     forcing.verify_forced and candidates.check_residual_wp asserted;
  3. components exactly as pipeline.py (build_comp_wp, cap 300,000, 60 s per component, enumeration budget 1,200 s
     from the start of the forcing), found worlds per seed 0, 1, 2 exactly as pipeline.py (MILP 60 s, world budget 180 s
     per seed, guarded child above 30,000 columns);
  4. exact certification as certify.py: certify_component on every enumerated, non-pinned component with a
     found world (all its found worlds), then no_world_component on the components without a found world;
  5. soundness: every certified copy (forced + pinned + proved lower bounds) is a copy of the sample with at least that
     multiplicity (asserted at every saved stage; a failure raises and stops the driver).
  The driver kills a sample (whole process tree) 300 s after its forcing finished; such a sample is recorded as
  'exact certification incomplete' (it is then not a sample with known F^cert).

usage: python -I -B -X utf8 experiments/dense_samples.py plan
       python -I -B -X utf8 experiments/dense_samples.py one <sid>
       python -I -B -X utf8 experiments/dense_samples.py drive [<sid> ...] [--workers 6] [--wall 300]
              (default: the 13 samples of LOCAL_CERT_SAMPLES)
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402
from hyrange import engine as EN  # noqa: E402
from hyrange import forcing as WC  # noqa: E402
from hyrange import candidates as WP  # noqa: E402
from hyrange import pipeline as MO  # noqa: E402
from hyrange import certify as CX  # noqa: E402

OUTD = CFG.RESULTS / "dense_samples"
SAMP = OUTD / "samples"
LOGD = OUTD / "logs"
PLAN = OUTD / "plan.json"
LOCAL_CERT_SAMPLES = ["enron_L6_s0", "enron_L8_s0", "enron_L8_s1", "enron_L8_s2", "enron_L12_s1", "enron_L16_s2",
                      "hschool_L16_s0", "hschool_L16_s1", "hschool_L16_s2", "hschool_L32_s1",
                      "pschool_L6_s0", "pschool_L6_s1", "pschool_L6_s2"]

# limits: defaults of pipeline.py (read from its argparse defaults, not changed)
FORCE_TL = 900.0
ENUM_BUDGET = 1200.0
WORLD_BUDGET = 180.0
MILP_TL = 60.0
CAP = EN.CAP


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def save_json(obj, path):
    """MB.save_json with retries: on Windows os.replace fails while the driver has the target open for reading."""
    for _ in range(200):
        try:
            MB.save_json(obj, path)
            return
        except PermissionError:
            time.sleep(0.05)
    MB.save_json(obj, path)


def load_json(path):
    for _ in range(200):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.05)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ samples
def sequence(src, seed):
    """The ordered line sequence whose prefixes are the samples of (src, seed)."""
    lines, _ = MB.target(src, "preserved")
    perm = np.random.default_rng(1000 + seed).permutation(len(lines))
    return [lines[i] for i in perm]


def prefix_degrees(seq):
    seen = set()
    vol = 0
    md = np.zeros(len(seq))
    for k, e in enumerate(seq):
        vol += len(e)
        seen.update(e)
        md[k] = vol / len(seen)
    return md                     # md[k-1] = mean degree of the first k lines


def choose_k(md, target):
    diff = np.abs(md - target)
    best = diff.min()
    k = int(np.nonzero(diff <= best + 1e-12)[0].max()) + 1
    return k, float(md[k - 1])


def parse_sid(sid):
    src, lev, seed = sid.rsplit("_", 2)
    assert lev.startswith("L") and seed.startswith("s"), sid
    return src, int(lev[1:]), int(seed[1:])


def plan(sids=None):
    P = load_json(PLAN) if PLAN.exists() else {}
    for sid in (sids or LOCAL_CERT_SAMPLES):
        src, lev, seed = parse_sid(sid)
        k, m = choose_k(prefix_degrees(sequence(src, seed)), lev)
        P[sid] = dict(sid=sid, src=src, level=lev, seed=seed, k=k, planned_mean_degree=m)
        print(sid, k, round(m, 3), flush=True)
    save_json(P, PLAN)


# ------------------------------------------------------------------ one sample
def run_one(sid):
    spec = load_json(PLAN)[sid]
    outp = SAMP / f"{sid}.json"
    t_script = time.perf_counter()
    seq = sequence(spec["src"], spec["seed"])
    S = seq[:spec["k"]]
    nodes = sorted(set(v for e in S for v in e))
    loc = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    lines_loc = [tuple(sorted(loc[v] for v in e)) for e in S]
    mult = Counter(lines_loc)
    adj, deg = WC.build_wdeg(lines_loc, n)
    vol = int(sum(deg))
    sizes = Counter(len(e) for e in lines_loc)
    res = dict(spec, stage="start", n=n, copies=len(lines_loc), distinct=len(mult), vol=vol,
               mean_degree=vol / n, mean_size=vol / len(lines_loc), max_size=max(sizes),
               limits=dict(force_tl=FORCE_TL, l_call_s=MO.WDForcerTL.L_CALL_S, enum_budget=ENUM_BUDGET,
                           world_budget=WORLD_BUDGET, milp_tl=MILP_TL, cap=CAP, enum_s_per_comp=EN.ENUM_S,
                           rank_max=EN.RANK_MAX, milp_guard_cols=MO.MILP_GUARD,
                           certify=dict(enum_max_cands=CX.ENUM_MAX_CANDS, enum_s=CX.ENUM_S, lp_s=CX.LP_S, ip_s=CX.IP_S,
                                   ip_comp_s=CX.IP_COMP_S, dfs_s=CX.DFS_S, dfslp_s=CX.DFSLP_S,
                                   dfs_comp_s=CX.DFS_COMP_S, find_s=CX.FIND_S, screen_s=CX.SCREEN_S)))
    assert abs(res["mean_degree"] - spec["planned_mean_degree"]) < 1e-9

    # ---- forcing (as pipeline.py)
    t_start = time.perf_counter()
    F = MO.WDForcerTL(adj, deg, rules=MO.RULES, time_limit=FORCE_TL, copy_adj=False).run()
    t_force = time.perf_counter() - t_start
    forced = Counter({tuple(sorted(int(v) for v in e)): int(m) for e, m in F.forced.items()})
    forced_vol = sum(len(e) * m for e, m in forced.items())
    ver = WC.verify_forced(F.forced, lines_loc)
    if not ver["ok"]:
        res["verify_forced"] = ver
        save_json(res, outp)
        raise AssertionError(("UNSOUND forcing", sid, ver))
    rok = WP.check_residual_wp(F, lines_loc, n)
    if not rok:
        save_json(res, outp)
        raise AssertionError(("UNSOUND residual", sid))
    res.update(stage="forced", forced_at_epoch=time.time(), t_force_s=t_force, force_complete=bool(F.complete),
               force_counts=dict(F.counts), l_call_stops=F.l_call_stops, l_deadline_stops=F.l_deadline_stops,
               forced_copies=int(sum(forced.values())), forced_vol_share=forced_vol / vol,
               certified_partial_share=forced_vol / vol, verify_forced_ok=True, residual_ok=True)
    save_json(res, outp)
    log(f"{sid}: n {n} copies {len(lines_loc)} md {vol / n:.2f}; forced {forced_vol / vol:.4f} in {t_force:.1f}s")

    cert = Counter(forced)

    def check_and_save(stage, extra_vol=0):
        badc = {e: m for e, m in cert.items() if mult.get(e, 0) < m}
        if badc:
            res["UNSOUND"] = len(badc)
            save_json(res, outp)
            raise AssertionError(("UNSOUND certified copies", sid, len(badc), list(badc.items())[:5]))
        cv = sum(len(e) * m for e, m in cert.items())
        res["stage"] = stage
        res["certified_partial_share"] = cv / vol
        res["certified_partial_copies"] = int(sum(cert.values()))
        save_json(res, outp)

    # ---- components (as pipeline.py)
    t1 = time.perf_counter()
    comp_nodes = EN._components(F.adj, n)
    comp_nodes.sort(key=len)
    comps = []
    for cn in comp_nodes:
        rem = t_start + ENUM_BUDGET - time.perf_counter()
        if rem <= 0:
            C = WP.build_comp_wp(cn, F.adj, F.deg, enumerate_=False)
            C.enum = "budget"
        else:
            C = WP.build_comp_wp(cn, F.adj, F.deg, cap=CAP, enum_s=min(EN.ENUM_S, rem))
        comps.append(C)
    assert forced_vol + sum(C.vol for C in comps) == vol
    res.update(t_components_s=time.perf_counter() - t1, comps=len(comps),
               largest_comp_nodes=max((len(C.nodes) for C in comps), default=0),
               enum_status=dict(Counter(C.enum for C in comps)),
               vol_not_enumerated_share=sum(int(C.vol) for C in comps if C.cands is None) / vol,
               cands_max=int(max((len(C.cands) for C in comps if C.cands is not None), default=0)),
               pinned_comps=sum(1 for C in comps if C.cands is not None and C.unique),
               pinned_vol_share=sum(int(C.vol) for C in comps if C.cands is not None and C.unique) / vol)
    for C in comps:
        if C.cands is not None and C.unique and C.xu is not None:
            for j in np.nonzero(C.xu)[0]:
                cert[tuple(sorted(int(C.nodes[i]) for i in C.cands[j]))] += int(C.xu[j])
    check_and_save("components")

    # ---- found worlds, seeds 0, 1, 2 (as pipeline.py)
    worlds = {ci: [] for ci in range(len(comps))}
    wstat = []
    t_worlds = []
    for seed in MB.SEEDS:
        rng = np.random.default_rng(seed)
        tw = time.perf_counter()
        status = Counter()
        for ci, C in enumerate(comps):
            if C.cands is None:
                status["not_enumerated:" + C.enum] += 1
                continue
            if not (C.unique and C.xu is not None) and time.perf_counter() - tw > WORLD_BUDGET:
                status["world_budget"] += 1
                continue
            x, st = MO.world_of(C, rng, MILP_TL)
            status[st] += 1
            if x is not None and st != "pinned":
                worlds[ci].append(x)
        t_worlds.append(time.perf_counter() - tw)
        wstat.append(dict(status))
        res.update(world_status=wstat, t_worlds_s=t_worlds)
        check_and_save(f"worlds_seed{seed}")

    # ---- exact certification (certify.certify_component), then no-world components
    t2 = time.perf_counter()
    treated, noworld = [], []
    crecs = []
    for ci, C in enumerate(comps):
        if C.cands is None or C.unique:
            continue
        if worlds[ci]:
            treated.append(ci)
        else:
            noworld.append(ci)
    res.update(treated_comps=len(treated), noworld_comps=len(noworld),
               noworld_vol_share=sum(int(comps[ci].vol) for ci in noworld) / vol)
    for ci in sorted(treated, key=lambda c: len(comps[c].cands)):
        C = comps[ci]
        r = CX.certify_component(C, worlds[ci])
        for c in r["cands"]:
            if c["lb"] > 0:
                cert[tuple(sorted(int(C.nodes[i]) for i in C.cands[c["j"]]))] += c["lb"]
        crecs.append(dict(index=ci, kind="treated", nodes=len(C.nodes), vol=int(C.vol), ncands=len(C.cands),
                          worlds=len(worlds[ci]), method=r["method"], all_decided=r["all_decided"],
                          new_certified_vol=int(sum(c["k"] * c["lb"] for c in r["cands"])),
                          t_s=r["t_total_s"]))
        res["cert_components"] = crecs
        check_and_save("cert_treated")
    res["t_cert_treated_s"] = time.perf_counter() - t2
    check_and_save("cert_treated_done")
    t3 = time.perf_counter()
    for ci in noworld:
        C = comps[ci]
        r, world = CX.no_world_component(C)
        for c in r["certified"]:
            cert[tuple(sorted(int(C.nodes[i]) for i in C.cands[c["j"]]))] += c["lb"]
        crecs.append(dict(index=ci, kind="no_world", nodes=len(C.nodes), vol=int(C.vol), ncands=len(C.cands),
                          find_world=r["find_world"], new_certified_vol=int(sum(c["k"] * c["lb"]
                                                                                for c in r["certified"])),
                          t_s=r["t_total_s"]))
        res["cert_components"] = crecs
        check_and_save("cert_noworld")
    res["t_cert_noworld_s"] = time.perf_counter() - t3
    check_and_save("done")
    res["exact_complete"] = True
    res["certified_share"] = res["certified_partial_share"]
    res["t_total_s"] = time.perf_counter() - t_start
    res["t_script_s"] = time.perf_counter() - t_script
    save_json(res, outp)
    log(f"{sid}: done, certified {res['certified_share']:.4f} in {res['t_total_s']:.1f}s")


# ------------------------------------------------------------------ driver
def kill_tree(pid):
    CFG.kill_tree(pid)


def drive(workers, wall, sids):
    P = load_json(PLAN)
    SAMP.mkdir(parents=True, exist_ok=True)
    LOGD.mkdir(parents=True, exist_ok=True)

    def finished(sid):
        p = SAMP / f"{sid}.json"
        if not p.exists():
            return False
        r = load_json(p)
        return r.get("stage") == "done" or r.get("wall_cut", False)
    pend = [s for s in sids if not finished(s)]
    pend.sort(key=lambda s: P[s]["k"])                 # small samples first
    log(f"drive: {len(pend)} pending of {len(sids)}, workers {workers}, wall {wall}s")
    running = {}
    failed = []
    t_drive = time.time()
    while (pend or running) and not failed:
        while pend and len(running) < workers:
            sid = pend.pop(0)
            p = SAMP / f"{sid}.json"
            if p.exists():
                p.unlink()
            lf = open(LOGD / f"{sid}.log", "w", encoding="utf-8")
            pr = subprocess.Popen([*CFG.module_cmd("hyrange.dense_samples"), "one",
                                   sid], stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE))
            running[sid] = (pr, lf, time.time())
        time.sleep(2.0)
        for sid in list(running):
            pr, lf, t_l = running[sid]
            rc = pr.poll()
            p = SAMP / f"{sid}.json"
            if rc is not None:
                lf.close()
                del running[sid]
                if rc != 0:
                    failed.append(sid)
                    log(f"FAILED {sid} rc {rc}")
                else:
                    log(f"done {sid} after {time.time() - t_l:.0f}s")
                continue
            try:
                r = load_json(p) if p.exists() else None
            except (json.JSONDecodeError, OSError):
                r = None
            cut = None
            if r is not None and "forced_at_epoch" in r and time.time() > r["forced_at_epoch"] + wall:
                cut = "wall after forcing"
            elif time.time() > t_l + FORCE_TL + 600 and (r is None or "forced_at_epoch" not in r):
                cut = "forcing did not return"
            if cut:
                kill_tree(pr.pid)
                pr.wait()
                lf.close()
                del running[sid]
                r = load_json(p) if p.exists() else dict(P[sid])
                r.update(wall_cut=True, cut_reason=cut, exact_complete=False, stage_at_cut=r.get("stage"),
                         certified_share=r.get("forced_vol_share"), wall_s=wall,
                         t_total_s=(r.get("t_force_s", 0.0) + wall))
                save_json(r, p)
                log(f"CUT {sid} at stage {r['stage_at_cut']} ({cut}); partial {r.get('certified_partial_share')}")
    if failed:
        for sid in running:
            kill_tree(running[sid][0].pid)
        log(f"STOPPED: failures {failed}")
        raise SystemExit(1)
    log(f"drive finished in {time.time() - t_drive:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wall", type=float, default=300.0)
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    if a.what == "plan":
        plan(a.args or None)
    elif a.what == "one":
        run_one(a.args[0])
    elif a.what == "drive":
        drive(a.workers, a.wall, a.args or LOCAL_CERT_SAMPLES)
    else:
        raise SystemExit("unknown " + a.what)


if __name__ == "__main__":
    main()
