"""Table IV: certified share under fewer forcing rules (rows No forcing, D1 + exact, HyRange).

Variants (forcing rule set, everything after it identical to HyRange):
  full : rules D1, A, B, P(=E), L  -> HyRange itself (its certified multiset is the reference of the comparison
                                      "certifies the same hyperedges as HyRange")
  d1   : rule D1 only               -> row D1 + exact
  none : no forcing                 -> row No forcing
Pipeline after the forcing = that of pipeline.py, unchanged:
  forcing pipeline.WDForcerTL (900 s, 10 s per rule-L call); residual components (smallest first) with
  candidates.build_comp_wp (Lemma 1 / Lemma 3 filters, cap 300,000 candidates and 60 s per component, rank test on
  <= 2,000 candidates, components reached after 1,200 s from the start not enumerated); per seed 0, 1, 2 the world of
  each component by pipeline.world_of (pinned if the rank test proves one world, else HiGHS with the random objective
  default_rng(seed), MILP 60 s, guarded child above 30,000 columns, world budget 180 s per seed).
followed by the exact certification of certify.py, unchanged (certify_component: enumeration <= 5,000
candidates / 30 s / 1e6 worlds, LP 60 s, IP 30 s per candidate and 900 s per component, refutation search 2 s + 60 s
per candidate and 900 s per component), applied to every residual component that is enumerated, not pinned, and has
a found world in some seed (the worlds of all seeds that found one are passed, as in certify.bench).  A
component with candidates whose world MILP was attempted and failed in every seed gets certify's no-world
procedure (feasibility MILP 1,800 s, LP screening 1,800 s, IP min 120 s / 600 s).
Stop rule (--cert-max-cands N): the certification of a component with more than N candidates is not run, its copies
stay undecided (the share is then a lower bound, dagger in Table IV).  Runs of Table IV:
  fewer_rules.py <ds> <mode> none|d1|full                       for the six benchmarks, except
  fewer_rules.py dblp preserved d1 --cert-max-cands 100000
  fewer_rules.py topcs <mode> none --cert-max-cands 30000 --out-suffix _skipgiant
  fewer_rules.py dblp reduced d1 --out-suffix _full             (no stop rule; 7,349 s of certification)
Ground truth is read only to assert soundness (forced copies, residual, every certified copy) and to score.

usage: python -I -B -X utf8 experiments/fewer_rules.py <ds> <preserved|reduced> <full|d1|none>
              [--cert-max-cands N] [--out-suffix S]
       python -I -B -X utf8 experiments/fewer_rules.py all [--procs 4] [--variants d1,none,full]
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
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

VARIANTS = {"full": ("D1", "A", "B", "P", "L"), "d1": ("D1",), "none": ()}
SPARSE = ["dblp", "topcs", "foursquare", "hosts", "crime", "directors"]     # long jobs first
FORCE_TL, ENUM_BUDGET, WORLD_BUDGET, MILP_TL = 900.0, 1200.0, 180.0, 60.0  # = pipeline defaults
OUTD = CFG.RESULTS / "fewer_rules"
LOGS = CFG.LOGS / "fewer_rules"


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def run(ds, mode, var, cert_max_cands=None, out_suffix=""):
    from hyrange import engine as EN
    from hyrange import forcing as WC
    from hyrange import candidates as WP
    from hyrange import pipeline as MO
    from hyrange import certify as CE
    rules = VARIANTS[var]
    vdir = OUTD / (var + out_suffix)
    outp = vdir / f"{ds}_{mode}.json"
    res = dict(ds=ds, name=MB.DISP[ds], mode=mode, variant=var, rules=list(rules), stage="start",
               cert_max_cands=cert_max_cands,
               force_tl=FORCE_TL, enum_budget=ENUM_BUDGET, world_budget=WORLD_BUDGET, milp_tl=MILP_TL, cap=EN.CAP,
               enum_s_per_comp=EN.ENUM_S, rank_max=EN.RANK_MAX, milp_guard_cols=MO.MILP_GUARD,
               cert_budgets=dict(enum_max_cands=CE.ENUM_MAX_CANDS, enum_s=CE.ENUM_S,
                                 enum_max_worlds=CE.ENUM_MAX_WORLDS, lp_s=CE.LP_S, ip_s=CE.IP_S,
                                 ip_comp_s=CE.IP_COMP_S, dfs_s=CE.DFS_S, dfslp_s=CE.DFSLP_S,
                                 dfs_comp_s=CE.DFS_COMP_S, noworld_find_s=CE.FIND_S, noworld_screen_s=CE.SCREEN_S))
    lines, truth = MB.target(ds, mode)
    nodes = sorted(set(v for e in lines for v in e))
    loc = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    adj, deg = WC.build_wdeg([tuple(loc[v] for v in e) for e in lines], n)
    vol = int(sum(deg))
    res.update(n=n, vol=vol)

    # ---------------- forcing + components (as pipeline.py)
    t_start = time.perf_counter()
    F = MO.WDForcerTL(adj, deg, rules=rules, time_limit=FORCE_TL, copy_adj=False).run()
    t_force = time.perf_counter() - t_start
    forced = Counter({tuple(sorted(int(v) for v in e)): int(m) for e, m in F.forced.items()})
    forced_vol = sum(len(e) * m for e, m in forced.items())
    res.update(stage="forced", t_force_s=t_force, force_complete=bool(F.complete), force_counts=dict(F.counts),
               l_call_stops=getattr(F, "l_call_stops", 0), forced_copies=int(sum(forced.values())),
               forced_vol=forced_vol, forced_vol_share=forced_vol / vol)
    log(f"{ds} {mode} {var}: forcing {t_force:.1f}s, vol share {forced_vol / vol:.4f}")

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
            C = WP.build_comp_wp(cn, F.adj, F.deg, cap=EN.CAP, enum_s=min(EN.ENUM_S, rem))
        comps.append(C)
    t_prep = time.perf_counter() - t1
    assert forced_vol + sum(C.vol for C in comps) == vol
    big = max(comps, key=lambda C: len(C.nodes)) if comps else None
    res.update(stage="prepared", t_components_enum_rank_s=t_prep, comps=len(comps),
               largest_comp_nodes=(len(big.nodes) if big else 0),
               largest_comp_cands=(len(big.cands) if big is not None and big.cands is not None else None),
               largest_comp_enum=(big.enum if big else None),
               cands_max=int(max((len(C.cands) for C in comps if C.cands is not None), default=0)),
               cands_total=int(sum(len(C.cands) for C in comps if C.cands is not None)),
               enum_status=dict(Counter(C.enum for C in comps)),
               comps_not_enumerated=sum(1 for C in comps if C.cands is None),
               vol_not_enumerated=int(sum(C.vol for C in comps if C.cands is None)),
               not_enumerated_nodes=[int(len(C.nodes)) for C in comps if C.cands is None],
               unique_comps=sum(1 for C in comps if C.unique),
               unique_vol=int(sum(C.vol for C in comps if C.unique)))
    log(f"components {len(comps)} in {t_prep:.1f}s: {res['enum_status']}, unique {res['unique_comps']}, "
        f"largest {res['largest_comp_nodes']} nodes / {res['largest_comp_cands']} cands")

    lines_loc = [tuple(loc[v] for v in e) for e in lines]
    ver = WC.verify_forced(F.forced, lines_loc)
    res["verify_forced"] = ver
    assert ver["ok"], ("UNSOUND forcing", ver)
    res["residual_equals_Wplus_of_truth_minus_forced"] = bool(WP.check_residual_wp(F, lines_loc, n))
    assert res["residual_equals_Wplus_of_truth_minus_forced"]
    del lines_loc
    MB.save_json(res, outp)

    # ---------------- worlds, three seeds (as pipeline.py, plus the world vectors kept per component)
    line_mult = Counter(lines)
    worlds = [[] for _ in comps]            # (seed, x) per component, non-pinned found worlds
    attempted = [0] * len(comps)            # seeds in which the world MILP ran
    res["seeds"] = []
    for seed in MB.SEEDS:
        rng = np.random.default_rng(seed)
        tw = time.perf_counter()
        cert = Counter(forced)
        unc = Counter()
        status = Counter()
        uncovered = 0
        for ci, C in enumerate(comps):
            if C.cands is None:
                status["not_enumerated:" + C.enum] += 1
                uncovered += int(C.vol)
                continue
            if not (C.unique and C.xu is not None) and time.perf_counter() - tw > WORLD_BUDGET:
                status["world_budget"] += 1
                uncovered += int(C.vol)
                continue
            x, st = MO.world_of(C, rng, MILP_TL)
            status[st] += 1
            if st != "pinned":
                attempted[ci] += 1
            if x is None:
                uncovered += int(C.vol)
                continue
            if st != "pinned":
                worlds[ci].append((seed, x))
            tgt = cert if st == "pinned" else unc
            for j in np.nonzero(x)[0]:
                tgt[tuple(sorted(int(C.nodes[i]) for i in C.cands[j]))] += int(x[j])
        t_world = time.perf_counter() - tw
        cert_o = Counter({tuple(nodes[i] for i in e): m for e, m in cert.items()})
        unc_o = Counter({tuple(nodes[i] for i in e): m for e, m in unc.items()})
        bad = {e: m for e, m in cert_o.items() if line_mult.get(e, 0) < m}
        assert not bad, ("UNSOUND certified copies", len(bad))
        out_h = cert_o + unc_o
        met = MB.metrics(out_h, truth)
        cvol = sum(len(e) * m for e, m in cert_o.items())
        row = dict(seed=seed, t_world_s=t_world, status=dict(status), uncovered_vol=uncovered,
                   uncovered_vol_share=uncovered / vol, certified_copies=int(sum(cert_o.values())),
                   certified_vol=cvol, certified_vol_share=cvol / vol, certified_sound=True,
                   uncertain_copies=int(sum(unc_o.values())),
                   uncertain_true_copies=int(sum(min(m, line_mult.get(e, 0)) for e, m in unc_o.items())),
                   t_total_s=t_force + t_prep + t_world, **met)
        res["seeds"].append(row)
        MB.save_hyperedges(out_h, vdir / "pred" / f"{ds}_{mode}_s{seed}.json")
        MB.save_json(res, outp)
        log(f"seed {seed}: {t_world:.1f}s {dict(status)} J {met['jaccard']:.4f} MJ {met['multi_jaccard']:.4f} "
            f"cert vol {cvol / vol:.4f} uncovered {uncovered / vol:.4f}")
    res["comps_no_world_attempted"] = sum(1 for ci, C in enumerate(comps) if C.cands is not None and not C.unique
                                          and not worlds[ci] and attempted[ci] > 0)
    res["comps_world_budget_only"] = sum(1 for ci, C in enumerate(comps) if C.cands is not None and not C.unique
                                         and not worlds[ci] and attempted[ci] == 0)

    # ---------------- exact certification (= certify.bench, on the components of this run)
    res["stage"] = "certifying"
    MB.save_json(res, outp)
    t_c0 = time.perf_counter()
    pinned_vol = 0
    certified = Counter(forced)             # local ids; forced + pinned + proved lb
    treated = []
    seeds_unc = [0, 0, 0]
    per_seed = [Counter() for _ in MB.SEEDS]
    res["components"] = []
    noworld_idx = []
    for ci, C in enumerate(comps):
        crec = dict(index=ci, nodes=len(C.nodes), vol=int(C.vol), enum_status=C.enum,
                    ncands=(len(C.cands) if C.cands is not None else None))
        if C.cands is None:
            crec.update(treated=False, why=f"not enumerated ({C.enum})")
            res["components"].append(crec)
            continue
        if C.unique and C.xu is not None:
            pinned_vol += int(C.vol)
            for j in np.nonzero(C.xu)[0]:
                certified[tuple(sorted(int(C.nodes[i]) for i in C.cands[j]))] += int(C.xu[j])
            continue                         # pinned components are not listed one by one (thousands on DBLP)
        if not worlds[ci]:
            crec.update(treated=False, attempted_seeds=attempted[ci],
                        why=("no found world (no-world procedure)" if attempted[ci] else
                             "world search budget reached before this component in every seed"))
            if attempted[ci]:
                noworld_idx.append(ci)
            res["components"].append(crec)
            continue
        xs = [x for _, x in worlds[ci]]
        if cert_max_cands is not None and len(C.cands) > cert_max_cands:
            # stop rule (> 3 h on one thread): this component is not certified; its copies stay undecided
            crec.update(treated=False, why=f"certification not run: {len(C.cands)} candidates > {cert_max_cands} "
                        "(stop rule)",
                        cert_skipped=True, seeds_with_world=[s for s, _ in worlds[ci]])
            for s, x in worlds[ci]:
                k = int(x.sum())
                per_seed[s].update(dict(uncertain_copies=k, undecided=k, cert_skipped_copies=k))
            res["components"].append(crec)
            continue
        t0 = time.perf_counter()
        r = CE.certify_component(C, xs)
        r.update(crec)
        r["treated"] = True
        r["seeds_with_world"] = [s for s, _ in worlds[ci]]
        r["per_seed"] = CE.comp_summary(C, r, xs, None)
        for (s, _), ps in zip(worlds[ci], r["per_seed"]):
            seeds_unc[s] += ps["uncertain_copies"]
            per_seed[s].update(ps)
        for c in r["cands"]:
            if c["lb"] > 0:
                certified[tuple(sorted(int(C.nodes[i]) for i in C.cands[c["j"]]))] += c["lb"]
        r.pop("cands")                       # per-candidate proofs are in certify's format; keep the summary
        res["components"].append(r)
        treated.append((C, r))
        if time.perf_counter() - t0 > 5:
            log(f"comp {ci}: {len(C.nodes)} nodes, {len(C.cands)} cands, {r['method']}, "
                f"{time.perf_counter() - t0:.1f}s")
    res["t_cert_components_s"] = float(sum(r["t_total_s"] for _, r in treated))
    res["t_cert_wall_s"] = time.perf_counter() - t_c0
    res["treated_components"] = len(treated)
    res["cert_methods"] = dict(Counter(r["method"] for _, r in treated))
    res["all_treated_decided"] = all(r["all_decided"] for _, r in treated)
    res["per_seed_cert"] = [dict(seed=s, **{k: int(v) for k, v in per_seed[s].items()}) for s in range(3)]
    for s in range(3):
        assert per_seed[s]["uncertain_copies"] == res["seeds"][s]["uncertain_copies"], \
            ("uncertain copies of treated components != uncertain copies of the worlds", s)
    MB.save_json(res, outp)

    # ---------------- no-world procedure (certify.no_world_component)
    t_n0 = time.perf_counter()
    res["noworld"] = []
    for ci in noworld_idx:
        C = comps[ci]
        log(f"comp {ci}: {len(C.nodes)} nodes, {len(C.cands)} cands, no world: no-world procedure")
        r, world = CE.no_world_component(C)
        r["index"] = ci
        for c in r["certified"]:
            certified[tuple(sorted(int(C.nodes[i]) for i in C.cands[c["j"]]))] += c["lb"]
        r["new_certified_copies"] = int(sum(c["lb"] for c in r["certified"]))
        r["new_certified_vol"] = int(sum(c["k"] * c["lb"] for c in r["certified"]))
        r.pop("lp_bounds", None)
        res["noworld"].append(r)
        MB.save_json(res, outp)
    res["t_noworld_s"] = time.perf_counter() - t_n0

    # ---------------- certified multiset, soundness (truth only here), shares
    cert_o = Counter({tuple(nodes[i] for i in e): int(m) for e, m in certified.items() if m > 0})
    bad = {e: m for e, m in cert_o.items() if line_mult.get(e, 0) < m}
    res["soundness"] = dict(certified_copies=int(sum(cert_o.values())), violations=len(bad))
    assert not bad, ("UNSOUND certified multiset", len(bad), list(bad.items())[:5])
    cv = int(sum(len(e) * m for e, m in cert_o.items()))
    res["certified_vol_before"] = forced_vol + pinned_vol
    res["certified_vol_share_before"] = (forced_vol + pinned_vol) / vol
    res["certified_vol_after"] = cv
    res["certified_vol_share_after"] = cv / vol
    res["certified_copies_after"] = int(sum(cert_o.values()))
    MB.save_hyperedges(cert_o, vdir / "certified" / f"{ds}_{mode}.json")
    res["stage"] = "done"
    res["t_script_s"] = time.perf_counter() - t_start
    MB.save_json(res, outp)
    log(f"done: certified before {res['certified_vol_share_before']:.6f} after {cv / vol:.6f}, treated "
        f"{len(treated)} comps, methods {res['cert_methods']}, cert {res['t_cert_components_s']:.1f}s, "
        f"no-world comps {len(noworld_idx)}")


def run_all(procs, variants):
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = [(ds, mode, v) for v in variants for ds in SPARSE for mode in MB.MODES
            if not (OUTD / v / f"{ds}_{mode}.json").exists()]        # started or finished before: skip
    running = []
    t0 = time.time()
    while jobs or running:
        for r in list(running):
            if r[0].poll() is not None:
                r[1].close()
                running.remove(r)
                log(f"end {r[2]} rc {r[0].returncode} ({time.time() - r[3]:.0f}s)")
        while jobs and len(running) < procs:
            ds, mode, v = jobs.pop(0)
            name = f"{v}_{ds}_{mode}"
            fh = open(LOGS / f"{name}.log", "w", encoding="utf-8")
            p = subprocess.Popen([sys.executable, "-I", "-B", "-X", "utf8", str(Path(__file__)), ds, mode, v],
                                 stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
            running.append((p, fh, name, time.time()))
            log(f"start {name}")
        time.sleep(2)
    log(f"all done in {time.time() - t0:.0f}s")


def main():
    if sys.argv[1] == "all":
        procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
        variants = (sys.argv[sys.argv.index("--variants") + 1].split(",") if "--variants" in sys.argv
                    else ["d1", "none", "full"])
        run_all(procs, variants)
        return
    ds, mode, var = sys.argv[1], sys.argv[2], sys.argv[3]
    cmc = int(sys.argv[sys.argv.index("--cert-max-cands") + 1]) if "--cert-max-cands" in sys.argv else None
    assert var in VARIANTS and mode in MB.MODES
    suf = sys.argv[sys.argv.index("--out-suffix") + 1] if "--out-suffix" in sys.argv else ""
    run(ds, mode, var, cmc, suf)


if __name__ == "__main__":
    main()
