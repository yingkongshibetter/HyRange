"""Rule ablation of the forcing on MARIOH's benchmark (Fig. 4; paragraphs Rules E and L, Inside the 2-core, and the
forced shares of paragraph Dense hypergraphs).

Per dataset and setting (input = the view (W, d) of the target lines, bench.target), the forced share of the
volume vol = sum_e |e| = sum_u d_u with the growing rule sets
    {D1}, {D1, A, B}, {D1, A, B, P}, {D1, A, B, P, L}      (json rule P = the pair-core rule, rule E in the experiments)
using the forcer of pipeline.py unchanged: pipeline.WDForcerTL (forcing.WDForcer + 10 s wall budget per rule-L call),
wall limit 900 s per run, default leaf / degree budgets.  Also:
  * 2-core of the target hypergraph: repeatedly remove one copy of a hyperedge that contains a node of degree one (in
    the current hypergraph); the volume removed is the volume outside the 2-core.  Asserted equal to the {D1} forced
    volume (D1 on (W, d) forces exactly the one hyperedge through a degree-one node), and the two multisets are
    compared as well.
  * Every forced copy of every rule set is checked against the target lines (forcing.verify_forced: a target line
    with at least that multiplicity); a failure raises.
  * The full rule set is compared with the forced multiset that pipeline.py wrote (forced/<ds>_<mode>_forced.json).
The target lines are used only to build the view and to check.

usage: python -I -B -X utf8 experiments/rule_ablation.py <ds> <mode>         -> results/bench/ablation/<ds>_<mode>.json
       python -I -B -X utf8 experiments/rule_ablation.py all [--procs 6]      runs the 20 jobs (<= procs processes), then summary
       python -I -B -X utf8 experiments/rule_ablation.py summary              -> ablation_summary.json and stdout
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402

RULESETS = (("D1",), ("D1", "A", "B"), ("D1", "A", "B", "P"), ("D1", "A", "B", "P", "L"))
RULE_ORDER = ("D1", "A", "B", "P", "L")
FORCE_TL = 900.0                      # pipeline.py --force-tl default
ABL = MB.OUT / "ablation"
LOGS = CFG.LOGS / "bench"
# long jobs first
ORDER = ["eu", "dblp", "hschool", "pschool", "enron", "topcs", "foursquare", "hosts", "crime", "directors"]


def two_core_outside(lines_loc, n):
    """Peel: while some node has degree one, remove the one hyperedge copy containing it.  Returns the removed
    multiset (the part outside the 2-core)."""
    H = Counter(lines_loc)
    inc = [[] for _ in range(n)]
    d = [0] * n
    for e, m in H.items():
        for x in e:
            d[x] += m
            inc[x].append(e)
    removed = Counter()
    stack = [u for u in range(n) if d[u] == 1]
    while stack:
        u = stack.pop()
        if d[u] != 1:
            continue
        e = next(f for f in inc[u] if H[f] > 0)
        assert H[e] == 1, "degree-one node in a repeated hyperedge"
        H[e] -= 1
        removed[e] += 1
        for x in e:
            d[x] -= 1
            if d[x] == 1:
                stack.append(x)
    return removed


def one(ds, mode):
    from hyrange import forcing as WC
    from hyrange import pipeline as MO
    lines, truth = MB.target(ds, mode)
    nodes = sorted(set(v for e in lines for v in e))
    loc = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    lines_loc = [tuple(loc[v] for v in e) for e in lines]
    adj, deg = WC.build_wdeg(lines_loc, n)
    vol = int(sum(deg))
    out = dict(ds=ds, name=MB.DISP[ds], mode=mode, n=n, lines=len(lines), vol=vol, force_tl=FORCE_TL,
               l_call_s=MO.WDForcerTL.L_CALL_S, forcer="pipeline.WDForcerTL", rows=[])
    # 2-core of the target hypergraph (truth used only for this reference quantity)
    t0 = time.perf_counter()
    rem = two_core_outside(lines_loc, n)
    out["two_core"] = dict(outside_copies=int(sum(rem.values())),
                           outside_vol=int(sum(len(e) * m for e, m in rem.items())), seconds=time.perf_counter() - t0)
    out["two_core"]["outside_vol_share"] = out["two_core"]["outside_vol"] / vol
    print(time.strftime("%H:%M:%S ") + f"{ds} {mode}: n {n}, vol {vol}, outside 2-core vol "
          f"{out['two_core']['outside_vol_share']:.6f}", flush=True)
    for rules in RULESETS:
        t0 = time.perf_counter()
        F = MO.WDForcerTL(adj, deg, rules=rules, time_limit=FORCE_TL, copy_adj=True).run()
        t = time.perf_counter() - t0
        ver = WC.verify_forced(F.forced, lines_loc)
        if not ver["ok"]:
            MB.save_json(out, ABL / f"{ds}_{mode}.json")
            raise AssertionError(("UNSOUND forcing", ds, mode, rules, ver))
        fv = int(sum(len(e) * m for e, m in F.forced.items()))
        row = dict(rules=list(rules), forced_copies=int(sum(F.forced.values())), forced_distinct=len(F.forced),
                   forced_vol=fv, forced_vol_share=fv / vol,
                   copies_per_rule={r: int(F.counts.get(r, 0)) for r in RULE_ORDER if r in rules},
                   seconds=t, complete=bool(F.complete), rounds=int(F.rounds), l_runs=int(F.l_runs),
                   l_skipped=int(F.l_skipped), l_call_stops=int(F.l_call_stops),
                   l_deadline_stops=int(F.l_deadline_stops), verified_against_truth=True,
                   invalid_copies=int(ver["invalid"]))
        out["rows"].append(row)
        if rules == ("D1",):
            row["equals_two_core_peel_multiset"] = (Counter({tuple(sorted(e)): m for e, m in F.forced.items()}) ==
                                                    Counter({tuple(sorted(e)): m for e, m in rem.items()}))
            assert fv == out["two_core"]["outside_vol"], ("D1 share != outside-2-core share", fv,
                                                          out["two_core"]["outside_vol"])
            out["d1_share_equals_two_core_share"] = True
        if rules == RULESETS[-1]:
            fp = MB.OUT / "forced" / f"{ds}_{mode}_forced.json"
            if fp.exists():
                ref = MB.load_hyperedges(fp)
                mine = Counter({tuple(nodes[i] for i in sorted(e)): int(m) for e, m in F.forced.items()})
                row["equals_pipeline_forced"] = (mine == ref)
                ou = json.load(open(MB.OUT / "ours" / f"{ds}_{mode}.json", encoding="utf-8"))
                row["pipeline_forced_vol_share"] = ou["forced_vol_share"]
                row["pipeline_l_call_stops"] = ou.get("l_call_stops")
        print(time.strftime("%H:%M:%S ") + f"  {'+'.join(rules):12s} share {fv / vol:.6f} copies "
              f"{row['forced_copies']} {row['copies_per_rule']} {t:.1f}s complete {F.complete} "
              f"lstops {F.l_call_stops}", flush=True)
        MB.save_json(out, ABL / f"{ds}_{mode}.json")
    out["done"] = True
    MB.save_json(out, ABL / f"{ds}_{mode}.json")


def run_all(procs):
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = [(ds, mode) for ds in ORDER for mode in MB.MODES]
    running = []
    while jobs or running:
        for r in list(running):
            if r[0].poll() is not None:
                r[1].close()
                running.remove(r)
                print(time.strftime("%H:%M:%S ") + f"end {r[2]} rc {r[0].returncode}", flush=True)
        while jobs and len(running) < procs:
            ds, mode = jobs.pop(0)
            fh = open(LOGS / f"ablation_{ds}_{mode}.log", "w", encoding="utf-8")
            p = subprocess.Popen([sys.executable, "-I", "-B", "-X", "utf8", str(Path(__file__)), ds, mode],
                                 stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE))
            running.append((p, fh, f"{ds}_{mode}"))
            print(time.strftime("%H:%M:%S ") + f"start {ds}_{mode}", flush=True)
        time.sleep(1)
    summary()


def summary():
    recs = []
    for mode in MB.MODES:
        for ds in MB.DATASETS:
            p = ABL / f"{ds}_{mode}.json"
            r = json.load(open(p, encoding="utf-8")) if p.exists() else None
            if not r or not r.get("done"):
                raise SystemExit(f"missing or unfinished: {p}")
            recs.append(r)
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               rule_sets=[list(r) for r in RULESETS],
               note=("Forced share of the volume vol = sum_e |e| per rule set; json rule 'P' = pair-core rule (rule E "
                     "in the experiments).  Forcer pipeline.WDForcerTL, wall limit 900 s per run, 10 s per rule-L call.  "
                     "two_core_outside_share = volume removed by peeling hyperedges at degree-one nodes of the target."),
               all_forced_copies_true=all(row["verified_against_truth"] for r in recs for row in r["rows"]),
               all_d1_equals_two_core=all(r.get("d1_share_equals_two_core_share") for r in recs),
               all_full_set_equals_pipeline=all(r["rows"][-1].get("equals_pipeline_forced") for r in recs),
               records=[])
    for r in recs:
        out["records"].append(dict(
            ds=r["ds"], name=r["name"], mode=r["mode"], vol=r["vol"], lines=r["lines"],
            two_core_outside_share=r["two_core"]["outside_vol_share"],
            two_core_outside_copies=r["two_core"]["outside_copies"],
            d1_equals_two_core=r["d1_share_equals_two_core_share"],
            d1_equals_two_core_multiset=r["rows"][0]["equals_two_core_peel_multiset"],
            shares={"+".join(row["rules"]): row["forced_vol_share"] for row in r["rows"]},
            forced_copies={"+".join(row["rules"]): row["forced_copies"] for row in r["rows"]},
            complete={"+".join(row["rules"]): row["complete"] for row in r["rows"]},
            full_set_equals_pipeline_forced=r["rows"][-1].get("equals_pipeline_forced"),
            rows=r["rows"]))
    MB.save_json(out, MB.OUT / "ablation_summary.json")
    R = {(r["ds"], r["mode"]): r for r in out["records"]}
    full = "+".join(RULESETS[-1])
    L = ["Forced share of the volume (%) with growing rule sets (rule P = rule E of the experiments); the share of {D1} is "
         "the volume outside the 2-core (asserted). Fig. 4 and the 2-core shares: experiments/fig4_rule_steps.py.", "",
         "| Setting | Dataset | outside the 2-core | {D1} | {D1,A,B} | {D1,A,B,E} | {D1,A,B,E,L} |",
         "|---|---|---|---|---|---|---|"]
    for mode in MB.MODES:
        for ds in MB.SIX:
            r = R[(ds, mode)]
            L.append(f"| {mode} | {r['name']} | {MB.pct1(r['two_core_outside_share'])} | " +
                     " | ".join(MB.pct1(r["shares"]["+".join(x)]) for x in RULESETS) + " |")
    dense_d1 = [R[(d, m)]["shares"]["D1"] for d in MB.DENSE for m in MB.MODES]
    dense_all = [R[(d, "preserved")]["shares"][full] for d in MB.DENSE]
    L += ["", f"Dense hypergraphs ({', '.join(MB.DISP[d] for d in MB.DENSE)}): rule D1 forces at most "
          f"{MB.pct1(max(dense_d1))}% of the volume; all rules force {MB.pct1(min(dense_all))}% to "
          f"{MB.pct1(max(dense_all))}% in the preserved setting."]
    print("\n".join(L))
    print(json.dumps({k: out[k] for k in ("all_forced_copies_true", "all_d1_equals_two_core",
                                           "all_full_set_equals_pipeline")}))


def main():
    if sys.argv[1] == "all":
        procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 6
        run_all(procs)
    elif sys.argv[1] == "summary":
        summary()
    else:
        one(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
