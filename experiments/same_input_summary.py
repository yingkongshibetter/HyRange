"""Summaries results/certification/cert_exact.json (Tables II, III and VI; paragraph Exact certification) and
same_input.json (Table III, rows IP and D1 + MARIOH; paragraph Same input), on the six benchmarks.  Reads only the
records written by certify.py (experiments/exact_certification.py), experiments/ip_baseline.py,
experiments/d1_marioh.py and the benchmark summaries; computes nothing new except aggregates.

usage: python -I -B -X utf8 experiments/same_input_summary.py cert | same | tables | all
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402

CERT_DIR = CFG.RESULTS / "certification"
CE = CERT_DIR / "cert_exact"
METRICS = ("jaccard", "multi_jaccard", "exact_recall", "precision")


def jl(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n=len(v),
                values=[float(x) for x in v])


# ------------------------------------------------------------------ exact certification
def comp_rows(rec):
    out = []
    for c in rec.get("components", []):
        if not c.get("treated"):
            continue
        out.append(dict(index=c["index"], nodes=c["nodes"], ncands=c["ncands"], vol=c["vol"], method=c["method"],
                        how=c.get("how_counts"), enum=c.get("enum"), t=c.get("t_total_s")))
    return out


def cert_bench_row(ds, mode):
    ou = jl(MB.OUT / "ours" / f"{ds}_{mode}.json")
    if ou is None:
        return None
    base = dict(ds=ds, name=MB.DISP[ds], mode=mode, residual_components=ou["comps"],
                forced_vol_share=ou["forced_vol_share"], largest_comp_nodes=ou.get("largest_comp_nodes"))
    s0 = ou["seeds"][0] if ou.get("seeds") else {}
    if ou["comps"] == 0:
        base.update(status="no residual component (everything forced)", treated_components=0, pinned_components=0,
                    certified_vol_share_before=s0.get("certified_vol_share", 1.0),
                    certified_vol_share_after=s0.get("certified_vol_share", 1.0), t_components_s=0.0)
        return base
    r = jl(CE / f"bench_{ds}_{mode}.json")
    if not r or not r.get("done"):
        base.update(status="MISSING")
        return base
    comps = comp_rows(r)
    base.update(status="treated", treated_components=r["treated_components"],
                pinned_components=sum(1 for c in r["components"] if not c.get("treated")
                                      and "pinned" in c.get("why", "")),
                component_methods=dict(Counter(c["method"] for c in comps)),
                single_world_components=sum(1 for c in comps if c["method"] == "enumeration"
                                            and (c.get("enum") or {}).get("worlds") == 1),
                certified_vol_share_before=r["certified_vol_share_before"],
                certified_vol_share_after=r["certified_vol_share_after"],
                per_seed=r["per_seed"], soundness=r["soundness"], t_components_s=sum(c["t"] or 0 for c in comps),
                t_script_s=r["t_script_s"], components=comps)
    return base


def cert():
    bench = [x for x in (cert_bench_row(ds, mode) for mode in MB.MODES for ds in MB.SIX) if x is not None]
    sound = all((r.get("soundness") or {}).get("violations", 0) == 0 for r in bench if r.get("status") == "treated")
    out = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        method=("Per treated component (enumerated, not pinned by the rank test): for each candidate e of the found "
                "worlds (seeds 0-2), lb(e) proved by exhaustive integer enumeration of all worlds (enum), by an "
                "LP-dual bound evaluated in exact rationals (lp: engine.best_rig), or by an exhaustive backtracking "
                "refutation of x_e <= ub-1 (dfs); ub(e) = min over exactly checked witness worlds. Copies of the "
                "found world: certain (first lb), provably not certain (beyond ub), undecided (between)."),
        budgets=dict(enum_s=30, enum_max_worlds=1_000_000, enum_max_cands=5000, lp_s=60, ip_s=30, ip_comp_s=900,
                     dfs_s=20, dfs_comp_s=900),
        all_new_certified_copies_true=sound, benchmark=bench)
    MB.save_json(out, CERT_DIR / "cert_exact.json")
    return out


# ------------------------------------------------------------------ same input (IP, D1 + MARIOH)
def same():
    S = jl(MB.OUT / "summary.json")
    base = {(r["ds"], r["mode"]): r for r in S["records"]}
    d1 = jl(CERT_DIR / "d1_marioh" / "summary.json")
    d1m = {(r["ds"], r["mode"]): r for r in (d1 or {}).get("records", [])}
    recs = []
    for mode in MB.MODES:
        for ds in MB.SIX:
            b = base.get((ds, mode))
            if b is None or "ours" not in b:
                continue
            ou = b["ours"]
            ours = dict(agg={m: ou["agg"][m] for m in METRICS}, t_s=ou["t_force_s"] + ou["t_rest_s"])
            mo = jl(CERT_DIR / "mono" / f"{ds}_{mode}.json")
            mono = None
            if mo is not None:
                sd = mo.get("seeds", [])
                mono = dict(built=mo["built"], build_fail=mo.get("build_fail"), candidates=mo.get("candidates"),
                            candidates_before_failure=mo.get("candidates_before_failure"),
                            rows=mo.get("rows"), t_build_s=mo["t_build_s"], done=bool(mo.get("done")) or not mo["built"],
                            seeds=[dict(seed=x["seed"], status=x["status"], world=x["world"],
                                        t_solve_s=x["t_solve_s"], t_total_s=x["t_total_s"],
                                        **{m: x.get(m) for m in METRICS}) for x in sd],
                            worlds=sum(1 for x in sd if x["world"]),
                            agg={m: agg([x.get(m, 0.0) if x["world"] else 0.0 for x in sd]) for m in METRICS},
                            t_total_s=agg([x["t_total_s"] for x in sd]) if sd else dict(mean=mo["t_build_s"],
                                                                                         std=0.0))
            dm = d1m.get((ds, mode))
            d1r = None
            if dm is not None:
                d1r = dict(d1_forced_vol_share=dm["d1"]["forced_vol_share"], seeds_run=len(dm["seeds"]),
                           agg={m: dm["agg"].get(m) for m in METRICS}, t_total_s=dm["agg"].get("t_total_s"))
            recs.append(dict(ds=ds, name=MB.DISP[ds], mode=mode, metric=MB.PAPER_METRIC[mode], ours=ours, mono=mono,
                             d1_marioh=d1r))
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), records=recs)
    MB.save_json(out, CERT_DIR / "same_input.json")
    return out


# ------------------------------------------------------------------ tables (stdout)
def tables():
    L = []
    C = jl(CERT_DIR / "cert_exact.json")
    H = jl(MB.OUT / "hybrid_summary.json")
    hy = {(r["ds"], r["mode"]): r for r in (H or {}).get("records", [])}
    if C:
        B = {(r["ds"], r["mode"]): r for r in C["benchmark"]}
        L += ["### Table II, row (W,d): certified share of the volume (%) after exact certification", "",
              "| Setting | " + " | ".join(MB.DISP[d] for d in MB.SIX) + " |", "|---|" + "---|" * len(MB.SIX)]
        for mode in MB.MODES:
            L.append(f"| {mode} | " + " | ".join(MB.pct1(B[(d, mode)].get("certified_vol_share_after"))
                                                 if (d, mode) in B else "-" for d in MB.SIX) + " |")
        L += ["", "### Table VI: exact certification on the benchmarks (copies: mean per seed)", "",
              "| Setting | Dataset | Comp. | Certified before (%) | after (%) | Certain | Ambiguous |",
              "|---|---|---|---|---|---|---|"]
        for mode in MB.MODES:
            for d in ("hosts", "dblp", "topcs"):
                r = B.get((d, mode))
                if not r or r.get("status") != "treated":
                    continue
                ps = r["per_seed"]
                mean = (lambda k: sum(s[k] for s in ps) / len(ps))
                f = (lambda v: f"{v:g}" if v == int(v) else f"{v:.1f}")
                L.append(f"| {mode} | {r['name']} | {r['residual_components']} | "
                         f"{100 * r['certified_vol_share_before']:.2f} | {100 * r['certified_vol_share_after']:.2f} | "
                         f"{f(mean('certain'))} | {f(mean('not_certain'))} |")
        tr = [r for r in C["benchmark"] if r.get("status") == "treated"]
        meth = Counter()
        for r in tr:
            meth.update(r["component_methods"])
        enum_n = meth.get("enumeration", 0)
        lp_n = sum(v for k, v in meth.items() if k.startswith("lp") or k.startswith("dfs"))
        wit_n = meth.get("witness worlds (nothing certain)", 0)
        und = max((s["undecided"] for r in tr for s in r["per_seed"]), default=0)
        L += ["", f"Residual components: at most {max(r['residual_components'] for r in C['benchmark'])}, of at most "
              f"{max((r.get('largest_comp_nodes') or 0) for r in C['benchmark'])} nodes; rank test settles "
              f"{sum(r.get('pinned_components', 0) for r in C['benchmark'])}; exact certification treats "
              f"{sum(r['treated_components'] for r in tr)}: enumeration {enum_n} (with a single world: "
              f"{sum(r.get('single_world_components', 0) for r in tr)}), dual bound {lp_n}, witness worlds {wit_n}; "
              f"undecided copies {und}.", ""]
        if hy:
            L += ["### Table III, row HyRange + cert.: time per seed (s) = HyRange + exact certification of the "
                  "worlds of all three seeds", "",
                  "| Setting | " + " | ".join(MB.DISP[d] for d in MB.SIX) + " |", "|---|" + "---|" * len(MB.SIX)]
            for mode in MB.MODES:
                cells = []
                for d in MB.SIX:
                    if (d, mode) not in hy or (d, mode) not in B:
                        cells.append("-")
                        continue
                    cells.append(MB.sec1(hy[(d, mode)]["agg"]["t_total_s"]["mean"] + (B[(d, mode)].get("t_components_s")
                                                                                     or 0.0)))
                L.append(f"| {mode} | " + " | ".join(cells) + " |")
            if ("dblp", "preserved") in B:
                L.append(f"\nDBLP, preserved: exact certification {MB.sec1(B[('dblp', 'preserved')]['t_components_s'])} s.")
            L.append("")
    S = jl(CERT_DIR / "same_input.json")
    if S:
        R = {(r["ds"], r["mode"]): r for r in S["records"]}
        for mode in MB.MODES:
            k = MB.PAPER_METRIC[mode]
            L += [f"### Table III, rows D1 + MARIOH and IP, {mode} "
                  f"({'multi-Jaccard' if k == 'multi_jaccard' else 'Jaccard'} x100, mean ± std; time per seed in s)", "",
                  "| | " + " | ".join(MB.DISP[d] for d in MB.SIX) + " |", "|---|" + "---|" * len(MB.SIX)]
            acc_d, acc_i, t_d, t_i = [], [], [], []
            for d in MB.SIX:
                r = R.get((d, mode))
                dm = r and r["d1_marioh"]
                m = r and r["mono"]
                acc_d.append(f"{MB.pct1(dm['agg'][k]['mean'])} ± {100 * dm['agg'][k]['std']:.1f}"
                             if dm and dm["agg"].get(k) else "-")
                t_d.append(MB.sec1(dm["t_total_s"]["mean"]) if dm and dm.get("t_total_s") else "-")
                acc_i.append(f"{MB.pct1(m['agg'][k]['mean'])} ± {100 * m['agg'][k]['std']:.1f}"
                             if m and m["worlds"] == 3 else "-")
                t_i.append(MB.sec1(m["t_total_s"]["mean"]) if m else "-")
            L += ["| D1 + MARIOH | " + " | ".join(acc_d) + " |", "| IP | " + " | ".join(acc_i) + " |",
                  "| time D1 + MARIOH | " + " | ".join(t_d) + " |", "| time IP | " + " | ".join(t_i) + " |", ""]
        for (d, mode), r in sorted(R.items()):
            m = r["mono"]
            if m and not m["built"]:
                L.append(f"IP, {r['name']} {mode}: does not build (over {3_000_000:,} candidates)"
                         if m["build_fail"] == "cap" else f"IP, {r['name']} {mode}: does not build ({m['build_fail']})")
            elif m and m["worlds"] == 0:
                L.append(f"IP, {r['name']} {mode}: no world within the limit, {m['candidates']:,} candidates")
        if hy:
            gaps = []
            for (d, mode), r in R.items():
                dm = r["d1_marioh"]
                k = MB.PAPER_METRIC[mode]
                if dm and dm["agg"].get(k) and (d, mode) in hy:
                    g = 100 * (hy[(d, mode)]["agg"][k]["mean"] - dm["agg"][k]["mean"])
                    if g >= 0.05:
                        gaps.append(g)
            if gaps:
                L.append(f"D1 + MARIOH below HyRange by {min(gaps):.1f} to {max(gaps):.1f} points where it is below.")
        L.append("")
    print("\n".join(L))


if __name__ == "__main__":
    w = sys.argv[1]
    if w in ("cert", "all"):
        cert()
    if w in ("same", "all"):
        same()
    if w in ("tables", "all"):
        tables()
