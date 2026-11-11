"""Summary of the runs on MARIOH's benchmark: results/bench/summary.json and datasets/<ds>_<mode>.json.

Per input: the target statistics (Table I), MARIOH's paper numbers (row MARIOH of Table III, six benchmarks), the
MARIOH rerun (row MARIOH (rerun) and its time), MARIOH + refutation (row MARIOH + Ref., paragraph Refutation; six
benchmarks) and HyRange (experiments/hyrange_bench.py).  The metric per setting is that of Table III: multi-Jaccard
(preserved) and Jaccard (reduced), x100, mean and sample std (ddof = 1) over the seeds 0, 1, 2.
Printed: the rows MARIOH, MARIOH (rerun) and MARIOH + Ref. of Table III with the time of the rerun, and the share of
MARIOH's wrong copies that the refutation removes.  The row HyRange is printed by experiments/hybrid.py summary.

usage: python -I -B -X utf8 experiments/bench_summary.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import bench as MB  # noqa: E402

METRICS = ("jaccard", "multi_jaccard", "exact_recall", "precision")


def jl(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def agg(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n=len(v),
                          values=[float(x) for x in v])
    return out


def one(ds, mode):
    lines, truth = MB.target(ds, mode)
    st = MB.target_stats(lines, truth)
    rec = dict(ds=ds, name=MB.DISP[ds], mode=mode, target_file=MB.target_file(ds, mode).name,
               train_file=MB.train_file(ds, mode).name, target=st)
    if ds in MB.SIX:
        rec["paper"] = dict(table=MB.PAPER[mode]["table"], metric=MB.PAPER_METRIC[mode],
                            mean=MB.PAPER[mode][ds][0], std=MB.PAPER[mode][ds][1])
    # MARIOH rerun
    mr = [jl(MB.OUT / "marioh" / f"{ds}_{mode}_s{s}.json") for s in MB.SEEDS]
    mr = [r for r in mr if r]
    rows = [dict(seed=r["seed"], **r["metrics"], t_total_s=r["t_total_s"], t_train_s=r["t_train_s"],
                 t_reconstruct_s=r["t_reconstruct_s"], projection_reproduced=r["sanity_projection_reproduced"])
            for r in mr]
    rec["marioh"] = dict(seeds=rows, agg=agg(rows, METRICS + ("t_total_s",)))
    # MARIOH + refutation (six benchmarks)
    va = jl(MB.OUT / "variants" / f"{ds}_{mode}.json")
    if va and ds in MB.SIX:
        rec["refutation"] = dict(seeds=va["refutation"], agg=agg(va["refutation"], METRICS + ("refuted_copies",)))
    # HyRange (forcing, components, rank test, one world per seed)
    ou = jl(MB.OUT / "ours" / f"{ds}_{mode}.json")
    if ou:
        seeds = ou.get("seeds", [])
        rec["ours"] = dict({k: v for k, v in ou.items() if k != "seeds"}, seeds=seeds,
                           agg=agg(seeds, METRICS + ("certified_vol_share", "t_world_s")))
        if seeds:
            tw = float(np.mean([r["t_world_s"] for r in seeds]))
            rec["ours"]["t_rest_s"] = ou.get("t_components_enum_rank_s", 0.0) + tw
            rec["ours"]["certified_sound_all_seeds"] = all(r.get("certified_sound") for r in seeds)
    MB.save_json(rec, MB.OUT / "datasets" / f"{ds}_{mode}.json")
    return rec


def ms(a, k):
    if not a or k not in a:
        return "-"
    return f"{MB.pct1(a[k]['mean'])} ± {100 * a[k]['std']:.1f}"


def table(recs, mode):
    k = MB.PAPER_METRIC[mode]
    L = [f"### Table III, {mode} ({'multi-Jaccard' if k == 'multi_jaccard' else 'Jaccard'} x100, mean ± std over "
         "three seeds; time per seed in s)", "",
         "| Method | " + " | ".join(MB.DISP[d] for d in MB.SIX) + " |", "|---|" + "---|" * len(MB.SIX)]
    R = {r["ds"]: r for r in recs if r["mode"] == mode}
    L.append("| MARIOH | " + " | ".join(f"{R[d]['paper']['mean']:.1f} ± {R[d]['paper']['std']:.1f}" if d in R else "-"
                                         for d in MB.SIX) + " |")
    L.append("| MARIOH (rerun) | " + " | ".join(ms(R[d]["marioh"]["agg"], k) if d in R else "-" for d in MB.SIX) + " |")
    L.append("| MARIOH + Ref. | " + " | ".join(ms(R[d].get("refutation", {}).get("agg"), k) if d in R else "-"
                                              for d in MB.SIX) + " |")
    L.append("| time MARIOH (rerun) | " + " | ".join(
        MB.sec1(R[d]["marioh"]["agg"]["t_total_s"]["mean"]) if d in R and "t_total_s" in R[d]["marioh"]["agg"] else "-"
        for d in MB.SIX) + " |")
    cells = []
    for d in MB.SIX:
        rf = R.get(d, {}).get("refutation")
        w = sum(x["marioh_wrong_copies"] for x in rf["seeds"]) if rf else 0
        f = sum(x["refuted_copies"] for x in rf["seeds"]) if rf else 0
        cells.append(f"{100 * f / w:.1f}" if w else "-")
    L.append("| wrong copies of the rerun removed by refutation (%) | " + " | ".join(cells) + " |")
    L.append("")
    return "\n".join(L)


def main():
    recs = []
    for mode in MB.MODES:
        for ds in MB.DATASETS:
            if not (MB.OUT / "ours" / f"{ds}_{mode}.json").exists() and \
                    not (MB.OUT / "marioh" / f"{ds}_{mode}_s0.json").exists():
                continue
            recs.append(one(ds, mode))
    sound = [r for r in recs if r.get("ours", {}).get("seeds")]
    summary = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                   soundness=dict(datasets_checked=len(sound),
                                  all_certified_copies_true=all(r["ours"]["certified_sound_all_seeds"] for r in sound),
                                  all_forced_verified=all(r["ours"].get("verify_forced", {}).get("ok") for r in sound),
                                  all_residuals_match=all(r["ours"].get("residual_equals_Wplus_of_truth_minus_forced")
                                                          for r in sound)),
                   records=recs)
    MB.save_json(summary, MB.OUT / "summary.json")
    print("\n".join(table(recs, mode) for mode in MB.MODES if any(r["mode"] == mode for r in recs)))
    print(json.dumps(summary["soundness"]))


if __name__ == "__main__":
    main()
