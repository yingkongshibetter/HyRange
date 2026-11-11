"""Table IV and paragraph Fewer forcing rules: certified share of the volume without forcing, with D1 + exact, and
with HyRange, from the outputs of experiments/fewer_rules.py -> results/fewer_rules/summary.json and stdout.

Runs used (results/fewer_rules/<dir>/<ds>_<mode>.json; see the docstring of fewer_rules.py):
  No forcing    : none/, MAG-TopCS from none_skipgiant/ (stop rule 30,000 candidates)
  D1 + exact    : d1/ (DBLP preserved with the stop rule 100,000 candidates), DBLP reduced from d1_full/ (no stop rule)
  HyRange       : results/certification/cert_exact.json certified_vol_share_after (row (W,d) of Table II); its exact
                  certification time per input is the sum of components[].t_total_s of
                  results/certification/cert_exact/bench_<ds>_<mode>.json
A dagger marks a lower bound: some copies stay undecided, a component exceeds the candidate cap, or the stop rule
skipped the certification of a component.  "Same hyperedges as HyRange" compares the certified multiset with that of
the variant full (HyRange through the same driver).

usage: python -I -B -X utf8 experiments/fewer_rules_summary.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402

BASE = CFG.RESULTS / "fewer_rules"
RUN_DIR = {("dblp", "reduced", "d1"): "d1_full", ("topcs", "preserved", "none"): "none_skipgiant",
           ("topcs", "reduced", "none"): "none_skipgiant"}


def ld(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def variant(ds, mode, v):
    vdir = RUN_DIR.get((ds, mode, v), v)
    r = ld(BASE / vdir / f"{ds}_{mode}.json")
    if r is None or r.get("stage") != "done":
        return None
    undecided = max((s.get("undecided", 0) for s in r.get("per_seed_cert", [])), default=0)
    skipped = sum(1 for c in r.get("components", []) if c.get("cert_skipped"))
    over_cap = r.get("comps_not_enumerated", 0)
    cs = BASE / vdir / "certified" / f"{ds}_{mode}.json"
    ref = BASE / "full" / "certified" / f"{ds}_{mode}.json"
    same = (MB.load_hyperedges(cs) == MB.load_hyperedges(ref)) if cs.exists() and ref.exists() else None
    return dict(run=f"{vdir}/{ds}_{mode}.json", after=r["certified_vol_share_after"], undecided_per_seed=undecided,
                skipped_components=skipped, over_cap_components=over_cap,
                lower_bound=bool(undecided or skipped or over_cap), largest_comp_nodes=r.get("largest_comp_nodes"),
                t_cert_s=r.get("t_cert_components_s"), same_certified_as_hyrange=same)


def hyrange(ds, mode):
    ce = ld(CFG.RESULTS / "certification" / "cert_exact.json")
    su = ld(MB.OUT / "summary.json")
    if ce is None:
        return None
    r = next((x for x in ce["benchmark"] if x["ds"] == ds and x["mode"] == mode), None)
    if r is None:
        return None
    ou = next((x for x in (su or {}).get("records", []) if x["ds"] == ds and x["mode"] == mode), {}).get("ours", {})
    return dict(after=r["certified_vol_share_after"], t_cert_s=r.get("t_components_s", 0.0),
                largest_comp_nodes=ou.get("largest_comp_nodes"), lower_bound=False)


def cell(x):
    if x is None:
        return "-"
    return MB.pct1(x["after"]) + ("†" if x["lower_bound"] else "")


def main():
    recs = []
    for mode in MB.MODES:
        for ds in MB.SIX:
            recs.append(dict(ds=ds, name=MB.DISP[ds], mode=mode, none=variant(ds, mode, "none"),
                             d1=variant(ds, mode, "d1"), hyrange=hyrange(ds, mode)))
    MB.save_json(dict(records=recs), BASE / "summary.json")
    R = {(r["ds"], r["mode"]): r for r in recs}
    L = ["### Table IV: certified share of the volume (%) under fewer forcing rules († lower bound)", "",
         "| | " + " | ".join(MB.DISP[d] for d in MB.SIX) + " |", "|---|" + "---|" * len(MB.SIX)]
    for mode in MB.MODES:
        for key, lab in (("none", "No forcing"), ("d1", "D1 + exact"), ("hyrange", "HyRange")):
            L.append(f"| {lab}, {mode} | " + " | ".join(cell(R[(d, mode)][key]) for d in MB.SIX) + " |")
    L.append("")
    x = R[("dblp", "preserved")]["d1"]
    if x:
        L.append(f"D1 + exact, DBLP, preserved: component of {x['largest_comp_nodes']:,} nodes, "
                 f"{MB.pct1(x['after'])}% certified")
    x, h = R[("dblp", "reduced")]["d1"], R[("dblp", "reduced")]["hyrange"]
    if x and h:
        L.append(f"D1 + exact, DBLP, reduced: component of {x['largest_comp_nodes']:,} nodes; same certified "
                 f"hyperedges as HyRange: {x['same_certified_as_hyrange']}; exact certification {MB.sec1(x['t_cert_s'])} s "
                 f"against {MB.sec1(h['t_cert_s'])} s; {x['undecided_per_seed']} copies per seed undecided")
    xs = [R[("topcs", m)]["d1"] for m in MB.MODES]
    hs = [R[("topcs", m)]["hyrange"] for m in MB.MODES]
    if all(xs) and all(hs):
        L.append(f"D1 + exact, MAG-TopCS: exact certification {MB.sec1(xs[0]['t_cert_s'])} s and "
                 f"{MB.sec1(xs[1]['t_cert_s'])} s (preserved, reduced) against "
                 f"{' and '.join(MB.sec1(h['t_cert_s']) for h in hs)} s")
    lc = [R[(d, m)]["hyrange"]["largest_comp_nodes"] for d in MB.SIX for m in MB.MODES
          if R[(d, m)]["hyrange"] and R[(d, m)]["hyrange"]["largest_comp_nodes"] is not None]
    if lc:
        L.append(f"With all rules the residual components have at most {max(lc)} nodes.")
    print("\n".join(L))


if __name__ == "__main__":
    main()
