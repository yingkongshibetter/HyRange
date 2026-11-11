"""HyRange output with labels on MARIOH's benchmark (Table III, row HyRange; paragraphs Accuracy and Dense
hypergraphs).

Method (input: the target's pair counts W and degrees d; MARIOH additionally gets its usual training hypergraph):
  1. forced multiset F of (W, d)                               (pipeline.py: WDForcerTL, rules D1, A, B, P, L)
  2. residual (W - W(F), d - d(F)), its connected components, and per component the exact integer world of pipeline.py
     (unique world if the rank test pins it, else a HiGHS world with the seed's random objective; the cap of 300,000
     candidates, 60 s enumeration, 60 s MILP and 180 s world budget of pipeline.py);
  3. MARIOH (authors' code, experiments/marioh_runner.py --query: training on the source hypergraph exactly as in the
     rerun, same params and seed) on the residual pair counts of the components that got NO world;
  output = F + worlds + MARIOH's hyperedges.  Labels: CERTIFIED = forced copies and copies of components proved to
  have one world; UNCERTAIN = copies of the other worlds; LEARNED = MARIOH's copies.
  On the six sparse benchmarks every component gets a world, so no MARIOH run is needed; on the four dense ones the
  residual is one component beyond the limits of the world search, and MARIOH completes all of it.

Reuse (nothing is recomputed that an earlier run already produced; every reused piece is checked):
  * F: forced/<ds>_<mode>_forced.json (pipeline.py).  Checked: every forced copy is a target line with at least that
    multiplicity, and W - W(F) equals forced/<ds>_<mode>_residual.json.
  * worlds: ours_pred/<ds>_<mode>_s<k>.json minus F, split by residual component.  Checked: every world copy lies in one
    component, each component's world reproduces its residual (W and d) exactly, the uncovered volume and the
    certified / uncertain copy counts equal ours/<ds>_<mode>.json.  Which components are pinned is recomputed with
    candidates.build_comp_wp (rank test, same cap and enumeration limit as pipeline.py); a pinned world must equal the
    reused one.
  * MARIOH on the no-world components: if they are all the components, learned/<ds>_<mode>_s<k>_pred.json
    (experiments/run_benchmark.py, kind learned) is reused after checking that its pair counts equal the query (MARIOH
    reproduces its input exactly), i.e. that it was run on this residual.  Otherwise MARIOH is run here on the
    restricted query (hybrid/marioh/...).
  * times: t_force_s, t_components_enum_rank_s and the seed's t_world_s from ours/<ds>_<mode>.json, plus MARIOH's
    t_total_s on the query.
Ground truth is used only to check and to score.

usage: python -I -B -X utf8 experiments/hybrid.py <ds> <mode>      -> results/bench/hybrid/<ds>_<mode>.json, hybrid_pred/
       python -I -B -X utf8 experiments/hybrid.py all              (sequential; MARIOH runs, if any, one at a time)
       python -I -B -X utf8 experiments/hybrid.py summary          -> hybrid_summary.json and the tables on stdout
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

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402

HY = MB.OUT / "hybrid"
HP = MB.OUT / "hybrid_pred"
METRICS = ("jaccard", "multi_jaccard", "exact_recall", "precision")
LABELS = ("certified", "uncertain", "learned")
TOL = 1e-12


def jl(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def vol_of(H):
    return int(sum(len(e) * m for e, m in H.items()))


def sub(A, B):
    out = Counter(A)
    for e, m in B.items():
        out[e] -= m
        assert out[e] >= 0, ("subtraction below zero", e)
        if out[e] == 0:
            del out[e]
    return out


def one(ds, mode):
    from hyrange import engine as EN
    from hyrange import forcing as WC
    from hyrange import candidates as WP
    t_script = time.perf_counter()
    lines, truth = MB.target(ds, mode)
    line_mult = Counter(lines)
    nodes = sorted(set(v for e in lines for v in e))
    loc = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    adj, deg = WC.build_wdeg([tuple(loc[v] for v in e) for e in lines], n)       # the view (W, d)
    vol = int(sum(deg))
    ou = jl(MB.OUT / "ours" / f"{ds}_{mode}.json")
    assert ou and ou.get("stage") == "done", "ours run missing"
    rec = dict(ds=ds, name=MB.DISP[ds], mode=mode, n=n, vol=vol, lines=len(lines),
               reused=dict(forced=f"forced/{ds}_{mode}_forced.json", ours=f"ours/{ds}_{mode}.json",
                           worlds=f"ours_pred/{ds}_{mode}_s<k>.json"))

    # ---- F (reused) and the residual
    F = MB.load_hyperedges(MB.OUT / "forced" / f"{ds}_{mode}_forced.json")
    bad = {e: m for e, m in F.items() if line_mult.get(e, 0) < m}
    assert not bad, ("UNSOUND forced copies", len(bad))
    Floc = Counter({tuple(loc[v] for v in e): m for e, m in F.items()})
    radj = [dict(a) for a in adj]
    rdeg = list(deg)
    for e, m in Floc.items():
        for x in e:
            rdeg[x] -= m
            for y in e:
                if y != x:
                    radj[x][y] -= m
    for x in range(n):
        assert rdeg[x] >= 0
        for y in [y for y, w in radj[x].items() if w <= 0]:
            assert radj[x][y] == 0
            del radj[x][y]
    resid = sorted((nodes[u], nodes[v], int(w)) for u in range(n) for v, w in radj[u].items() if u < v)
    resid_file = [tuple(t) for t in jl(MB.OUT / "forced" / f"{ds}_{mode}_residual.json")]
    assert resid == resid_file, "W - W(F) differs from the residual file of pipeline.py"
    assert vol_of(F) == ou["forced_vol"]
    rec["forced_copies"] = int(sum(F.values()))
    rec["forced_vol_share"] = vol_of(F) / vol

    comps = EN._components(radj, n)
    comps = [np.asarray(c) for c in sorted(comps, key=len)]
    assert len(comps) == ou["comps"], ("component count differs from pipeline", len(comps), ou["comps"])
    comp_of = {}
    for ci, c in enumerate(comps):
        for u in c:
            comp_of[int(u)] = ci
    comp_vol = [int(sum(rdeg[int(u)] for u in c)) for c in comps]
    assert vol_of(F) + sum(comp_vol) == vol
    rec["comps"] = len(comps)
    rec["comp_vol"] = comp_vol

    def comp_residual(ci):
        cn = [int(u) for u in comps[ci]]
        W = {(min(u, v), max(u, v)): w for u in cn for v, w in radj[u].items()}
        return W, {u: rdeg[u] for u in cn}

    # ---- per seed: split the reused HyRange output by component
    seeds_world = []
    for s in MB.SEEDS:
        P = MB.load_hyperedges(MB.OUT / "ours_pred" / f"{ds}_{mode}_s{s}.json")
        Wd = sub(P, F)                                         # world copies (original ids)
        per = [Counter() for _ in comps]
        for e, m in Wd.items():
            cs = {comp_of.get(loc[v]) for v in e}
            assert len(cs) == 1 and None not in cs, ("world copy across components", e)
            per[cs.pop()][tuple(loc[v] for v in e)] += m
        for ci, Wc in enumerate(per):
            if Wc:
                W, d = comp_residual(ci)
                aw, dw = WC.build_wdeg([e for e, m in Wc.items() for _ in range(m)], n)
                W2 = {(min(u, v), max(u, v)): w for u in range(n) for v, w in aw[u].items()}
                d2 = {u: x for u, x in enumerate(dw) if x}
                assert W2 == W and d2 == d, ("reused world does not reproduce its component", ci)
        seeds_world.append(per)

    # pinned components (rank test, as pipeline.py) among those that got a world in some seed
    pinned = set()
    t_rank = time.perf_counter()
    for ci in range(len(comps)):
        if any(per[ci] for per in seeds_world):
            C = WP.build_comp_wp(comps[ci], radj, rdeg, cap=EN.CAP, enum_s=EN.ENUM_S)
            if C.unique and C.xu is not None:
                pinned.add(ci)
                xw = Counter()
                for j in np.nonzero(C.xu)[0]:
                    xw[tuple(sorted(int(C.nodes[i]) for i in C.cands[j]))] += int(C.xu[j])
                for per in seeds_world:
                    assert per[ci] == xw, "pinned world differs from the reused world"
    rec["t_recheck_pinned_s"] = time.perf_counter() - t_rank
    rec["pinned_comps"] = sorted(pinned)
    assert len(pinned) == ou["unique_comps"], ("pinned count differs", len(pinned), ou["unique_comps"])

    rec["seeds"] = []
    HP.mkdir(parents=True, exist_ok=True)
    for k, s in enumerate(MB.SEEDS):
        per = seeds_world[k]
        osd = ou["seeds"][k]
        assert osd["seed"] == s
        world_ci = [ci for ci in range(len(comps)) if per[ci]]
        none_ci = [ci for ci in range(len(comps)) if not per[ci]]
        assert sum(comp_vol[ci] for ci in none_ci) == osd["uncovered_vol"], "uncovered volume differs from pipeline"
        cert_l = Counter(Floc)
        unc_l = Counter()
        for ci in world_ci:
            (cert_l if ci in pinned else unc_l).update(per[ci])
        cert = Counter({tuple(nodes[i] for i in e): m for e, m in cert_l.items()})
        unc = Counter({tuple(nodes[i] for i in e): m for e, m in unc_l.items()})
        assert sum(cert.values()) == osd["certified_copies"] and sum(unc.values()) == osd["uncertain_copies"]
        # query = residual pairs inside the no-world components
        Q = sorted((nodes[u], nodes[v], int(w)) for ci in none_ci for u in map(int, comps[ci])
                   for v, w in radj[u].items() if u < v)
        row = dict(seed=s, comps_world=len(world_ci), comps_pinned=sum(1 for ci in world_ci if ci in pinned),
                   comps_found=sum(1 for ci in world_ci if ci not in pinned), comps_no_world=len(none_ci),
                   no_world_vol=int(sum(comp_vol[ci] for ci in none_ci)), query_pairs=len(Q),
                   query_weight_sum=int(sum(w for _, _, w in Q)), ours_status=osd["status"])
        learned = Counter()
        t_mar = 0.0
        if not none_ci:
            row["source"] = "worlds"                  # every component got a world: no MARIOH run
        elif len(none_ci) == len(comps):
            assert Q == resid_file
            gj = jl(MB.OUT / "learned" / f"{ds}_{mode}_s{s}.json")
            learned = MB.load_hyperedges(MB.OUT / "learned" / f"{ds}_{mode}_s{s}_pred.json")
            WG = MB.pair_counts(learned)
            assert {k_: w for k_, w in WG.items()} == {(u, v): w for u, v, w in Q}, \
                "the MARIOH run was not made on this residual"
            assert gj["query_pairs"] == len(Q) and gj["query_weight_sum"] == row["query_weight_sum"]
            assert gj["sanity_projection_reproduced"]
            t_mar = float(gj["t_total_s"])
            row["source"] = "learned"                 # no component got a world: MARIOH on the whole residual
            row["marioh_run"] = f"learned/{ds}_{mode}_s{s}.json (reused)"
        else:
            qp = HY / "query" / f"{ds}_{mode}_s{s}.json"
            MB.save_json([list(t) for t in Q], qp)
            mo = HY / "marioh" / f"{ds}_{mode}_s{s}.json"
            mp = HY / "marioh" / f"{ds}_{mode}_s{s}_pred.json"
            if not mo.exists():
                mo.parent.mkdir(parents=True, exist_ok=True)
                cmd = [str(MB.MAR_PY), "-I", "-B", "-X", "utf8", str(HERE / "marioh_runner.py"), "--data", ds, "--mode",
                       mode, "--seed", str(s), "--query", str(qp), "--out", str(mo), "--pred", str(mp)]
                with open(CFG.LOGS / "bench" / f"hybrid_marioh_{ds}_{mode}_s{s}.log", "w",
                          encoding="utf-8") as fh:
                    subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE), check=True)
            mj = jl(mo)
            learned = MB.load_hyperedges(mp)
            assert dict(MB.pair_counts(learned)) == {(u, v): w for u, v, w in Q}
            t_mar = float(mj["t_total_s"])
            row["source"] = "learned_partial"
            row["marioh_run"] = str(mo.relative_to(MB.OUT))
        out_h = cert + unc + learned
        # SOUNDNESS: every certified copy is a target line with at least that multiplicity
        badc = {e: m for e, m in cert.items() if line_mult.get(e, 0) < m}
        if badc:
            raise AssertionError(("UNSOUND certified copies", ds, mode, s, len(badc)))
        row["certified_sound"] = True
        met = MB.metrics(out_h, truth)
        row.update(met)
        tot = sum(out_h.values())
        # true copies per label: target lines (with multiplicity) are allotted to certified copies first; uncertain and
        # learned copies lie in different components, so they never share a node set
        left = sub(line_mult, Counter({e: min(m, line_mult[e]) for e, m in cert.items()}))
        for lab, H in (("certified", cert), ("uncertain", unc), ("learned", learned)):
            c = int(sum(H.values()))
            tr = int(sum(min(m, line_mult.get(e, 0)) for e, m in H.items())) if lab == "certified" else \
                int(sum(min(m, left.get(e, 0)) for e, m in H.items()))
            row[f"{lab}_copies"] = c
            row[f"{lab}_vol"] = vol_of(H)
            row[f"{lab}_copy_share"] = c / tot if tot else 0.0
            row[f"{lab}_vol_share_of_output"] = vol_of(H) / vol_of(out_h) if out_h else 0.0
            row[f"{lab}_true_copies"] = tr
            row[f"{lab}_true_share"] = tr / c if c else None
        assert row["certified_true_copies"] == row["certified_copies"]
        row["certified_vol_share"] = row["certified_vol"] / vol          # of vol (same in every world)
        row["t_force_s"] = float(ou["t_force_s"])
        row["t_components_enum_rank_s"] = float(ou["t_components_enum_rank_s"])
        row["t_world_s"] = float(osd["t_world_s"])
        row["t_marioh_s"] = t_mar
        row["t_total_s"] = row["t_force_s"] + row["t_components_enum_rank_s"] + row["t_world_s"] + t_mar
        # check against the stored numbers of pipeline.py
        if row["source"] == "worlds":
            row["equals_pipeline_metrics"] = all(abs(met[m] - osd[m]) <= TOL for m in METRICS)
            assert row["equals_pipeline_metrics"]
        MB.save_json({lab: [[list(e), int(m)] for e, m in sorted(H.items())]
                      for lab, H in (("certified", cert), ("uncertain", unc), ("learned", learned))},
                     HP / f"{ds}_{mode}_s{s}.json")
        rec["seeds"].append(row)
        print(time.strftime("%H:%M:%S ") + f"{ds} {mode} s{s}: {row['source']}, world {len(world_ci)} "
              f"(pinned {row['comps_pinned']}), no world {len(none_ci)}; J {met['jaccard']:.4f} MJ "
              f"{met['multi_jaccard']:.4f}; cert/unc/learn copies {row['certified_copies']}/"
              f"{row['uncertain_copies']}/{row['learned_copies']}; t {row['t_total_s']:.1f}s", flush=True)
    rec["sources"] = sorted({r["source"] for r in rec["seeds"]})
    if "learned" in rec["sources"]:
        rec["reused"]["marioh"] = f"learned/{ds}_{mode}_s<k>_pred.json (MARIOH on the whole residual)"
    rec["t_script_s"] = time.perf_counter() - t_script
    rec["done"] = True
    MB.save_json(rec, HY / f"{ds}_{mode}.json")


# ------------------------------------------------------------------ summary
def agg(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n=len(v),
                          values=[float(x) for x in v])
    return out


def rng(vals, f=MB.pct1):
    vals = [v for v in vals if v is not None]
    return f"{f(min(vals))} to {f(max(vals))}" if vals else "-"


def summary():
    S = jl(MB.OUT / "summary.json")
    base = {(r["ds"], r["mode"]): r for r in S["records"]}
    keys = METRICS + tuple(f"{lab}_{x}" for lab in LABELS for x in ("true_share", "copies")) + \
        ("certified_vol_share", "t_total_s")
    recs = []
    for mode in MB.MODES:
        for ds in MB.DATASETS:
            h = jl(HY / f"{ds}_{mode}.json")
            if not h or not h.get("done"):
                print(f"missing: hybrid {ds} {mode}")
                continue
            b = base[(ds, mode)]
            recs.append(dict(ds=ds, name=MB.DISP[ds], mode=mode, sources=h["sources"], comps=h["comps"],
                             forced_vol_share=h["forced_vol_share"], agg=agg(h["seeds"], keys), seeds=h["seeds"],
                             marioh_rerun_agg={m: b["marioh"]["agg"][m] for m in METRICS if m in b["marioh"]["agg"]},
                             mean_degree=b["target"]["mean_degree"]))
    allsound = all(r["certified_sound"] for x in recs for r in x["seeds"])
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               method=("forced hyperedges + the exact world of every residual component that got one (pipeline.py) + "
                       "MARIOH (authors' code, trained on the source hypergraph as in its protocol) on the residual "
                       "pair counts of the components that got no world. Labels: certified (forced or a pinned "
                       "component), uncertain (other worlds), learned (MARIOH)."),
               all_certified_copies_true=allsound, records=recs)
    MB.save_json(out, MB.OUT / "hybrid_summary.json")
    R = {(x["ds"], x["mode"]): x for x in recs}
    L = []
    six = [d for d in MB.SIX if all((d, m) in R for m in MB.MODES)]
    if six:
        for mode in MB.MODES:
            k = MB.PAPER_METRIC[mode]
            L += [f"### Table III, row HyRange, {mode} ({'multi-Jaccard' if k == 'multi_jaccard' else 'Jaccard'} x100, "
                  "mean ± std over three seeds; time per seed in s)", "",
                  "| | " + " | ".join(MB.DISP[d] for d in six) + " |", "|---|" + "---|" * len(six),
                  "| HyRange | " + " | ".join(f"{MB.pct1(R[(d, mode)]['agg'][k]['mean'])} ± "
                                            f"{100 * R[(d, mode)]['agg'][k]['std']:.1f}" for d in six) + " |",
                  "| time HyRange | " + " | ".join(MB.sec1(R[(d, mode)]["agg"]["t_total_s"]["mean"]) for d in six) + " |",
                  ""]
        acc = [R[(d, m)]["agg"][MB.PAPER_METRIC[m]]["mean"] for d in six for m in MB.MODES]
        L.append(f"HyRange scores {rng(acc)} on the six benchmarks; at least "
                 f"{MB.pct1(min(R[(d, m)]['agg']['precision']['mean'] for d in six for m in MB.MODES))}% of its "
                 "distinct hyperedges lie in the target (precision).")
        if ("dblp", "preserved") in R:
            b = base[("dblp", "preserved")]["ours"]
            L.append(f"DBLP, preserved: forcing {MB.sec1(b['t_force_s'])} s of the "
                     f"{MB.sec1(R[('dblp', 'preserved')]['agg']['t_total_s']['mean'])} s per seed.")
        L.append("")
    dense = [d for d in MB.DENSE if all((d, m) in R for m in MB.MODES)]
    if dense:
        L += ["### Dense hypergraphs (HyRange = forced copies + MARIOH on the residual, label learned)", "",
              "| | " + " | ".join(MB.DISP[d] for d in dense) + " |", "|---|" + "---|" * len(dense)]
        for mode in MB.MODES:
            k = MB.PAPER_METRIC[mode]
            L.append(f"| HyRange, {mode} | " + " | ".join(MB.pct1(R[(d, mode)]["agg"][k]["mean"]) for d in dense) + " |")
            L.append(f"| MARIOH (rerun), {mode} | " + " | ".join(
                MB.pct1(R[(d, mode)]["marioh_rerun_agg"][k]["mean"]) if k in R[(d, mode)]["marioh_rerun_agg"] else "-"
                for d in dense) + " |")
        L.append("")
        L.append(f"Mean node degree (preserved): {rng([R[(d, 'preserved')]['mean_degree'] for d in dense], lambda v: f'{v:.1f}')}.")
        L.append(f"Certified copies in the target: "
                 f"{rng([R[(d, m)]['agg']['certified_true_share']['mean'] for d in dense for m in MB.MODES])}%; "
                 f"learned copies in the target: "
                 f"{rng([R[(d, m)]['agg']['learned_true_share']['mean'] for d in dense for m in MB.MODES])}%.")
        L.append("")
    print("\n".join(L))
    print(json.dumps({"all_certified_copies_true": allsound}))


def main():
    if sys.argv[1] == "summary":
        summary()
    elif sys.argv[1] == "all":
        for mode in MB.MODES:
            for ds in MB.DATASETS:
                one(ds, mode)
        summary()
    else:
        one(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
