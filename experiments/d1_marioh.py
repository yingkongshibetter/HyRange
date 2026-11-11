"""D1 + MARIOH on MARIOH's benchmark (Table III, row D1 + MARIOH).

Method, on the six benchmarks:
  1. F1 = hyperedges forced by rule D1 alone on the view (W, d) of the target lines: pipeline.WDForcerTL with
     rules=("D1",), wall limit 900 s (the forcer of pipeline.py / experiments/rule_ablation.py, unchanged);
  2. MARIOH (authors' code via experiments/marioh_runner.py --query, trained on the source hypergraph exactly as in the
     rerun: params.py settings of the dataset and mode, same seed 0, 1, 2) on the residual pair counts W - W(F1);
  3. output = F1 + MARIOH's hyperedges.  Time = D1 forcing + MARIOH's t_total_s.
Ground truth is read only to check the forcing (verify_forced) and to score.

usage: python -I -B -X utf8 experiments/d1_marioh.py prep            -> results/certification/d1_marioh/forced/*  (12 inputs)
       python -I -B -X utf8 experiments/d1_marioh.py jobs            -> appends the 36 MARIOH command lines to
                                                                        results/certification/queue/jobs.jsonl (run each one)
       python -I -B -X utf8 experiments/d1_marioh.py summary         -> results/certification/d1_marioh/summary.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402

CERT_DIR = CFG.RESULTS / "certification"
D1O = CERT_DIR / "d1_marioh"
FLAGS = ["-I", "-B", "-X", "utf8"]
LIMIT_S = 4 * 3600.0


def prep():
    from hyrange import forcing as WC
    from hyrange import pipeline as MO
    (D1O / "forced").mkdir(parents=True, exist_ok=True)
    for mode in MB.MODES:
        for ds in MB.SIX:
            lines, truth = MB.target(ds, mode)
            nodes = sorted(set(v for e in lines for v in e))
            loc = {v: i for i, v in enumerate(nodes)}
            n = len(nodes)
            lines_loc = [tuple(loc[v] for v in e) for e in lines]
            adj, deg = WC.build_wdeg(lines_loc, n)
            vol = int(sum(deg))
            t0 = time.perf_counter()
            F = MO.WDForcerTL(adj, deg, rules=("D1",), time_limit=900.0, copy_adj=False).run()
            t = time.perf_counter() - t0
            ver = WC.verify_forced(F.forced, lines_loc)
            assert ver["ok"], ("UNSOUND D1 forcing", ds, mode, ver)
            assert WC_check_residual(F, lines_loc, n), "residual mismatch"
            forced = Counter({tuple(nodes[i] for i in sorted(e)): int(m) for e, m in F.forced.items()})
            resid = sorted((nodes[u], nodes[v], int(w)) for u in range(n) for v, w in F.adj[u].items() if u < v)
            MB.save_hyperedges(forced, D1O / "forced" / f"{ds}_{mode}_forced.json")
            MB.save_json(resid, D1O / "forced" / f"{ds}_{mode}_residual.json")
            fv = int(sum(len(e) * m for e, m in forced.items()))
            MB.save_json(dict(ds=ds, mode=mode, n=n, vol=vol, t_force_d1_s=t, complete=bool(F.complete),
                              forced_copies=int(sum(forced.values())), forced_vol=fv, forced_vol_share=fv / vol,
                              residual_pairs=len(resid), residual_weight=int(sum(w for _, _, w in resid)),
                              verify_forced=ver),
                         D1O / "forced" / f"{ds}_{mode}_info.json")
            print(f"{ds} {mode}: D1 forced {sum(forced.values())} copies, vol share {fv / vol:.4f}, "
                  f"{len(resid)} residual pairs, {t:.2f}s", flush=True)


def WC_check_residual(F, lines_loc, n):
    from hyrange import candidates as WP
    return WP.check_residual_wp(F, lines_loc, n)


def jobs():
    q = CERT_DIR / "queue" / "jobs.jsonl"
    q.parent.mkdir(parents=True, exist_ok=True)
    (D1O / "marioh").mkdir(parents=True, exist_ok=True)
    # long jobs first
    order = ["dblp", "topcs", "foursquare", "hosts", "crime", "directors"]
    with open(q, "a", encoding="utf-8") as f:
        for ds in order:
            for mode in MB.MODES:
                for s in MB.SEEDS:
                    out = D1O / "marioh" / f"{ds}_{mode}_s{s}.json"
                    pred = D1O / "marioh" / f"{ds}_{mode}_s{s}_pred.json"
                    cmd = [str(MB.MAR_PY)] + FLAGS + [str(HERE / "marioh_runner.py"), "--data", ds, "--mode", mode,
                                                      "--seed", str(s), "--query",
                                                      str(D1O / "forced" / f"{ds}_{mode}_residual.json"),
                                                      "--out", str(out), "--pred", str(pred)]
                    f.write(json.dumps(dict(name=f"d1m_{ds}_{mode}_s{s}", cmd=cmd, w=1, prio=20, group="marioh",
                                            timeout=LIMIT_S,
                                            done=str(out))) + "\n")


def summary():
    import numpy as np
    status = {}
    sp = CERT_DIR / "queue" / "status.jsonl"
    if sp.exists():
        for line in sp.read_text(encoding="utf-8").splitlines():
            ev = json.loads(line)
            if ev["name"].startswith("d1m_"):
                status[ev["name"]] = ev
    recs = []
    for mode in MB.MODES:
        for ds in MB.SIX:
            lines, truth = MB.target(ds, mode)
            info = json.load(open(D1O / "forced" / f"{ds}_{mode}_info.json", encoding="utf-8"))
            F = MB.load_hyperedges(D1O / "forced" / f"{ds}_{mode}_forced.json")
            Q = {(u, v): w for u, v, w in json.load(open(D1O / "forced" / f"{ds}_{mode}_residual.json",
                                                          encoding="utf-8"))}
            rows = []
            skipped = []
            for s in MB.SEEDS:
                jo = D1O / "marioh" / f"{ds}_{mode}_s{s}.json"
                jp = D1O / "marioh" / f"{ds}_{mode}_s{s}_pred.json"
                if not jo.exists():
                    ev = status.get(f"d1m_{ds}_{mode}_s{s}")
                    skipped.append(dict(seed=s, event=(ev or {}).get("event", "not run"), wall=(ev or {}).get("wall")))
                    continue
                rj = json.load(open(jo, encoding="utf-8"))
                G = MB.load_hyperedges(jp)
                WG = MB.pair_counts(G)
                assert dict(WG) == Q, ("MARIOH output does not reproduce the D1 residual", ds, mode, s)
                assert rj["sanity_projection_reproduced"]
                met = MB.metrics(F + G, truth)
                rows.append(dict(seed=s, t_marioh_s=rj["t_total_s"], t_total_s=info["t_force_d1_s"] + rj["t_total_s"],
                                 residual_copies=int(sum(G.values())), **met))
            agg = {}
            for k in ("jaccard", "multi_jaccard", "exact_recall", "precision", "t_total_s"):
                v = [r[k] for r in rows]
                if v:
                    agg[k] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                  values=v)
            recs.append(dict(ds=ds, name=MB.DISP[ds], mode=mode, d1=info, seeds=rows, skipped=skipped, agg=agg))
    MB.save_json(dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), records=recs), D1O / "summary.json")
    return recs


if __name__ == "__main__":
    {"prep": prep, "jobs": jobs, "summary": summary}[sys.argv[1]]()
