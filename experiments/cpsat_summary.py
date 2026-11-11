"""The CP-SAT sentence of paragraph Same input: the integer program solved by CP-SAT (experiments/cpsat_baseline.py,
one worker) on Hosts and MAG-TopCS in both settings and DBLP in the reduced setting, next to the HiGHS runs of
experiments/ip_baseline.py.  Checks that CP-SAT returns worlds of the same accuracy as HiGHS in every seed and prints
the range of its time per seed.

usage: python -I -B -X utf8 experiments/cpsat_summary.py   -> results/cpsat/summary.json and the sentence on stdout
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402
D = CFG.RESULTS / "cpsat"
H = CFG.RESULTS / "certification" / "mono"
INPUTS = [("hosts", "preserved"), ("hosts", "reduced"), ("topcs", "preserved"), ("topcs", "reduced"),
          ("dblp", "reduced")]


def main():
    out = []
    for ds, mode in INPUTS:
        runs = sorted((json.loads(p.read_text(encoding="utf-8")) for p in D.glob(f"{ds}_{mode}_w1_s*.json")),
                      key=lambda r: r["seed"])
        hp = H / f"{ds}_{mode}.json"
        if not runs or not hp.exists():
            print(f"{MB.DISP[ds]} {mode}: missing")
            continue
        hs = {s["seed"]: s for s in json.loads(hp.read_text(encoding="utf-8")).get("seeds", [])}
        k = MB.PAPER_METRIC[mode]
        same = all(r.get("exact_check") and int(r["seed"]) in hs and r.get(k) == hs[int(r["seed"])].get(k)
                   for r in runs)
        out.append(dict(ds=ds, mode=mode, seeds=[r["seed"] for r in runs], status=sorted(set(r["status"] for r in runs)),
                        t_total=[r["t_total_s"] for r in runs], metric=k, values=[r.get(k) for r in runs],
                        highs_values=[hs.get(int(r["seed"]), {}).get(k) for r in runs], same_accuracy_as_highs=same))
        print(f"{MB.DISP[ds]} {mode}: status {'/'.join(out[-1]['status'])}, same accuracy as HiGHS in every seed: "
              f"{same}, time per seed {', '.join(f'{t:.1f}' for t in out[-1]['t_total'])} s")
    (D / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    if out:
        ts = [t for r in out for t in r["t_total"]]
        print(f"CP-SAT: {min(ts):.1f} s to {max(ts):.1f} s per seed")


if __name__ == "__main__":
    main()
