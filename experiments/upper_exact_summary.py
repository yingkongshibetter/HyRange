"""Collect results/upper_exact/jobs/*.json into summary.json (no computation): Table II, row Upper; paragraph
Upper-bound mode; the HyRange shares at t = 5 of Table V (read by experiments/exports_table.py).

usage: python -I -B -X utf8 experiments/upper_exact_summary.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402
OUT = CFG.RESULTS / "upper_exact"
JOBS = OUT / "jobs"
EXPORTS = ["dblp", "openalex_db", "topcs", "openalex_cs_size3plus", "openalex_bio"]


def load(name):
    p = JOBS / (name + ".json")
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def main():
    S = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), bench={}, export={})
    for mode in MB.MODES:
        for ds in MB.SIX:
            S["bench"][f"{ds}_{mode}"] = load(f"bench_{ds}_{mode}")
    for src in EXPORTS:
        S["export"][f"{src}_t5"] = load(f"export_{src}_5")
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(S, f, indent=1)
    B = {k: r for k, r in S["bench"].items() if r and r.get("stage") == "done"}
    L = ["### Table II, row Upper: certified share of the volume (%), upper-bound mode", "",
         "| Setting | " + " | ".join(MB.DISP[d] for d in MB.SIX) + " |", "|---|" + "---|" * len(MB.SIX)]
    for mode in MB.MODES:
        L.append(f"| {mode} | " + " | ".join(MB.pct1(B[f"{d}_{mode}"]["cert_vol_share"]) if f"{d}_{mode}" in B else "-"
                                             for d in MB.SIX) + " |")
    if B:
        f = [r["forced_vol_share"] for r in B.values()]
        c = [r["cert_vol_share"] for r in B.values()]
        L += ["", f"Upper-bound mode: {MB.pct1(min(c))}% to {MB.pct1(max(c))}% certified, {MB.pct1(min(f))}% to "
              f"{MB.pct1(max(f))}% by forcing alone; no copy left undecided on "
              f"{sum(1 for r in B.values() if r['fcert_proved'])} of the {len(B)} inputs; false certified copies "
              f"{sum(r['truth']['false_certified_copies'] for r in B.values())}."]
        ce = CFG.RESULTS / "certification" / "cert_exact.json"
        if ce.exists():
            ex = {f"{r['ds']}_{r['mode']}": r["certified_vol_share_after"]
                  for r in json.load(open(ce, encoding="utf-8"))["benchmark"]}
            gap = max(100 * (ex[k] - r["cert_vol_share"]) for k, r in B.items() if k in ex)
            L.append(f"Largest gap to the share with exact degrees (row (W,d)): {gap:.2f} points.")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
