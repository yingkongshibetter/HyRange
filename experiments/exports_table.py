"""Table V and paragraph Exports with a document threshold, from stored outputs only.

Inputs (read only):
  results/exports/sim_export.json     records[] (source, t, upper, cliques): rows Pair counts (upper.w_only_vol_share)
                                      and Forcing (upper.cert_vol_share) of Table V, false copies (upper.false_copies)
  results/upper_exact/summary.json    export.<source>_t5: row HyRange at t = 5 (cert_vol_share) and its false copies
                                      (truth.false_certified_copies)
Shares are of the trace volume (sum of the sizes of the multi-member traces e & S).
Writes results/exports/exports_table.json and prints Table V.

usage: python -I -B -X utf8 experiments/exports_table.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402

SIM = CFG.RESULTS / "exports" / "sim_export.json"
UEX = CFG.RESULTS / "upper_exact" / "summary.json"
OUT = CFG.RESULTS / "exports"
ORDER = ["dblp", "topcs", "openalex_db", "openalex_cs_size3plus", "openalex_bio"]
PAPER = {"dblp": "DBLP", "topcs": "MAG-TopCS", "openalex_db": "OA-DB", "openalex_cs_size3plus": "OA-CS",
         "openalex_bio": "OA-Bio"}
THRESH = [1, 2, 3, 5, 10]


def main():
    sim = json.load(open(SIM, encoding="utf-8"))
    assert sim["complete"] is True and sim["upper_soundness_violations"] == []
    rec = {(r["source"], r["t"]): r for r in sim["records"]}
    uex = json.load(open(UEX, encoding="utf-8"))["export"] if UEX.exists() else {}
    rows = []
    for src in ORDER:
        for t in THRESH:
            u = rec[(src, t)]["upper"]
            assert u["stage"] == "done" and u["force_complete"] is True and u["exception"] is None
            row = dict(source=src, name=PAPER[src], t=t, pair_counts_share=u["w_only_vol_share"],
                       forcing_share=u["cert_vol_share"], forcing_false_copies=u["false_copies"])
            h = uex.get(f"{src}_t{t}")
            if h is not None and h.get("stage") == "done":
                assert abs(h["forced_vol_share"] - u["cert_vol_share"]) < 1e-12      # same export, same forcing
                row.update(hyrange_share=h["cert_vol_share"],
                           hyrange_false_copies=h["truth"]["false_certified_copies"])
            rows.append(row)
    json.dump(dict(generated_by="experiments/exports_table.py", rows=rows), open(OUT / "exports_table.json", "w",
                                                                                   encoding="utf-8"), indent=1)
    R = {(r["source"], r["t"]): r for r in rows}
    L = ["### Table V: certified share of the trace volume (%) in the exports", "",
         "| | t | " + " | ".join(PAPER[s] for s in ORDER) + " |", "|---|---|" + "---|" * len(ORDER)]
    for key, lab in (("pair_counts_share", "Pair counts"), ("forcing_share", "Forcing")):
        for t in THRESH:
            L.append(f"| {lab} | {t} | " + " | ".join(MB.pct1(R[(s, t)][key]) for s in ORDER) + " |")
    L.append("| HyRange | 5 | " + " | ".join(MB.pct1(R[(s, 5)].get("hyrange_share")) for s in ORDER) + " |")
    L.append("")
    L.append(f"False copies certified by forcing in the 25 exports: {sum(r['forcing_false_copies'] for r in rows)}; "
             f"by HyRange at t = 5: {sum(r.get('hyrange_false_copies', 0) for r in rows if r['t'] == 5)}.")
    L.append(f"Forcing certifies more than the pair counts alone in "
             f"{sum(1 for r in rows if r['forcing_share'] > r['pair_counts_share'])} of the 25 exports.")
    t5 = [R[(s, 5)] for s in ORDER]
    if all("hyrange_share" in r for r in t5):
        L.append(f"t = 5: HyRange {MB.pct1(min(r['hyrange_share'] for r in t5))}% to "
                 f"{MB.pct1(max(r['hyrange_share'] for r in t5))}%, pair counts alone "
                 f"{MB.pct1(min(r['pair_counts_share'] for r in t5))}% to {MB.pct1(max(r['pair_counts_share'] for r in t5))}%.")
    print("\n".join(L))


if __name__ == "__main__":
    main()
