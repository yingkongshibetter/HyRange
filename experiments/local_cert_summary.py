"""Paragraph Local certification: the levels on the 24 benchmark inputs (local_cert.py m4) and on the 13 dense samples
with known F^cert (local_cert.py sample, runs _node and _lv) -> results/local_cert/summary.json and stdout.

Benchmarks (results/local_cert/m4/<ds>_<setting>_<exact|upper>.json): forced share with the five rules (old_share), after
the levels at single nodes and at the node sets of single candidates (edge_share), the certified share of exact
certification as in the experiments (paper_certified: Table II, rows (W,d) and Upper), and the copies left to exact
certification before and after the levels (residual_before / residual_after copies).
Samples (results/local_cert/samples/<sid>_lv.json): forced share with the five rules and after the levels with up to
eight nodes beyond a candidate (stage e8), over the share of F^cert (ref_certified_share, dense_samples.py).

usage: python -I -B -X utf8 experiments/local_cert_summary.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402
from hyrange import dense_samples as RD  # noqa: E402
OUTD = CFG.RESULTS / "local_cert"


def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def stage_share(r, st):
    for s in r.get("snapshots", []):
        if s["stage"] == st:
            return s["forced_share"]
    return None


def main():
    out = dict(m4=[], samples=[])
    for md in ("exact", "upper"):
        for st in MB.MODES:
            for ds in MB.SIX:
                r = load(OUTD / "m4" / f"{ds}_{st}_{md}.json")
                if r is None or r.get("stage") != "done":
                    print(f"missing: m4 {ds} {st} {md}")
                    continue
                final = r["edge_share"] if r["edge_share"] is not None else r["node_share"]
                out["m4"].append(dict(ds=ds, setting=st, mode=md, old=r["old_share"], final=final,
                                      paper_certified=r.get("paper_certified"),
                                      reaches_paper=(r.get("paper_certified") is not None
                                                     and abs(final - r["paper_certified"]) < 1e-12),
                                      resid_copies_before=r["residual_before"]["copies"],
                                      resid_copies_after=r["residual_after"]["copies"],
                                      sound=r["soundness"]["verify_forced_ok"] and r["soundness"]["residual_ok"]))
    for sid in RD.LOCAL_CERT_SAMPLES:
        rl = load(OUTD / "samples" / f"{sid}_lv.json")
        if rl is None or rl.get("stage") != "done":
            print(f"missing: sample {sid}")
            continue
        cert = rl["ref_certified_share"]
        e8 = stage_share(rl, "e8")
        out["samples"].append(dict(sid=sid, mean_degree=rl["mean_degree"], certain=cert, old=rl["old_forced_share"],
                                   levels=e8, old_of_certain=rl["old_forced_share"] / cert, levels_of_certain=e8 / cert,
                                   reaches_certain=abs(e8 - cert) < 1e-12,
                                   sound=rl["soundness"]["verify_forced_ok"] and rl["soundness"]["residual_ok"]))
    with open(OUTD / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    L = []
    M = out["m4"]
    if M:
        L += ["| Mode | Setting | Dataset | five rules (%) | + levels (%) | exact certification (%) | copies left to exact "
              "certification: before / after |", "|---|---|---|---|---|---|---|"]
        for x in M:
            L.append(f"| {x['mode']} | {x['setting']} | {MB.DISP[x['ds']]} | {MB.pct1(x['old'])} | {MB.pct1(x['final'])} "
                     f"| {MB.pct1(x['paper_certified'])} | {x['resid_copies_before']} / {x['resid_copies_after']} |")
        up = [x for x in M if x["mode"] == "upper"]
        rat = [x["resid_copies_before"] / x["resid_copies_after"] for x in M if x["resid_copies_before"]]
        L += ["", f"The levels reach the certified share of exact certification on {sum(x['reaches_paper'] for x in M)} "
              f"of the {len(M)} inputs.",
              f"Upper-bound mode, forced without a world: {MB.pct1(min(x['old'] for x in up))}%-"
              f"{MB.pct1(max(x['old'] for x in up))}% -> {MB.pct1(min(x['final'] for x in up))}%-"
              f"{MB.pct1(max(x['final'] for x in up))}%." if up else "",
              f"Copies left to exact certification: {min(rat):.1f} to {max(rat):.1f} times fewer on the {len(rat)} inputs "
              "with a residual." if rat else "", ""]
    S = out["samples"]
    if S:
        L += ["| Sample | mean degree | five rules (% of F^cert) | + levels (% of F^cert) |", "|---|---|---|---|"]
        for x in S:
            L.append(f"| {x['sid']} | {x['mean_degree']:.0f} | {MB.pct1(x['old_of_certain'])} | "
                     f"{MB.pct1(x['levels_of_certain'])} |")
        L += ["", f"{len(S)} samples, mean degree {min(x['mean_degree'] for x in S):.0f} to "
              f"{max(x['mean_degree'] for x in S):.0f}: the five rules force {MB.pct1(min(x['old_of_certain'] for x in S))}%"
              f"-{MB.pct1(max(x['old_of_certain'] for x in S))}% of the certain volume, the levels "
              f"{MB.pct1(min(x['levels_of_certain'] for x in S))}%-{MB.pct1(max(x['levels_of_certain'] for x in S))}%, "
              f"all of it on {sum(x['reaches_certain'] for x in S)} samples."]
    print("\n".join(L))


if __name__ == "__main__":
    main()
