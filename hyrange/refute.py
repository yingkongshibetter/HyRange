"""MARIOH + refutation on the six benchmarks (Table III, row MARIOH + Ref.; paragraph Refutation).

Inputs: the forced multiset F of the view (W, d) (pipeline.py) and MARIOH's reconstructions R
(experiments/marioh_runner.py, seeds 0-2).
  refutation: R minus the copies that no hypergraph consistent with (W, d) can contain.  With rW = W - W(F) and
              rd = d - d(F) (both from the view), the copies of a node set e in a world are at most
              F(e) + ub(e), ub(e) = min(min_{u<v in e} rW_uv, min_{u in e} rd_u) (0 if |e| < 2), because the worlds
              are exactly F + (worlds of the residual).  So max(0, R(e) - F(e) - ub(e)) copies of e are dropped.
Ground truth is used only to score, to count MARIOH's wrong copies, and to check that the refutation never drops a
true copy (asserted).

usage: python -I -B -X utf8 experiments/refutation.py <ds> <mode>   ->  results/bench/variants/<ds>_<mode>.json
"""
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
from hyrange import bench as MB  # noqa: E402


def refute(R, F, rW, rd):
    kept = Counter()
    refuted = 0
    for e, m in R.items():
        f = min(m, F.get(e, 0))
        extra = m - f
        if len(e) < 2:
            ub = 0
        else:
            ub = min(min(rW.get(p, 0) for p in combinations(e, 2)), min(rd.get(u, 0) for u in e))
            ub = max(0, ub)
        drop = max(0, extra - ub)
        refuted += drop
        if m - drop > 0:
            kept[e] = m - drop
    return kept, refuted


def main():
    ds, mode = sys.argv[1], sys.argv[2]
    lines, truth = MB.target(ds, mode)
    H = Counter(lines)
    fp = MB.OUT / "forced" / f"{ds}_{mode}_forced.json"
    if not fp.exists():
        print("no forced multiset for", ds, mode)
        return
    F = MB.load_hyperedges(fp)
    assert all(H.get(e, 0) >= m for e, m in F.items()), "forced copy not in the target"
    W, d = MB.pair_counts(H), MB.degrees(H)          # the view (W, d); H itself is not used below except to score
    WF, dF = MB.pair_counts(F), MB.degrees(F)
    rW = {k: W[k] - WF.get(k, 0) for k in W}
    rd = {u: d[u] - dF.get(u, 0) for u in d}
    assert min(rW.values(), default=0) >= 0 and min(rd.values(), default=0) >= 0
    out = dict(ds=ds, mode=mode, forced_copies=int(sum(F.values())), refutation=[])
    for s in MB.SEEDS:
        pm = MB.OUT / "marioh" / f"{ds}_{mode}_s{s}_pred.json"
        if pm.exists():
            R = MB.load_hyperedges(pm)
            kept, ref = refute(R, F, rW, rd)
            # soundness of the refutation: no true copy is dropped
            lost = sum(max(0, min(m, H.get(e, 0)) - kept.get(e, 0)) for e, m in R.items())
            assert lost == 0, ("refutation dropped true copies", lost)
            wrong = sum(max(0, m - H.get(e, 0)) for e, m in R.items())
            out["refutation"].append(dict(seed=s, marioh_copies=int(sum(R.values())), marioh_wrong_copies=int(wrong),
                                          refuted_copies=int(ref), **MB.metrics(kept, truth)))
    MB.save_json(out, MB.OUT / "variants" / f"{ds}_{mode}.json")
    for r in out["refutation"]:
        print(ds, mode, "refutation", r["seed"], {m: round(r[m], 4) for m in ("jaccard", "multi_jaccard")},
              f"refuted {r['refuted_copies']} of {r['marioh_wrong_copies']} wrong copies")


if __name__ == "__main__":
    main()
