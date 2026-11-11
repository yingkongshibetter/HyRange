"""Paragraph Exact certification, last sentence: the share of the copies labeled ambiguous (copies of the found world
beyond ub(e)) that lie in the target, per seed over the inputs with ambiguous copies (Hosts, DBLP, MAG-TopCS, both
settings).

Inputs (all stored, nothing rerun): results/certification/cert_exact/bench_<ds>_<mode>.json (components[].cands: j, k,
x per seed, lb, ub; per_seed not_certain), results/bench/forced/<ds>_<mode>_forced.json (F),
results/bench/ours_pred/<ds>_<mode>_s<k>.json (output hypergraph per seed), and the MARIOH target files
(bench.target).  The node set of each candidate is recomputed with candidates.build_comp_wp on its residual component.

usage: python -I -B -X utf8 experiments/label_precision.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402
from hyrange import engine as EN  # noqa: E402
from hyrange import forcing as WC  # noqa: E402
from hyrange import candidates as WP  # noqa: E402

CE = CFG.RESULTS / "certification" / "cert_exact"
rows = []
for ds in ["hosts", "dblp", "topcs"]:
    for mode in ["preserved", "reduced"]:
        rec = json.load(open(CE / f"bench_{ds}_{mode}.json", encoding="utf-8"))
        lines, truth = MB.target(ds, mode)
        T = Counter(lines)                      # multiplicities of the lines the input is counted over
        nodes = sorted(set(v for e in lines for v in e))
        loc = {v: i for i, v in enumerate(nodes)}
        n = len(nodes)
        adj, deg = WC.build_wdeg([tuple(loc[v] for v in e) for e in lines], n)
        F = MB.load_hyperedges(MB.OUT / "forced" / f"{ds}_{mode}_forced.json")
        radj = [dict(a) for a in adj]
        for e, m in F.items():
            le = [loc[v] for v in e]
            for x in le:
                for y in le:
                    if y != x:
                        radj[x][y] -= m
        for x in range(n):
            for y in [y for y, w in radj[x].items() if w <= 0]:
                del radj[x][y]
        rdeg = list(deg)
        for e, m in F.items():
            for v in e:
                rdeg[loc[v]] -= m
        comps = sorted(EN._components(radj, n), key=len)
        assert len(comps) == rec["comps"]
        comp_of = {int(u): ci for ci, c in enumerate(comps) for u in c}
        # residual world per seed, as node sets of original labels
        X = [defaultdict(Counter) for _ in MB.SEEDS]   # X[s][ci][e] = residual multiplicity
        for s in MB.SEEDS:
            P = MB.load_hyperedges(MB.OUT / "ours_pred" / f"{ds}_{mode}_s{s}.json")
            for e, m in P.items():
                r = m - F.get(e, 0)
                if r > 0:
                    cs = {comp_of[loc[v]] for v in e}
                    assert len(cs) == 1
                    X[s][cs.pop()][e] += r
        per_seed = [dict(amb=0, amb_in_T=0) for _ in MB.SEEDS]
        for crec in rec["components"]:
            if not crec.get("treated"):
                continue
            ci = crec["index"]
            assert len(comps[ci]) == crec["nodes"]
            C = WP.build_comp_wp(comps[ci], radj, rdeg, cap=EN.CAP, enum_s=EN.ENUM_S)
            assert C.cands is not None and len(C.cands) == crec["ncands"]
            for c in crec["cands"]:
                e = tuple(sorted(nodes[int(C.nodes[i])] for i in C.cands[c["j"]]))
                assert len(e) == c["k"]
                assert [X[s][ci].get(e, 0) for s in MB.SEEDS] == c["x"], (ds, mode, ci, e)
                f = F.get(e, 0)
                t = T.get(e, 0)
                for s in MB.SEEDS:
                    x = X[s][ci].get(e, 0)
                    if x > c["ub"]:
                        per_seed[s]["amb"] += x - c["ub"]
                        per_seed[s]["amb_in_T"] += max(0, min(f + x, t) - (f + c["ub"]))
        for s in MB.SEEDS:
            assert per_seed[s]["amb"] == rec["per_seed"][s]["not_certain"], (ds, mode, s, per_seed[s])
            rows.append((MB.DISP[ds], mode, s, per_seed[s]["amb"], per_seed[s]["amb_in_T"]))

print("input | setting | seed | ambiguous copies | of which in the target")
for r in rows:
    print(" | ".join(str(v) for v in r))
shares = []
for s in MB.SEEDS:
    a = sum(r[3] for r in rows if r[2] == s)
    t = sum(r[4] for r in rows if r[2] == s)
    shares.append(t / a)
    print(f"seed {s}: {t} of {a} ambiguous copies lie in the target ({100 * t / a:.1f}%)")
print(f"over the three seeds: {100 * min(shares):.1f}% to {100 * max(shares):.1f}%")
