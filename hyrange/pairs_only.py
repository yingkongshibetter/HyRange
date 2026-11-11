"""What the pair counts W alone certify (Table II, row W; paragraphs Pair counts alone and Dense hypergraphs), on the
ten MARIOH benchmarks in both settings.

Per setting, from W of the target lines (no degrees):
  c(u, v) = max(0, w_uv - sum_{z != u, v} min(w_uz, w_vz))        (MARIOH's Lemma 1-2 closed form)
copies of the two-member hyperedge {u, v} are in every hypergraph whose pair counts are W (hyperedges of >= 2
members, repeats allowed).  the experiments proves that this is exactly the certain multiplicity under W alone: the
capped form max(0, w_uv - sum_z min(w_uz, w_vz, w_uv)) is the exact minimum (witness world: all pairs except
t_z triangles {u, v, z}), and the cap changes nothing once the value is positive (a positive value needs every
term below w_uv).  No hyperedge of >= 3 members is certain under W alone (it can be replaced by its pairs).  Both
forms are computed here and asserted equal.
Reported: certified copies, certified volume 2 * copies and its share of vol = sum_u d_u, checked against the truth
(every certified copy is a target line with at least that multiplicity; asserted), and the share of the volume held by
the target's hyperedges of two members (the most that W alone can certify).

usage: python -I -B -X utf8 experiments/pairs_only.py      -> results/certification/w_only.json
"""
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402

CERT_DIR = CFG.RESULTS / "certification"


def certified_pairs(adj):
    """{(u, v): c(u, v) > 0} with the closed form, and the same with the cap w_uv (asserted equal)."""
    out = {}
    nd = 0
    for u, au in adj.items():
        for v, w in au.items():
            if u >= v:
                continue
            av = adj[v]
            small, big = (au, av) if len(au) <= len(av) else (av, au)
            s = s_cap = 0
            for z in small:
                if z == u or z == v or z not in big:
                    continue
                m = min(au[z], av[z])
                s += m
                s_cap += min(m, w)
            c = max(0, w - s)
            c_cap = max(0, w - s_cap)
            if c != c_cap:
                nd += 1
            if c > 0:
                out[(u, v)] = c
    assert nd == 0, ("closed form differs from the capped (exact) form", nd)
    return out


def main():
    recs = []
    for mode in MB.MODES:
        for ds in MB.DATASETS:
            t0 = time.perf_counter()
            lines, truth = MB.target(ds, mode)
            line_mult = Counter(lines)
            adj = {}
            deg = Counter()
            for e, m in line_mult.items():
                for x in e:
                    deg[x] += m
                    ax = adj.setdefault(x, {})
                    for y in e:
                        if y != x:
                            ax[y] = ax.get(y, 0) + m
            vol = int(sum(deg.values()))
            cert = certified_pairs(adj)
            t = time.perf_counter() - t0
            bad = {p: c for p, c in cert.items() if line_mult.get(p, 0) < c}
            assert not bad, ("UNSOUND W-only certification", ds, mode, len(bad))
            copies = int(sum(cert.values()))
            true_pair_copies = int(sum(m for e, m in line_mult.items() if len(e) == 2))
            recs.append(dict(ds=ds, name=MB.DISP[ds], mode=mode, vol=vol, lines=len(lines),
                             target_two_member_copies=true_pair_copies,
                             target_two_member_vol_share=2 * true_pair_copies / vol,
                             w_only_certified_copies=copies, w_only_certified_distinct=len(cert),
                             w_only_certified_vol=2 * copies, w_only_certified_vol_share=2 * copies / vol,
                             w_only_certified_true=True, seconds=t))
    MB.save_json(dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                      method=("W-only certain multiplicity of two-member hyperedges, closed form max(0, w_uv - "
                              "sum_z min(w_uz, w_vz)) (= the exact minimum over the worlds of W); "
                              "nothing of >= 3 members is certain under W alone"),
                      all_sound=True, records=recs), CERT_DIR / "w_only.json")
    R = {(r["ds"], r["mode"]): r for r in recs}
    print("### Table II, row W: certified share of the volume (%) from the pair counts alone\n")
    print("| Setting | " + " | ".join(MB.DISP[d] for d in MB.SIX) + " |\n|---|" + "---|" * len(MB.SIX))
    for mode in MB.MODES:
        print(f"| {mode} | " + " | ".join(MB.pct1(R[(d, mode)]["w_only_certified_vol_share"]) for d in MB.SIX) + " |")
    six = [R[(d, m)] for d in MB.SIX for m in MB.MODES]
    print(f"\nSix benchmarks: at most {MB.pct1(max(r['w_only_certified_vol_share'] for r in six))}% with W alone; "
          f"hyperedges of two members hold at most {MB.pct1(max(r['target_two_member_vol_share'] for r in six))}% of "
          "the volume.")
    dense = [R[(d, "preserved")]["w_only_certified_vol_share"] for d in MB.DENSE]
    print(f"Dense hypergraphs, preserved: {MB.pct1(min(dense))}% to {MB.pct1(max(dense))}% with W alone.")


if __name__ == "__main__":
    main()
