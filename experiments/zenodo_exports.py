"""A deposited VOSviewer co-authorship map (Zenodo record 22729673; no raw records, no ground truth): the last
sentences of paragraph Exports with a document threshold.

Input: the VOSviewer map file and network file of the deposit (data/get_zenodo.py downloads them; parsed as plain
text only, never executed).
  * items: id and weight<Documents> from the map file; links: "id1 id2 strength" lines of the network file
    (duplicates and reversed duplicates are summed as VOSviewer does);
  * upper-bound-mode forcing exactly as upper.py (forcing_upper.WDForcer(adj, D, singleton=True), rules D1, A, B, P,
    wall limit 900 s), D_u = Documents_u;
  * pair counts alone: the closed form max(0, W_uv - sum_z min(W_uz, W_vz)).
Output: aggregate counts only -> results/exports/real/<name>/result.json: authors, links, certified copies (and how
many have three or more authors), the share of the total link strength covered by the pairs of the certified copies,
and the share certified from the pair counts alone.

usage: python -I -B -X utf8 experiments/zenodo_exports.py zenodo22729673_authors
           --map data/zenodo/zenodo22729673_authors/vosviewer_authors_map.txt
           --net data/zenodo/zenodo22729673_authors/vosviewer_authors_network.txt
"""
import csv
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
OUT = CFG.RESULTS / "exports" / "real"
UPPER_RULES = ("D1", "A", "B", "P")
FORCE_TL = 900.0
EPS = 1e-9


def read_text(p):
    b = Path(p).read_bytes()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        enc = "utf-16"
    elif b[:3] == b"\xef\xbb\xbf":
        enc = "utf-8-sig"
    elif len(b) > 1 and b[1:2] == b"\x00":
        enc = "utf-16-le"
    else:
        enc = "utf-8"
    try:
        return b.decode(enc)
    except UnicodeDecodeError:
        return b.decode("latin-1")


def rows_of(p):
    lines = [ln for ln in read_text(p).splitlines() if ln.strip() != ""]
    delim = next((d for d in ("\t", ";", ",") if d in lines[0]), "\t")
    return list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delim))


def to_num(s):
    try:
        return float(s.strip())
    except ValueError:
        return None


def parse_map(p):
    rows = rows_of(p)
    col = {h.strip(): i for i, h in enumerate(rows[0])}
    items = []
    for r in rows[1:]:
        if r:
            items.append(dict(id=r[col["id"]].strip(), documents=to_num(r[col["weight<Documents>"]])))
    return items


def parse_net(p):
    return [(r[0].strip(), r[1].strip(), to_num(r[2]) if len(r) > 2 else 1.0) for r in rows_of(p) if len(r) >= 2]


def closed_form_pairs(W):
    """sum over pairs of max(0, W_uv - sum_z min(W_uz, W_vz))."""
    tot = 0
    for u, au in enumerate(W):
        for v, w in au.items():
            if u < v:
                av = W[v]
                small, big = (au, av) if len(au) <= len(av) else (av, au)
                ub = 0
                for z, x in small.items():
                    if z != u and z != v and z in big:
                        ub += min(au[z], av[z])
                        if ub >= w:
                            break
                tot += max(0, w - ub)
    return int(tot)


def run(name, items, links):
    idx = {it["id"]: i for i, it in enumerate(items)}
    n = len(items)
    W = [dict() for _ in range(n)]
    for a, b, s in links:
        u, v = idx[a], idx[b]
        if u != v:
            W[u][v] = W[u].get(v, 0.0) + s
            W[v][u] = W[v].get(u, 0.0) + s
    strengths = [w for u, a in enumerate(W) for v, w in a.items() if u < v]
    assert all(abs(w - round(w)) < EPS for w in strengths), "non-integer link strengths"
    assert all(it["documents"] is not None and abs(it["documents"] - round(it["documents"])) < EPS for it in items)
    adj = [{v: int(round(w)) for v, w in a.items()} for a in W]
    D = [int(round(it["documents"])) for it in items]
    sumW = sum(int(round(w)) for w in strengths)
    from hyrange import forcing_upper
    F = forcing_upper.WDForcer(adj, D, rules=UPPER_RULES, time_limit=FORCE_TL, copy_adj=True, singleton=True)
    t0 = time.perf_counter()
    F.run()
    t = time.perf_counter() - t0
    forced = Counter({tuple(sorted(e)): int(m) for e, m in F.forced.items() if m > 0})
    pair_cov = int(sum(m * len(e) * (len(e) - 1) // 2 for e, m in forced.items()))
    cf = closed_form_pairs(adj)
    return dict(name=name, n_authors=n, n_links=len(strengths), sum_link_strength=sumW,
                upper_forcing=dict(rules=list(UPPER_RULES), t_force_s=round(t, 3), complete=bool(F.complete),
                                   certified_copies=int(sum(forced.values())),
                                   certified_copies_ge3_authors=int(sum(m for e, m in forced.items() if len(e) >= 3)),
                                   pair_coverage=pair_cov, share_of_total_link_strength=pair_cov / sumW),
                pairs_only_closed_form=dict(certified_pair_copies=cf, share_of_total_link_strength=cf / sumW))


def main():
    a = sys.argv[1:]
    name = a[0]
    opts = {a[i]: a[i + 1] for i in range(1, len(a) - 1, 2)}
    res = run(name, parse_map(opts["--map"]), parse_net(opts["--net"]))
    outp = OUT / name / "result.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    u, p = res["upper_forcing"], res["pairs_only_closed_form"]
    print(f"{res['n_authors']} authors, {res['n_links']} links; forcing certifies {u['certified_copies']} copies, "
          f"{u['certified_copies_ge3_authors']} of them with three or more authors, covering "
          f"{100 * u['share_of_total_link_strength']:.1f}% of the total link strength, against "
          f"{100 * p['share_of_total_link_strength']:.1f}% from the pair counts alone")


if __name__ == "__main__":
    main()
