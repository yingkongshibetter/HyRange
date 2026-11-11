"""HyRange on a small input in one call (exact mode): forcing, residual components, candidates and rank test, one
world per component, and exact certification of that world (the steps of pipeline.py and certify.py, with their
default limits).

    from hyrange.quick import input_of, certify_input
    W, d = input_of([(0, 1, 2), (2, 3)], 4)      # pair counts and degrees of a hypergraph
    out = certify_input(W, d, 4)
    out["certified"]   # Counter of node tuples: the copies that lie in every hypergraph with these W and d

The input is the co-occurrence matrix only: W maps pairs (u, v), u < v, to counts, and d lists the degrees of the
nodes 0..n-1.  Raises forcing.Inconsistent if the rules meet a contradiction (the input has no world).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from hyrange import candidates as WP
from hyrange import certify as CX
from hyrange import engine as EN
from hyrange import forcing as WC
from hyrange import pipeline as MO


def input_of(hyperedges, n):
    """(W, d) of a multiset of hyperedges (tuples of node ids 0..n-1, at least two members each)."""
    adj, deg = WC.build_wdeg([tuple(sorted(e)) for e in hyperedges], n)
    W = {(u, v): w for u in range(n) for v, w in adj[u].items() if u < v}
    return W, list(deg)


def certify_input(W, d, n, seed=0, milp_tl=60.0):
    """Certified copies of the input (W, d) in the exact mode.  Returns a dict with
    forced (Counter), certified (Counter, forced plus the copies proved by the rank test and exact certification),
    world (Counter, one world of the input, None if a component got no world), undecided (copies of that world
    neither proved certain nor refuted) and components (number of residual components)."""
    adj = [dict() for _ in range(n)]
    for (u, v), w in W.items():
        if w:
            adj[u][v] = int(w)
            adj[v][u] = int(w)
    F = MO.WDForcerTL(adj, [int(x) for x in d], rules=MO.RULES, time_limit=900, copy_adj=True).run()
    forced = Counter({tuple(sorted(int(v) for v in e)): int(m) for e, m in F.forced.items() if m > 0})
    cert, world = Counter(forced), Counter(forced)
    undecided = 0
    rng = np.random.default_rng(seed)
    comps = EN._components(F.adj, n)
    for nodes in comps:
        C = WP.build_comp_wp(nodes, F.adj, F.deg)
        if C.cands is None:
            world = None
            continue
        if C.unique:
            x = C.xu
        else:
            x, _ = MO.world_of(C, rng, milp_tl)
        if x is None:
            world = None
            continue
        sets = {j: tuple(sorted(int(C.nodes[i]) for i in C.cands[j])) for j in np.nonzero(x)[0]}
        if world is not None:
            for j, e in sets.items():
                world[e] += int(x[j])
        if C.unique:
            for j, e in sets.items():
                cert[e] += int(x[j])
            continue
        rec = CX.certify_component(C, [x])
        for c in rec["cands"]:
            if c["lb"] > 0:
                cert[tuple(sorted(int(C.nodes[i]) for i in C.cands[c["j"]]))] += c["lb"]
            undecided += c["ub"] - c["lb"]
    return dict(forced=forced, certified=cert, world=world, undecided=undecided, components=len(comps))
