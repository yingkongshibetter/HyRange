"""Brute-force enumeration of all worlds of a small input (library code of the tests and of hyrange.levels_check).

A world is a multiset of multi-member node sets (size >= 2) with pair counts exactly W and degree == D (exact mode)
resp. degree <= D (upper-bound mode; the rest of D is one-member groups).  Enumeration by pair-pivot branching: pick
the smallest positive residual pair, branch over every node set containing it whose pairs are all positive, recurse;
worlds deduplicated as sorted multisets.
"""
from collections import Counter
from itertools import combinations


def worlds(adj, D, n, exact):
    """All multisets of multi-member node sets with pair counts adj and degree == D (exact) / <= D (upper)."""
    out = set()
    subsets = [s for k in range(2, n + 1) for s in combinations(range(n), k)]

    def rec(res, deg, cur):
        pos = [(u, v) for (u, v) in combinations(range(n), 2) if res.get((u, v), 0) > 0]
        if not pos:
            if all(x == 0 for x in deg) if exact else all(x >= 0 for x in deg):
                out.add(tuple(sorted(cur)))
            return
        u, v = pos[0]
        for s in subsets:
            if u in s and v in s and all(res.get(p, 0) > 0 for p in combinations(s, 2)) and all(deg[x] > 0 for x in s):
                r2 = dict(res)
                for p in combinations(s, 2):
                    r2[p] -= 1
                d2 = list(deg)
                for x in s:
                    d2[x] -= 1
                rec(r2, d2, cur + [s])

    res0 = {(u, v): w for u in range(n) for v, w in adj[u].items() if u < v}
    rec(res0, list(D), [])
    return [Counter(w) for w in sorted(out)]
