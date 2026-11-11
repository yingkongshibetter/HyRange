"""Shared setup of the tests: repository root on sys.path, outputs in a temporary folder, and a small end-to-end
certification (forcing, components, candidates, rank test, one world, exact certification) of an input given as a
list of hyperedges."""
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HYRANGE_RESULTS", tempfile.mkdtemp(prefix="hyrange_test_"))

from hyrange import brute as BR  # noqa: E402
from hyrange import forcing as WC  # noqa: E402
from hyrange import quick as Q  # noqa: E402


def input_of(edges, n):
    """Pair counts (adjacency dicts) and degrees of a multiset of hyperedges on nodes 0..n-1."""
    return WC.build_wdeg([tuple(sorted(e)) for e in edges], n)


def certify(edges, n, seed=0):
    """HyRange in the exact mode (hyrange.quick.certify_input) on the input of `edges`, which are used only to build
    the input.  Returns (forced, certified, undecided)."""
    W, d = Q.input_of(edges, n)
    out = Q.certify_input(W, d, n, seed=seed)
    assert out["world"] is not None, "a component got no world"
    return out["forced"], out["certified"], out["undecided"]


def certain_by_brute_force(edges, n):
    """F^cert = minimum multiplicity over all worlds of the exact input, by brute force (hyrange.brute.worlds)."""
    adj, deg = input_of(edges, n)
    worlds = BR.worlds(adj, deg, n, exact=True)
    common = None
    for w in worlds:
        common = Counter(w) if common is None else (common & Counter(w))
    return worlds, (common or Counter())


def nodes_of(names):
    """'abc' -> (0, 1, 2)."""
    return tuple(sorted("abcdefgh".index(ch) for ch in names))
