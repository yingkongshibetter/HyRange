"""Brute-force soundness on small random inputs: every certified copy lies in every world, and when no copy is left
undecided the certified multiset equals F^cert, the copies common to all worlds.

  test_exact_mode_random        HyRange in the exact mode (forcing, rank test, world, exact certification) against
                                hyrange.brute.worlds on random multihypergraphs with repeated hyperedges;
  test_upper_mode_validation    exact certification in the upper-bound mode and with degree bounds
                                (hyrange.upper_exact.validate) against brute-force enumeration;
  test_levels_validation        the levels (hyrange.levels_check.one) against the exact certain multiset.
The sizes are small so that the file runs in about a minute.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import certain_by_brute_force, certify  # noqa: E402

N_EXACT = 150
N_UPPER = 60
SEEDS_LEVELS = range(6)


def random_input(rng):
    n = rng.randint(4, 7)
    H = []
    for _ in range(rng.randint(2, 6)):
        if H and rng.random() < 0.25:
            H.append(rng.choice(H))
        else:
            H.append(tuple(sorted(rng.sample(range(n), rng.randint(2, min(4, n))))))
    return n, H


def test_exact_mode_random():
    rng = random.Random(20261111)
    multi = equal = 0
    for _ in range(N_EXACT):
        n, H = random_input(rng)
        worlds, fcert = certain_by_brute_force(H, n)
        assert Counter(H) in [Counter(w) for w in worlds]
        forced, cert, undecided = certify(H, n, seed=rng.randint(0, 2))
        for e, m in cert.items():
            assert m <= fcert.get(e, 0), ("certified copy outside a world", H, e, m)
        for e, m in forced.items():
            assert m <= fcert.get(e, 0), ("forced copy outside a world", H, e, m)
        if undecided == 0:
            assert +cert == +fcert, ("no undecided copy but certified != F^cert", H)
            equal += 1
        multi += len(worlds) > 1
    assert multi > 0 and equal > 0


def test_upper_mode_validation():
    from hyrange import upper_exact as RU
    RU.validate(N_UPPER)
    res = json.load(open(RU.OUT / "validation.json", encoding="utf-8"))
    assert res["complete"] and res["n_failures"] == 0, res["failures"][:3]
    for cfg, c in res["per_config"].items():
        assert c.get("soundness_violations", 0) == 0 and c.get("equality_violations", 0) == 0, cfg


def test_levels_validation():
    from hyrange import levels_check as V
    for s in SEEDS_LEVELS:
        r = V.one(s)
        assert not r.get("fails"), (s, r.get("fails"))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print("ok", name)
