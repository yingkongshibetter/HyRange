"""MARIOH benchmark inputs and metrics (Lee, Lee, Shin, "MARIOH: Multiplicity-Aware Hypergraph Reconstruction",
ICDE 2025, arXiv 2504.00522v2).

Shared by the pipeline, the MARIOH runner (MARIOH's environment), the refutation and the summaries.  Pure Python +
numpy only, so it imports in both environments.

Protocol (MARIOH's main.py / utils/data_processing.py, unchanged):
  preserved : target = data/<ds>/test_dup.txt, one hyperedge per line, repeats kept (preprocess_data_preserved: gt is
              a list); pair multiplicities W_uv counted over all lines.
  reduced   : target = data/<ds>/test.txt (preprocess_data: gt is a SET of distinct node sets); W_uv counted over all
              lines (so a repeated line, which exists only in eu/test.txt (4 lines), still counts twice in W).
  Lines with <= 1 node are dropped by MARIOH's loader; none of the 40 files has such a line.
Degree input: d_u = number of lines (hyperedges of >= 2 members) that contain u, counted over the same lines as W,
so (W, d) is exactly A^T A of the line incidence A.  No one-member hyperedges exist, so the exact mode applies.
Metrics (MARIOH's utils/evaluation.py): Jaccard over distinct node sets, multi-Jaccard sum min / sum max over the
multisets (truth = gt: a list in preserved mode, a set in reduced mode), plus exact recall |R & T| / |T| and
precision |R & T| / |R| over distinct node sets (R = distinct predicted sets, T = distinct true sets).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from itertools import combinations
from pathlib import Path

from hyrange import config as CFG  # noqa: E402

MAR = CFG.MARIOH                      # clone of the MARIOH repository (its data/ folder holds the ten datasets)
MAR_PY = CFG.MARIOH_PYTHON            # Python of the MARIOH environment
OUT = CFG.RESULTS / "bench"
DATASETS = ["enron", "pschool", "hschool", "crime", "hosts", "directors", "foursquare", "dblp", "eu", "topcs"]
DISP = {"enron": "Enron", "pschool": "P.School", "hschool": "H.School", "crime": "Crime", "hosts": "Hosts",
        "directors": "Directors", "foursquare": "Foursquare", "dblp": "DBLP", "eu": "Eu", "topcs": "MAG-TopCS"}
MODES = ["preserved", "reduced"]
SIX = ["crime", "hosts", "directors", "foursquare", "dblp", "topcs"]      # the six benchmarks of Tables I to VI
DENSE = ["enron", "pschool", "hschool", "eu"]                            # paragraph Dense hypergraphs

# Row MARIOH of Table III: the numbers of MARIOH's paper (arXiv 2504.00522v2), x100, mean +- std as printed there
# (its Table II: Jaccard, multiplicity-reduced setting; its Table III: multi-Jaccard, multiplicity-preserved setting).
PAPER = {
    "reduced": {"table": "MARIOH's Table II (Jaccard x100, multiplicity-reduced)",
                "crime": (100.00, 0.00), "hosts": (61.52, 2.63), "directors": (100.00, 0.00),
                "foursquare": (99.19, 0.34), "dblp": (98.13, 0.06), "topcs": (98.04, 0.12)},
    "preserved": {"table": "MARIOH's Table III (multi-Jaccard x100, multiplicity-preserved)",
                  "crime": (100.00, 0.00), "hosts": (58.41, 15.22), "directors": (100.00, 0.00),
                  "foursquare": (98.88, 0.46), "dblp": (97.48, 0.05), "topcs": (97.62, 0.03)},
}
PAPER_METRIC = {"reduced": "jaccard", "preserved": "multi_jaccard"}
SEEDS = (0, 1, 2)


def target_file(ds, mode):
    return MAR / "data" / ds / ("test_dup.txt" if mode == "preserved" else "test.txt")


def train_file(ds, mode):
    return MAR / "data" / ds / ("train_dup.txt" if mode == "preserved" else "train.txt")


def read_lines(path):
    """Hyperedges as sorted int tuples, one per line, lines of <= 1 node dropped (as MARIOH's loader)."""
    out = []
    with open(path, "r") as f:
        for line in f:
            nodes = list(map(int, line.strip().split()))
            if len(nodes) <= 1:
                continue
            out.append(tuple(sorted(nodes)))
    return out


def target(ds, mode):
    """(lines, truth): lines = the multiset whose projection is the input (W and d are counted over it);
    truth = Counter of MARIOH's gt (list in preserved mode, set in reduced mode)."""
    lines = read_lines(target_file(ds, mode))
    truth = Counter(lines) if mode == "preserved" else Counter(set(lines))
    return lines, truth


def pair_counts(H):
    """W of a Counter of hyperedges."""
    W = Counter()
    for e, m in H.items():
        for u, v in combinations(e, 2):
            W[(u, v)] += m
    return W


def degrees(H):
    d = Counter()
    for e, m in H.items():
        for u in e:
            d[u] += m
    return d


def metrics(pred, truth):
    """pred: iterable of node tuples (copies) or a Counter; truth: Counter.  MARIOH's Jaccard / multi-Jaccard plus
    exact recall / precision over distinct node sets."""
    P = pred if isinstance(pred, Counter) else Counter(tuple(sorted(int(x) for x in e)) for e in pred)
    P = Counter({e: m for e, m in P.items() if m > 0})
    Ps, Ts = set(P), set(truth)
    inter = len(Ps & Ts)
    union = len(Ps | Ts)
    keys = Ps | Ts
    smin = sum(min(P.get(k, 0), truth.get(k, 0)) for k in keys)
    smax = sum(max(P.get(k, 0), truth.get(k, 0)) for k in keys)
    return dict(jaccard=inter / union if union else 0.0,
                multi_jaccard=smin / smax if smax else 0.0,
                exact_recall=inter / len(Ts) if Ts else 0.0,
                precision=inter / len(Ps) if Ps else 0.0,
                n_pred_copies=int(sum(P.values())), n_pred_distinct=len(Ps), n_hit_distinct=inter)


def target_stats(lines, truth):
    H = Counter(lines)
    nodes = set(v for e in lines for v in e)
    sizes = Counter(len(e) for e in lines)
    dsz = Counter(len(e) for e in truth)
    vol = sum(len(e) for e in lines)
    return dict(nodes=len(nodes), hyperedge_copies=len(lines), distinct_hyperedges=len(H),
                truth_copies=int(sum(truth.values())), truth_distinct=len(truth),
                avg_multiplicity=len(lines) / len(H) if H else 0.0,
                size_hist_copies={int(k): int(v) for k, v in sorted(sizes.items())},
                size_hist_distinct={int(k): int(v) for k, v in sorted(dsz.items())},
                share_size2_copies=sizes.get(2, 0) / len(lines) if lines else 0.0,
                share_size2_distinct=dsz.get(2, 0) / len(truth) if truth else 0.0,
                mean_size=vol / len(lines) if lines else 0.0, max_size=max(sizes) if sizes else 0,
                mean_degree=vol / len(nodes) if nodes else 0.0, vol=vol,
                distinct_pairs=len(pair_counts(H)), pair_weight_sum=int(sum(pair_counts(H).values())))


def pct1(x):
    """A share x in [0, 1] as the experiments prints it: percent with one decimal (half up), '>99.9' for [99.95, 100)."""
    if x is None:
        return "-"
    from decimal import Decimal, ROUND_HALF_UP
    v = 100.0 * float(x)
    if 99.95 <= v < 100.0:
        return ">99.9"
    return str(Decimal(repr(v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def sec1(t):
    """Seconds as the experiments prints them: '<0.1', one decimal below 100 s, whole seconds with separators above."""
    if t is None:
        return "-"
    from decimal import Decimal, ROUND_HALF_UP
    if t < 0.05:
        return "<0.1"
    if t >= 100:
        return f"{int(Decimal(repr(float(t))).quantize(Decimal('1'), rounding=ROUND_HALF_UP)):,}"
    return str(Decimal(repr(float(t))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def save_hyperedges(H, path):
    """Counter of tuples -> json list of [members, multiplicity] (sorted)."""
    save_json([[list(e), int(m)] for e, m in sorted(H.items()) if m > 0], path)


def load_hyperedges(path):
    with open(path, "r", encoding="utf-8") as f:
        return Counter({tuple(int(x) for x in e): int(m) for e, m in json.load(f)})
