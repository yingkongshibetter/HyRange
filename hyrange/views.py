"""Loader of the three OpenAlex samples (OA-DB, OA-CS, OA-Bio) used by the co-authorship exports.

Each sample is a folder <HYRANGE_OPENALEX>/<name>/ with incidence.npz, a scipy CSR matrix whose rows are OpenAlex
works and whose columns are authors (node_mapping.csv and hyperedge_mapping.csv give the OpenAlex ids).
Hyperedges with fewer than two members are dropped (none occur in the samples).
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

from hyrange import config as CFG

SAMPLES = ("openalex_db", "openalex_cs_size3plus", "openalex_bio")


def hyperedges_from_incidence(H):
    """Rows of a CSR incidence (rows = hyperedges, columns = nodes) as sorted tuples of node ids, size >= 2."""
    H = H.tocsr()
    out = []
    for i in range(H.shape[0]):
        e = H.indices[H.indptr[i]:H.indptr[i + 1]]
        if len(e) >= 2:
            out.append(tuple(sorted(int(x) for x in e)))
    return out


class Sample:
    """One sample: H = 0/1 int64 CSR incidence (works x authors) with sorted column indices."""

    def __init__(self, name, H):
        self.name = name
        self.H = H

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.H.indptr).astype(np.int64)


def setup_env(threads=6):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, str(threads))


def load(ds):
    """(sample, list of hyperedges) of one OpenAlex sample."""
    setup_env()
    if ds not in SAMPLES:
        raise ValueError(ds)
    H = sp.load_npz(CFG.OPENALEX / ds / "incidence.npz").tocsr()
    H = H.copy()
    H.data[:] = 1
    H = H.astype(np.int64)
    H.sort_indices()
    D = Sample(ds, H)
    return D, hyperedges_from_incidence(D.H)
