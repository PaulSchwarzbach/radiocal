"""Shared MeasurementGraph -> range-matrix utilities for the method implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from radiocal import MeasurementGraph

# Paper gauge convention (see radiocal README / frames.fix_gauge): origin,
# x-axis and positive-y anchor. Same ids exist in both Torgau campaigns.
# Also used as the default deterministic anchor ordering of all methods.
GAUGE_TRIPLE = ("5BB6", "4A23", "8B05")


def default_anchor_order(graph: MeasurementGraph) -> list[str]:
    """Gauge-triple anchors first (where present), remaining anchors sorted."""
    anchors = sorted(graph.nodes.loc[graph.nodes["role"] == "anchor", "id"])
    head = [a for a in GAUGE_TRIPLE if a in anchors]
    return head + [a for a in anchors if a not in head]


def estimation_order(graph: MeasurementGraph) -> list[str]:
    """Deterministic node ordering of everything a method estimates:
    the anchor order plus any mobile nodes that appear in the edges
    (mobile_assisted / slam_bipartite views; a quasi-static mobile is just
    one extra unknown). On views without mobile edges this equals
    default_anchor_order exactly, so fully_meshed behaviour is unchanged.
    """
    order = default_anchor_order(graph)
    mobiles = sorted(graph.nodes.loc[graph.nodes["role"] == "mobile", "id"])
    used = set(graph.edges["from_id"]) | set(graph.edges["to_id"])
    return order + [m for m in mobiles if m in used]


def graph_components(M: np.ndarray) -> int:
    """Connected components of the pair graph of a symmetric range matrix
    (finite off-diagonal entries = edges). 1 = connected."""
    n = M.shape[0]
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    iu, ju = np.triu_indices(n, k=1)
    for i, j in zip(iu[np.isfinite(M[iu, ju])], ju[np.isfinite(M[iu, ju])]):
        parent[find(int(i))] = find(int(j))
    return len({find(i) for i in range(n)})


def pair_range_matrix(graph: MeasurementGraph, anchor_order: list[str],
                      agg: str = "median") -> np.ndarray:
    """n x n symmetric matrix of session-aggregated undirected ranges.

    Each entry is `agg` over all per-epoch symmetrized ranges of that pair;
    NaN where a pair never ranged.
    """
    sym = graph.symmetrized()
    per_pair = sym.groupby(["node_a", "node_b"])["range_m"].agg(agg)
    idx = {a: i for i, a in enumerate(anchor_order)}
    M = np.full((len(anchor_order), len(anchor_order)), np.nan)
    for (a, b), r in per_pair.items():
        if a in idx and b in idx:
            M[idx[a], idx[b]] = M[idx[b], idx[a]] = r
    return M


def pair_counts(M: np.ndarray) -> tuple[int, int]:
    """(available, total) undirected pairs of a symmetric range matrix."""
    iu = np.triu_indices(M.shape[0], k=1)
    return int(np.isfinite(M[iu]).sum()), len(iu[0])


# Prior scale where a node's surveyed sigma_m is NaN. The nodes table is the
# convention's definition site (surveyed anchors carry 0.01 m); this constant
# only backstops incomplete metadata. Shared by the prior-anchored estimator
# and the CRLB machinery so both consume identical prior information.
PRIOR_SIGMA_FALLBACK_M = 0.01


def known_anchor_priors(graph: MeasurementGraph, order: list[str],
                        fallback_sigma_m: float = PRIOR_SIGMA_FALLBACK_M
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Known-anchor unary priors of a (masked) view, in estimation order.

    Returns (idx, xy, sigma_m): positions in ``order`` of the anchors that
    kept a positional prior under the mask, their surveyed coordinates, and
    the per-node prior scale (nodes-table sigma_m, ``fallback_sigma_m``
    where NaN). Mobile-node priors are excluded: a mobile's surveyed marker
    is truth bookkeeping, not reference information of the anchor problem.
    """
    p = graph.priors()
    p = p[(p["role"] == "anchor") & p["id"].isin(order)]
    pos = {a: i for i, a in enumerate(order)}
    idx = np.array([pos[a] for a in p["id"]], dtype=int)
    o = np.argsort(idx)
    xy = p[["x", "y"]].to_numpy(float)[o]
    sig = p["sigma_m"].to_numpy(float)[o]
    sig = np.where(np.isfinite(sig), sig, fallback_sigma_m)
    return idx[o], xy, sig
