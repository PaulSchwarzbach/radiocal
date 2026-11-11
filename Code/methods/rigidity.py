"""2-D rigidity certificates of anchor pair graphs (multi_hop family).

Whether a set of pairwise ranges determines the anchor geometry at all is a
property of the pair graph, not of any estimator: the thinned views of the
multihop family need a structural certificate before method RMSE numbers
mean anything. This module provides the standard 2-D hierarchy:

- generic rigidity: rank of the rigidity matrix R (one row per edge,
  R[e, 2i:2i+2] = p_i - p_j and the negative at j) equals 2n - 3, the
  Asimow-Roth condition. Rank is a generic property, but the SURVEYED
  embedding may be non-generic (near-collinear triples etc.), so the rank
  is evaluated at seeded random perturbations of the surveyed coordinates
  (rank can only drop at special positions; the max over a few draws is the
  generic rank almost surely). Deterministic via a fixed seed.
- redundant rigidity: still generically rigid after removing any single
  edge (brute force over edges; n <= 12 here, so cost is negligible).
- vertex 3-connectivity: connected after removing any two vertices, brute
  force over vertex pairs + BFS. By convention this requires n >= 4
  (complete graphs K_2, K_3 are not 3-connected).
- generic GLOBAL rigidity (unique realization up to rigid motion +
  reflection): the Jackson-Jordan theorem - for n >= 4 a graph is
  generically globally rigid in 2-D iff it is redundantly rigid AND
  3-connected. For n <= 3 global rigidity holds iff the graph is complete
  (K_1, K_2, K_3); disconnected graphs are never rigid.

Inputs everywhere: ``coords`` (n, >= 2) array-like of anchor positions
(only x, y are used) and ``pairs`` as an iterable of undirected index pairs
(i, j); duplicates, orientation and self-loops are normalized away.
"""

from __future__ import annotations

import numpy as np

PERTURB_REL = 1e-3   # perturbation scale relative to the coordinate span
N_DRAWS = 3          # rank = max over this many seeded perturbations


def _norm_pairs(pairs) -> list[tuple[int, int]]:
    """Sorted unique undirected pairs (i < j), self-loops dropped."""
    out = {(min(int(i), int(j)), max(int(i), int(j)))
           for i, j in pairs if int(i) != int(j)}
    return sorted(out)


def _rigidity_matrix(P: np.ndarray, pairs: list[tuple[int, int]]
                     ) -> np.ndarray:
    """The (m, 2n) 2-D rigidity matrix at the embedding P."""
    R = np.zeros((len(pairs), 2 * len(P)))
    for e, (i, j) in enumerate(pairs):
        d = P[i] - P[j]
        R[e, 2 * i:2 * i + 2] = d
        R[e, 2 * j:2 * j + 2] = -d
    return R


def rigidity_rank(coords, pairs, seed: int = 0) -> int:
    """Generic rank of the 2-D rigidity matrix.

    Evaluated at N_DRAWS seeded random perturbations of ``coords`` (scale
    PERTURB_REL x coordinate span) and maximized: the rank only drops on a
    measure-zero set of embeddings, so the max is the generic rank a.s.
    """
    P0 = np.asarray(coords, dtype=float)[:, :2]
    plist = _norm_pairs(pairs)
    if not plist:
        return 0
    span = float(max(np.ptp(P0, axis=0).max(), 1.0))
    rng = np.random.default_rng(seed)
    rank = 0
    for _ in range(N_DRAWS):
        P = P0 + rng.normal(0.0, PERTURB_REL * span, P0.shape)
        rank = max(rank, int(np.linalg.matrix_rank(_rigidity_matrix(P,
                                                                    plist))))
    return rank


def is_generically_rigid(coords, pairs) -> bool:
    """Asimow-Roth: rank == 2n - 3 (n >= 2; n <= 1 is trivially rigid)."""
    n = len(np.asarray(coords, dtype=float))
    if n <= 1:
        return True
    return rigidity_rank(coords, pairs) == 2 * n - 3


def is_redundantly_rigid(coords, pairs) -> bool:
    """Generically rigid, and still rigid after removing any single edge."""
    plist = _norm_pairs(pairs)
    if not is_generically_rigid(coords, plist):
        return False
    for e in range(len(plist)):
        if not is_generically_rigid(coords, plist[:e] + plist[e + 1:]):
            return False
    return True


def _connected(adj: list[set], alive: set[int]) -> bool:
    """BFS connectivity of the induced subgraph on ``alive`` vertices."""
    if not alive:
        return True
    start = next(iter(alive))
    seen = {start}
    stack = [start]
    while stack:
        for w in adj[stack.pop()]:
            if w in alive and w not in seen:
                seen.add(w)
                stack.append(w)
    return seen == alive


def n_components(pairs, n: int) -> int:
    """Connected components of the pair graph on n vertices (isolated
    vertices count as components)."""
    adj = [set() for _ in range(n)]
    for i, j in _norm_pairs(pairs):
        adj[i].add(j)
        adj[j].add(i)
    left = set(range(n))
    comps = 0
    while left:
        start = left.pop()
        seen = {start}
        stack = [start]
        while stack:
            for w in adj[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        left -= seen
        comps += 1
    return comps


def is_3_connected(pairs, n: int) -> bool:
    """Vertex 3-connectivity, brute force: n >= 4, connected, and connected
    after removing every pair of vertices (fine at the n <= 12 here)."""
    if n < 4:
        return False
    adj = [set() for _ in range(n)]
    for i, j in _norm_pairs(pairs):
        adj[i].add(j)
        adj[j].add(i)
    if not _connected(adj, set(range(n))):
        return False
    for u in range(n):
        for v in range(u + 1, n):
            if not _connected(adj, set(range(n)) - {u, v}):
                return False
    return True


def global_rigidity_certificate(coords, pairs) -> dict:
    """Full 2-D certificate of a pair graph at (a generic perturbation of)
    the given embedding: component count, generic / redundant rigidity,
    3-connectivity and the Jackson-Jordan global-rigidity verdict."""
    P = np.asarray(coords, dtype=float)
    n = len(P)
    plist = _norm_pairs(pairs)
    comps = n_components(plist, n)
    gen = is_generically_rigid(P, plist) if comps == 1 else False
    red = is_redundantly_rigid(P, plist) if gen else False
    three = is_3_connected(plist, n)
    if n < 4:
        glob = (comps <= 1) and len(plist) == n * (n - 1) // 2
    else:
        glob = red and three
    return {"n": n, "n_edges": len(plist), "n_components": comps,
            "generically_rigid": gen, "redundantly_rigid": red,
            "three_connected": three, "globally_rigid": glob}


def certificate_class(cert: dict) -> str:
    """One-word ambiguity class of a certificate: disconnected / flexible /
    rigid_not_global / globally_rigid."""
    if cert["n_components"] > 1:
        return "disconnected"
    if not cert["generically_rigid"]:
        return "flexible"
    if not cert["globally_rigid"]:
        return "rigid_not_global"
    return "globally_rigid"
