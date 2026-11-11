"""MDS anchor self-calibration from the pairwise range matrix (baseline).

Two variants on the session-aggregated symmetric range matrix:

- ``classical``: Torgerson/Gower double centering + eigendecomposition, the
  MDS-MAP recipe of Shang et al. (2003) - unobserved pairs are completed with
  shortest-path distances over the measured range graph before centering.
- ``smacof_gaussian`` (alias ``smacof``): stress majorization (de Leeuw's
  SMACOF), which needs no completion - missing pairs simply get weight
  zero - initialized from the classical solution.
- ``smacof_huber`` / ``smacof_cauchy``: IRLS-robust weighted SMACOF - the
  stress weights are multiplied by the shared benchmark kernel weights
  w(r / sigma) (kernels.py: huber k = 1.345, cauchy c = 2.385, session
  sigma from kernels.session_sigma), recomputed every Guttman iteration.
  This is the MDS cell of the factorial estimator x kernel benchmark.

Unlike the closed-form construction, every anchor is placed as long as the
range graph is connected, and all observed pairs enter the estimate, so a
single NLOS-biased range is averaged against instead of adopted. NLOS still
shows up globally: inflated ranges make the squared-distance matrix
non-Euclidean, visible as negative eigenvalue mass and residual stress.

The solution frame is gauge-free (radio_free) and chirality-blind; evaluation
aligns with reflection allowed, as for all methods.

Stopping rule (both SMACOF solvers): the Guttman iterations stop when the
minimized objective changes by less than TOL = 1e-13 relative between two
iterations, or when no coordinate moves by more than STEP_TOL = 1e-12 m
(stagnation guard at machine precision, reached on noise-free data).
MAX_ITER = 1e6 is a safety cap only: majorization converges linearly and
needs thousands of iterations on sparse views (up to 3.2e5 on the thinned
benchmark views), so a small fixed budget returns unconverged coordinates
there. info["converged"] records whether the rule, not the cap, ended the
run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radiocal import MeasurementGraph

from common import default_anchor_order, pair_counts, pair_range_matrix

MAX_ITER = 1_000_000  # safety cap; the stopping rule ends every benchmark cell
TOL = 1e-13           # relative objective change between two iterations
STEP_TOL = 1e-12      # max coordinate update in m (stagnation guard)


class DisconnectedGraphError(ValueError):
    """The measured range graph has more than one component, so shortest-path
    completion (and everything initialized from it) is undefined. A ValueError
    subclass so existing handlers keep working; the benchmark driver records
    it as the structured cell status "disconnected" (multihop thinning at low
    k produces it by construction - an outcome, not a failure)."""


def complete_distances(M: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill unobserved pairs with shortest-path distances (Floyd-Warshall)
    over the measured range graph. Returns (complete matrix, n filled).
    Raises DisconnectedGraphError if the range graph is disconnected."""
    D = np.where(np.isfinite(M), M, np.inf)
    np.fill_diagonal(D, 0.0)
    for k in range(D.shape[0]):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    if np.isinf(D).any():
        raise DisconnectedGraphError(
            "range graph is disconnected; MDS needs one component")
    iu = np.triu_indices(M.shape[0], k=1)
    return D, int((~np.isfinite(M[iu])).sum())


def classical_mds(D: np.ndarray, dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Torgerson MDS of a complete distance matrix.

    Returns (coords (n, dim), full eigenvalue spectrum descending). Negative
    eigenvalues (non-Euclidean noise/NLOS) are clipped to zero for the
    embedding but reported unclipped in the spectrum.
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    X = V[:, :dim] * np.sqrt(np.clip(w[:dim], 0.0, None))
    return X, w


def normalized_stress(X: np.ndarray, M: np.ndarray) -> float:
    """Normalized raw stress over the observed pairs of M (denominator
    sum of squared dissimilarities, the sklearn convention; Kruskal's
    stress-1 normalizes by the configuration distances instead)."""
    iu, ju = np.triu_indices(M.shape[0], k=1)
    obs = np.isfinite(M[iu, ju])
    delta = M[iu, ju][obs]
    d = np.linalg.norm(X[iu[obs]] - X[ju[obs]], axis=1)
    return float(np.sqrt(((d - delta) ** 2).sum() / (delta ** 2).sum()))


def smacof(M: np.ndarray, X0: np.ndarray, max_iter: int = MAX_ITER,
           tol: float = TOL) -> tuple[np.ndarray, float, int, bool]:
    """Weighted SMACOF: minimize raw stress over observed pairs only.

    Guttman-transform iterations from X0 until the normalized stress changes
    by less than tol relative, or no coordinate moves by more than STEP_TOL.
    Returns (coords, final stress-1, iterations used, converged).
    """
    W = np.isfinite(M).astype(float)
    np.fill_diagonal(W, 0.0)
    delta = np.where(np.isfinite(M), M, 0.0)
    Vp = np.linalg.pinv(np.diag(W.sum(axis=1)) - W)
    X = X0.copy()
    stress = normalized_stress(X, M)
    n_iter, converged = 0, False
    for n_iter in range(1, max_iter + 1):
        d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            B = -np.where(d > 0.0, delta / d, 0.0) * W
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(axis=1))
        X_new = Vp @ B @ X
        step = float(np.abs(X_new - X).max())
        X = X_new
        prev, stress = stress, normalized_stress(X, M)
        if abs(prev - stress) < tol * max(stress, 1e-12) or step < STEP_TOL:
            converged = True
            break
    return X, stress, n_iter, converged


def smacof_robust(M: np.ndarray, X0: np.ndarray, kernel: str = "huber",
                  sigma: float = 1.0, max_iter: int = MAX_ITER,
                  tol: float = TOL) -> tuple[np.ndarray, float, int, bool]:
    """IRLS-robust weighted SMACOF under the shared benchmark kernels.

    The observation weights are multiplied by the kernel weights
    w(r / sigma) of the current residuals and the Guttman transform is
    recomputed with the reweighted V every iteration ("gaussian" reduces to
    plain smacof()). Stops when the robust cost of an iteration changes by
    less than tol relative, or no coordinate moves by more than STEP_TOL.
    Returns (coords, stress-1 over observed pairs, iterations, converged).
    """
    import kernels as kern  # deferred: kernels.py imports mds at load time

    W_obs = np.isfinite(M).astype(float)
    np.fill_diagonal(W_obs, 0.0)
    delta = np.where(np.isfinite(M), M, 0.0)
    iu = np.triu_indices(M.shape[0], k=1)
    X = X0.copy()
    prev, step = None, np.inf
    n_iter, converged = 0, False
    for n_iter in range(1, max_iter + 1):
        d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        u = (d - delta) / sigma
        cost = float((W_obs * kern.rho(u, kernel))[iu].sum())
        if prev is not None and (abs(prev - cost) < tol * max(cost, 1e-12)
                                 or step < STEP_TOL):
            converged = True
            break
        prev = cost
        W = W_obs * kern.weight(u, kernel)
        Vp = np.linalg.pinv(np.diag(W.sum(axis=1)) - W)
        with np.errstate(divide="ignore", invalid="ignore"):
            B = -np.where(d > 0.0, delta / d, 0.0) * W
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(axis=1))
        X_new = Vp @ B @ X
        step = float(np.abs(X_new - X).max())
        X = X_new
    return X, normalized_stress(X, M), n_iter, converged


def mds_anchors(graph: MeasurementGraph, variant: str = "smacof",
                anchor_order: list[str] | None = None, agg: str = "median",
                dim: int = 2, sigma_mult: float = 1.0,
                sigma_m: float | None = None
                ) -> tuple[pd.DataFrame, dict]:
    """Run MDS self-calibration on a (masked) MeasurementGraph.

    variant: "classical", "smacof_gaussian" (alias "smacof"),
    "smacof_huber" or "smacof_cauchy". sigma_mult scales the shared session
    sigma (scale-sweep diagnostics; only the robust variants depend on it).
    sigma_m, if given, is the absolute noise scale (any multiplier already
    applied) - the benchmark driver passes the session's fully_meshed-view
    sigma so every mask of a session shares one scale; None keeps the
    legacy self-computed sigma of this graph's own view.
    Returns (est_df with id, x, y; info dict with pair availability,
    eigenvalue spectrum, stress, sigma, iterations, converged flag).
    """
    import kernels as kern  # deferred: kernels.py imports mds at load time

    if variant == "smacof":
        variant = "smacof_gaussian"
    order = default_anchor_order(graph) if anchor_order is None else list(anchor_order)
    M = pair_range_matrix(graph, order, agg=agg)
    D, n_completed = complete_distances(M)
    X, eig = classical_mds(D, dim=dim)
    available, total = pair_counts(M)
    info = {"anchor_order": order,
            "n_pairs_available": available,
            "n_pairs_total": total,
            "n_pairs_completed": n_completed,
            "eigenvalues": eig,
            "neg_eig_frac": float(np.abs(eig[eig < 0]).sum() / np.abs(eig).sum()),
            "sigma_m": np.nan,
            "n_iter": 0}
    if variant == "smacof_gaussian":
        X, stress, n_iter, converged = smacof(M, X)
        info.update(stress=stress, n_iter=n_iter, converged=converged)
    elif variant in ("smacof_huber", "smacof_cauchy"):
        sigma = (sigma_m if sigma_m is not None
                 else kern.session_sigma(graph, order, agg=agg) * sigma_mult)
        X, stress, n_iter, converged = smacof_robust(
            M, X, kernel=variant.split("_")[1], sigma=sigma)
        info.update(stress=stress, n_iter=n_iter, converged=converged,
                    sigma_m=sigma)
    elif variant == "classical":
        info["stress"] = normalized_stress(X, M)
    else:
        raise ValueError(f"unknown MDS variant {variant!r}")
    est = pd.DataFrame({"id": order, "x": X[:, 0], "y": X[:, 1]})
    return est, info
