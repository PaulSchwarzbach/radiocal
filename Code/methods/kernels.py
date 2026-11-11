"""Shared per-range loss kernels and the session noise scale.

The factorial benchmark compares estimator classes under an identical noise
model: every method uses the same dimensionless kernel rho(u), u = r / sigma,
and the same data-derived session scale sigma. Kernel constants are fixed
a-priori at the 95 % Gaussian-efficiency values of the M-estimation
literature - nothing is tuned against the evaluation metric.

    gaussian : rho(u) = u^2 / 2                        (L2; w = 1)
    huber    : quadratic core, linear tails, k = 1.345 (w = min(1, k/|u|))
    cauchy   : rho(u) = (c^2/2) log(1 + (u/c)^2), c = 2.385
               (redescending; w = 1 / (1 + (u/c)^2))

Session scale:

    sigma = max(sigma_scatter, sigma_init)

where sigma_scatter is the median over pairs of the per-pair epoch std of
symmetrized ranges (pairs with >= 10 epochs) and sigma_init is 1.4826 * MAD
of the range residuals of a classical-MDS embedding of the session-median
matrix. The scatter alone measures temporal precision and under-states the
per-pair systematic biases on clean sessions (set_b: 0.022 m scatter
vs 0.3-0.5 m biases); the init-residual MAD provides the geometric floor.
Values: set_b/s03 0.129 m, set_a/s21 1.621 m.

Sigma doctrine (benchmark driver, since the mask-family extension): sigma is
computed ONCE per session from its fully_meshed view and shared by every mask
of that session. Sigma measures the radio hardware and environment, which a
topology mask does not change; a per-mask sigma would confound the family's
connectivity/information axis, is undefined on thinned views that
disconnect (init_residual_scale needs one component), and measures motion
instead of noise on bipartite views. On fully_meshed itself the value is
unchanged (the regression gate pins it). session_sigma() itself still
computes from whatever graph it is handed; the legacy per-method runners
pass their (fully_meshed) view as before.

Usage per estimator family:
- batch NLS / IRLS (factor graph, robust SMACOF): weight w(r/sigma).
- filters (EKF, PF, grid): per-range negative log-likelihood rho(r/sigma)
  (gaussian gives the standard N(0, sigma^2) update; robust kernels give
  the corresponding pseudo-likelihoods).
"""

from __future__ import annotations

import numpy as np

from radiocal import MeasurementGraph

import mds
from common import pair_range_matrix

KERNELS = ("gaussian", "huber", "cauchy")
HUBER_K = 1.345
CAUCHY_C = 2.385
MIN_PAIR_EPOCHS = 10
SCATTER_FLOOR_M = 0.0  # the init-residual MAD is the effective floor


def rho(u: np.ndarray, kernel: str) -> np.ndarray:
    """Dimensionless loss rho(u), u = residual / sigma."""
    u = np.asarray(u, dtype=float)
    if kernel == "gaussian":
        return 0.5 * u ** 2
    if kernel == "huber":
        a = np.abs(u)
        return np.where(a <= HUBER_K, 0.5 * u ** 2,
                        HUBER_K * (a - 0.5 * HUBER_K))
    if kernel == "cauchy":
        return 0.5 * CAUCHY_C ** 2 * np.log1p((u / CAUCHY_C) ** 2)
    raise ValueError(f"unknown kernel {kernel!r}")


def weight(u: np.ndarray, kernel: str) -> np.ndarray:
    """IRLS / likelihood weight w(u) = psi(u) / u of the kernel."""
    u = np.asarray(u, dtype=float)
    if kernel == "gaussian":
        return np.ones_like(u)
    a = np.abs(u)
    if kernel == "huber":
        w = np.ones_like(u)
        big = a > HUBER_K
        w[big] = HUBER_K / a[big]
        return w
    if kernel == "cauchy":
        return 1.0 / (1.0 + (u / CAUCHY_C) ** 2)
    raise ValueError(f"unknown kernel {kernel!r}")


def scatter_scale(graph: MeasurementGraph,
                  min_count: int = MIN_PAIR_EPOCHS) -> float:
    """Median over pairs of the per-pair epoch std of symmetrized ranges."""
    grp = graph.symmetrized().groupby(["node_a", "node_b"])["range_m"]
    stds = grp.std()[grp.count() >= min_count]
    return float(max(stds.median(), SCATTER_FLOOR_M))


def init_residual_scale(M: np.ndarray) -> float:
    """1.4826 * MAD of the pair-range residuals of a classical-MDS embedding
    of the (shortest-path-completed) session matrix."""
    D, _ = mds.complete_distances(M)
    X0, _ = mds.classical_mds(D, dim=2)
    iu, ju = np.triu_indices(M.shape[0], k=1)
    obs = np.isfinite(M[iu, ju])
    d = np.linalg.norm(X0[iu[obs]] - X0[ju[obs]], axis=1)
    r = d - M[iu, ju][obs]
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def session_sigma(graph: MeasurementGraph, anchor_order: list[str],
                  agg: str = "median") -> float:
    """The shared session noise scale sigma (see module docstring)."""
    M = pair_range_matrix(graph, anchor_order, agg=agg)
    return max(scatter_scale(graph), init_residual_scale(M))
