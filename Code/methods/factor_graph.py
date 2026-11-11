"""Factor-graph batch NLS anchor self-calibration.

Batch nonlinear least squares over the session-aggregated symmetric range
matrix: every observed anchor pair contributes one range factor

    r_ij(X) = ||x_i - x_j|| - m_ij

and the 2-D anchor coordinates X are estimated by minimizing sum_ij
rho(r_ij) with Gauss-Newton / Levenberg-Marquardt (analytic Jacobian).
The gauge is fixed by parameterization: anchor 0 of the ordering sits at
the origin, anchor 1 is constrained to y = 0 (only its x is a variable),
which removes the three planar gauge DOF. The leftover reflection is
resolved by the evaluation alignment, as for all methods.

Variants (per-range kernel rho, realized as IRLS weights inside the LM
loop) come from the shared benchmark kernel module ``kernels``:

- ``gaussian`` (alias ``l2``): quadratic - the plain NLS reference,
  equivalent in objective to SMACOF's raw stress over observed pairs.
- ``huber``: quadratic core, linear tails - bounds the pull of
  NLOS-inflated ranges without rejecting them.
- ``cauchy``: redescending influence, asymptotically ignores gross
  outliers.

Kernel constants (huber k = 1.345, cauchy c = 2.385, dimensionless) and
the session noise scale sigma (kernels.session_sigma: max of pair epoch
scatter and MDS-init residual MAD) are shared by all methods, so the
benchmark is factorial in estimator class x kernel under one noise model.

Initialization: classical MDS on the shortest-path-completed matrix
(mds.complete_distances + mds.classical_mds), rigidly moved into the gauge
parameterization.

Solver parameters (documented defaults): convergence when an accepted step
changes the robust cost by less than TOL = 1e-12 relative or moves no
parameter by more than 1e-10; damping lambda starts at 1e-3, x10 on
rejection, /10 on acceptance, Marquardt-scaled with diag(J^T W J).
MAX_ITER = 10000 outer LM iterations is a safety cap only: the canonical
views converge within 100 iterations, the random-thinning views need up to
about 1900, and info["converged"] records whether the rule ended the run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radiocal import MeasurementGraph, frames

from common import (default_anchor_order, known_anchor_priors, pair_counts,
                    pair_range_matrix)
import kernels
import mds

KERNELS = kernels.KERNELS  # ("gaussian", "huber", "cauchy")

MAX_ITER = 10_000  # safety cap; the convergence rule ends every benchmark cell
TOL = 1e-12       # relative robust-cost decrease of an accepted step
STEP_TOL = 1e-10  # max parameter change of an accepted step
LAM0, LAM_FACTOR, LAM_MIN, LAM_MAX = 1e-3, 10.0, 1e-12, 1e12


def _canonical_kernel(kernel: str) -> str:
    """Accept the historical alias "l2" for the gaussian kernel."""
    return "gaussian" if kernel == "l2" else kernel


def _rho(r: np.ndarray, kernel: str, sigma: float) -> np.ndarray:
    """Robust loss of a metric residual under the shared kernels."""
    return kernels.rho(r / sigma, kernel)


def _weights(r: np.ndarray, kernel: str, sigma: float) -> np.ndarray:
    """IRLS weights w = psi(u)/u of a metric residual, u = r / sigma."""
    return kernels.weight(r / sigma, kernel)


# ------------------------------------------------- gauge parameterization

def gauge_coords(X: np.ndarray) -> np.ndarray:
    """Rigidly move coords into the gauge: node 0 at the origin, node 1 on
    the x-axis (rotation only, no reflection - chirality is preserved)."""
    Xg = X - X[0]
    ang = np.arctan2(Xg[1, 1], Xg[1, 0])
    c, s = np.cos(ang), np.sin(ang)
    Xg = Xg @ np.array([[c, -s], [s, c]])
    Xg[0] = 0.0
    Xg[1, 1] = 0.0
    return Xg


def _pack(X: np.ndarray) -> np.ndarray:
    """(n, 2) gauged coords -> parameter vector of length 2n - 3."""
    return np.concatenate(([X[1, 0]], X[2:].ravel()))


def _unpack(theta: np.ndarray, n: int) -> np.ndarray:
    """Parameter vector -> (n, 2) coords with the gauge rows restored."""
    X = np.zeros((n, 2))
    X[1, 0] = theta[0]
    X[2:] = theta[1:].reshape(n - 2, 2)
    return X


def initial_coords(M: np.ndarray) -> np.ndarray:
    """Classical-MDS initialization (shortest-path completion), gauged."""
    D, _ = mds.complete_distances(M)
    X0, _ = mds.classical_mds(D, dim=2)
    return gauge_coords(X0)


# ------------------------------------------------------------- NLS solver

def _residuals(X: np.ndarray, pairs: np.ndarray, d_obs: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = X[pairs[:, 0]] - X[pairs[:, 1]]
    d = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
    return d - d_obs, diff, d


_GAUGE_COLS = (0, 1, 3)  # node 0 x/y and node 1 y: the gauge-fixed components


def _jacobian_full(diff: np.ndarray, d: np.ndarray, pairs: np.ndarray, n: int
                   ) -> np.ndarray:
    """Analytic Jacobian dr/dX of the stacked residuals, m x 2n, node i on
    columns 2i, 2i + 1, no gauge reduction.

    dr_ij/dx_i = (x_i - x_j)/d, dr_ij/dx_j = -(x_i - x_j)/d. Shared with the
    Fisher-information machinery (crlb.range_fim); up to a row scaling by d
    it is the rigidity matrix of the embedded graph (rigidity.py).
    """
    u = diff / d[:, None]
    m = len(d)
    J = np.zeros((m, 2 * n))
    rows = np.arange(m)
    for sign, nodes in ((1.0, pairs[:, 0]), (-1.0, pairs[:, 1])):
        J[rows, 2 * nodes] += sign * u[:, 0]
        J[rows, 2 * nodes + 1] += sign * u[:, 1]
    return J


def _jacobian(diff: np.ndarray, d: np.ndarray, pairs: np.ndarray, n: int
              ) -> np.ndarray:
    """Gauge-reduced Jacobian dr/dtheta, m x (2n - 3): the full Jacobian
    with the _GAUGE_COLS columns removed."""
    return np.delete(_jacobian_full(diff, d, pairs, n), _GAUGE_COLS, axis=1)


def solve_nls(M: np.ndarray, X0: np.ndarray, kernel: str = "gaussian",
              sigma: float = 1.0, max_iter: int = MAX_ITER, tol: float = TOL
              ) -> tuple[np.ndarray, dict]:
    """Levenberg-Marquardt over the observed pairs of a symmetric range
    matrix, from gauged initial coords X0.

    sigma is the shared session noise scale (kernels.session_sigma); the
    gaussian kernel is scale-invariant in its minimizer. Returns
    (coords (n, 2), info) with info holding n_iter, cost0/cost (robust cost
    sum rho(r/sigma)), converged flag, and the final per-pair residuals and
    IRLS weights aligned with info["pairs"].
    """
    kernel = _canonical_kernel(kernel)
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}")
    n = M.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    obs = np.isfinite(M[iu, ju])
    pairs = np.column_stack([iu[obs], ju[obs]])
    d_obs = M[pairs[:, 0], pairs[:, 1]]

    theta = _pack(X0)
    r, diff, d = _residuals(_unpack(theta, n), pairs, d_obs)
    cost = float(_rho(r, kernel, sigma).sum())
    cost0 = cost
    lam = LAM0
    n_iter, converged = 0, False

    for n_iter in range(1, max_iter + 1):
        w = _weights(r, kernel, sigma)
        J = _jacobian(diff, d, pairs, n)
        A = J.T @ (J * w[:, None])
        g = J.T @ (w * r)
        D = np.diag(np.maximum(np.diag(A), 1e-12))
        accepted = False
        while lam <= LAM_MAX:
            try:
                dtheta = np.linalg.solve(A + lam * D, -g)
            except np.linalg.LinAlgError:
                lam *= LAM_FACTOR
                continue
            cand = theta + dtheta
            r_c, diff_c, d_c = _residuals(_unpack(cand, n), pairs, d_obs)
            cost_c = float(_rho(r_c, kernel, sigma).sum())
            if cost_c < cost:
                accepted = True
                if (cost - cost_c <= tol * max(cost, 1e-12)
                        or np.abs(dtheta).max() < STEP_TOL):
                    converged = True
                theta, r, diff, d, cost = cand, r_c, diff_c, d_c, cost_c
                lam = max(lam / LAM_FACTOR, LAM_MIN)
                break
            lam *= LAM_FACTOR
        if not accepted:  # no downhill step at any damping: stationary
            converged = True
        if converged:
            break

    info = {"n_iter": n_iter, "cost0": cost0, "cost": cost,
            "converged": converged, "pairs": pairs, "resid": r,
            "weights": _weights(r, kernel, sigma)}
    return _unpack(theta, n), info


# ------------------------------------------------- anchored NLS solver

def solve_nls_anchored(M: np.ndarray, X0: np.ndarray,
                       kernel: str = "gaussian", sigma: float = 1.0, *,
                       prior_idx: np.ndarray, prior_xy: np.ndarray,
                       prior_sigma: np.ndarray, max_iter: int = MAX_ITER,
                       tol: float = TOL) -> tuple[np.ndarray, dict]:
    """Levenberg-Marquardt over the observed pairs plus Gaussian unary
    priors on the known anchors, full 2n parameterization (no gauge
    reduction).

    Whitened algebra throughout: range rows carry the IRLS kernel weights
    at scale sigma; prior rows are always Gaussian at their own
    prior_sigma, since surveyed coordinates are never robustified. The
    Marquardt damping regularizes whatever gauge directions the priors
    leave free (|K| <= 1). Cost = sum rho(r/sigma)
    + 0.5 sum ((x_k - p_k)/prior_sigma_k)^2, the negative log posterior of
    the MAP objective up to constants. A separate function rather than a
    flag on solve_nls, so the blind path's algebra stays byte-identical.
    """
    kernel = _canonical_kernel(kernel)
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}")
    n = M.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    obs = np.isfinite(M[iu, ju])
    pairs = np.column_stack([iu[obs], ju[obs]])
    d_obs = M[pairs[:, 0], pairs[:, 1]]
    k_idx = np.asarray(prior_idx, dtype=int)
    p_xy = np.asarray(prior_xy, dtype=float).reshape(len(k_idx), 2)
    p_sig = np.asarray(prior_sigma, dtype=float)
    cols = np.stack([2 * k_idx, 2 * k_idx + 1], axis=1).ravel()  # x,y slots

    def cost_of(X: np.ndarray, r: np.ndarray) -> float:
        pr = (X[k_idx] - p_xy) / p_sig[:, None] if len(k_idx) else 0.0
        return float(_rho(r, kernel, sigma).sum()
                     + 0.5 * np.sum(np.square(pr)))

    X = X0.copy()
    r, diff, d = _residuals(X, pairs, d_obs)
    cost = cost_of(X, r)
    cost0 = cost
    lam = LAM0
    n_iter, converged = 0, False

    for n_iter in range(1, max_iter + 1):
        w = _weights(r, kernel, sigma)
        J = _jacobian_full(diff, d, pairs, n)
        A = J.T @ (J * (w / sigma**2)[:, None])
        g = J.T @ (w * r) / sigma**2
        if len(k_idx):
            pr = ((X[k_idx] - p_xy) / p_sig[:, None] ** 2).ravel()
            A[cols, cols] += np.repeat(1.0 / p_sig**2, 2)
            g[cols] += pr
        D = np.diag(np.maximum(np.diag(A), 1e-12))
        accepted = False
        while lam <= LAM_MAX:
            try:
                dx = np.linalg.solve(A + lam * D, -g)
            except np.linalg.LinAlgError:
                lam *= LAM_FACTOR
                continue
            cand = X + dx.reshape(n, 2)
            r_c, diff_c, d_c = _residuals(cand, pairs, d_obs)
            cost_c = cost_of(cand, r_c)
            if cost_c < cost:
                accepted = True
                if (cost - cost_c <= tol * max(cost, 1e-12)
                        or np.abs(dx).max() < STEP_TOL):
                    converged = True
                X, r, diff, d, cost = cand, r_c, diff_c, d_c, cost_c
                lam = max(lam / LAM_FACTOR, LAM_MIN)
                break
            lam *= LAM_FACTOR
        if not accepted:  # no downhill step at any damping: stationary
            converged = True
        if converged:
            break

    info = {"n_iter": n_iter, "cost0": cost0, "cost": cost,
            "converged": converged, "pairs": pairs, "resid": r,
            "weights": _weights(r, kernel, sigma)}
    return X, info


# --------------------------------------------- offset-state NLS solver

OFFSET_MODES = ("zero", "common", "device")
# prior width of the per-device offsets: the nominal ranging accuracy of the
# DWM1001 modules (10 cm, MDEK1001 documentation), not a tuned value
OFFSET_PRIOR_WIDTH_M = 0.10


def _offset_columns(pairs: np.ndarray, n: int, offset: str) -> np.ndarray:
    """dr/d(offset states), m x n_b: one column of ones for the common
    offset b, plus the columns e_i + e_j of the per-device offsets delta in
    device mode. Constant, since the range model is linear in the offsets."""
    m = len(pairs)
    cols = []
    if offset in ("common", "device"):
        cols.append(np.ones((m, 1)))
    if offset == "device":
        E = np.zeros((m, n))
        rows = np.arange(m)
        E[rows, pairs[:, 0]] = 1.0
        E[rows, pairs[:, 1]] = 1.0
        cols.append(E)
    return np.hstack(cols) if cols else np.zeros((m, 0))


def solve_nls_offset(M: np.ndarray, X0: np.ndarray, kernel: str = "gaussian",
                     sigma: float = 1.0, *, offset: str = "common",
                     prior_width: float = OFFSET_PRIOR_WIDTH_M,
                     max_iter: int = MAX_ITER, tol: float = TOL
                     ) -> tuple[np.ndarray, dict]:
    """Levenberg-Marquardt with range-offset states.

    Range model r_ij = ||x_i - x_j|| + b + delta_i + delta_j - m_ij over the
    observed pairs, gauge-reduced coordinates as in solve_nls.

    - offset="zero": no offset states. The algebra is that of solve_nls
      operation for operation, which the reproduction check relies on.
    - offset="common": one common offset b, unconstrained.
    - offset="device": b plus one offset delta_i per node under the Gaussian
      prior delta_i ~ N(0, prior_width^2). The prior removes the direction
      (b + c, delta - c/2) that the ranges alone cannot resolve.

    The unwhitened algebra of solve_nls is kept (every term multiplied by
    sigma^2), so the prior rows carry precision sigma^2 / prior_width^2.
    Cost = sum rho(r/sigma) + 0.5 sum (delta_i / prior_width)^2.
    """
    kernel = _canonical_kernel(kernel)
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}")
    if offset not in OFFSET_MODES:
        raise ValueError(f"unknown offset mode {offset!r}")
    n = M.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    obs = np.isfinite(M[iu, ju])
    pairs = np.column_stack([iu[obs], ju[obs]])
    d_obs = M[pairs[:, 0], pairs[:, 1]]

    B = _offset_columns(pairs, n, offset)
    n_x, n_b = 2 * n - 3, B.shape[1]
    prec = np.zeros(n_b)
    if offset == "device":
        if not prior_width or prior_width <= 0:
            raise ValueError("device offsets need a positive prior_width")
        prec[1:] = sigma ** 2 / prior_width ** 2

    def resid(theta: np.ndarray):
        r, diff, d = _residuals(_unpack(theta[:n_x], n), pairs, d_obs)
        if n_b:
            r = r + B @ theta[n_x:]
        return r, diff, d

    def cost_of(theta: np.ndarray, r: np.ndarray) -> float:
        c = float(_rho(r, kernel, sigma).sum())
        if n_b:
            c += 0.5 * float(np.sum(prec * theta[n_x:] ** 2)) / sigma ** 2
        return c

    theta = np.concatenate((_pack(X0), np.zeros(n_b)))
    r, diff, d = resid(theta)
    cost = cost_of(theta, r)
    cost0 = cost
    lam = LAM0
    n_iter, converged = 0, False

    for n_iter in range(1, max_iter + 1):
        w = _weights(r, kernel, sigma)
        J = _jacobian(diff, d, pairs, n)
        if n_b:
            J = np.hstack((J, B))
        A = J.T @ (J * w[:, None])
        g = J.T @ (w * r)
        if n_b:
            A[n_x:, n_x:] += np.diag(prec)
            g[n_x:] += prec * theta[n_x:]
        D = np.diag(np.maximum(np.diag(A), 1e-12))
        accepted = False
        while lam <= LAM_MAX:
            try:
                dtheta = np.linalg.solve(A + lam * D, -g)
            except np.linalg.LinAlgError:
                lam *= LAM_FACTOR
                continue
            cand = theta + dtheta
            r_c, diff_c, d_c = resid(cand)
            cost_c = cost_of(cand, r_c)
            if cost_c < cost:
                accepted = True
                if (cost - cost_c <= tol * max(cost, 1e-12)
                        or np.abs(dtheta).max() < STEP_TOL):
                    converged = True
                theta, r, diff, d, cost = cand, r_c, diff_c, d_c, cost_c
                lam = max(lam / LAM_FACTOR, LAM_MIN)
                break
            lam *= LAM_FACTOR
        if not accepted:  # no downhill step at any damping: stationary
            converged = True
        if converged:
            break

    offsets = theta[n_x:]
    info = {"n_iter": n_iter, "cost0": cost0, "cost": cost,
            "converged": converged, "pairs": pairs, "resid": r,
            "weights": _weights(r, kernel, sigma),
            "offset_common_m": float(offsets[0]) if n_b else 0.0,
            "offset_device_m": (offsets[1:].copy() if offset == "device"
                                else np.zeros(0))}
    return _unpack(theta[:n_x], n), info


# ------------------------------------------------------------- interface

def factor_graph_anchors(graph: MeasurementGraph, variant: str = "huber",
                         anchor_order: list[str] | None = None,
                         agg: str = "median", sigma_mult: float = 1.0,
                         sigma_m: float | None = None,
                         X0: np.ndarray | None = None
                         ) -> tuple[pd.DataFrame, dict]:
    """Run factor-graph NLS self-calibration on a (masked) MeasurementGraph.

    variant: "gaussian" (alias "l2"), "huber" or "cauchy". sigma_mult scales
    the shared session sigma (scale-sweep diagnostics; 1.0 = benchmark).
    sigma_m, if given, is the absolute noise scale (any multiplier already
    applied) - the benchmark driver passes the session's fully_meshed-view
    sigma so every mask of a session shares one scale; None keeps the
    legacy self-computed sigma of this graph's own view.
    X0, if given, is an (n, 2) start geometry in anchor order (any frame, it
    is moved into the gauge) that replaces the classical-MDS initialization;
    the benchmark never passes it, run_controls.py uses it for the
    oracle-start control.
    Returns (est_df with id, x, y; info dict with pair availability, the
    shared noise scale, LM iteration count, initial/final robust cost, and a
    per-pair weight table ``pair_weights`` with columns node_a, node_b,
    range_m, resid_m, weight).
    """
    variant = _canonical_kernel(variant)
    order = default_anchor_order(graph) if anchor_order is None else list(anchor_order)
    M = pair_range_matrix(graph, order, agg=agg)
    X0 = (initial_coords(M) if X0 is None
          else gauge_coords(np.asarray(X0, dtype=float)))

    sigma = (sigma_m if sigma_m is not None
             else kernels.session_sigma(graph, order, agg=agg) * sigma_mult)

    X, sol = solve_nls(M, X0, kernel=variant, sigma=sigma)

    pairs = sol["pairs"]
    pw = pd.DataFrame({
        "node_a": [order[i] for i in pairs[:, 0]],
        "node_b": [order[j] for j in pairs[:, 1]],
        "range_m": M[pairs[:, 0], pairs[:, 1]],
        "resid_m": sol["resid"],
        "weight": sol["weights"]})

    available, total = pair_counts(M)
    info = {"anchor_order": order,
            "n_pairs_available": available,
            "n_pairs_total": total,
            "sigma_m": sigma,
            "n_iter": sol["n_iter"],
            "cost0": sol["cost0"],
            "cost": sol["cost"],
            "converged": sol["converged"],
            "pair_weights": pw}
    est = pd.DataFrame({"id": order, "x": X[:, 0], "y": X[:, 1]})
    return est, info


def factor_graph_anchored(graph: MeasurementGraph, variant: str = "huber",
                          anchor_order: list[str] | None = None,
                          agg: str = "median", sigma_mult: float = 1.0,
                          sigma_m: float | None = None
                          ) -> tuple[pd.DataFrame, dict]:
    """Prior-anchored factor-graph NLS on a (masked) MeasurementGraph.

    The known set is the mask's surviving anchor priors
    (common.known_anchor_priors), the same reference information the CRLB
    machinery consumes. Initialization maps the classical-MDS coords onto
    the priors: rigid Umeyama fit with reflection allowed for |K| >= 2
    (which selects the chirality branch nearer the priors), translation
    onto the seed for |K| = 1, and the blind gauge unchanged for |K| = 0,
    the consistency arm of the prior sweep. Everything else (kernels,
    constants, sigma doctrine, aggregation) matches factor_graph_anchors.
    """
    variant = _canonical_kernel(variant)
    order = default_anchor_order(graph) if anchor_order is None else list(anchor_order)
    M = pair_range_matrix(graph, order, agg=agg)
    X0 = initial_coords(M)

    sigma = (sigma_m if sigma_m is not None
             else kernels.session_sigma(graph, order, agg=agg) * sigma_mult)

    idx, p_xy, p_sig = known_anchor_priors(graph, order)
    if len(idx) >= 2:
        tf = frames.umeyama(X0[idx], p_xy, with_scale=False,
                            allow_reflection=True)
        X0 = tf.apply(X0)
    elif len(idx) == 1:
        X0 = X0 - X0[idx[0]] + p_xy[0]

    X, sol = solve_nls_anchored(M, X0, kernel=variant, sigma=sigma,
                                prior_idx=idx, prior_xy=p_xy,
                                prior_sigma=p_sig)

    pairs = sol["pairs"]
    pw = pd.DataFrame({
        "node_a": [order[i] for i in pairs[:, 0]],
        "node_b": [order[j] for j in pairs[:, 1]],
        "range_m": M[pairs[:, 0], pairs[:, 1]],
        "resid_m": sol["resid"],
        "weight": sol["weights"]})

    available, total = pair_counts(M)
    prior_err = (np.linalg.norm(X[idx] - p_xy, axis=1) if len(idx)
                 else np.array([]))
    info = {"anchor_order": order,
            "n_pairs_available": available,
            "n_pairs_total": total,
            "sigma_m": sigma,
            "n_iter": sol["n_iter"],
            "cost0": sol["cost0"],
            "cost": sol["cost"],
            "converged": sol["converged"],
            "prior_ids": [order[i] for i in idx],
            "n_priors_used": int(len(idx)),
            "prior_max_err_m": (float(prior_err.max()) if len(prior_err)
                                else float("nan")),
            "pair_weights": pw}
    est = pd.DataFrame({"id": order, "x": X[:, 0], "y": X[:, 1]})
    return est, info


def factor_graph_offset(graph: MeasurementGraph, variant: str = "huber",
                        anchor_order: list[str] | None = None,
                        agg: str = "median", sigma_mult: float = 1.0,
                        sigma_m: float | None = None, *,
                        offset: str = "common",
                        prior_width: float = OFFSET_PRIOR_WIDTH_M
                        ) -> tuple[pd.DataFrame, dict]:
    """Factor-graph NLS with range-offset states on a (masked)
    MeasurementGraph (solve_nls_offset). Initialization, kernels, sigma
    doctrine and aggregation match factor_graph_anchors, and the offsets
    start at zero. info adds the offset mode, the estimated common offset
    b, and in device mode the per-node offsets delta (by node ID), their
    prior width and their standard deviation over the nodes.
    """
    variant = _canonical_kernel(variant)
    order = default_anchor_order(graph) if anchor_order is None else list(anchor_order)
    M = pair_range_matrix(graph, order, agg=agg)
    X0 = initial_coords(M)

    sigma = (sigma_m if sigma_m is not None
             else kernels.session_sigma(graph, order, agg=agg) * sigma_mult)

    X, sol = solve_nls_offset(M, X0, kernel=variant, sigma=sigma,
                              offset=offset, prior_width=prior_width)

    pairs = sol["pairs"]
    pw = pd.DataFrame({
        "node_a": [order[i] for i in pairs[:, 0]],
        "node_b": [order[j] for j in pairs[:, 1]],
        "range_m": M[pairs[:, 0], pairs[:, 1]],
        "resid_m": sol["resid"],
        "weight": sol["weights"]})

    available, total = pair_counts(M)
    info = {"anchor_order": order,
            "n_pairs_available": available,
            "n_pairs_total": total,
            "sigma_m": sigma,
            "n_iter": sol["n_iter"],
            "cost0": sol["cost0"],
            "cost": sol["cost"],
            "converged": sol["converged"],
            "offset_mode": offset,
            "offset_common_m": sol["offset_common_m"],
            "pair_weights": pw}
    if offset == "device":
        dev = sol["offset_device_m"]
        info["offset_device_m"] = dict(zip(order, dev.tolist()))
        info["offset_prior_width_m"] = float(prior_width)
        info["offset_device_std_m"] = float(np.std(dev))
    est = pd.DataFrame({"id": order, "x": X[:, 0], "y": X[:, 1]})
    return est, info
