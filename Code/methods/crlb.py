"""Gauge-quotient Cramer-Rao bounds for the benchmark views (paper Sec. IV-E).

Fisher information of the stacked 2-D anchor coordinates under the zero-mean
Gaussian range model at the session noise scale sigma, evaluated at the
surveyed geometry:

    F = J^T J / sigma^2  +  sum_{k in K} I_2 / sigma_k^2  (per-seed blocks)

with J the full (un-reduced) residual Jacobian shared with the batch solver
(factor_graph._jacobian_full). Two bounds per view:

- aligned bound  sqrt(tr(F^+) / n): the bound on the quotient by the gauge
  (F^+ = Moore-Penrose pseudo-inverse), matching the reflection-allowed
  rigid-alignment RMSE metric (Ash & Moses 2008).
- absolute bound sqrt(tr(F^-1) / n): defined once the nullity is zero
  (generically |K| >= 2), matching evaluation without alignment.

Both are Gaussian bounds at the session sigma (paper Sec. IV-E). They limit
neither the error of a single estimate nor that of a biased estimator, and
on biased ranges they are no floor, since the per-pair bias breaks their
model (paper Sec. VI-C).

Nullity bookkeeping (d = 2): the blind gauge nullity is 3 (two translations,
one rotation; the reflection is discrete and invisible to a local bound),
one seed leaves 1 (rotation about the seed), two non-degenerate seeds leave
0. A nullity above expected_nullity(n_priors) means the view has continuous
flexes (below rigidity); the aligned bound then conditions on the flex modes
as well, and run_crlb records the view as status "flexible".
"""

from __future__ import annotations

import numpy as np

from radiocal import MeasurementGraph

from common import (estimation_order, known_anchor_priors, pair_counts,
                    pair_range_matrix)
from factor_graph import _jacobian_full, _offset_columns

# eigenvalues below RANK_TOL * lambda_max count as the nullspace
RANK_TOL = 1e-9


def range_fim(X: np.ndarray, pairs: np.ndarray, sigma_m: float) -> np.ndarray:
    """Range-factor Fisher information, 2n x 2n, at geometry X (n, 2)."""
    diff = X[pairs[:, 0]] - X[pairs[:, 1]]
    d = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
    J = _jacobian_full(diff, d, pairs, X.shape[0])
    return (J.T @ J) / sigma_m**2


def prior_fim(n: int, known_idx: np.ndarray,
              prior_sigma_m: np.ndarray) -> np.ndarray:
    """Unary-prior information: I_2 / sigma_k^2 on each seed's block."""
    F = np.zeros((2 * n, 2 * n))
    for i, s in zip(known_idx, prior_sigma_m):
        F[2 * i, 2 * i] = F[2 * i + 1, 2 * i + 1] = 1.0 / s**2
    return F


def expected_nullity(n_priors: int) -> int:
    """Gauge nullity of a rigid view given the seed count (d = 2)."""
    return {0: 3, 1: 1}.get(n_priors, 0)


def gauge_basis(X: np.ndarray) -> np.ndarray:
    """Orthonormal basis (2n x 3) of the infinitesimal rigid motions at
    geometry X: two translations and the rotation about the centroid."""
    n = X.shape[0]
    t1 = np.zeros(2 * n)
    t2 = np.zeros(2 * n)
    t1[0::2] = 1.0
    t2[1::2] = 1.0
    Xc = X - X.mean(axis=0)
    r = np.empty(2 * n)
    r[0::2] = -Xc[:, 1]
    r[1::2] = Xc[:, 0]
    U, _ = np.linalg.qr(np.column_stack([t1, t2, r]))
    return U


def bound(F: np.ndarray, X: np.ndarray | None = None,
          tol: float = RANK_TOL) -> dict:
    """Rank/nullity and the aligned / absolute RMSE bounds of a FIM.

    crlb_rmse_m bounds the metric the benchmark aligns: with geometry X it
    is the relative-error form tr(P F^+ P) with P projecting out the rigid
    motions at X (Ash & Moses 2008), so it matches the Procrustes-aligned
    RMSE at every |K| - the alignment absorbs frame error whether or not
    priors bound it. Without X (blind pseudo-inverse trace) the two forms
    coincide, since the nullspace then spans exactly the rigid motions.
    crlb_abs_rmse_m is the unprojected trace, defined once the nullity is
    zero. crlb_weak_mode_m is the weakest identifiable mode's contribution
    to the unprojected per-anchor bound, sqrt(1/(lambda_min * n)): near the
    rigidity edge a single near-flex mode can carry most of the bound. Under
    the Gaussian model the bound limits the covariance of an unbiased
    estimator, for a blind one unbiased up to the gauge and to first order,
    and neither the error of a single estimate nor that of a biased
    estimator, so an empirical error below it is no evidence of bias, least
    of all when one weak mode carries it (paper, Sec. IV-E).
    """
    lam, V = np.linalg.eigh(F)
    cut = tol * lam[-1]
    pos = lam > cut
    n = F.shape[0] // 2
    nullity = int((~pos).sum())
    inv_tr = float(np.sum(1.0 / lam[pos]))
    if X is None:
        rel_tr = inv_tr
    else:
        U = gauge_basis(X)
        Vp = V[:, pos]
        # tr(P F^+ P) = sum_i ||P v_i||^2 / lam_i, P = I - U U^T
        pv = 1.0 - np.sum((U.T @ Vp) ** 2, axis=0)
        rel_tr = float(np.sum(pv / lam[pos]))
    weak = float(np.sqrt(1.0 / (lam[pos].min() * n)))
    return {"fim_rank": int(pos.sum()),
            "fim_nullity": nullity,
            "crlb_rmse_m": float(np.sqrt(rel_tr / n)),
            "crlb_abs_rmse_m": (float(np.sqrt(inv_tr / n)) if nullity == 0
                                else float("nan")),
            "crlb_weak_mode_m": weak}


def view_bound(g: MeasurementGraph, ref, sigma_m: float,
               use_priors: bool = True) -> dict:
    """FIM bounds of one masked view at the surveyed geometry.

    ref is the unmasked session's surveyed reference
    (evaluation.surveyed_reference: columns id, x, y, z). Anchors of the
    view's estimation order must all be surveyed; mobile nodes are not part
    of the anchor FIM (pose-chain information is out of scope, paper
    Sec. IV-E). With use_priors the mask's surviving anchor priors enter as
    information blocks (common.known_anchor_priors, the same reference
    information the prior-anchored estimator consumes); without, the bound
    is the blind one - the caller matches the flag to what the view's
    method roster actually consumes, since a mask may expose priors that a
    blind roster ignores (the canonical multi_hop subset).
    """
    order = [a for a in estimation_order(g) if a in set(ref["id"])]
    anchors = set(g.nodes.loc[g.nodes["role"] == "anchor", "id"])
    order = [a for a in order if a in anchors]
    xy = ref.set_index("id").loc[order, ["x", "y"]].to_numpy(float)
    if not np.isfinite(xy).all():
        raise ValueError("unsurveyed anchor in view; FIM undefined")
    M = pair_range_matrix(g, order)
    iu, ju = np.triu_indices(len(order), k=1)
    obs = np.isfinite(M[iu, ju])
    pairs = np.column_stack([iu[obs], ju[obs]])
    if use_priors:
        idx, _, sig = known_anchor_priors(g, order)
    else:
        idx, sig = np.array([], dtype=int), np.array([])
    F = range_fim(xy, pairs, sigma_m) + prior_fim(len(order), idx, sig)
    out = bound(F, X=xy)
    available, total = pair_counts(M)
    out.update(n_anchors=len(order), n_pairs=available,
               n_pairs_total=total, n_priors=len(idx),
               sigma_m=float(sigma_m))
    return out


def offset_view_bound(g: MeasurementGraph, ref, sigma_m: float,
                      offset: str = "common",
                      prior_width: float | None = None) -> dict:
    """Blind bound of one anchor view with range-offset nuisance states
    (factor_graph.solve_nls_offset).

    Joint Fisher information of the coordinates and the offsets under the
    range model ||x_i - x_j|| + b + delta_i + delta_j, with the Gaussian
    prior 1 / prior_width^2 on each delta in device mode, reduced to the
    coordinates by the Schur complement F_xx - F_xb F_bb^-1 F_bx, then
    passed to bound(). offset="zero" returns the blind view_bound value.
    Anchors only, no anchor priors, as for the blind roster.
    """
    order = [a for a in estimation_order(g) if a in set(ref["id"])]
    anchors = set(g.nodes.loc[g.nodes["role"] == "anchor", "id"])
    order = [a for a in order if a in anchors]
    xy = ref.set_index("id").loc[order, ["x", "y"]].to_numpy(float)
    if not np.isfinite(xy).all():
        raise ValueError("unsurveyed anchor in view; FIM undefined")
    M = pair_range_matrix(g, order)
    n = len(order)
    iu, ju = np.triu_indices(n, k=1)
    obs = np.isfinite(M[iu, ju])
    pairs = np.column_stack([iu[obs], ju[obs]])
    diff = xy[pairs[:, 0]] - xy[pairs[:, 1]]
    d = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
    J = _jacobian_full(diff, d, pairs, n)
    B = _offset_columns(pairs, n, offset)
    F = (J.T @ J) / sigma_m**2
    if B.shape[1]:
        F_xb = (J.T @ B) / sigma_m**2
        F_bb = (B.T @ B) / sigma_m**2
        if offset == "device":
            F_bb[1:, 1:] += np.eye(n) / prior_width**2
        F = F - F_xb @ np.linalg.solve(F_bb, F_xb.T)
        F = 0.5 * (F + F.T)
    out = bound(F, X=xy)
    available, total = pair_counts(M)
    out.update(n_anchors=n, n_pairs=available, n_pairs_total=total,
               sigma_m=float(sigma_m))
    return out
