"""Pose-chain robust factor graph on bipartite and mobile-assisted views.

Batch nonlinear least squares over ALL per-epoch mobile-anchor ranges of the
view, jointly estimating the mobile pose chain and the anchor coordinates:

    theta = [pose_0 .. pose_{E-1} (2 each), anchor_0 .. anchor_{n-1} (2 each)]

Residuals:

- range factors (one per symmetrized mobile-anchor measurement):
  f = (||p_e - a_k|| - r) / sigma, robustified via the shared factorial
  kernel axis (kernels.py: gaussian / huber k = 1.345 / cauchy c = 2.385) as
  IRLS weights w(f) inside the LM loop; cost = sum rho(f).
- anchor-anchor range factors, PRESENT ONLY where the view provides
  inter-anchor pairs (the mobile_assisted mask on a dynamic session; the
  slam_bipartite mask strips them): f = (||a_i - a_j|| - r_ij) / sigma with
  r_ij the per-pair session median (the static suite's aggregation), under
  the same kernel axis. The bipartite and mobile-assisted runs of one
  session therefore differ in exactly the anchor-anchor factor set - the
  edge-set contrast the benchmark is built to isolate.
- motion factors (one per consecutive epoch pair, per axis, always
  Gaussian - the kernel axis applies to range factors only):
  f = (p_{e+1} - p_e) / q_step with q_step = V_MAX_MS * median epoch dt
  (slam_common; random walk on position, no velocity state).

sigma is the ABSOLUTE session noise scale handed in by the benchmark driver
(the session's fully_meshed-view kernels.session_sigma); sigma_m=None falls
back to slam_common.fallback_sigma for standalone use only.

Solver: Levenberg-Marquardt with the same discipline and tolerances as the
static factor_graph.py (cost tol 1e-12 relative, step tol 1e-10, lambda
1e-3 with factor-10 raise/lower, Marquardt scaling with diag(J^T W J)) and
its own iteration cap MAX_ITER = 100 (the benchmark cells converge within
36), but on
SPARSE normal equations (scipy.sparse): the pose chain is block-tridiagonal
and each range factor couples one pose block to one anchor block
(arrowhead structure; s01 ~ 3589 epochs -> ~7200 parameters). The 3-dof
planar gauge null space (translation + rotation) is left free and handled
by the LM damping; evaluation aligns the estimate, as for all methods.

Initialization: slam_common.bipartite_init (joint classical MDS over
anchors + every 25th epoch pose on the shortest-path-completed bipartite
distance matrix; the same init family as the static suite). The pose stride
affects the init only - every epoch enters the solve.

Output per the frozen registry contract: est_df holds the ANCHOR estimates
only (id, x, y); info carries "trajectory" (DataFrame seq, t, x, y - the
final smoothed pose chain in the solver frame), "sigma_m", and scalar
diagnostics (n_poses, n_updates, n_iter, cost0, cost, converged, init,
init_pose_stride, q_step_m).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from radiocal import MeasurementGraph

import kernels
import slam_common

KERNELS = kernels.KERNELS  # ("gaussian", "huber", "cauchy")

# LM discipline mirrored from factor_graph.py (stated there a-priori)
MAX_ITER = 100
TOL = 1e-12       # relative robust-cost decrease of an accepted step
STEP_TOL = 1e-10  # max parameter change of an accepted step
LAM0, LAM_FACTOR, LAM_MIN, LAM_MAX = 1e-3, 10.0, 1e-12, 1e12


def _cost(theta: np.ndarray, E: int, n: int, e: np.ndarray, k: np.ndarray,
          r: np.ndarray, kernel: str, sigma: float, q: float,
          aa: dict | None = None
          ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Robust total cost and the range-factor geometry at theta.

    Returns (cost, f_range, unit (m, 2), f_aa, unit_aa): f_range is the
    whitened mobile-anchor range residual with unit the pose-minus-anchor
    direction (predicted distance floored at 1e-12 as in factor_graph.py);
    f_aa / unit_aa are the anchor-anchor analogues (empty arrays when the
    view carries no inter-anchor pairs).
    """
    P = theta[:2 * E].reshape(E, 2)
    A = theta[2 * E:].reshape(n, 2)
    diff = P[e] - A[k]
    d = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
    f = (d - r) / sigma
    fm = (P[1:] - P[:-1]) / q
    cost = float(kernels.rho(f, kernel).sum() + 0.5 * (fm ** 2).sum())
    if aa is not None:
        diff_aa = A[aa["i"]] - A[aa["j"]]
        d_aa = np.maximum(np.linalg.norm(diff_aa, axis=1), 1e-12)
        f_aa = (d_aa - aa["r"]) / sigma
        u_aa = diff_aa / d_aa[:, None]
        cost += float(kernels.rho(f_aa, kernel).sum())
    else:
        f_aa = np.empty(0)
        u_aa = np.empty((0, 2))
    return cost, f, diff / d[:, None], f_aa, u_aa


def _motion_normal(E: int, n: int, q: float) -> sp.csr_matrix:
    """Constant Gauss-Newton normal matrix of the Gaussian motion factors.

    Per axis and consecutive epoch pair the residual (p_{e+1} - p_e)/q adds
    the block [[1, -1], [-1, 1]] / q^2 - block-tridiagonal over the pose
    chain, zero on the anchor block.
    """
    N = 2 * (E + n)
    inv_q2 = 1.0 / q ** 2
    i0 = np.arange(2 * (E - 1))            # pose e,  axis interleaved
    i1 = i0 + 2                            # pose e+1, same axis
    rows = np.concatenate([i0, i1, i0, i1])
    cols = np.concatenate([i0, i1, i1, i0])
    data = np.concatenate([np.full(2 * (E - 1), inv_q2),
                           np.full(2 * (E - 1), inv_q2),
                           np.full(2 * (E - 1), -inv_q2),
                           np.full(2 * (E - 1), -inv_q2)])
    return sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


def solve_slam(stream: dict, anchors0: np.ndarray, poses0: np.ndarray,
               kernel: str, sigma: float, q: float,
               max_iter: int = MAX_ITER,
               aa: dict | None = None) -> tuple[np.ndarray, np.ndarray,
                                                dict]:
    """Sparse LM over the pose chain + anchors. Returns (P, A, info).

    ``aa`` (optional): anchor-anchor factor arrays i/j/r as produced by
    slam_common.anchor_pair_arrays - adds one robust range factor per
    inter-anchor pair, coupling two anchor blocks (mobile-assisted views).
    """
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}")
    e, k, r = stream["e"], stream["k"], stream["r"]
    E, n = len(stream["seqs"]), len(anchors0)
    N = 2 * (E + n)
    theta = np.concatenate([poses0.ravel(), anchors0.ravel()])

    # range-factor parameter indices: pose x/y, anchor x/y -> (m, 4)
    idx4 = np.column_stack([2 * e, 2 * e + 1,
                            2 * E + 2 * k, 2 * E + 2 * k + 1])
    rows16 = np.repeat(idx4, 4, axis=1).ravel()
    cols16 = np.tile(idx4, (1, 4)).ravel()

    # anchor-anchor factor indices: anchor i x/y, anchor j x/y -> (m_aa, 4)
    if aa is not None:
        idx4_aa = np.column_stack(
            [2 * E + 2 * aa["i"], 2 * E + 2 * aa["i"] + 1,
             2 * E + 2 * aa["j"], 2 * E + 2 * aa["j"] + 1])
    else:
        idx4_aa = np.empty((0, 4), dtype=int)
    rows16_aa = np.repeat(idx4_aa, 4, axis=1).ravel()
    cols16_aa = np.tile(idx4_aa, (1, 4)).ravel()

    A_motion = _motion_normal(E, n, q)
    inv_q2 = 1.0 / q ** 2

    cost, f, u, f_aa, u_aa = _cost(theta, E, n, e, k, r, kernel, sigma, q, aa)
    cost0 = cost
    lam = LAM0
    n_iter, converged = 0, False

    for n_iter in range(1, max_iter + 1):
        w = kernels.weight(f, kernel)
        # whitened range jacobian rows: [ux, uy, -ux, -uy] / sigma
        g4 = np.column_stack([u, -u]) / sigma
        data16 = (w[:, None, None] * g4[:, :, None]
                  * g4[:, None, :]).ravel()
        w_aa = kernels.weight(f_aa, kernel)
        g4_aa = np.column_stack([u_aa, -u_aa]) / sigma
        data16_aa = (w_aa[:, None, None] * g4_aa[:, :, None]
                     * g4_aa[:, None, :]).ravel()
        A = sp.coo_matrix((np.concatenate([data16, data16_aa]),
                           (np.concatenate([rows16, rows16_aa]),
                            np.concatenate([cols16, cols16_aa]))),
                          shape=(N, N)).tocsr() + A_motion

        grad = np.bincount(idx4.ravel(),
                           weights=((w * f)[:, None] * g4).ravel(),
                           minlength=N)
        grad += np.bincount(idx4_aa.ravel(),
                            weights=((w_aa * f_aa)[:, None] * g4_aa).ravel(),
                            minlength=N)
        P = theta[:2 * E].reshape(E, 2)
        dm = (P[1:] - P[:-1]) * inv_q2      # J_m^T f_m per pose pair, axis
        gm = np.zeros((E, 2))
        gm[:-1] -= dm
        gm[1:] += dm
        grad[:2 * E] += gm.ravel()

        diagA = np.maximum(A.diagonal(), 1e-12)
        accepted = False
        while lam <= LAM_MAX:
            H = A + sp.diags(lam * diagA)
            try:
                dtheta = spla.spsolve(H.tocsc(), -grad)
            except RuntimeError:
                lam *= LAM_FACTOR
                continue
            if not np.all(np.isfinite(dtheta)):
                lam *= LAM_FACTOR
                continue
            cand = theta + dtheta
            cost_c, f_c, u_c, f_aa_c, u_aa_c = _cost(cand, E, n, e, k, r,
                                                     kernel, sigma, q, aa)
            if cost_c < cost:
                accepted = True
                if (cost - cost_c <= TOL * max(cost, 1e-12)
                        or np.abs(dtheta).max() < STEP_TOL):
                    converged = True
                theta, cost = cand, cost_c
                f, u, f_aa, u_aa = f_c, u_c, f_aa_c, u_aa_c
                lam = max(lam / LAM_FACTOR, LAM_MIN)
                break
            lam *= LAM_FACTOR
        if not accepted:  # no downhill step at any damping: stationary
            converged = True
        if converged:
            break

    P = theta[:2 * E].reshape(E, 2)
    A_est = theta[2 * E:].reshape(n, 2)
    info = {"n_iter": n_iter, "cost0": cost0, "cost": cost,
            "converged": converged}
    return P, A_est, info


def fg_slam_anchors(graph: MeasurementGraph, variant: str = "huber",
                    sigma_m: float | None = None,
                    pose_stride: int = slam_common.POSE_STRIDE,
                    max_iter: int = MAX_ITER) -> tuple[pd.DataFrame, dict]:
    """Run pose-chain factor-graph SLAM on a slam_bipartite or
    mobile_assisted view.

    variant: "gaussian", "huber" or "cauchy" (shared kernel axis, range
    factors only). sigma_m: absolute session noise scale from the driver;
    None self-computes for standalone use (see slam_common.fallback_sigma).
    Anchor-anchor pairs present in the view (mobile_assisted) enter as
    additional robust range factors and strengthen the init's anchor block;
    on bipartite views the factor set is unchanged. Returns (est_df with
    the ANCHOR rows id, x, y; info per the frozen registry contract,
    including the per-epoch "trajectory").
    """
    order, mobile_id, stream = slam_common.order_and_stream(graph)
    sigma = (sigma_m if sigma_m is not None
             else slam_common.fallback_sigma(graph, order, mobile_id))
    q = slam_common.q_step_m(stream["t"])
    aa = slam_common.anchor_pair_arrays(graph, order)

    anchors0, poses0, init_info = slam_common.bipartite_init(
        stream, len(order), pose_stride=pose_stride,
        aa_matrix=None if aa is None else aa["matrix"])
    P, A, sol = solve_slam(stream, anchors0, poses0, variant, sigma, q,
                           max_iter=max_iter, aa=aa)

    est = pd.DataFrame({"id": order, "x": A[:, 0], "y": A[:, 1]})
    info = {"anchor_order": order, "mobile_id": mobile_id,
            "sigma_m": sigma, "q_step_m": q,
            "n_poses": len(stream["seqs"]),
            "n_epochs_used": len(stream["seqs"]),
            "n_updates": len(stream["r"]),
            "n_aa_pairs": 0 if aa is None else int(len(aa["r"])),
            "trajectory": slam_common.trajectory_frame(stream, P),
            **init_info, **sol}
    return est, info
