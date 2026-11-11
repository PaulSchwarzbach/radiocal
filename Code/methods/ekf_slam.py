"""Sequential EKF-SLAM on the bipartite view.

State = [mobile x, y | anchor_0 x, y | ... | anchor_{n-1} x, y] (2 + 2n).
The filter runs once, causally, over the ranging epochs of a slam_bipartite
view:

- predict (between epochs): random walk on the mobile position - the mobile
  block of P grows by q_step^2 * I with q_step = V_MAX_MS * median epoch dt
  (slam_common; no velocity state). The anchor blocks get ZERO process
  noise, the same static-anchor convention as the static-suite ekf.py.
- update (within an epoch): each symmetrized mobile-anchor range is a
  scalar measurement h = ||p - a_k||, processed sequentially in anchor
  order (later updates are linearized at the already-updated state, as in
  ekf.py). The kernel axis uses the robust R-inflation convention of
  ekf.py: u = innovation / sigma, w = kernels.weight(u, kernel),
  R_eff = sigma^2 / w (update skipped when w < 1e-12); "gaussian" is the
  standard EKF with R = sigma^2.

sigma is the ABSOLUTE session noise scale handed in by the benchmark driver
(the session's fully_meshed-view kernels.session_sigma); sigma_m=None falls
back to slam_common.fallback_sigma for standalone use only.

Initialization: slam_common.bipartite_init (joint classical MDS over
anchors + strided epoch poses); the state starts at [first init pose,
init anchors] with diagonal covariance sigma0_m = 2.0 per coordinate (the
same generous prior as ekf.py). No gauge is fixed - the filter drifts in
the init frame and evaluation aligns, as for all methods.

Verified behaviour (set_b, 2026-07-07): the causal filter is
init-basin-limited. From the surveyed-anchor (oracle) init it converges on
both dyn sessions (0.38-0.88 m anchor RMSE, kernel-ordered); from the
production bipartite-MDS init it converges on s02 (init 1.05 m RMSE) but
not on s01 (init 1.52 m RMSE with two anchors ~3 m off, a ~20 sigma
innovation at sigma = 0.131 m: the robust weights collapse on exactly the
updates that could repair the init, and the gaussian variant freezes the
inconsistent geometry into an overconfident P). The batch fg_slam, from
the SAME init, re-linearizes over 15-35 LM iterations and recovers on both
sessions - a finding of the family comparison, not a bug.

Output per the frozen registry contract: est_df = final ANCHOR means only
(id, x, y, plus the posterior per-coordinate sigmas); info carries
"trajectory" (DataFrame seq, t, x, y - the CAUSAL filtered mobile estimate
after each epoch's updates, in the solver frame), "sigma_m", and scalar
diagnostics (n_poses, n_updates, init, init_pose_stride, q_step_m).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radiocal import MeasurementGraph

import kernels
import slam_common

KERNELS = kernels.KERNELS  # ("gaussian", "huber", "cauchy")

SIGMA0_M = 2.0  # initial per-coordinate std (m), as in ekf.py


def run_filter(stream: dict, x0: np.ndarray, P0: np.ndarray, sigma: float,
               q: float, kernel: str) -> dict:
    """The causal filter pass. x0/P0 over [mobile(2) | anchors(2n)]."""
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}")
    x = x0.copy()
    P = P0.copy()
    R = sigma ** 2
    e, k, r = stream["e"], stream["k"], stream["r"]
    E = len(stream["seqs"])
    starts = np.searchsorted(e, np.arange(E), side="left")
    ends = np.searchsorted(e, np.arange(E), side="right")

    traj = np.empty((E, 2))
    n_updates = 0
    for ep in range(E):
        if ep > 0:  # random walk on the mobile block only
            P[0, 0] += q ** 2
            P[1, 1] += q ** 2
        for p in range(starts[ep], ends[ep]):
            ka = int(k[p])
            ia = 2 + 2 * ka
            d = x[0:2] - x[ia:ia + 2]
            h = np.hypot(d[0], d[1])
            if h < 1e-9:
                continue
            u = d / h
            idxs = np.array([0, 1, ia, ia + 1])
            vals = np.array([u[0], u[1], -u[0], -u[1]])
            PHt = P[:, idxs] @ vals
            HPH = vals @ PHt[idxs]
            nu = r[p] - h
            if kernel == "gaussian":
                R_eff = R
            else:
                w = float(kernels.weight(np.array([nu / sigma]), kernel)[0])
                if w < 1e-12:
                    continue
                R_eff = R / w
            S = HPH + R_eff
            K = PHt / S
            x = x + K * nu
            P = P - np.outer(K, PHt)
            n_updates += 1
        P = 0.5 * (P + P.T)  # keep symmetric (per epoch is enough)
        traj[ep] = x[0:2]
    return {"x": x, "P": P, "traj": traj, "n_updates": n_updates}


def ekf_slam_anchors(graph: MeasurementGraph, variant: str = "gaussian",
                     sigma_m: float | None = None,
                     pose_stride: int = slam_common.POSE_STRIDE,
                     sigma0_m: float = SIGMA0_M
                     ) -> tuple[pd.DataFrame, dict]:
    """Run EKF-SLAM on a slam_bipartite view.

    variant: "gaussian", "huber" or "cauchy" (shared kernel axis, realized
    as the R-inflation of ekf.py). sigma_m: absolute session noise scale
    from the driver; None self-computes for standalone use (see
    slam_common.fallback_sigma). Returns (est_df with the ANCHOR rows id,
    x, y and posterior sigmas; info per the frozen registry contract,
    including the causal per-epoch "trajectory").
    """
    order, mobile_id, stream = slam_common.order_and_stream(graph)
    sigma = (sigma_m if sigma_m is not None
             else slam_common.fallback_sigma(graph, order, mobile_id))
    q = slam_common.q_step_m(stream["t"])
    n = len(order)

    anchors0, poses0, init_info = slam_common.bipartite_init(
        stream, n, pose_stride=pose_stride)
    x0 = np.concatenate([poses0[0], anchors0.ravel()])
    P0 = np.eye(2 + 2 * n) * sigma0_m ** 2

    out = run_filter(stream, x0, P0, sigma, q, variant)

    A = out["x"][2:].reshape(n, 2)
    sig = np.sqrt(np.diag(out["P"]))[2:].reshape(n, 2)
    est = pd.DataFrame({"id": order, "x": A[:, 0], "y": A[:, 1],
                        "sigma_x_m": sig[:, 0], "sigma_y_m": sig[:, 1]})
    info = {"anchor_order": order, "mobile_id": mobile_id,
            "sigma_m": sigma, "q_step_m": q, "sigma0_m": sigma0_m,
            "n_poses": len(stream["seqs"]),
            "n_epochs_used": len(stream["seqs"]),
            "n_updates": out["n_updates"],
            "trajectory": slam_common.trajectory_frame(stream, out["traj"]),
            **init_info}
    return est, info
