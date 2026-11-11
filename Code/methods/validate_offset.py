"""Validation of the offset-state factor graph and its bound.

Uses the surveyed set_b/s03 geometry (read-only) with synthetic ranges built
from the exact distances, so the true offsets are known:

  (i)   offset fixed at zero: solve_nls_offset equals solve_nls bit for bit;
  (ii)  noise-free ranges with one common offset: the common mode recovers
        the offset and the geometry exactly;
  (iii) noise-free ranges with per-device offsets: the device mode recovers
        every link offset b + delta_i + delta_j and the geometry (the split
        between b and the deltas is set by the prior, not by the data);
  (iv)  bound ordering: blind <= common <= device, and the device bound
        grows with the prior width;
  (v)   Monte-Carlo closure, common mode: the aligned RMSE of the gaussian
        offset solver under Gaussian noise approaches the offset-aware
        bound from above.

    python validate_offset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import crlb
import evaluation
import factor_graph as fg

DATA = Path(__file__).resolve().parents[2] / "Data"
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


def aligned_rmse(X: np.ndarray, ref: np.ndarray) -> float:
    est = pd.DataFrame({"id": [str(i) for i in range(len(X))],
                        "x": X[:, 0], "y": X[:, 1]})
    truth = pd.DataFrame({"id": [str(i) for i in range(len(X))],
                          "x": ref[:, 0], "y": ref[:, 1]})
    return evaluation.evaluate_estimate(est, truth)["rmse_m"]


def main() -> int:
    ref = evaluation.surveyed_reference(DATA / "set_b" / "s03")
    X = ref[["x", "y"]].to_numpy(float)
    n = len(X)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    np.fill_diagonal(D, np.nan)
    rng = np.random.default_rng(20260930)

    # (i) zero mode == blind solver on a noisy problem
    E = rng.normal(0, 0.05, (n, n))
    M = D + np.triu(E, 1) + np.triu(E, 1).T
    X0 = fg.initial_coords(M)
    for k in fg.KERNELS:
        Xb, ib = fg.solve_nls(M, X0, kernel=k, sigma=0.05)
        Xz, iz = fg.solve_nls_offset(M, X0, kernel=k, sigma=0.05,
                                     offset="zero")
        check(np.array_equal(Xb, Xz) and ib["n_iter"] == iz["n_iter"],
              f"(i) zero offset equals solve_nls bit for bit ({k})")

    # (ii) common offset, noise-free
    b = 0.15
    M = D + b
    np.fill_diagonal(M, np.nan)
    Xc, ic = fg.solve_nls_offset(M, fg.initial_coords(M), kernel="gaussian",
                                 sigma=0.05, offset="common")
    check(abs(ic["offset_common_m"] - b) < 1e-6
          and aligned_rmse(Xc, X) < 1e-6,
          f"(ii) common offset recovered: b {ic['offset_common_m']:.6f} m, "
          f"geometry {aligned_rmse(Xc, X):.1e} m")

    # (iii) per-device offsets, noise-free, wide prior
    delta = rng.normal(0, 0.05, n)
    L = b + delta[:, None] + delta[None, :]
    M = D + L
    np.fill_diagonal(M, np.nan)
    Xd, idv = fg.solve_nls_offset(M, fg.initial_coords(M),
                                  kernel="gaussian", sigma=0.05,
                                  offset="device", prior_width=10.0)
    Lhat = (idv["offset_common_m"] + idv["offset_device_m"][:, None]
            + idv["offset_device_m"][None, :])
    iu = np.triu_indices(n, 1)
    link_err = np.abs(Lhat[iu] - L[iu]).max()
    check(link_err < 1e-4 and aligned_rmse(Xd, X) < 1e-4,
          f"(iii) link offsets recovered to {link_err:.1e} m, geometry "
          f"{aligned_rmse(Xd, X):.1e} m (prior width 10 m)")

    # (iv) bound ordering on the synthetic view
    pairs = np.column_stack(iu)
    diff = X[pairs[:, 0]] - X[pairs[:, 1]]
    d = np.linalg.norm(diff, axis=1)
    J = fg._jacobian_full(diff, d, pairs, n)

    def bnd(offset, w=None, sigma=0.05):
        B = fg._offset_columns(pairs, n, offset)
        F = J.T @ J / sigma**2
        if B.shape[1]:
            Fxb, Fbb = J.T @ B / sigma**2, B.T @ B / sigma**2
            if offset == "device":
                Fbb[1:, 1:] += np.eye(n) / w**2
            F = F - Fxb @ np.linalg.solve(Fbb, Fxb.T)
        return crlb.bound(0.5 * (F + F.T), X=X)["crlb_rmse_m"]

    b0, bc = bnd("zero"), bnd("common")
    bd = [bnd("device", w) for w in (0.03, 0.10, 0.30)]
    check(b0 <= bc <= bd[0] <= bd[1] <= bd[2],
          f"(iv) bounds ordered: blind {b0:.4f} <= common {bc:.4f} <= "
          f"device {bd[0]:.4f} / {bd[1]:.4f} / {bd[2]:.4f} m")

    # (v) Monte-Carlo closure, common mode, gaussian noise
    sigma, trials, err2 = 0.05, 200, []
    for _ in range(trials):
        E = rng.normal(0, sigma, (n, n))
        M = D + b + np.triu(E, 1) + np.triu(E, 1).T
        np.fill_diagonal(M, np.nan)
        Xm, _ = fg.solve_nls_offset(M, fg.initial_coords(M),
                                    kernel="gaussian", sigma=sigma,
                                    offset="common")
        err2.append(aligned_rmse(Xm, X) ** 2)
    mc = float(np.sqrt(np.mean(err2)))
    bound_c = bnd("common", sigma=sigma)
    check(0.9 * bound_c <= mc <= 1.25 * bound_c,
          f"(v) MC closure: RMSE {mc:.4f} m against the common-offset "
          f"bound {bound_c:.4f} m over {trials} trials")

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILURES: ' + str(len(FAILURES))}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
