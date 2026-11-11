"""Grid-filter anchor self-calibration (reference method).

Port of the histogram + probability-grid filters from the IOPS
auto-positioning study (module ``iops_autopositioning_qorvo/positioning.py``
of the study code, not released, classes ``histogram_filter`` and
``probability_grid_anchor``, driven as in ``main_autocalibration.py``),
lifted onto the ``radiocal.MeasurementGraph``
interface and onto the shared noise model of
``kernels.py`` (one session sigma, a-priori kernel constants).

Pipeline (2-D, gauge = radio_conv): anchor 0 is fixed at the origin; anchor 1
lives on a 1-D histogram over its x-coordinate (distance along the baseline)
with per-epoch updates from the measured 0-1 range; every anchor a >= 2
holds a 2-D probability grid, initialized once via the closed-form triple
from its ranges to anchors 0 and 1 (``meas_init``) and then updated per
epoch against every anchor that already has an estimate. Estimates update in
place during the sequential per-epoch sweep, so earlier anchors feed later
ones exactly like the original loop. The per-anchor estimate is the
probability-weighted mean of the cells within RADIUS_EST_M around the MAP
cell (original ``estimate()``).

Variants (the factorial cells of this estimator class):

- ``gaussian`` / ``huber`` / ``cauchy``: textbook independent-measurement
  product update; per-range log-likelihood = -kernels.rho(r / sigma, kernel)
  with the shared session sigma, accumulated in the log domain. The same
  kernel drives the 1-D histogram of anchor 1.
- ``mixture``: the mechanism peculiarity of the original IOPS code, kept for
  study: ``measurement()`` SUMS Gaussian likelihood components over the
  available reference anchors (p += term per reference) before multiplying
  into the posterior - a mixture ("consistent with at least one reference"),
  not a product. Components are Gaussian at the shared sigma; reported as
  "mixture(gaussian)".

Deviations from the original (each one deliberate; effects reported in
results/grid_filter/summary.md):

1.  Config constants -> shared/data-derived module constants. The original
    loaded a precomputed grid file (grid_office_4_6_02.npy: 4 x 6 m, 0.2 m
    cells, 600 cells) and fixed likelihood stds in config.py
    (RANGE_STD_HIST = 1 m, RANGE_STD_GRID_ANCHORS = 0.5 m). Here the grid
    is a square box of half-width (max median range around anchor 0 +
    MARGIN_M) at GRID_RES_M resolution, and the likelihood scale is the
    shared session sigma of kernels.session_sigma (an earlier version used a
    method-private per-pair scatter estimate).
2.  Histogram span. The original fixed the 1-D histogram to [0, 6] m with
    100 cells - far too small for these sites. The span is [0, grid
    half-width] at GRID_RES_M resolution.
3.  ``motion()`` dropped, floor kept. The site is static; of the original
    motion step only the probability-floor regularization survives, applied
    after every update. The original floored at the absolute value 1e-5,
    which on its 600-cell grid equals 6e-3 x the uniform density; a fixed
    1e-5 on a 10^4..10^5-cell grid would exceed uniform density and act as
    a dominant per-epoch forgetting factor, so the ratio is preserved:
    floor = P_FLOOR_REL / n_cells.
4.  ``meas_init`` for every anchor >= 2, with uniform-start fallback. The
    original applied the closed-form init only to anchor index 2 and let
    later anchors start from a uniform grid; here every grid anchor first
    tries the closed-form init from its ranges to anchors 0 and 1, and if
    that triple is unavailable or its radicand is negative (anchors near
    the 0-1 baseline, where histogram quantization of d01 can push
    d2^2 - x^2 < 0), the anchor starts from the uniform grid as soon as at
    least 3 references are visible - the original's behavior for anchors
    >= 3. The init bump keeps the original's 0.5 m std (SIGMA_INIT_M):
    a single-epoch trilateration is much less precise than the ranging
    noise, and the init is part of the historical pipeline, outside the
    kernel axis.
5.  Anchor 1 stays on the 1-D histogram (y = 0) for the whole run. The
    original promoted it to a 2-D grid anchor after its init phase, letting
    the gauge baseline drift off the x-axis; evaluation aligns either way.
6.  Reference-variance inflation removed. The original inflated
    the per-reference likelihood variance by the reference anchor's
    posterior variance (std**2 + var_ref**0.5, dimensionally inconsistent);
    an earlier version used a corrected std**2 + var_ref. The factorial design needs
    one shared noise scale per session, so all updates now use the plain
    shared sigma; posterior variances are kept as diagnostics only.
7.  Posterior variance windowed. The original took the weighted variance
    over the whole grid; with the floor mass spread over a large grid that
    alone contributes O(10 m^2), so the variance is computed over the same
    RADIUS_EST_M window around the MAP cell that the estimate uses (also
    for the 1-D histogram).
8.  No self-range term. The original zeroed the diagonal of the per-epoch
    range matrix, so each anchor's update contained a range-0 pseudo-
    measurement to its own previous estimate; this port is driven by
    measured edges only, where no self-edge exists.
9.  Epoch subsampling. Updates run on every stride-th epoch with stride =
    max(1, n_epochs // TARGET_UPDATE_EPOCHS); the site is static and the
    posterior converges long before that. The stride is reported in the
    info dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radiocal import MeasurementGraph

import kernels
from common import default_anchor_order

GRID_RES_M = 0.25           # cell size; cells grow quadratically with 1/res
MARGIN_M = 5.0              # box margin beyond max median range around anchor 0
SIGMA_INIT_M = 0.5          # std of the one-time closed-form init bump (original cfg)
RADIUS_EST_M = 4.0          # estimate/variance window radius (original cfg)
VAR_ANCHOR0_M2 = 0.01       # reported posterior variance of the fixed anchor 0
P_FLOOR_REL = 6e-3          # probability floor as fraction of the uniform density
TARGET_UPDATE_EPOCHS = 75   # epoch-subsampling target (stride-stable,
                            # keeps the 4-variant table under ~3 min/session)

VARIANTS = ("gaussian", "huber", "cauchy", "mixture")


def _kernel_of(variant: str) -> str:
    """Kernel driving the per-range loss; the mixture components are
    Gaussian at the shared sigma."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown grid_filter variant {variant!r}")
    return "gaussian" if variant == "mixture" else variant


def grid_half_width(sym: pd.DataFrame, anchor0: str) -> float:
    """Half-width of the square grid: max median range of any pair involving
    anchor 0 (fallback: global max) plus MARGIN_M."""
    med = sym.groupby(["node_a", "node_b"])["range_m"].median()
    inv0 = [r for (a, b), r in med.items() if anchor0 in (a, b)]
    base = max(inv0) if inv0 else float(med.max())
    return float(base + MARGIN_M)


class _Hist1D:
    """Port of ``histogram_filter``: 1-D grid over anchor 1's x, per-epoch
    kernelized product updates, MAP estimate, windowed weighted variance."""

    def __init__(self, half: float, res: float, sigma: float,
                 kernel: str) -> None:
        self.h = np.arange(0.0, half + res / 2.0, res)
        self.p = np.full(len(self.h), 1.0 / len(self.h))
        self.sigma = sigma
        self.kernel = kernel
        self.floor = P_FLOOR_REL / len(self.h)
        self.est: float | None = None
        self.var = np.nan
        self.n_updates = 0

    def update(self, d: float) -> None:
        like = np.exp(-kernels.rho((d - self.h) / self.sigma, self.kernel))
        p = self.p * like
        s = p.sum()
        if s > 0.0 and np.isfinite(s):
            p /= s
            p = np.maximum(p, self.floor)
            self.p = p / p.sum()
            self.n_updates += 1
        i = int(np.argmax(self.p))
        self.est = float(self.h[i])
        w = np.abs(self.h - self.h[i]) < RADIUS_EST_M
        pw = self.p[w]
        self.var = float(np.average((self.h[w] - self.est) ** 2, weights=pw))


class _Grid2D:
    """Port of ``probability_grid_anchor`` on a shared flat (gx, gy) grid."""

    def __init__(self, gx: np.ndarray, gy: np.ndarray, sigma: float,
                 variant: str) -> None:
        self.gx, self.gy = gx, gy
        self.n = len(gx)
        self.p = np.full(self.n, 1.0 / self.n)
        self.sigma = sigma
        self.variant = variant
        self.kernel = _kernel_of(variant)
        self.floor = P_FLOOR_REL / self.n
        self.initialized = False
        self.init_mode: str | None = None
        self.est: np.ndarray | None = None
        self.var = np.nan
        self.map_vs_mean = np.nan
        self.n_updates = 0
        self.n_zero_updates = 0

    def meas_init(self, d1: float, d2: float, d3: float) -> bool:
        """Closed-form triple init (original ``meas_init``): d1 = baseline
        0-1, d2 = range to anchor 0, d3 = range to anchor 1; +y branch."""
        x = (d1 ** 2 + d2 ** 2 - d3 ** 2) / (2.0 * d1)
        if d2 ** 2 <= x ** 2:
            return False
        y = np.sqrt(d2 ** 2 - x ** 2)
        dist = np.hypot(self.gx - x, self.gy - y)
        p = self.p * np.exp(-dist ** 2 / (2.0 * SIGMA_INIT_M ** 2))
        s = p.sum()
        if s <= 0.0 or not np.isfinite(s):
            return False
        self.p = p / s
        self.initialized = True
        self.init_mode = "triple"
        return True

    def measurement(self, d: np.ndarray, ref_xy: np.ndarray) -> None:
        """One epoch update from ranges d (m,) to reference positions ref_xy
        (m, 2), all at the shared sigma.

        variant "mixture" sums Gaussian likelihood components over the
        references (original IOPS update); the kernel variants accumulate
        -rho(residual / sigma, kernel) in the log domain (product update).
        """
        if self.variant == "mixture":
            like = np.zeros(self.n)
            for k in range(len(d)):
                dist = np.hypot(self.gx - ref_xy[k, 0], self.gy - ref_xy[k, 1])
                u = (d[k] - dist) / self.sigma
                m = np.abs(u) < 8.0
                like[m] += np.exp(-0.5 * u[m] ** 2)
            p = self.p * like
        else:
            ll = np.log(np.maximum(self.p, 1e-300))
            for k in range(len(d)):
                dist = np.hypot(self.gx - ref_xy[k, 0], self.gy - ref_xy[k, 1])
                ll -= kernels.rho((d[k] - dist) / self.sigma, self.kernel)
            ll -= ll.max()
            p = np.exp(ll)
        s = p.sum()
        if s <= 0.0 or not np.isfinite(s):
            self.n_zero_updates += 1
            return
        p /= s
        p = np.maximum(p, self.floor)
        self.p = p / p.sum()
        self.n_updates += 1

    def estimate(self) -> None:
        """Original ``estimate()``: probability-weighted mean of the cells
        within RADIUS_EST_M around the MAP cell; variance over that window."""
        i = int(np.argmax(self.p))
        w = ((self.gx - self.gx[i]) ** 2 + (self.gy - self.gy[i]) ** 2
             < RADIUS_EST_M ** 2)
        pw = self.p[w]
        s = pw.sum()
        ex = float((pw * self.gx[w]).sum() / s)
        ey = float((pw * self.gy[w]).sum() / s)
        self.est = np.array([ex, ey])
        self.var = float(0.5 * ((pw * (self.gx[w] - ex) ** 2).sum() / s
                                + (pw * (self.gy[w] - ey) ** 2).sum() / s))
        self.map_vs_mean = float(np.hypot(self.gx[i] - ex, self.gy[i] - ey))


def run_grid_filter(sym: pd.DataFrame, order: list[str], variant: str,
                    sigma: float, half: float | None = None,
                    res: float = GRID_RES_M, stride: int | None = None
                    ) -> tuple[pd.DataFrame, dict]:
    """Run the grid-filter pipeline on a symmetrized epoch stream.

    sym needs columns seq, node_a, node_b, range_m (grouping is by seq only;
    set_a has NaN t). sigma is the shared session noise scale
    (kernels.session_sigma). Returns (est_df with id, x, y plus per-anchor
    posterior_std_m / n_update_epochs / map_vs_mean_m, info dict).
    """
    _kernel_of(variant)
    n = len(order)
    idx = {a: i for i, a in enumerate(order)}
    df = sym[sym["node_a"].isin(idx) & sym["node_b"].isin(idx)]
    if half is None:
        half = grid_half_width(df, order[0])

    seqs = np.sort(df["seq"].unique())
    if stride is None:
        stride = max(1, len(seqs) // TARGET_UPDATE_EPOCHS)
    used = seqs[::stride]
    df = df[df["seq"].isin(set(used))]

    ax = np.arange(-half, half + res / 2.0, res)
    gxm, gym = np.meshgrid(ax, ax, indexing="ij")
    gx, gy = gxm.ravel(), gym.ravel()

    hist = _Hist1D(half, res, sigma, _kernel_of(variant))
    grids = {order[a]: _Grid2D(gx, gy, sigma, variant) for a in range(2, n)}

    est = np.full((n, 2), np.nan)
    var = np.full(n, np.nan)
    est[0] = (0.0, 0.0)
    var[0] = VAR_ANCHOR0_M2
    have = np.zeros(n, dtype=bool)
    have[0] = True

    ia = df["node_a"].map(idx).to_numpy()
    ib = df["node_b"].map(idx).to_numpy()
    rr = df["range_m"].to_numpy(float)
    sq = df["seq"].to_numpy()
    o = np.argsort(sq, kind="stable")
    ia, ib, rr, sq = ia[o], ib[o], rr[o], sq[o]
    starts = np.flatnonzero(np.r_[True, sq[1:] != sq[:-1]])
    bounds = np.r_[starts, len(sq)]

    R = np.full((n, n), np.nan)
    for e in range(len(starts)):
        lo, hi = bounds[e], bounds[e + 1]
        R[:] = np.nan
        R[ia[lo:hi], ib[lo:hi]] = rr[lo:hi]
        R[ib[lo:hi], ia[lo:hi]] = rr[lo:hi]

        if np.isfinite(R[0, 1]):
            hist.update(R[0, 1])
            est[1] = (hist.est, 0.0)
            var[1] = hist.var
            have[1] = True

        for a in range(2, n):
            gridd = grids[order[a]]
            refs = np.flatnonzero(have & np.isfinite(R[a]) & (np.arange(n) != a))
            if not gridd.initialized:
                if (have[1] and np.isfinite(R[a, 0]) and np.isfinite(R[a, 1])
                        and est[1, 0] > 0.0):
                    gridd.meas_init(est[1, 0], R[a, 0], R[a, 1])
                if not gridd.initialized:
                    # original behavior for anchors >= 3: start from the
                    # uniform grid once enough references are visible
                    if len(refs) >= 3:
                        gridd.initialized = True
                        gridd.init_mode = "uniform"
                    else:
                        continue
            if len(refs):
                gridd.measurement(R[a, refs], est[refs])
            gridd.estimate()
            est[a] = gridd.est
            var[a] = gridd.var
            have[a] = True

    rows = []
    for a in range(n):
        if not have[a]:
            continue
        if a == 0:
            nup, mvm = 0, 0.0
        elif a == 1:
            nup, mvm = hist.n_updates, 0.0
        else:
            nup, mvm = grids[order[a]].n_updates, grids[order[a]].map_vs_mean
        rows.append({"id": order[a], "x": est[a, 0], "y": est[a, 1],
                     "posterior_std_m": float(np.sqrt(var[a])),
                     "n_update_epochs": nup, "map_vs_mean_m": mvm})
    info = {"anchor_order": order,
            "variant": variant,
            "kernel": _kernel_of(variant),
            "grid_res_m": res,
            "grid_half_m": float(half),
            "n_cells": len(gx),
            "hist_cells": len(hist.h),
            "sigma_m": float(sigma),
            "stride": int(stride),
            "n_epochs_total": int(len(seqs)),
            "n_epochs_used": int(len(used)),
            "n_placed": int(have.sum()),
            "init_mode": {order[a]: grids[order[a]].init_mode
                          for a in range(2, n)},
            "n_zero_updates": {order[a]: grids[order[a]].n_zero_updates
                               for a in range(2, n)}}
    return pd.DataFrame(rows), info


def grid_filter_anchors(graph: MeasurementGraph, variant: str = "cauchy",
                        anchor_order: list[str] | None = None,
                        sigma: float | None = None,
                        **kwargs) -> tuple[pd.DataFrame, dict]:
    """Run the grid filter on a (masked) MeasurementGraph.

    variant: "gaussian" / "huber" / "cauchy" (kernelized product update) or
    "mixture" (original IOPS sum update, Gaussian components). sigma
    defaults to the shared session scale kernels.session_sigma. Returns
    (est_df with id, x, y and per-anchor diagnostics, info dict).
    """
    order = (default_anchor_order(graph) if anchor_order is None
             else list(anchor_order))
    if sigma is None:
        sigma = kernels.session_sigma(graph, order)
    return run_grid_filter(graph.symmetrized(), order, variant, sigma,
                           **kwargs)
