"""Per-anchor conditional bootstrap particle filter (baseline).

A joint particle filter over the 2n-dimensional anchor-coordinate state is
intractable (the particle count needed grows exponentially with dimension).
Instead each anchor carries its own small particle set and is conditioned on
point estimates (current posterior means) of the other anchors - the same
per-anchor conditional decomposition as the original IOPS grid pipeline
(Material/.../iops_autopositioning_qorvo) and as the per-node marginal
filters of the sensor-network localization literature (nonparametric BP /
Monte-Carlo localization variants that sample each node's marginal instead
of the joint).

Structure (2-D, gauge = radio_conv, anchors in common.default_anchor_order):

- anchor 0 is fixed at the origin;
- anchor 1 lives on the x-axis and carries a 1-D particle set over its x
  coordinate;
- every anchor a >= 2 carries a 2-D particle set of N = 2000 particles.

The ranging epochs are swept sequentially (group key: seq, never t - t is
NaN in set_a). Within an epoch every anchor's particle weights are
multiplied by the likelihood of that epoch's symmetrized ranges to the
current posterior means of its conditioning set, Gauss-Seidel style (means
already updated in this epoch are used as soon as available).

Sweeps over the anchor list:

- sweep 1: anchor a conditions only on anchors before it in the order
  (already-processed anchors), so early anchors never see information from
  late ones - the conditioning-order sensitivity inherent to one pass;
- sweep 2 (default SWEEPS = 2): particle sets are re-scattered around the
  sweep-1 means with the init sigma and the epoch stream is replayed with
  every anchor conditioning on all other anchors. The re-scatter is needed
  because after thousands of epochs the clouds are millimeter-tight and the
  post-resampling jitter alone cannot move them to a corrected position.

Initialization (deliberate deviation from the original IOPS per-anchor
pipeline, which seeds each anchor from its own closed-form construction):
the default init "mds20" scatters the particle clouds (Gaussian, sigma
INIT_SIGMA_M = 2 m) around a classical-MDS solution of the median range
matrix of the first MDS_INIT_EPOCHS = 20 epochs (shortest-path completion
+ Torgerson embedding via mds.complete_distances / mds.classical_mds,
gauge-fixed afterwards to anchor 0 at the origin and anchor 1 on the
positive x-axis). The EKF initializes from the same
classical-MDS-on-first-20-epochs recipe and the factor graph from the
classical-MDS embedding of the full-session median matrix, so the
benchmark compares estimation behavior within one init family. init="closed_form" keeps the
historical closed-form init (faithful variant, disambiguate=False - the
y-sign test is itself fooled at the NLOS-rich 2024 site, see
results/closed_form/summary.md) as a documented option; the grid filter
retains it as the faithful port. When an anchor has no finite init
position (closed-form failure; a disconnected 20-epoch range graph falls
back to the closed form), particles are drawn uniformly from the annulus
around anchor 0 with radius = its median range to anchor 0 +-
ANNULUS_HALF_WIDTH_M = 3 m (uniform disk of the largest observed range if
even that range is missing). Because a static-state bootstrap filter can
only select among its initial particles (plus the small jitter random
walk), anchors whose init is much further off than ~2-3 sigma cannot be
recovered; with the closed-form init this masked the Cauchy variant's
stream behavior at set_a (init-mode capture, quantified in
results/particle_filter/summary.md).

Likelihood: a per-epoch range r to a
conditioning anchor at distance d contributes the per-range log-weight
-kernels.rho((r - d) / sigma, variant) from the shared kernels module -
variants "gaussian" (L2), "huber" (quadratic core, linear tails,
k = 1.345) and "cauchy" (redescending pseudo-likelihood, c = 2.385),
identical dimensionless kernels and a-priori constants across all
estimator classes of the benchmark. sigma = kernels.session_sigma, the
shared data-derived session scale (0.129 m at set_b, 1.621 m at
set_a); the sigma_mult argument scales it for the scale-sweep
diagnostic only. An earlier version used a per-method scale (median per-pair epoch
std floored at 0.05 m, Cauchy scale gamma = that sigma), which
confounded the kernel scale with the estimator class; the shared scale
removes that confound.

Resampling: systematic resampling whenever ESS < N/2, followed by Gaussian
jitter with sigma JITTER_SIGMA_M = 0.02 m against particle depletion.
Weights are kept in log space. All randomness comes from one
np.random.default_rng(seed) stream, seed 0 by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radiocal import MeasurementGraph

from common import default_anchor_order, pair_counts, pair_range_matrix
from closed_form import solve_closed_form
import kernels
import mds

N_PARTICLES = 2000        # particles per anchor
INIT_METHOD = "mds20"     # default init; "closed_form" = historical option
MDS_INIT_EPOCHS = 20      # epochs entering the classical-MDS init matrix
INIT_SIGMA_M = 2.0        # Gaussian scatter around the init positions
ANNULUS_HALF_WIDTH_M = 3.0  # annulus half width for the no-init-position fallback
JITTER_SIGMA_M = 0.02     # post-resampling jitter against depletion
ESS_FRACTION = 0.5        # resample when ESS < ESS_FRACTION * N
DEPLETION_ESS = 0.01      # ESS < DEPLETION_ESS * N counts as a depletion event
SWEEPS = 2                # default sweeps over the anchor list (see docstring)
TRACE_EVERY = 50          # convergence snapshot every this many epochs

VARIANTS = kernels.KERNELS  # ("gaussian", "huber", "cauchy")


def _epoch_measurements(graph: MeasurementGraph, order: list[str]
                        ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten the symmetrized edge stream into per-(epoch, anchor) slices.

    Every undirected per-epoch range is duplicated in both directions and
    the rows are sorted by (epoch, anchor index). Returns (n_epochs, other-
    anchor index array, range array, boundary array b) such that the
    measurements of anchor a in epoch e are rows b[e*n+a] : b[e*n+a+1].
    """
    idx = {a: i for i, a in enumerate(order)}
    sym = graph.symmetrized()
    sym = sym[sym["node_a"].isin(idx) & sym["node_b"].isin(idx)]
    ia = sym["node_a"].map(idx).to_numpy()
    ib = sym["node_b"].map(idx).to_numpy()
    r = sym["range_m"].to_numpy(dtype=float)
    seq = sym["seq"].to_numpy()
    useq = np.unique(seq)                      # sorted epoch keys
    e = np.searchsorted(useq, seq)
    i = np.concatenate([ia, ib])
    j = np.concatenate([ib, ia])
    rr = np.concatenate([r, r])
    ee = np.concatenate([e, e])
    n = len(order)
    key = ee * n + i
    perm = np.argsort(key, kind="stable")
    j, rr, key = j[perm], rr[perm], key[perm]
    bounds = np.searchsorted(key, np.arange(len(useq) * n + 1))
    return len(useq), j, rr, bounds


def _mds20_coords(graph: MeasurementGraph, order: list[str],
                  n_init_epochs: int = MDS_INIT_EPOCHS) -> np.ndarray:
    """Classical-MDS init positions from the first `n_init_epochs` epochs.

    Median range matrix of the first epochs (by seq), shortest-path
    completed, Torgerson-embedded, then gauge-fixed: anchor 0 at the
    origin, anchor 1 rotated onto the positive x-axis. Raises ValueError
    if the 20-epoch range graph is disconnected (caller falls back to the
    closed form).
    """
    idx = {a: i for i, a in enumerate(order)}
    sym = graph.symmetrized()
    sym = sym[sym["node_a"].isin(idx) & sym["node_b"].isin(idx)]
    first = np.sort(sym["seq"].unique())[:n_init_epochs]
    sub = sym[sym["seq"].isin(first)]
    n = len(order)
    M = np.full((n, n), np.nan)
    for (a, b), r in sub.groupby(["node_a", "node_b"])["range_m"].median().items():
        M[idx[a], idx[b]] = M[idx[b], idx[a]] = r
    D, _ = mds.complete_distances(M)
    X, _ = mds.classical_mds(D, dim=2)
    X = X - X[0]
    c, s = np.cos(np.arctan2(X[1, 1], X[1, 0])), np.sin(np.arctan2(X[1, 1], X[1, 0]))
    return X @ np.array([[c, -s], [s, c]])


def _init_particles(graph: MeasurementGraph, order: list[str], M: np.ndarray,
                    n_particles: int, rng: np.random.Generator,
                    init: str = INIT_METHOD) -> tuple[list, np.ndarray, dict]:
    """Initial particle sets and means (init "mds20" or "closed_form").

    Returns (particles, means (n, 2), init_method per anchor index).
    particles[0] is None (fixed), particles[1] is 1-D (x only), the rest
    are (N, 2). See module docstring for the fallbacks.
    """
    if init not in ("mds20", "closed_form"):
        raise ValueError(f"unknown init {init!r}")
    n = M.shape[0]
    coords, placed, label = None, [False] * n, "closed_form"
    if init == "mds20":
        try:
            coords = _mds20_coords(graph, order)
            placed, label = [True] * n, "mds20"
        except ValueError:      # disconnected 20-epoch graph
            coords = None
    if coords is None:
        coords, placed = solve_closed_form(M, disambiguate=False)
    means = np.zeros((n, 2))
    particles: list = [None] * n
    method = {}

    r_max = float(np.nanmax(M)) if np.isfinite(M).any() else 50.0
    for a in range(1, n):
        if a == 1:
            if placed[1] and np.isfinite(coords[1, 0]):
                x1 = coords[1, 0]
                method[1] = label
            elif np.isfinite(M[0, 1]):
                x1 = M[0, 1]
                method[1] = "closed_form"
            else:
                x1 = r_max / 2.0
                method[1] = "disk"
            particles[1] = x1 + rng.normal(0.0, INIT_SIGMA_M, n_particles)
            means[1] = (x1, 0.0)
            continue
        if placed[a] and np.isfinite(coords[a]).all():
            particles[a] = coords[a] + rng.normal(0.0, INIT_SIGMA_M, (n_particles, 2))
            method[a] = label
        else:
            theta = rng.uniform(0.0, 2.0 * np.pi, n_particles)
            if np.isfinite(M[0, a]):
                r0 = M[0, a]
                rad = rng.uniform(max(0.1, r0 - ANNULUS_HALF_WIDTH_M),
                                  r0 + ANNULUS_HALF_WIDTH_M, n_particles)
                method[a] = "annulus"
            else:
                rad = r_max * np.sqrt(rng.uniform(0.0, 1.0, n_particles))
                method[a] = "disk"
            particles[a] = np.column_stack([rad * np.cos(theta),
                                            rad * np.sin(theta)])
        means[a] = particles[a].mean(axis=0)
    return particles, means, method


def _systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: indices drawn with one uniform offset."""
    n = len(w)
    u = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w), u)


def particle_filter_anchors(graph: MeasurementGraph, variant: str = "gaussian",
                            anchor_order: list[str] | None = None,
                            n_particles: int = N_PARTICLES, sweeps: int = SWEEPS,
                            seed: int = 0, trace_every: int = TRACE_EVERY,
                            agg: str = "median", init: str = INIT_METHOD,
                            sigma_mult: float = 1.0,
                            sigma_m: float | None = None
                            ) -> tuple[pd.DataFrame, dict]:
    """Run the conditional bootstrap particle filter on a MeasurementGraph.

    variant: "gaussian", "huber" or "cauchy" (kernels.KERNELS); init:
    "mds20" (default) or "closed_form"; sigma_mult scales the shared
    session sigma (scale-sweep diagnostic only, 1.0 = committed config).
    sigma_m, if given, is the absolute noise scale (any multiplier already
    applied) - the benchmark driver passes the session's fully_meshed-view
    sigma so every mask of a session shares one scale; None keeps the
    legacy self-computed sigma of this graph's own view.
    Returns (est_df with id, x, y, posterior_std_m; info dict with the
    sigma used, ESS statistics, resampling / depletion counts, init
    methods and the convergence trace as a list of (cumulative epoch
    index, coords (n, 2)) snapshots).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown particle filter variant {variant!r}")
    order = default_anchor_order(graph) if anchor_order is None else list(anchor_order)
    n = len(order)
    rng = np.random.default_rng(seed)

    sigma = (sigma_m if sigma_m is not None
             else kernels.session_sigma(graph, order, agg=agg) * sigma_mult)
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"invalid likelihood sigma {sigma!r}")
    M = pair_range_matrix(graph, order, agg=agg)
    particles, means, init_method = _init_particles(graph, order, M,
                                                    n_particles, rng, init=init)
    logw = [None] + [np.zeros(n_particles) for _ in range(1, n)]

    n_epochs, jj_all, rr_all, bounds = _epoch_measurements(graph, order)
    ess_thresh = ESS_FRACTION * n_particles

    ess_sum = np.zeros(n)
    ess_min = np.full(n, np.inf)
    n_updates = np.zeros(n, dtype=int)
    n_resample = np.zeros(n, dtype=int)
    n_depletion = np.zeros(n, dtype=int)
    trace: list[tuple[int, np.ndarray]] = []
    weights = [None] + [np.full(n_particles, 1.0 / n_particles) for _ in range(1, n)]

    epoch_counter = 0
    for sweep in range(sweeps):
        if sweep > 0:
            # re-scatter around current means so refined late anchors can
            # correct early ones (see module docstring)
            particles[1] = means[1, 0] + rng.normal(0.0, INIT_SIGMA_M, n_particles)
            for a in range(2, n):
                particles[a] = means[a] + rng.normal(0.0, INIT_SIGMA_M,
                                                     (n_particles, 2))
            for a in range(1, n):
                logw[a] = np.zeros(n_particles)
                weights[a] = np.full(n_particles, 1.0 / n_particles)

        for e in range(n_epochs):
            base = e * n
            for a in range(1, n):
                lo, hi = bounds[base + a], bounds[base + a + 1]
                if lo == hi:
                    continue
                jj, rv = jj_all[lo:hi], rr_all[lo:hi]
                if sweep == 0:                 # condition on earlier anchors only
                    m = jj < a
                    if not m.any():
                        continue
                    jj, rv = jj[m], rv[m]
                if a == 1:                     # 1-D state on the x-axis
                    dx = particles[1][:, None] - means[jj, 0][None, :]
                    d = np.sqrt(dx * dx + (means[jj, 1] ** 2)[None, :])
                else:
                    dx = particles[a][:, 0:1] - means[jj, 0][None, :]
                    dy = particles[a][:, 1:2] - means[jj, 1][None, :]
                    d = np.sqrt(dx * dx + dy * dy)
                z = (rv[None, :] - d) / sigma
                ll = -kernels.rho(z, variant).sum(axis=1)

                lw = logw[a] + ll
                lw -= lw.max()
                w = np.exp(lw)
                w /= w.sum()
                ess = 1.0 / (w * w).sum()
                ess_sum[a] += ess
                ess_min[a] = min(ess_min[a], ess)
                n_updates[a] += 1
                if ess < DEPLETION_ESS * n_particles:
                    n_depletion[a] += 1
                if ess < ess_thresh:
                    keep = _systematic_resample(w, rng)
                    jit_shape = n_particles if a == 1 else (n_particles, 2)
                    particles[a] = particles[a][keep] + rng.normal(
                        0.0, JITTER_SIGMA_M, jit_shape)
                    logw[a] = np.zeros(n_particles)
                    w = np.full(n_particles, 1.0 / n_particles)
                    n_resample[a] += 1
                else:
                    logw[a] = lw
                weights[a] = w
                if a == 1:
                    means[1] = (float(w @ particles[1]), 0.0)
                else:
                    means[a] = w @ particles[a]
            epoch_counter += 1
            if epoch_counter % trace_every == 0:
                trace.append((epoch_counter, means.copy()))
    if not trace or trace[-1][0] != epoch_counter:
        trace.append((epoch_counter, means.copy()))

    post_std = np.zeros(n)
    for a in range(1, n):
        w = weights[a]
        if a == 1:
            mu = w @ particles[1]
            post_std[a] = np.sqrt(w @ (particles[1] - mu) ** 2)
        else:
            mu = w @ particles[a]
            var = w @ ((particles[a] - mu) ** 2)
            post_std[a] = np.sqrt(var.sum())

    est = pd.DataFrame({"id": order, "x": means[:, 0], "y": means[:, 1],
                        "posterior_std_m": post_std})
    available, total = pair_counts(M)
    with np.errstate(invalid="ignore"):
        ess_mean = np.where(n_updates > 0, ess_sum / np.maximum(n_updates, 1), np.nan)
    info = {"anchor_order": order,
            "variant": variant,
            "init": init,
            "sigma_m": sigma,
            "sigma_mult": sigma_mult,
            "n_particles": n_particles,
            "sweeps": sweeps,
            "seed": seed,
            "n_epochs": n_epochs,
            "n_pairs_available": available,
            "n_pairs_total": total,
            "init_method": {order[a]: init_method[a] for a in range(1, n)},
            "n_resample": dict(zip(order, n_resample.tolist())),
            "n_depletion": dict(zip(order, n_depletion.tolist())),
            "ess_mean": dict(zip(order, np.round(ess_mean, 1).tolist())),
            "ess_min": dict(zip(order, np.where(np.isinf(ess_min), np.nan,
                                                ess_min).round(1).tolist())),
            "posterior_std_m": dict(zip(order, post_std.tolist())),
            "trace": trace}
    return est, info
