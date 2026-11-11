"""Sequential EKF anchor self-calibration over the ranging epochs.

The state is the stacked 2-D coordinates of all anchors of a static site,
minus the gauge: anchor 0 of the ordering is fixed at the origin and is not
in the state, anchor 1 contributes only its x coordinate (y = 0 fixed), so
the state dimension is 2n - 3. Each symmetrized per-epoch pair range is a
scalar measurement h = ||x_i - x_j||, processed sequentially within the
epoch (the state is updated between pairs of the same epoch, so later pairs
are linearized at the already-updated estimate; within an epoch the pairs
arrive in lexicographic (node_a, node_b) order as produced by
``MeasurementGraph.symmetrized``).

Initialization (``init`` parameter): the default ``"mds"`` is a classical-
MDS embedding (shortest-path completed) of the median range matrix of the
first ``init_epochs`` epochs, re-gauged onto the state convention (anchor 0
to the origin, anchor 1 onto the +x axis) - the same init family as the
batch factor-graph method, so the benchmark compares batch vs sequential
estimation from comparable starting points. ``"closed_form"`` (disambiguated
sequential trilateration on the same matrix, classical-MDS fill-in for
anchors it cannot place) is kept as a documented option; on the NLOS-rich
set_a site it is reflected and 17.8 m wrong, and the filter cannot
escape it with the default prior. Initial covariance is diagonal with a
generous ``sigma0_m`` (default 2 m) per state coordinate.

Noise model: every method shares the session
noise scale sigma of ``kernels.session_sigma`` (0.129 m on set_b,
1.621 m on set_a) and the a-priori kernel constants of ``kernels``;
nothing is tuned per method. Process noise is zero (the anchors are static,
the state is a constant).

Variants = kernel axis plus one mechanism variant:

- ``gaussian``: standard EKF update, R = sigma^2.
- ``huber``, ``cauchy``: M-estimation Kalman filtering via IRLS-style R
  inflation per scalar update - u = innovation / sigma (dimensionless),
  w = kernels.weight(u, kernel), R_eff = sigma^2 / w, so as w -> 0 the
  effective measurement variance diverges and the update vanishes. This is
  the filter analog of the IRLS weight in batch M-estimation; the weight
  is evaluated once at the predicted innovation of each measurement (one
  IRLS step per scalar update, not iterated).
- ``gated``: gaussian kernel plus innovation gating - an update is skipped
  when its normalized innovation squared nu^2 / S exceeds ``gate_nis``
  (default 9, the 3-sigma chi-square threshold for 1 degree of freedom).
  Per-pair gating counts are reported so the benchmark can check whether
  the persistent NLOS pairs are the ones rejected.

The solution frame is the gauge of the init; evaluation aligns with
reflection allowed, as for all methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radiocal import MeasurementGraph

import closed_form
import kernels
import mds
from common import default_anchor_order

VARIANTS = ("gaussian", "huber", "cauchy", "gated")

# Chi-square 3-sigma threshold for a scalar (1-DOF) innovation.
DEFAULT_GATE_NIS = 9.0


# ---------------------------------------------------------------------------
# state <-> coordinates (gauge-reduced parameterization)

def state_indices(k: int) -> tuple[int, int]:
    """State indices (ix, iy) of anchor k in the ordering; -1 = gauge-fixed.

    Anchor 0: both fixed. Anchor 1: x at state[0], y fixed. Anchor k >= 2:
    (x, y) at state[2k-3], state[2k-2].
    """
    if k == 0:
        return -1, -1
    if k == 1:
        return 0, -1
    return 2 * k - 3, 2 * k - 2


def coords_to_state(coords: np.ndarray) -> np.ndarray:
    """Reduce gauge-conform coordinates (anchor 0 at origin, anchor 1 on the
    x-axis) to the (2n-3)-dim state vector."""
    n = coords.shape[0]
    s = np.empty(2 * n - 3)
    s[0] = coords[1, 0]
    for k in range(2, n):
        s[2 * k - 3], s[2 * k - 2] = coords[k]
    return s


def state_to_coords(s: np.ndarray) -> np.ndarray:
    """Expand the state vector back to full (n, 2) coordinates."""
    n = (len(s) + 3) // 2
    coords = np.zeros((n, 2))
    coords[1, 0] = s[0]
    for k in range(2, n):
        coords[k] = s[2 * k - 3], s[2 * k - 2]
    return coords


# ---------------------------------------------------------------------------
# initialization

def _rigid_align(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid 2-D Umeyama fit A -> B (reflection allowed, no scale).

    Returns (R, t) with B ~ A @ R.T + t. Used only to splice the MDS fill-in
    into the closed-form init frame.
    """
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = (U @ Vt).T
    return R, cb - R @ ca


def _init_matrix(sym: pd.DataFrame, order: list[str], init_epochs: int
                 ) -> tuple[np.ndarray, int]:
    """Per-pair median range matrix of the first ``init_epochs`` epochs of a
    symmetrized edge list. Returns (matrix, number of epochs used)."""
    seqs = np.sort(sym["seq"].unique())[:init_epochs]
    sub = sym[sym["seq"].isin(seqs)]
    idx = {a: i for i, a in enumerate(order)}
    n = len(order)
    M = np.full((n, n), np.nan)
    med = sub.groupby(["node_a", "node_b"])["range_m"].median()
    for (a, b), r in med.items():
        if a in idx and b in idx:
            M[idx[a], idx[b]] = M[idx[b], idx[a]] = r
    return M, int(len(seqs))


def _regauge(coords: np.ndarray) -> np.ndarray:
    """Move coordinates onto the state convention: anchor 0 to the origin,
    anchor 1 onto the +x axis (y exactly zero, as the reduced state needs)."""
    coords = coords - coords[0]
    ang = np.arctan2(coords[1, 1], coords[1, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    coords = coords @ np.array([[c, -s], [s, c]]).T
    coords[1, 1] = 0.0
    return coords


def initial_coords_mds(sym: pd.DataFrame, order: list[str],
                       init_epochs: int = 20) -> tuple[np.ndarray, dict]:
    """Default init: classical MDS of the first ``init_epochs`` epochs.

    Torgerson embedding of the shortest-path-completed per-pair median
    matrix (the MDS-MAP recipe, same init family as the batch factor-graph
    method), re-gauged onto the state convention. Returns (coords (n, 2),
    info dict).
    """
    M, n_used = _init_matrix(sym, order, init_epochs)
    D, _ = mds.complete_distances(M)
    X, _ = mds.classical_mds(D, dim=2)
    return _regauge(X), {"init": "mds", "init_epochs": n_used,
                         "mds_filled": []}


def initial_coords_closed_form(sym: pd.DataFrame, order: list[str],
                               init_epochs: int = 20) -> tuple[np.ndarray, dict]:
    """Optional init: disambiguated closed-form on the first ``init_epochs``
    epochs; anchors it cannot place are filled from a classical-MDS
    embedding of the same matrix, rigidly aligned to the placed anchors.

    Kept for the init-dependence study: on set_a this init is
    reflected and 17.8 m wrong (the y-sign disambiguation is fooled by
    NLOS), which traps the filter - see results/ekf/summary.md.
    """
    M, n_used = _init_matrix(sym, order, init_epochs)
    coords, placed = closed_form.solve_closed_form(M, disambiguate=True)
    placed = np.asarray(placed)
    mds_filled: list[str] = []
    if not placed.all():
        D, _ = mds.complete_distances(M)
        X, _ = mds.classical_mds(D, dim=2)
        R, t = _rigid_align(X[placed], coords[placed])
        coords[~placed] = X[~placed] @ R.T + t
        mds_filled = [order[k] for k in np.flatnonzero(~placed)]
    info = {"init": "closed_form", "init_epochs": n_used,
            "closed_form_placed": {order[k]: bool(placed[k])
                                   for k in range(len(order))},
            "mds_filled": mds_filled}
    return _regauge(coords), info


def initial_coords(sym: pd.DataFrame, order: list[str], init_epochs: int = 20,
                   method: str = "mds") -> tuple[np.ndarray, dict]:
    """Dispatch to the requested init family ("mds" default, "closed_form")."""
    if method == "mds":
        return initial_coords_mds(sym, order, init_epochs)
    if method == "closed_form":
        return initial_coords_closed_form(sym, order, init_epochs)
    raise ValueError(f"unknown init method {method!r}")


# ---------------------------------------------------------------------------
# filter

def run_filter(seq: np.ndarray, ia: np.ndarray, ib: np.ndarray,
               r: np.ndarray, x0: np.ndarray, sigma0_m: float,
               sigma_m: float, kernel: str = "gaussian",
               gate_nis: float | None = None,
               trace_every: int = 50) -> dict:
    """Sequential EKF over pre-extracted per-epoch pair measurements.

    seq/ia/ib/r are parallel arrays sorted by seq (epoch counter, anchor
    indices into the ordering, symmetrized range). sigma_m is the shared
    session noise scale; kernel selects the per-update measurement model
    (gaussian: R = sigma^2; huber/cauchy: R_eff = sigma^2 / w(nu/sigma),
    the M-estimation convention of the module docstring). gate_nis = None
    disables gating; the gate NIS uses the gaussian S. Process noise is
    zero (static anchors). Returns a dict with the final state, covariance,
    per-pair update/gate counts and coordinate snapshots every
    ``trace_every`` epochs.
    """
    m = len(x0)
    n = (m + 3) // 2
    x = x0.copy()
    P = np.eye(m) * sigma0_m ** 2
    R = sigma_m ** 2

    n_total = np.zeros((n, n), dtype=int)   # attempted updates per pair
    n_gated = np.zeros((n, n), dtype=int)   # gated-out updates per pair
    gate_log: list[tuple[int, int, int]] = []  # (epoch_index, i, j) of gated
    snapshots: list[tuple[int, np.ndarray]] = []

    # Epoch boundaries (seq is sorted).
    starts = np.flatnonzero(np.r_[True, seq[1:] != seq[:-1]])
    ends = np.r_[starts[1:], len(seq)]

    sidx = np.array([state_indices(k) for k in range(n)])  # (n, 2)

    for e, (lo, hi) in enumerate(zip(starts, ends)):
        for p in range(lo, hi):
            i, j, z = ia[p], ib[p], r[p]
            ci = np.array([0.0, 0.0]) if i == 0 else (
                np.array([x[0], 0.0]) if i == 1 else x[2 * i - 3: 2 * i - 1])
            cj = np.array([0.0, 0.0]) if j == 0 else (
                np.array([x[0], 0.0]) if j == 1 else x[2 * j - 3: 2 * j - 1])
            d = ci - cj
            h = np.hypot(d[0], d[1])
            if h < 1e-9:
                continue
            u = d / h
            # Sparse Jacobian: <=4 nonzero entries.
            idxs, vals = [], []
            for (kx, ky), sgn in (((sidx[i][0], sidx[i][1]), 1.0),
                                  ((sidx[j][0], sidx[j][1]), -1.0)):
                if kx >= 0:
                    idxs.append(kx)
                    vals.append(sgn * u[0])
                if ky >= 0:
                    idxs.append(ky)
                    vals.append(sgn * u[1])
            idxs = np.asarray(idxs)
            vals = np.asarray(vals)

            PHt = P[:, idxs] @ vals
            HPH = vals @ PHt[idxs]
            nu = z - h
            n_total[i, j] += 1
            if gate_nis is not None and nu * nu / (HPH + R) > gate_nis:
                n_gated[i, j] += 1
                gate_log.append((e + 1, i, j))
                continue
            if kernel == "gaussian":
                R_eff = R
            else:
                w = float(kernels.weight(np.array([nu / sigma_m]), kernel)[0])
                if w < 1e-12:
                    continue
                R_eff = R / w
            S = HPH + R_eff
            K = PHt / S
            x = x + K * nu
            P = P - np.outer(K, PHt)
        P = 0.5 * (P + P.T)  # keep symmetric (per epoch is enough)
        if (e + 1) % trace_every == 0:
            snapshots.append((e + 1, state_to_coords(x)))

    n_epochs = len(starts)
    if not snapshots or snapshots[-1][0] != n_epochs:
        snapshots.append((n_epochs, state_to_coords(x)))
    return {"x": x, "P": P, "coords": state_to_coords(x),
            "n_epochs": n_epochs, "snapshots": snapshots,
            "n_total": n_total, "n_gated": n_gated, "gate_log": gate_log}


# ---------------------------------------------------------------------------
# public entry point

def ekf_from_pairs(sym: pd.DataFrame, order: list[str], sigma_m: float,
                   variant: str = "gaussian", init: str = "mds",
                   init_epochs: int = 20, sigma0_m: float = 2.0,
                   gate_nis: float = DEFAULT_GATE_NIS,
                   trace_every: int = 50) -> tuple[pd.DataFrame, dict]:
    """Run the EKF on a symmetrized edge list (columns seq, node_a, node_b,
    range_m) with an explicitly given shared noise scale sigma_m. Split out
    from ``ekf_anchors`` so synthetic streams can drive the identical
    pipeline (init + filter) in validation.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown EKF variant {variant!r}")
    kernel = "gaussian" if variant == "gated" else variant
    idx = {a: i for i, a in enumerate(order)}
    sym = sym[sym["node_a"].isin(idx) & sym["node_b"].isin(idx)]
    sym = sym.sort_values(["seq", "node_a", "node_b"], kind="stable")

    coords0, init_info = initial_coords(sym, order, init_epochs=init_epochs,
                                        method=init)

    seq = sym["seq"].to_numpy()
    ia = sym["node_a"].map(idx).to_numpy(dtype=int)
    ib = sym["node_b"].map(idx).to_numpy(dtype=int)
    r = sym["range_m"].to_numpy(dtype=float)

    out = run_filter(seq, ia, ib, r, coords_to_state(coords0),
                     sigma0_m=sigma0_m, sigma_m=sigma_m, kernel=kernel,
                     gate_nis=gate_nis if variant == "gated" else None,
                     trace_every=trace_every)

    n = len(order)
    coords = out["coords"]
    sig = np.sqrt(np.diag(out["P"]))
    sx = np.zeros(n)
    sy = np.zeros(n)
    for k in range(n):
        kx, ky = state_indices(k)
        sx[k] = sig[kx] if kx >= 0 else 0.0
        sy[k] = sig[ky] if ky >= 0 else 0.0
    est = pd.DataFrame({"id": order, "x": coords[:, 0], "y": coords[:, 1],
                        "sigma_x_m": sx, "sigma_y_m": sy})

    gate_rows = []
    for i in range(n):
        for j in range(n):
            if out["n_total"][i, j] > 0:
                gate_rows.append({"node_a": order[i], "node_b": order[j],
                                  "n_total": int(out["n_total"][i, j]),
                                  "n_gated": int(out["n_gated"][i, j])})
    gate_stats = pd.DataFrame(gate_rows)
    gate_stats["gated_frac"] = gate_stats["n_gated"] / gate_stats["n_total"]
    gate_stats = gate_stats.sort_values("gated_frac", ascending=False,
                                        kind="stable").reset_index(drop=True)
    n_total = int(out["n_total"].sum())
    n_gated = int(out["n_gated"].sum())
    info = {"anchor_order": list(order), "variant": variant,
            "kernel": kernel, "sigma0_m": sigma0_m, "sigma_m": sigma_m,
            "gate_nis": gate_nis if variant == "gated" else None,
            "n_epochs": out["n_epochs"], "n_updates": n_total,
            "n_gated": n_gated,
            "gated_frac": n_gated / n_total if n_total else 0.0,
            "gate_stats": gate_stats, "snapshots": out["snapshots"],
            "gate_log": pd.DataFrame(
                [(ep, order[i], order[j]) for ep, i, j in out["gate_log"]],
                columns=["epoch_index", "node_a", "node_b"]),
            "init": init_info}
    return est, info


def ekf_anchors(graph: MeasurementGraph, variant: str = "gaussian",
                anchor_order: list[str] | None = None,
                sigma_m: float | None = None, init: str = "mds",
                init_epochs: int = 20, sigma0_m: float = 2.0,
                gate_nis: float = DEFAULT_GATE_NIS,
                trace_every: int = 50) -> tuple[pd.DataFrame, dict]:
    """Run the EKF self-calibration on a (masked) MeasurementGraph.

    variant: "gaussian", "huber", "cauchy" or "gated"; init: "mds"
    (default) or "closed_form"; sigma_m: shared session noise scale,
    defaults to ``kernels.session_sigma`` of this graph (override only for
    scale sweeps). Returns (est_df with id, x, y and the posterior
    per-coordinate sigma; info dict with the scale used, gating statistics,
    init bookkeeping and coordinate snapshots every ``trace_every`` epochs
    for the convergence trace).
    """
    order = default_anchor_order(graph) if anchor_order is None else list(anchor_order)
    if sigma_m is None:
        sigma_m = kernels.session_sigma(graph, order)
    return ekf_from_pairs(graph.symmetrized(), order, sigma_m=sigma_m,
                          variant=variant, init=init,
                          init_epochs=init_epochs, sigma0_m=sigma0_m,
                          gate_nis=gate_nis, trace_every=trace_every)
