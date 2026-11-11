"""Closed-form sequential-trilateration anchor self-calibration (baseline).

Port of ``closed_form`` from the IOPS auto-positioning study
(module ``iops_autopositioning_qorvo/positioning.py`` of the study code, not
released), lifted onto the ``radiocal.MeasurementGraph`` interface.

Construction (2-D, gauge = radio_conv): anchor 0 defines the origin, anchor 1
the positive x-axis via the measured range d01, and every further anchor a is
intersected from its ranges to anchors 0 and 1:

    x_a = (x_1^2 + d_a0^2 - d_a1^2) / (2 x_1),   y_a = +-sqrt(d_a0^2 - x_a^2)

Deliberate deviation from the original: the sign of y_a is ambiguous from two
ranges alone. The original always takes +y (anchors on the far side of the
0-1 baseline get mirrored); with ``disambiguate=True`` the sign is resolved
against the measured range to an already-placed off-axis anchor (normally
anchor 2), which is the literature-standard closed-form. Both variants are
kept so the benchmark can show the effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radiocal import MeasurementGraph

from common import default_anchor_order, pair_counts, pair_range_matrix


def solve_closed_form(M: np.ndarray, disambiguate: bool = True
                      ) -> tuple[np.ndarray, list[bool]]:
    """Sequential construction from a symmetric range matrix.

    Returns (coords (n, 2) with NaN rows for failures, placed flags).
    Placement of anchor a >= 2 requires finite d01, d_a0, d_a1 and a positive
    radicand d_a0^2 - x_a^2 (same strict check as the original port source).
    """
    n = M.shape[0]
    coords = np.full((n, 2), np.nan)
    placed = [False] * n

    coords[0] = (0.0, 0.0)
    placed[0] = True
    if n > 1 and np.isfinite(M[0, 1]):
        coords[1] = (M[0, 1], 0.0)
        placed[1] = True

    for a in range(2, n):
        if not (placed[1] and np.isfinite(M[a, 0]) and np.isfinite(M[a, 1])):
            continue
        x1 = coords[1, 0]
        xc = (x1 ** 2 + M[a, 0] ** 2 - M[a, 1] ** 2) / (2.0 * x1)
        rad = M[a, 0] ** 2 - xc ** 2
        if rad <= 0.0:
            continue
        yc = np.sqrt(rad)
        if disambiguate and yc > 0.0:
            yc *= _y_sign(M, coords, placed, a, xc, yc)
        coords[a] = (xc, yc)
        placed[a] = True
    return coords, placed


def _y_sign(M: np.ndarray, coords: np.ndarray, placed: list[bool],
            a: int, xc: float, yc: float) -> float:
    """Resolve the +-y ambiguity of anchor a against an off-axis anchor.

    Uses the first already-placed anchor with y != 0 (normally anchor 2, which
    keeps +y by the gauge convention) to which a measured range exists; +1 if
    no such anchor is available.
    """
    for r in range(2, a):
        if placed[r] and abs(coords[r, 1]) > 1e-9 and np.isfinite(M[a, r]):
            d_plus = np.hypot(xc - coords[r, 0], yc - coords[r, 1])
            d_minus = np.hypot(xc - coords[r, 0], -yc - coords[r, 1])
            return 1.0 if abs(d_plus - M[a, r]) <= abs(d_minus - M[a, r]) else -1.0
    return 1.0


def closed_form_anchors(graph: MeasurementGraph, anchor_order: list[str] | None = None,
                        disambiguate: bool = True, agg: str = "median"
                        ) -> tuple[pd.DataFrame, dict]:
    """Run the closed-form baseline on a (masked) MeasurementGraph.

    Returns (est_df with columns id, x, y for placed anchors; info dict with
    the order used, per-anchor placed flags and pair availability).
    """
    order = default_anchor_order(graph) if anchor_order is None else list(anchor_order)
    M = pair_range_matrix(graph, order, agg=agg)
    coords, placed = solve_closed_form(M, disambiguate=disambiguate)
    est = pd.DataFrame({"id": order,
                        "x": coords[:, 0], "y": coords[:, 1]})[placed]
    available, total = pair_counts(M)
    info = {"anchor_order": order,
            "placed": dict(zip(order, placed)),
            "n_pairs_available": available,
            "n_pairs_total": total}
    return est.reset_index(drop=True), info


def epoch_success_rate(graph: MeasurementGraph, anchor_order: list[str],
                       disambiguate: bool = False) -> pd.DataFrame:
    """Diagnostic port of the original per-epoch success bookkeeping.

    Runs the construction on each epoch's own range matrix and reports, per
    anchor, the fraction of epochs in which it could be placed.
    """
    idx = {a: i for i, a in enumerate(anchor_order)}
    n = len(anchor_order)
    success = np.zeros(n, dtype=int)
    n_epochs = 0
    for _, df in graph.symmetrized().groupby("seq", sort=True):
        M = np.full((n, n), np.nan)
        for a, b, r in zip(df["node_a"], df["node_b"], df["range_m"]):
            if a in idx and b in idx:
                M[idx[a], idx[b]] = M[idx[b], idx[a]] = r
        _, placed = solve_closed_form(M, disambiguate=disambiguate)
        success += np.asarray(placed, dtype=int)
        n_epochs += 1
    return pd.DataFrame({"id": anchor_order,
                         "success_rate": success / max(n_epochs, 1),
                         "n_epochs": n_epochs})
