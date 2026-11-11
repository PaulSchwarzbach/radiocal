"""Mobile-assisted family evaluation: score the roving tag against its marker.

The mobile_assisted views add one prior-less quasi-static unknown (the
roving tag; set_a quasi-static sessions: 8B05 on a surveyed marker) to the anchor
problem. The static estimators treat it as one extra node in the estimation
order; the anchor metric stays evaluation.evaluate_estimate's Procrustes
RMSE, whose transform is fitted on the ANCHORS only (the surveyed reference
contains no mobile, align merges on common ids). This module adds the family
metric on top: apply that anchor-fitted transform to the mobile's estimate
and report the xy distance to the surveyed marker.

Truth quality: the radiocal loader does not act on session.yaml
``truth_quality`` ("verified" / "unreliable", set by the truth adjudication
of 04_build_set_a.py: a multilateration screen, then a direct range test at
the marker for the sessions it flags; all 20 quasi-static sessions are
verified since 2026-09-30). The flag rides here into every benchmark row so
the summarizer can exclude unreliable sessions from aggregates while the
raw rows stay in the table.

Driver interface (frozen, run_benchmark.py):
  session_context(session_dir) -> ctx        once per (session, mask)
  evaluate_mobile(est, result, ctx)
      -> ({"mobile_error_m", "mobile_truth_quality"}, mobile_rows | None)
mobile_rows is one per-node estimate row shaped like the anchor rows of
reporting.per_anchor_table (the driver adds the key columns); None whenever
the mobile has no finite estimate - never raise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from radiocal import MeasurementGraph, load_session

MOBILE_ROW_COLUMNS = ["id", "x_est", "y_est", "x_aligned", "y_aligned",
                      "x_ref", "y_ref", "z_ref", "error_m"]


def _context_from_graph(graph: MeasurementGraph) -> dict:
    """Evaluation context from an UNMASKED session graph: the mobile id, its
    single surveyed-marker truth position and the session's truth quality.
    Shared by the driver path (session_context) and the synthetic validation
    path (validate_mobile.py), so both score through identical code."""
    mobiles = list(graph.nodes.loc[graph.nodes["role"] == "mobile", "id"])
    if len(mobiles) != 1:
        raise ValueError(f"expected exactly one mobile node, got {mobiles}")
    mobile_id = mobiles[0]
    tr = graph.truth[graph.truth["node_id"] == mobile_id]
    marker = None
    if len(tr):
        row = tr.iloc[0]
        x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        if np.isfinite(x) and np.isfinite(y):
            marker = (x, y, z)
    return {"mobile_id": mobile_id, "marker": marker,
            "truth_quality": str(graph.meta.get("truth_quality", ""))}


def session_context(session_dir: str | Path) -> dict:
    """Driver hook: context of one session, from the unmasked load."""
    return _context_from_graph(load_session(session_dir))


def evaluate_mobile(est: pd.DataFrame, result: dict, ctx: dict
                    ) -> tuple[dict, pd.DataFrame | None]:
    """Score the mobile node of one benchmark cell.

    est is the method's estimate (id, x, y) including the mobile; result is
    evaluation.evaluate_estimate's dict with the anchor-fitted "tf". Returns
    the summary metrics plus the mobile's estimate row for the
    benchmark_estimates table (None if the mobile has no finite estimate or
    the session carries no usable marker).
    """
    metrics = {"mobile_error_m": float("nan"),
               "mobile_truth_quality": ctx.get("truth_quality", "")}
    mobile_id, marker = ctx.get("mobile_id"), ctx.get("marker")
    if mobile_id is None or marker is None:
        return metrics, None
    hit = est.loc[est["id"] == mobile_id]
    if not len(hit):
        return metrics, None
    x_est, y_est = float(hit.iloc[0]["x"]), float(hit.iloc[0]["y"])
    if not (np.isfinite(x_est) and np.isfinite(y_est)):
        return metrics, None
    x_al, y_al = result["tf"].apply([[x_est, y_est]])[0]
    err = float(np.hypot(x_al - marker[0], y_al - marker[1]))
    metrics["mobile_error_m"] = err
    rows = pd.DataFrame([{
        "id": mobile_id, "x_est": x_est, "y_est": y_est,
        "x_aligned": float(x_al), "y_aligned": float(y_al),
        "x_ref": marker[0], "y_ref": marker[1], "z_ref": marker[2],
        "error_m": err,
    }], columns=MOBILE_ROW_COLUMNS)
    return metrics, rows
