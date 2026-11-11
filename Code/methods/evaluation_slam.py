"""Trajectory evaluation of the slam_bipartite family (driver hook).

The benchmark driver calls, per the frozen contract of run_benchmark.py:

    ctx = session_context(session_dir)                    # once per session
    traj_rows, metrics = evaluate_trajectory(trajectory_df, result, ctx)

where ``trajectory_df`` is the method's per-epoch mobile estimate (seq, t,
x, y) in the SOLVER frame and ``result`` is evaluation.evaluate_estimate's
dict - its ``result["tf"]`` (the 2-D anchor alignment est->ENU, reflection
allowed) also maps the trajectory into ENU, since mobile and anchors live
in one solver frame.

Truth handling (set_b Vicon, BURSTY - ~20 % coverage, gaps up to
36 s): the Vicon stream is mapped to ENU with the session's vicon->enu
transform and segmented at gaps > MAX_GAP_S; truth is NEVER interpolated.
The transform is read from the session.yaml slot ``transform_vicon_to_local``
when non-null, else from results/transforms/vicon_to_enu_<campaign>_
<session>.yaml (estimate_vicon_transform.py); if neither exists,
session_context raises a RuntimeError telling the user to run
estimate_vicon_transform.py first (the driver records the error cells).

Metrics (constants stated once, a-priori):

- MAX_GAP_S = 0.5: truth rows with dt <= MAX_GAP_S share a segment. The
  value is far longer than the spacing of the poses within a 120 Hz burst
  (about 8 ms), so a segment ends only at a gap of the bursty truth.
- MATCH_TOL_S = 0.06: checkpoint matching - each estimate epoch is matched
  to the NEAREST truth pose within this tolerance (shorter than the period
  of the ranging epochs, about 0.08 s at the 12.6 Hz of the walks, so about
  three quarters of it, and comparable to the ~0.1 s clock uncertainty).
  ate_rmse_m =
  RMSE of the xy error over matched epochs; n_truth_matched = their count.
- RPE_DELTA_S = 1.0, RPE_DT_TOL_S = 0.25: within-segment relative pose
  error - for each matched epoch i, the matched epoch j of the SAME truth
  segment whose time offset is closest to RPE_DELTA_S (accepted when
  |dt - RPE_DELTA_S| < RPE_DT_TOL_S); the pair error is
  ||(p_est_j - p_est_i) - (p_true_j - p_true_i)|| in ENU, and rpe_rmse_m
  their RMSE. RPE is insensitive to the translational part of the alignment
  (the rotation of the anchor-fitted transform still enters) and measures
  local trajectory consistency (drift at the 1 s horizon).

traj_rows follows the frozen schema exactly: seq, t, x_est, y_est (solver
frame), x_enu, y_enu (alignment applied), x_ref, y_ref (matched ENU truth
or NaN), segment (truth segment id or -1), err_m (xy error or NaN).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from radiocal import frames

MAX_GAP_S = 0.5     # truth segmentation gap
MATCH_TOL_S = 0.06  # nearest-truth checkpoint match tolerance
RPE_DELTA_S = 1.0   # relative-pose horizon
RPE_DT_TOL_S = 0.25 # acceptance window around the horizon

TRANSFORM_DIR = Path(__file__).resolve().parent / "results" / "transforms"

TRAJ_COLUMNS = ["seq", "t", "x_est", "y_est", "x_enu", "y_enu",
                "x_ref", "y_ref", "segment", "err_m"]


def _load_transform(sdir: Path, meta: dict) -> frames.Transform:
    """The session's vicon->enu transform: session.yaml slot first, then
    the provenance YAML of estimate_vicon_transform.py."""
    slot = meta.get("transform_vicon_to_local")
    if slot:
        return frames.Transform.from_dict(slot)
    campaign = meta.get("campaign", sdir.parent.name)
    session = meta.get("session_id", sdir.name)
    path = TRANSFORM_DIR / f"vicon_to_enu_{campaign}_{session}.yaml"
    if path.exists():
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
        return frames.Transform.from_dict(d)
    raise RuntimeError(
        f"no vicon->enu transform for {campaign}/{session}: the session.yaml "
        f"slot transform_vicon_to_local is null and {path} does not exist - "
        f"run estimate_vicon_transform.py --session {campaign}/{session} "
        f"first")


def context_from_truth(truth: pd.DataFrame,
                       tf: frames.Transform | None) -> dict:
    """Evaluation context from a truth table (columns t, x, y[, z]) and an
    optional truth-frame -> ENU transform (None = already ENU). Shared by
    session_context and the synthetic validation harness."""
    tr = truth.sort_values("t").reset_index(drop=True)
    t = tr["t"].to_numpy(dtype=float)
    if tf is not None:
        cols = ["x", "y", "z"] if tf.dim == 3 else ["x", "y"]
        xy = tf.apply(tr[cols].to_numpy(dtype=float))[:, :2]
    else:
        xy = tr[["x", "y"]].to_numpy(dtype=float)
    segment = np.cumsum(np.r_[0, (np.diff(t) > MAX_GAP_S).astype(int)])
    return {"truth_t": t, "truth_xy": xy, "segment": segment,
            "transform": tf, "n_segments": int(segment[-1]) + 1 if len(t)
            else 0}


def session_context(session_dir: str | Path) -> dict:
    """Load truth + transform of a session and prepare the ENU truth
    segments. Raises RuntimeError when no vicon->enu transform exists."""
    sdir = Path(session_dir)
    meta = yaml.safe_load((sdir / "session.yaml").read_text(encoding="utf-8"))
    tf = _load_transform(sdir, meta)
    truth = pd.read_parquet(sdir / "groundtruth.parquet")
    mobile_id = (meta.get("vicon") or {}).get("node_id")
    if mobile_id is not None:
        truth = truth[truth["node_id"] == mobile_id]
    if not len(truth):
        raise RuntimeError(f"{sdir}: session has no ground-truth poses")
    ctx = context_from_truth(truth, tf)
    ctx["session_dir"] = str(sdir)
    return ctx


def _match_truth(t_est: np.ndarray, ctx: dict
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Nearest truth row per estimate epoch. Returns (row index, matched
    mask); a match must be within MATCH_TOL_S (never interpolated, so a
    match can never cross a truth gap)."""
    tt = ctx["truth_t"]
    j = np.searchsorted(tt, t_est)
    j = np.clip(j, 1, len(tt) - 1)
    j = np.where(np.abs(tt[j - 1] - t_est) <= np.abs(tt[j] - t_est),
                 j - 1, j)
    return j, np.abs(tt[j] - t_est) <= MATCH_TOL_S


def evaluate_trajectory(trajectory_df: pd.DataFrame, result: dict,
                        ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Score a solver-frame trajectory against the segmented ENU truth.

    ``result["tf"]`` (2-D anchor alignment) maps the estimate to ENU.
    Returns (traj_rows per the frozen schema, metrics dict with ate_rmse_m,
    rpe_rmse_m, n_truth_matched).
    """
    tr = trajectory_df.sort_values("t").reset_index(drop=True)
    t = tr["t"].to_numpy(dtype=float)
    xy_est = tr[["x", "y"]].to_numpy(dtype=float)
    xy_enu = result["tf"].apply(xy_est)

    j, ok = _match_truth(t, ctx)
    xy_ref = np.full((len(t), 2), np.nan)
    xy_ref[ok] = ctx["truth_xy"][j[ok]]
    segment = np.where(ok, ctx["segment"][j], -1)
    err = np.linalg.norm(xy_enu - xy_ref, axis=1)  # NaN where unmatched

    n_matched = int(ok.sum())
    ate = (float(np.sqrt(np.nanmean(err[ok] ** 2))) if n_matched
           else float("nan"))

    # within-segment RPE at the RPE_DELTA_S horizon
    mi = np.flatnonzero(ok)
    rpe_sq: list[float] = []
    if len(mi) >= 2:
        tm = t[mi]
        for a in range(len(mi)):
            b = np.searchsorted(tm, tm[a] + RPE_DELTA_S)
            best, best_off = -1, RPE_DT_TOL_S
            for c in (b - 1, b):
                if a < c < len(mi):
                    off = abs(tm[c] - tm[a] - RPE_DELTA_S)
                    if off < best_off:
                        best, best_off = c, off
            if best < 0 or segment[mi[best]] != segment[mi[a]]:
                continue
            ia, ib = mi[a], mi[best]
            d = ((xy_enu[ib] - xy_enu[ia])
                 - (xy_ref[ib] - xy_ref[ia]))
            rpe_sq.append(float(d @ d))
    rpe = float(np.sqrt(np.mean(rpe_sq))) if rpe_sq else float("nan")

    traj_rows = pd.DataFrame({
        "seq": tr["seq"].to_numpy(), "t": t,
        "x_est": xy_est[:, 0], "y_est": xy_est[:, 1],
        "x_enu": xy_enu[:, 0], "y_enu": xy_enu[:, 1],
        "x_ref": xy_ref[:, 0], "y_ref": xy_ref[:, 1],
        "segment": segment.astype(int), "err_m": err,
    })[TRAJ_COLUMNS]
    metrics = {"ate_rmse_m": ate, "rpe_rmse_m": rpe,
               "n_truth_matched": float(n_matched)}
    return traj_rows, metrics
