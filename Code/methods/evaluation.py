"""Evaluation shared by the method implementations.

Reference = surveyed anchor coordinates of the *unmasked* session (topology
masks such as fully_meshed strip the priors from the view the method sees).
Metric = per-anchor residual after rigid Umeyama alignment with reflection
allowed (radiocal.frames.align), reported in the reference (ENU) frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from radiocal import frames, load_session


class UndefinedConstructionError(ValueError):
    """Too few placed nodes for the alignment-based metric: the estimator's
    construction is undefined on this view (e.g. closed_form placing fewer
    than two anchors). The driver records these cells as status="undefined",
    a documented structural outcome distinct from crashes."""


def surveyed_reference(session_dir: str | Path) -> pd.DataFrame:
    """Surveyed anchors of the unmasked session (columns id, x, y, z)."""
    g = load_session(session_dir)
    ref = g.priors()
    ref = ref[ref["role"] == "anchor"]
    return ref[["id", "x", "y", "z"]].reset_index(drop=True)


def evaluate_estimate(est: pd.DataFrame, ref: pd.DataFrame, dim: int = 2) -> dict:
    """Align an estimate (id, x, y[, z]) to the surveyed reference and score it.

    Returns tf (Transform est->ref), resid (id, error_m), rmse_m, max_error_m,
    n_aligned / n_reference and whether the best fit needed a reflection.

    Raises UndefinedConstructionError when fewer than ``dim`` estimated nodes
    have a reference coordinate (mirrors the frames.align guard, typed so the
    driver can distinguish the outcome from a crash).
    """
    common = est["id"].isin(ref["id"]) & est[["x", "y"]].notna().all(axis=1)
    if int(common.sum()) < dim:
        raise UndefinedConstructionError(
            f"only {int(common.sum())} common ids for a {dim}-D alignment")
    tf, resid = frames.align(est, ref, dim=dim, with_scale=False,
                             allow_reflection=True)
    return {"tf": tf,
            "resid": resid,
            "rmse_m": resid.attrs["rmse_m"],
            "max_error_m": float(resid["error_m"].max()),
            "n_aligned": len(resid),
            "n_reference": len(ref),
            "is_reflection": tf.is_reflection()}


def evaluate_absolute(est: pd.DataFrame, ref: pd.DataFrame,
                      known_ids: list[str]) -> dict | None:
    """Score an estimate in the reference frame, without alignment.

    Defined only when the priors the estimator consumed determine the
    frame: fewer than two known anchors returns None (the absolute error
    is gauge-contaminated; the aligned metric stays valid). With exactly
    two, the reflection about the line through the two known reference
    coordinates is undecidable from range data, so the estimate is scored
    as-is and mirrored about that line, the smaller RMSE is reported, and
    abs_reflection marks the mirrored branch winning - the same doctrine
    as the aligned metric's released reflection. From three on the frame
    is fully determined and the direct residuals are reported.
    """
    ref_ids = set(ref["id"])
    known = [k for k in known_ids if k in ref_ids]
    if len(known) < 2:
        return None
    common = est["id"].isin(ref_ids) & est[["x", "y"]].notna().all(axis=1)
    if not int(common.sum()):
        return None
    e = est[common].set_index("id")
    r = ref.set_index("id").loc[e.index]
    E = e[["x", "y"]].to_numpy(float)
    R = r[["x", "y"]].to_numpy(float)

    def rmse(err: np.ndarray) -> float:
        return float(np.sqrt((err ** 2).mean()))

    err = np.linalg.norm(E - R, axis=1)
    out = {"abs_rmse_m": rmse(err), "abs_max_error_m": float(err.max()),
           "abs_reflection": False}
    if len(known) == 2:
        rk = ref.set_index("id")
        a = rk.loc[known[0], ["x", "y"]].to_numpy(float)
        b = rk.loc[known[1], ["x", "y"]].to_numpy(float)
        u = b - a
        nrm = np.linalg.norm(u)
        if nrm > 1e-9:
            u = u / nrm
            H = 2.0 * np.outer(u, u) - np.eye(2)  # mirror across the baseline
            err_m = np.linalg.norm((E - a) @ H + a - R, axis=1)
            if rmse(err_m) < out["abs_rmse_m"]:
                out = {"abs_rmse_m": rmse(err_m),
                       "abs_max_error_m": float(err_m.max()),
                       "abs_reflection": True}
    return out
