"""UWB cross-check of the vicon->enu registration of a set_b dynamic session.

The Vicon truth of the dyn sessions lives in the Vicon system's own local
frame; the slam_bipartite evaluation needs it in campaign ENU. The canonical
registration is the WAND TIE: Code/datasets/radiocal/scripts/02_build_set_b.py
computes it from the two surveyed wand points (nodes.yaml control points
WAND-0-0 and WAND-001) and writes it into the transform_vicon_to_local slot of
session.yaml. It uses no UWB range, so the trajectory reference does not
depend on the data under test. This script fits an independent registration
from the radio data and reports how far it lies from the wand tie:

1. Robust 3-D multilateration of the mobile (DD98) per ranging epoch, from
   the UNMASKED graph and the surveyed anchors: epochs with >=
   MIN_EPOCH_RANGES symmetrized mobile-anchor ranges are solved by LM with
   huber IRLS weights (shared kernels.py constants), sigma = the session's
   fully_meshed-view kernels.session_sigma. The anchors are coplanar to 0.04 m
   (z = 4.54-4.58 m), so a per-epoch solution has an exact mirror about the
   anchor plane; the init sits INIT_BELOW_M below the anchor centroid,
   selecting the physical branch (the tag is carried at ~1 m, the anchors
   hang at ~4.6 m) a-priori.
2. Epochs whose metric residual RMS exceeds RMS_GATE_NSIG * sigma are
   dropped (NLOS-corrupted fixes).
3. Surviving epochs are matched to the nearest Vicon pose on the shared
   clock, |dt| <= MATCH_TOL_S; no interpolation across truth gaps.
4. Rigid Umeyama vicon->enu (frames.umeyama, with_scale=False,
   allow_reflection=False) with iterated trimming: refit on the pairs whose
   residual norm is <= TRIM_NSIG * max(robust scale, sigma), robust scale =
   1.4826 * MAD of the norms, until the inlier set is stable. The sigma
   floor matters: the pair residuals are dominated by the systematic
   multilateration bias field of the site (~0.2 m, from the known 0.3-0.5 m
   per-pair range biases), not by Gaussian scatter, so an unfloored MAD
   trim collapses onto whichever Vicon burst happens to agree best with the
   current fit (root-caused on s01: 31/400 "inliers" in one cluster and a
   yaw 1.9 deg off s02, while the untrimmed fits of the two sessions agree
   to 0.24 deg). With the floor, trimming removes gross outliers only and
   the bias field is averaged over.

Adopted parameterization: PLANAR - yaw about z from the 2-D (xy) Umeyama
plus a z offset (median of z_enu - z_vicon over the xy inliers). Decided by
the evidence (s01/s02, 2026-07-07): the coplanar anchor geometry makes
the multilaterated z soft (planar-fit z residual RMSE 0.39-0.41 m vs
0.12-0.17 m per xy axis), so the full 3-D Umeyama fits a spurious roll /
pitch to that z noise - +2.1/-0.2 deg on s01 but +3.4/-1.8 deg on s02,
i.e. not reproducible across sessions of one rigid setup - while improving
the xy residuals only marginally (s02: 0.104/0.124 m vs 0.121/0.134 m).
The Vicon frame is physically leveled; the planar fit keeps it level, and
its yaw agrees with the 3-D fit to ~0.1 deg on both sessions. Both fits
are printed; the planar one is written. The z offset inherits the z
softness (s01 vs s02 differ by 0.26 m); the slam evaluation is 2-D, so
only yaw and t_xy enter the benchmark metrics.

Output: results/transforms/vicon_to_enu_<campaign>_<session>.yaml =
frames.Transform.to_dict() (src "vicon", dst "enu") of the UWB fit + a
provenance block that holds the comparison with the wand tie: delta yaw,
delta translation at the Vicon origin, and the xy residual RMSE of the
matched fixes under both registrations. The script never writes session.yaml
(until 2026-09-30 an option wrote this fit there; the wand tie replaced it).

Usage:
    python estimate_vicon_transform.py --session set_b/s01
    python estimate_vicon_transform.py \
        --session set_b/s01,set_b/s02 --static-crosscheck
    python estimate_vicon_transform.py --synthetic-selftest

Multiple sessions run in one call and end with a consistency report (delta
yaw, delta translation between the estimates - target < 1 deg / < 0.10 m).
--static-crosscheck multilaterates the static DD98 of set_b/s03
once (median ranges) and compares it against the transform-mapped Vicon
centroid of that session, per estimated transform.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from radiocal import frames, load_session

import evaluation
import kernels
from common import default_anchor_order

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
OUT_DEFAULT = Path(__file__).resolve().parent / "results" / "transforms"

MIN_EPOCH_RANGES = 4   # symmetrized mobile-anchor ranges needed per epoch
INIT_BELOW_M = 2.0     # init this far below the anchor centroid (mirror pick)
RMS_GATE_NSIG = 3.0    # drop epochs with residual RMS > this * sigma
MATCH_TOL_S = 0.06     # nearest-Vicon-pose match tolerance
TRIM_NSIG = 3.0        # Umeyama trimming threshold in robust sigmas
TRIM_MAX_ITER = 10     # trim scale is floored at sigma (see docstring)

# LM discipline as in factor_graph.py
MLAT_MAX_ITER = 50
MLAT_TOL = 1e-12
MLAT_STEP_TOL = 1e-10
LAM0, LAM_FACTOR, LAM_MAX = 1e-3, 10.0, 1e12


# ------------------------------------------------------------ multilateration

def multilaterate(A: np.ndarray, r: np.ndarray, sigma: float,
                  p0: np.ndarray) -> tuple[np.ndarray, float]:
    """Robust (huber-IRLS) LM point fix from anchor positions A (m, 3) and
    ranges r (m,). Returns (p (3,), metric residual RMS)."""
    p = p0.astype(float).copy()

    def resid(pt):
        d = np.maximum(np.linalg.norm(A - pt, axis=1), 1e-12)
        return d - r, (pt - A) / d[:, None]

    f, J = resid(p)
    cost = float(kernels.rho(f / sigma, "huber").sum())
    lam = LAM0
    for _ in range(MLAT_MAX_ITER):
        w = kernels.weight(f / sigma, "huber") / sigma ** 2
        H = J.T @ (J * w[:, None])
        g = J.T @ (w * f)
        D = np.diag(np.maximum(np.diag(H), 1e-12))
        accepted, converged = False, False
        while lam <= LAM_MAX:
            try:
                dp = np.linalg.solve(H + lam * D, -g)
            except np.linalg.LinAlgError:
                lam *= LAM_FACTOR
                continue
            f_c, J_c = resid(p + dp)
            cost_c = float(kernels.rho(f_c / sigma, "huber").sum())
            if cost_c < cost:
                accepted = True
                if (cost - cost_c <= MLAT_TOL * max(cost, 1e-12)
                        or np.abs(dp).max() < MLAT_STEP_TOL):
                    converged = True
                p, f, J, cost = p + dp, f_c, J_c, cost_c
                lam = max(lam / LAM_FACTOR, 1e-12)
                break
            lam *= LAM_FACTOR
        if not accepted or converged:
            break
    return p, float(np.sqrt((f ** 2).mean()))


def multilaterate_stream(g, mobile_id: str, sigma: float) -> pd.DataFrame:
    """Per-epoch robust DD98 fixes from the unmasked graph.

    Returns a DataFrame (seq, t, x, y, z, rms_m, n_ranges) of every epoch
    with >= MIN_EPOCH_RANGES symmetrized mobile-anchor ranges.
    """
    anchors = g.nodes[(g.nodes["role"] == "anchor")
                      & (g.nodes["prior_type"] != "none")]
    aidx = {a: i for i, a in enumerate(anchors["id"])}
    A = anchors[["x", "y", "z"]].to_numpy(dtype=float)
    centroid = A.mean(axis=0)
    p0 = centroid - np.array([0.0, 0.0, INIT_BELOW_M])

    sym = g.symmetrized()
    is_ma = (sym["node_a"] == mobile_id) & sym["node_b"].isin(aidx)
    is_am = (sym["node_b"] == mobile_id) & sym["node_a"].isin(aidx)
    sym = sym[is_ma | is_am].copy()
    sym["anchor"] = np.where(sym["node_a"] == mobile_id,
                             sym["node_b"], sym["node_a"])

    rows = []
    for seq, df in sym.groupby("seq", sort=True):
        if len(df) < MIN_EPOCH_RANGES:
            continue
        ks = df["anchor"].map(aidx).to_numpy(dtype=int)
        p, rms = multilaterate(A[ks], df["range_m"].to_numpy(dtype=float),
                               sigma, p0)
        rows.append((seq, float(df["t"].min()), p[0], p[1], p[2], rms,
                     len(df)))
    return pd.DataFrame(rows, columns=["seq", "t", "x", "y", "z", "rms_m",
                                       "n_ranges"])


# ----------------------------------------------------------------- alignment

def match_to_truth(fixes: pd.DataFrame, truth: pd.DataFrame,
                   tol_s: float = MATCH_TOL_S) -> pd.DataFrame:
    """Nearest-pose match of epoch fixes to the Vicon stream (no
    interpolation). Returns matched rows with vicon and enu coordinates."""
    tr = truth.sort_values("t")
    tt = tr["t"].to_numpy(dtype=float)
    txyz = tr[["x", "y", "z"]].to_numpy(dtype=float)
    te = fixes["t"].to_numpy(dtype=float)
    j = np.searchsorted(tt, te)
    j = np.clip(j, 1, len(tt) - 1)
    j = np.where(np.abs(tt[j - 1] - te) <= np.abs(tt[j] - te), j - 1, j)
    ok = np.abs(tt[j] - te) <= tol_s
    out = fixes[ok].reset_index(drop=True).rename(
        columns={"x": "x_enu", "y": "y_enu", "z": "z_enu"})
    out[["x_vic", "y_vic", "z_vic"]] = txyz[j[ok]]
    out["dt_s"] = tt[j[ok]] - te[ok]
    return out


def trimmed_umeyama(src: np.ndarray, dst: np.ndarray, scale_floor: float
                    ) -> tuple[frames.Transform, np.ndarray]:
    """frames.umeyama (rigid, no reflection) with iterated trimming at
    TRIM_NSIG * max(1.4826 * MAD of the residual norms, scale_floor).

    ``scale_floor`` is the session ranging noise scale sigma: residual
    structure below it is the site's systematic multilateration bias field,
    which must be averaged over, not trimmed (see module docstring).
    Returns (transform, final inlier mask).
    """
    keep = np.ones(len(src), dtype=bool)
    tf = frames.umeyama(src, dst, with_scale=False, allow_reflection=False,
                        src="vicon", dst="enu")
    for _ in range(TRIM_MAX_ITER):
        res = np.linalg.norm(tf.apply(src) - dst, axis=1)
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = res <= TRIM_NSIG * max(s, scale_floor)
        if new.sum() < max(3, src.shape[1]):
            break
        if (new == keep).all():
            break
        keep = new
        tf = frames.umeyama(src[keep], dst[keep], with_scale=False,
                            allow_reflection=False, src="vicon", dst="enu")
    return tf, keep


def yaw_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def rollpitch_deg(R: np.ndarray) -> tuple[float, float]:
    """ZYX Euler roll/pitch of a 3-D rotation (deg)."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return float(np.degrees(roll)), float(np.degrees(pitch))


def fit_transforms(matched: pd.DataFrame, sigma: float) -> dict:
    """Fit the full 3-D and the planar (yaw + z offset) vicon->enu
    transforms with identical trimming; return both plus diagnostics."""
    src = matched[["x_vic", "y_vic", "z_vic"]].to_numpy(dtype=float)
    dst = matched[["x_enu", "y_enu", "z_enu"]].to_numpy(dtype=float)

    tf3, in3 = trimmed_umeyama(src, dst, sigma)
    res3 = tf3.apply(src) - dst

    tf2, in2 = trimmed_umeyama(src[:, :2], dst[:, :2], sigma)
    tz = float(np.median((dst[in2, 2] - src[in2, 2])))
    Rp = np.eye(3)
    Rp[:2, :2] = tf2.R
    tp = np.array([tf2.t[0], tf2.t[1], tz])
    tfp = frames.Transform(Rp, tp, 1.0, src="vicon", dst="enu")
    resp = tfp.apply(src) - dst

    def rmse(res, keep):
        r = res[keep]
        ax = np.sqrt((r ** 2).mean(axis=0))
        return {"x": float(ax[0]), "y": float(ax[1]), "z": float(ax[2]),
                "xy": float(np.sqrt((r[:, :2] ** 2).sum(axis=1).mean())),
                "xyz": float(np.sqrt((r ** 2).sum(axis=1).mean()))}

    roll3, pitch3 = rollpitch_deg(tf3.R)
    return {
        "tf_3d": tf3, "tf_planar": tfp,
        "rmse_3d": rmse(res3, in3), "rmse_planar": rmse(resp, in2),
        "n_inliers_3d": int(in3.sum()), "n_inliers_planar": int(in2.sum()),
        "yaw_3d_deg": yaw_deg(tf3.R), "yaw_planar_deg": yaw_deg(tfp.R),
        "roll_3d_deg": roll3, "pitch_3d_deg": pitch3,
    }


# ----------------------------------------------------------------- pipeline

def _write_yaml_with_retry(path: Path, payload: dict,
                           attempts: int = 6) -> None:
    """Atomic YAML write with backoff (the dir is cloud-synced; Windows
    transiently locks files, WinError 5/32)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False,
                                  default_flow_style=None),
                   encoding="utf-8")
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.5 * 2 ** i)


def wand_comparison(tf_uwb: frames.Transform, meta: dict,
                    matched: pd.DataFrame) -> dict | None:
    """Difference between the UWB fit and the wand tie stored in
    session.yaml, and the xy residual RMSE of the matched fixes under
    each (None if the session carries no stored transform)."""
    slot = meta.get("transform_vicon_to_local")
    if not slot:
        return None
    tf_w = frames.Transform.from_dict(slot)
    dyaw = (yaw_deg(tf_uwb.R) - yaw_deg(tf_w.R) + 180.0) % 360.0 - 180.0
    dt = tf_uwb.t - tf_w.t
    src = matched[["x_vic", "y_vic", "z_vic"]].to_numpy(dtype=float)
    dst = matched[["x_enu", "y_enu"]].to_numpy(dtype=float)

    def xy_rmse(tf):
        r = tf.apply(src)[:, :2] - dst
        return float(np.sqrt((r ** 2).sum(axis=1).mean()))

    return {"wand_method": (meta.get("transform_provenance") or {}).get("method"),
            "delta_yaw_deg": float(dyaw),
            "delta_t_m": [float(v) for v in dt],
            "delta_t_xy_m": float(np.hypot(dt[0], dt[1])),
            "xy_rmse_uwb_fit_m": xy_rmse(tf_uwb),
            "xy_rmse_wand_tie_m": xy_rmse(tf_w)}


def estimate_session(session: str, out_dir: Path) -> dict:
    """The full pipeline of one session. Returns the report dict."""
    campaign, name = session.split("/")
    sdir = DATA / campaign / name
    g = load_session(sdir)
    mobile_id = g.meta["vicon"]["node_id"]
    g_full = g.apply_mask("fully_meshed")
    sigma = kernels.session_sigma(g_full, default_anchor_order(g_full))

    fixes = multilaterate_stream(g, mobile_id, sigma)
    gate = RMS_GATE_NSIG * sigma
    kept = fixes[fixes["rms_m"] <= gate].reset_index(drop=True)
    truth = g.truth[g.truth["node_id"] == mobile_id]
    matched = match_to_truth(kept, truth)
    if len(matched) < 10:
        raise RuntimeError(f"{session}: only {len(matched)} matched epochs")

    fits = fit_transforms(matched, sigma)
    tf = fits["tf_planar"]  # adopted parameterization (module docstring)

    print(f"-- {session}: sigma={sigma:.3f} m, "
          f"{len(fixes)} epochs multilaterated, "
          f"{len(kept)} kept (RMS gate {gate:.3f} m), "
          f"{len(matched)} matched to Vicon (|dt|<={MATCH_TOL_S}s)")
    print(f"   mlat residual RMS: median {fixes['rms_m'].median():.3f} m, "
          f"p90 {fixes['rms_m'].quantile(0.9):.3f} m")
    r3, rp = fits["rmse_3d"], fits["rmse_planar"]
    print(f"   3-D fit    : yaw {fits['yaw_3d_deg']:+8.3f} deg, "
          f"roll {fits['roll_3d_deg']:+.3f} pitch {fits['pitch_3d_deg']:+.3f} deg, "
          f"rmse x/y/z {r3['x']:.3f}/{r3['y']:.3f}/{r3['z']:.3f} m "
          f"({fits['n_inliers_3d']} inliers)")
    print(f"   planar fit : yaw {fits['yaw_planar_deg']:+8.3f} deg, "
          f"rmse x/y/z {rp['x']:.3f}/{rp['y']:.3f}/{rp['z']:.3f} m "
          f"({fits['n_inliers_planar']} inliers)  [ADOPTED]")
    print(f"   translation: [{tf.t[0]:.3f}, {tf.t[1]:.3f}, {tf.t[2]:.3f}] m")
    cmp = wand_comparison(tf, g.meta, matched)
    if cmp:
        print(f"   vs wand tie: delta yaw {cmp['delta_yaw_deg']:+.3f} deg, "
              f"delta t [{cmp['delta_t_m'][0]:+.3f}, {cmp['delta_t_m'][1]:+.3f}, "
              f"{cmp['delta_t_m'][2]:+.3f}] m, xy RMSE of the matched fixes "
              f"{cmp['xy_rmse_uwb_fit_m']:.3f} m (UWB fit) / "
              f"{cmp['xy_rmse_wand_tie_m']:.3f} m (wand tie)")

    payload = tf.to_dict()
    payload["provenance"] = {
        "script": "estimate_vicon_transform.py",
        "date": date.today().isoformat(),
        "session": session,
        "parameterization": "planar_yaw_plus_z_offset",
        "sigma_m": float(sigma),
        "n_epochs_multilaterated": int(len(fixes)),
        "n_epochs_kept": int(len(kept)),
        "rms_gate_m": float(gate),
        "n_matched": int(len(matched)),
        "n_inliers": fits["n_inliers_planar"],
        "yaw_deg": fits["yaw_planar_deg"],
        "residual_rmse_m": fits["rmse_planar"],
        "residual_rmse_3d_fit_m": fits["rmse_3d"],
        "yaw_3d_deg": fits["yaw_3d_deg"],
        "roll_3d_deg": fits["roll_3d_deg"],
        "pitch_3d_deg": fits["pitch_3d_deg"],
        "mlat_resid_rms_median_m": float(fixes["rms_m"].median()),
        "role": "cross-check; the canonical registration is the wand tie "
                "in session.yaml",
        "wand_comparison": cmp,
    }
    out = out_dir / f"vicon_to_enu_{campaign}_{name}.yaml"
    _write_yaml_with_retry(out, payload)
    print(f"   wrote {out}")

    return {"session": session, "tf": tf, "fits": fits,
            "n_matched": len(matched), "sigma": sigma, "wand": cmp,
            "tf_wand": (frames.Transform.from_dict(g.meta["transform_vicon_to_local"])
                        if g.meta.get("transform_vicon_to_local") else None)}


def consistency_report(reports: list[dict]) -> None:
    """Pairwise delta yaw / delta translation of the estimated transforms."""
    print("-- consistency")
    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            a, b = reports[i], reports[j]
            dyaw = (a["fits"]["yaw_planar_deg"]
                    - b["fits"]["yaw_planar_deg"])
            dyaw = (dyaw + 180.0) % 360.0 - 180.0
            dt = a["tf"].t - b["tf"].t
            print(f"   {a['session']} vs {b['session']}: "
                  f"delta yaw {dyaw:+.3f} deg, "
                  f"delta t [{dt[0]:+.3f}, {dt[1]:+.3f}, {dt[2]:+.3f}] m, "
                  f"|dt_xy| {np.hypot(dt[0], dt[1]):.3f} m")


def static_crosscheck(reports: list[dict],
                      session: str = "set_b/s03") -> None:
    """Single-point check: multilaterate the static DD98 of s03 from
    per-pair median ranges and compare against the transform-mapped Vicon
    centroid (translation-only check; one point constrains no rotation)."""
    campaign, name = session.split("/")
    g = load_session(DATA / campaign / name)
    mobile_id = g.meta["vicon"]["node_id"]
    g_full = g.apply_mask("fully_meshed")
    sigma = kernels.session_sigma(g_full, default_anchor_order(g_full))

    anchors = g.nodes[(g.nodes["role"] == "anchor")
                      & (g.nodes["prior_type"] != "none")]
    aidx = {a: i for i, a in enumerate(anchors["id"])}
    A = anchors[["x", "y", "z"]].to_numpy(dtype=float)
    sym = g.symmetrized()
    is_ma = (sym["node_a"] == mobile_id) & sym["node_b"].isin(aidx)
    is_am = (sym["node_b"] == mobile_id) & sym["node_a"].isin(aidx)
    sym = sym[is_ma | is_am].copy()
    sym["anchor"] = np.where(sym["node_a"] == mobile_id,
                             sym["node_b"], sym["node_a"])
    med = sym.groupby("anchor")["range_m"].median()
    ks = med.index.map(aidx).to_numpy(dtype=int)
    p0 = A.mean(axis=0) - np.array([0.0, 0.0, INIT_BELOW_M])
    p_enu, rms = multilaterate(A[ks], med.to_numpy(dtype=float), sigma, p0)

    truth = g.truth[g.truth["node_id"] == mobile_id]
    c_vic = truth[["x", "y", "z"]].median().to_numpy(dtype=float)
    print(f"-- s03 cross-check: DD98 mlat ENU "
          f"[{p_enu[0]:.3f}, {p_enu[1]:.3f}, {p_enu[2]:.3f}] m "
          f"(residual RMS {rms:.3f} m, {len(ks)} anchors), "
          f"Vicon centroid ({len(truth)} poses)")
    via = [(f"UWB fit of {r['session']}", r["tf"]) for r in reports]
    wand = next((r["tf_wand"] for r in reports if r.get("tf_wand")), None)
    if wand is not None:
        via.append(("wand tie", wand))
    for label, tf in via:
        mapped = tf.apply(c_vic[None, :])[0]
        d = mapped - p_enu
        print(f"   via {label}: mapped "
              f"[{mapped[0]:.3f}, {mapped[1]:.3f}, {mapped[2]:.3f}] m, "
              f"delta [{d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f}] m, "
              f"|d_xy| {np.hypot(d[0], d[1]):.3f} m")


# ----------------------------------------------------------------- selftest

def synthetic_selftest() -> None:
    """Exact-recovery test of the full pipeline: synthetic ranges (3-D,
    noiseless) from a KNOWN planar vicon->enu transform must reproduce R
    and t to < 1e-6."""
    import synthetic

    ref = evaluation.surveyed_reference(DATA / "set_b" / "s03")
    yaw = np.radians(37.0)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([12.3, -4.5, 0.8])
    tf_true = frames.Transform(R, t, 1.0, src="vicon", dst="enu")

    traj = synthetic.synthetic_trajectory(ref, duration_s=120.0)
    g = synthetic.synthetic_slam_graph(ref, traj, sigma=0.0,
                                       vicon_tf=tf_true, truth_z=1.0,
                                       coverage=None, ranges_3d=True)
    mobile_id = "DD98"
    sigma = 0.05  # nominal scale; exact ranges make the huber core exact

    fixes = multilaterate_stream(g, mobile_id, sigma)
    assert fixes["rms_m"].max() < 1e-8, "mlat did not recover exactly"
    matched = match_to_truth(fixes, g.truth)
    assert len(matched) == len(fixes), "matching lost epochs"
    fits = fit_transforms(matched, sigma)
    tf = fits["tf_planar"]
    dR = np.abs(tf.R - tf_true.R).max()
    dt = np.abs(tf.t - tf_true.t).max()
    print(f"-- synthetic selftest: n={len(matched)}, "
          f"|dR|_max={dR:.2e}, |dt|_max={dt:.2e}")
    assert dR < 1e-6 and dt < 1e-6, "transform recovery failed"
    print("   PASS (planar transform recovered to < 1e-6)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", default="",
                    help="comma-separated campaign/session list, e.g. "
                         "set_b/s01,set_b/s02")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--static-crosscheck", action="store_true")
    ap.add_argument("--synthetic-selftest", action="store_true")
    args = ap.parse_args()

    if args.synthetic_selftest:
        synthetic_selftest()
        if not args.session:
            return 0

    if not args.session:
        ap.error("--session or --synthetic-selftest required")

    reports = []
    for session in [s.strip() for s in args.session.split(",") if s.strip()]:
        reports.append(estimate_session(session, Path(args.out)))
    if len(reports) > 1:
        consistency_report(reports)
    if args.static_crosscheck:
        static_crosscheck(reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
