"""Ground-truth error budget of the benchmark.

Collects the terms the data supply and the nominal manufacturer figures into
results/transforms/error_budget.yaml. Read-only on everything else.

Set B (Vicon truth of the dynamic sessions s01, s02):
- registration: the wand tie (session.yaml) against the UWB-based fit of
  estimate_vicon_transform.py (results/transforms/): delta yaw, delta
  translation at the Vicon origin, the xy RMSE of the multilaterated fixes
  under each, and the fg_slam cauchy ATE under each registration (the
  trajectory table holds the wand-tie truth; the UWB-fit truth follows from
  the two planar transforms);
- wand baseline: the yaw uncertainty of a 0.47 m baseline under the nominal
  survey point accuracy, and what it moves at the p95 radius of the covered
  poses around the Vicon origin;
- clock: the Vicon speed at covered epochs times the clock-offset
  uncertainty recorded in session.yaml;
- coverage: the share of estimate epochs with Vicon truth, and the speed of
  the fg_slam cauchy estimate on covered against uncovered epochs. Added for
  the Part II figure pass (2026-10-10): the truth bursts under the
  evaluation's gap rule, the Vicon frame-counter rate across every gap (a
  logging loss leaves the counter running), the ATE of each SLAM cell
  reweighted from the covered epochs to the walk's share of stationary and
  moving epochs, and the per-epoch median absolute range residual of the
  aligned track against the surveyed anchors on covered and uncovered
  epochs. The static s03 adds its duty cycle only (it has no registration).

Set A (surveyed markers of the quasi-static sessions): the tag height of
the truth adjudication (Data/set_a/truth_adjudication.csv) against the
surveyed height of the 8B05 antenna above marker P01 in s21/s22.

Nominal figures (manufacturer specifications, D10) are quoted with their
sources, not derived.

    python gt_error_budget.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from radiocal import frames, load_session

import evaluation
import slam_common
from evaluation_slam import MATCH_TOL_S, MAX_GAP_S

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "Data"
RESULTS = HERE / "results"
OUT = RESULTS / "transforms" / "error_budget.yaml"
SESSIONS = ("s01", "s02")
SPEED_HORIZON_S = 1.0  # displacement horizon of the speed estimates
STATIONARY_MPS = 0.2   # estimate speed below which an epoch counts as a stop
SLAM_CELLS = tuple((m, v) for m in ("fg_slam", "ekf_slam")
                   for v in ("gaussian", "huber", "cauchy"))

# nominal manufacturer figures (D10): quoted, not derived
NOMINAL = {
    "total_station": {
        "instrument": "Leica TS16 (angle class of the unit not recorded)",
        "distance_prism": "1 mm + 1.5 ppm",
        "distance_reflectorless": "2 mm + 2 ppm",
        "angle": "1\" to 5\" by instrument class",
        "source": "Leica Viva TS16 data sheet, https://leica-geosystems.com/-/media/files/leicageosystems/products/datasheets/leica_viva_ts16_ds.ashx",
    },
    "uwb_module": {
        "module": "Qorvo/Decawave DWM1001 (MDEK1001 kit)",
        "ranging_accuracy": "to within 10 cm",
        "antenna_delay": "pre-calibrated per design and stored in OTP; per-module calibration recommended for greater accuracy",
        "phase_centre": "no figure given",
        "source": "DWM1001 data sheet v1.5 (Decawave 2017), https://store.qorvo.com/datasheets/qorvo/dwm1001datasheet.pdf",
    },
}
SURVEY_POINT_SIGMA_M = 0.002  # nominal, reflectorless distance term


def planar(tf: frames.Transform) -> tuple[np.ndarray, np.ndarray]:
    return tf.R[:2, :2], tf.t[:2]


def speeds(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Speed over SPEED_HORIZON_S around each sample (NaN at the ends)."""
    half = SPEED_HORIZON_S / 2
    x0 = np.interp(t - half, t, xy[:, 0]), np.interp(t - half, t, xy[:, 1])
    x1 = np.interp(t + half, t, xy[:, 0]), np.interp(t + half, t, xy[:, 1])
    v = np.hypot(x1[0] - x0[0], x1[1] - x0[1]) / SPEED_HORIZON_S
    v[(t - half < t[0]) | (t + half > t[-1])] = np.nan
    return v


def truth_bursts(truth: pd.DataFrame) -> dict:
    """Truth bursts under the evaluation's gap rule (MAX_GAP_S), and the
    Vicon frame-counter rate across each gap and within the bursts."""
    t = truth.t.to_numpy(float)
    f = truth.frame.to_numpy(float)
    dt, df = np.diff(t), np.diff(f)
    gap = dt > MAX_GAP_S
    seg = np.r_[0, np.cumsum(gap)]
    dur = np.array([t[seg == s][-1] - t[seg == s][0]
                    for s in range(seg[-1] + 1)])
    rate = df[gap] / dt[gap]
    return {"bursts": int(seg[-1] + 1),
            "burst_median_s": round(float(np.median(dur)), 2),
            "burst_max_s": round(float(dur.max()), 2),
            "gap_max_s": round(float(dt[gap].max()), 1),
            "vicon_rate_across_gaps_hz": [round(float(rate.min()), 1),
                                          round(float(rate.max()), 1)],
            "vicon_unit_steps_in_bursts": round(float((df[~gap] == 1).mean()), 3)}


def matched(t_est: np.ndarray, t_truth: np.ndarray) -> np.ndarray:
    """Estimate epochs with a truth pose within MATCH_TOL_S (the
    evaluation's nearest-checkpoint rule)."""
    j = np.clip(np.searchsorted(t_truth, t_est), 1, len(t_truth) - 1)
    near = np.minimum(np.abs(t_truth[j - 1] - t_est), np.abs(t_truth[j] - t_est))
    return near <= MATCH_TOL_S


def static_duty_cycle(sid: str, mobile: str = "DD98") -> dict:
    """Coverage of a static set B session: tag ranging epochs with a truth
    pose (no registration needed, timestamps only)."""
    sdir = DATA / "set_b" / sid
    rng = pd.read_parquet(sdir / "ranges.parquet")
    tag = rng[(rng.from_id == mobile) | (rng.to_id == mobile)]
    t_ep = tag.groupby("seq").t.median().sort_values().to_numpy(float)
    truth = pd.read_parquet(sdir / "groundtruth.parquet").sort_values("t")
    cov = matched(t_ep, truth.t.to_numpy(float))
    out = {"epochs": int(len(t_ep)), "covered": int(cov.sum()),
           "share": round(float(cov.mean()), 3)}
    out.update(truth_bursts(truth))
    return out


def cell_coverage_terms(cell: pd.DataFrame, moving: pd.Series,
                        stream: dict, anchors_xy: np.ndarray) -> dict:
    """ATE at covered epochs split by stops and motion, reweighted to the
    walk's shares, and the range residual of the aligned track on covered
    and uncovered epochs (median over epochs of the per-epoch median
    absolute residual against the surveyed anchors)."""
    c = cell.set_index("seq")
    mv = moving.reindex(c.index)
    cov = c.x_ref.notna()
    sq = c.err_m ** 2
    known = mv.notna()
    out = {"ate_m": round(float(np.sqrt(sq[cov].mean())), 3)}
    ate_rw = 0.0
    for name, sel in (("moving", mv == 1.0), ("stationary", mv == 0.0)):
        mse = float(sq[cov & sel].mean())
        out[f"ate_{name}_m"] = round(float(np.sqrt(mse)), 3)
        ate_rw += float((sel & known).sum() / known.sum()) * mse
    out["ate_reweighted_m"] = round(float(np.sqrt(ate_rw)), 3)

    seqs = np.asarray(stream["seqs"])
    pos = c.loc[seqs, ["x_enu", "y_enu"]].to_numpy(float)
    e, k, r = stream["e"], stream["k"], stream["r"]
    res = np.abs(r - np.linalg.norm(pos[e] - anchors_xy[k], axis=1))
    per_epoch = pd.Series(res).groupby(e).median()
    cov_ep = cov.loc[seqs].to_numpy()[per_epoch.index]
    out["range_resid_covered_m"] = round(float(per_epoch[cov_ep].median()), 3)
    out["range_resid_uncovered_m"] = round(float(per_epoch[~cov_ep].median()), 3)
    return out


def main() -> int:
    traj_all = pd.read_parquet(RESULTS / "benchmark_trajectories.parquet")
    traj_all = traj_all[traj_all["mask"] == "slam_bipartite"]
    traj = traj_all[(traj_all.method == "fg_slam")
                    & (traj_all.variant == "cauchy")]
    bench = pd.read_parquet(RESULTS / "benchmark.parquet")
    out = {"set_b": {}, "set_a": {}, "nominal": NOMINAL}

    for sid in SESSIONS:
        sdir = DATA / "set_b" / sid
        meta = yaml.safe_load((sdir / "session.yaml").read_text(encoding="utf-8"))
        tf_w = frames.Transform.from_dict(meta["transform_vicon_to_local"])
        uwb = yaml.safe_load((RESULTS / "transforms" /
                              f"vicon_to_enu_set_b_{sid}.yaml")
                             .read_text(encoding="utf-8"))
        tf_u = frames.Transform.from_dict(uwb)
        cmp = uwb["provenance"]["wand_comparison"]

        tr = traj[traj.session == sid].sort_values("t")
        cov = tr.x_ref.notna().to_numpy()
        enu = tr[["x_enu", "y_enu"]].to_numpy(float)
        ref_w = tr[["x_ref", "y_ref"]].to_numpy(float)
        Rw, tw = planar(tf_w)
        Ru, tu = planar(tf_u)
        ref_u = (ref_w - tw) @ Rw @ Ru.T + tu  # same Vicon pose, UWB fit
        ate_w = float(np.sqrt(np.nanmean(np.sum((enu - ref_w)[cov] ** 2, 1))))
        ate_u = float(np.sqrt(np.nanmean(np.sum((enu - ref_u)[cov] ** 2, 1))))
        row = bench[(bench.campaign == "set_b") & (bench.session == sid)
                    & (bench["mask"] == "slam_bipartite")
                    & (bench.method == "fg_slam") & (bench.variant == "cauchy")]
        assert len(row) == 1 and abs(float(row.ate_rmse_m.iloc[0]) - ate_w) < 1e-9, \
            f"{sid}: trajectory table and benchmark ATE disagree"

        # radius of the covered poses around the Vicon origin (wand point)
        vic_xy = (ref_w[cov] - tw) @ Rw  # back to the Vicon frame
        r95 = float(np.percentile(np.hypot(vic_xy[:, 0], vic_xy[:, 1]), 95))
        baseline = float(meta["transform_provenance"]["baseline_m"])
        sig_yaw = float(np.sqrt(2.0) * SURVEY_POINT_SIGMA_M / baseline)

        # speeds: Vicon at covered epochs, estimate on covered / uncovered
        truth = pd.read_parquet(sdir / "groundtruth.parquet").sort_values("t")
        v_vic = speeds(truth.t.to_numpy(float), truth[["x", "y"]].to_numpy(float))
        t_est = tr.t.to_numpy(float)
        v_est = speeds(t_est, enu)
        sig_t = float(meta["clock"]["uncertainty_s"])

        out["set_b"][sid] = {
            "registration": {
                "delta_yaw_deg": round(cmp["delta_yaw_deg"], 3),
                "delta_t_xy_m": round(cmp["delta_t_xy_m"], 3),
                "delta_z_m": round(cmp["delta_t_m"][2], 3),
                "mlat_xy_rmse_uwb_fit_m": round(cmp["xy_rmse_uwb_fit_m"], 3),
                "mlat_xy_rmse_wand_tie_m": round(cmp["xy_rmse_wand_tie_m"], 3),
                "ate_fg_slam_cauchy_wand_tie_m": round(ate_w, 3),
                "ate_fg_slam_cauchy_uwb_fit_m": round(ate_u, 3),
            },
            "wand_baseline": {
                "baseline_m": round(baseline, 3),
                "nominal_point_sigma_m": SURVEY_POINT_SIGMA_M,
                "yaw_sigma_deg": round(float(np.degrees(sig_yaw)), 3),
                "radius_p95_m": round(r95, 2),
                "effect_at_radius_p95_m": round(r95 * sig_yaw, 3),
            },
            "clock": {
                "offset_uncertainty_s": sig_t,
                "vicon_speed_median_mps": round(float(np.nanmedian(v_vic)), 2),
                "vicon_speed_p95_mps": round(float(np.nanpercentile(v_vic, 95)), 2),
                "effect_median_m": round(float(np.nanmedian(v_vic)) * sig_t, 3),
                "effect_p95_m": round(float(np.nanpercentile(v_vic, 95)) * sig_t, 3),
            },
            "coverage": {
                "epochs": int(len(tr)), "covered": int(cov.sum()),
                "share": round(float(cov.mean()), 3),
                "estimate_speed_covered_mps": round(float(np.nanmedian(v_est[cov])), 2),
                "estimate_speed_uncovered_mps": round(float(np.nanmedian(v_est[~cov])), 2),
            },
        }

        # bursts, counter rate, stop/motion reweighting, range consistency
        moving = pd.Series(np.where(np.isnan(v_est), np.nan,
                                    (v_est >= STATIONARY_MPS).astype(float)),
                           index=tr.seq.to_numpy())
        g = load_session(sdir).apply_mask("slam_bipartite")
        order, _, stream = slam_common.order_and_stream(g)
        anchors_xy = (evaluation.surveyed_reference(sdir).set_index("id")
                      .loc[order, ["x", "y"]].to_numpy(float))
        seqs = np.asarray(stream["seqs"])
        assert set(seqs) <= set(tr.seq), f"{sid}: stream epochs missing in the track"
        cells = {}
        for m, v in SLAM_CELLS:
            cell = traj_all[(traj_all.session == sid) & (traj_all.method == m)
                            & (traj_all.variant == v)].sort_values("t")
            cells[f"{m}_{v}"] = cell_coverage_terms(cell, moving, stream,
                                                    anchors_xy)
            b_ate = float(bench[(bench.campaign == "set_b") & (bench.session == sid)
                                & (bench["mask"] == "slam_bipartite")
                                & (bench.method == m) & (bench.variant == v)]
                          .ate_rmse_m.item())
            assert abs(cells[f"{m}_{v}"]["ate_m"] - b_ate) < 5e-4, \
                f"{sid} {m} {v}: ATE {cells[f'{m}_{v}']['ate_m']} != {b_ate}"
        ref_pos = tr.set_index("seq").loc[seqs, ["x_ref", "y_ref"]].to_numpy(float)
        e, k, r = stream["e"], stream["k"], stream["r"]
        res_truth = np.abs(r - np.linalg.norm(ref_pos[e] - anchors_xy[k], axis=1))
        per_truth = pd.Series(res_truth).groupby(e).median().dropna()
        known = moving.notna()
        cov_s = pd.Series(cov, index=tr.seq.to_numpy())
        extra = truth_bursts(truth)
        extra.update({
            "stationary_threshold_mps": STATIONARY_MPS,
            "stationary_share_all": round(float((moving[known] == 0).mean()), 2),
            "stationary_share_covered": round(
                float((moving[known & cov_s] == 0).mean()), 2),
            "range_resid_truth_m": round(float(per_truth.median()), 3),
            "cells": cells,
        })
        out["set_b"][sid]["coverage"].update(extra)
        print(f"set_b/{sid}: {extra['bursts']} bursts, counter "
              f"{extra['vicon_rate_across_gaps_hz']} Hz across gaps; ATE "
              + ", ".join(f"{n} {c['ate_m']:.3f}->{c['ate_reweighted_m']:.3f}"
                          for n, c in cells.items()))
        print(f"set_b/{sid}: ATE {ate_w:.3f} m (wand tie) / {ate_u:.3f} m (UWB fit); "
              f"yaw sigma {np.degrees(sig_yaw):.2f} deg -> {r95 * sig_yaw:.3f} m at "
              f"r95 {r95:.1f} m; clock {np.nanmedian(v_vic) * sig_t:.3f} m; covered "
              f"{cov.mean():.0%}, speed {np.nanmedian(v_est[cov]):.2f} / "
              f"{np.nanmedian(v_est[~cov]):.2f} m/s")

    out["set_b"]["s03"] = {"coverage": static_duty_cycle("s03")}
    print(f"set_b/s03 (static): covered {out['set_b']['s03']['coverage']['share']:.1%}")

    adj = pd.read_csv(DATA / "set_a" / "truth_adjudication.csv")
    nodes = yaml.safe_load((DATA / "set_a" / "nodes.yaml").read_text(encoding="utf-8"))
    z8 = next(n["prior"]["xyz_local"][2] for n in nodes["nodes"] if n["id"] == "8B05")
    zp = next(c["xyz_local"][2] for c in nodes["control_points"] if c["name"] == "P01")
    out["set_a"] = {
        "tag_height_above_marker_m": float(adj.tag_height_above_marker_m.iloc[0]),
        "common_range_bias_m": float(adj.common_range_bias_m.iloc[0]),
        "surveyed_8B05_antenna_above_P01_m": round(z8 - zp, 3),
        "sessions_verified": int((adj.truth_quality == "verified").sum()),
        "sessions": int(len(adj)),
    }
    print(f"set_a: tag {out['set_a']['tag_height_above_marker_m']:.2f} m above the "
          f"marker (surveyed antenna {z8 - zp:.3f} m), "
          f"{out['set_a']['sessions_verified']}/{len(adj)} verified")

    OUT.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
