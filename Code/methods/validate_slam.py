"""Validation of the slam_bipartite family (rerunnable; exit 0 = all green).

Four parts, each printing verified numbers:

1. Static-tag degeneracy: an explicit construction of two DIFFERENT anchor
   geometries that generate IDENTICAL bipartite range streams from one
   static tag (each anchor rotated independently about the tag). This pins
   the family rule "slam_bipartite is dyn-only" (family_masks.FAMILY_MOTIONS)
   as a verified statement, not an assumption.
2. Exact recovery through the PRODUCTION path: surveyed s03 anchors,
   synthetic lissajous trajectory, noiseless superset graph -> the real
   slam_bipartite mask -> the registry adapters -> evaluation.evaluate_
   estimate + evaluation_slam.evaluate_trajectory. Gates: fg_slam aligned
   anchor RMSE < 1e-2 m AND checkpoint ATE < 1e-2 m for all three kernels
   (the tolerance absorbs the motion-prior bias: the Gaussian random-walk
   prior slightly shrinks the true motion even at zero range noise);
   ekf_slam anchor RMSE < 0.1 m and ATE < 0.1 m (a causal filter carries
   transient init error). Also reports per-epoch anchor visibility and the
   fg_slam runtime on the full-length problem.
3. ATE/RPE code check on a synthetic session with a KNOWN vicon->enu
   transform and gappy (bursty) truth coverage: the undistorted estimate
   must score ATE = RPE = 0; an injected linear drift alpha * t must
   reproduce the ANALYTIC ATE (alpha * rms of matched t) and RPE (alpha *
   the 1 s horizon) exactly; no truth match may cross a coverage gap
   (epochs outside coverage stay unmatched, RPE pairs stay within
   segments). The 10 Hz rate makes the RPE horizon land exactly on an
   epoch, so the expectation is unambiguous.
4. The synthetic selftest of estimate_vicon_transform.py (multilateration
   + matching + trimmed Umeyama recover a known planar transform to
   < 1e-6).

Run from Code/methods:  python validate_slam.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from radiocal import frames

import estimate_vicon_transform
import evaluation
import evaluation_slam
import family_masks
import synthetic
from registry import REGISTRY

DATA = Path(__file__).resolve().parents[2] / "Data"
SIGMA_M = 0.1  # nominal noise scale handed to the methods (exact ranges)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"   {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- part 1

def part1_static_degeneracy() -> None:
    """Two different anchor geometries, identical static-tag range stream."""
    print("== part 1: static-tag degeneracy (slam_bipartite is dyn-only)")
    rng = np.random.default_rng(7)
    n = 6
    tag = np.zeros(2)
    rho = rng.uniform(3.0, 12.0, n)           # tag-anchor distances
    phi = rng.uniform(0.0, 2 * np.pi, n)      # geometry A angles
    dphi = rng.uniform(0.5, 2.5, n)           # independent per-anchor spins
    A = tag + np.column_stack([rho * np.cos(phi), rho * np.sin(phi)])
    B = tag + np.column_stack([rho * np.cos(phi + dphi),
                               rho * np.sin(phi + dphi)])
    rA = np.linalg.norm(A - tag, axis=1)
    rB = np.linalg.norm(B - tag, axis=1)
    print("   construction: tag at the origin; anchor k of geometry B is "
          "anchor k of geometry A rotated by an independent angle about "
          "the tag - every epoch of a static tag repeats the same ranges "
          "rho_k, so the two streams are indistinguishable.")
    print(f"   rho_k = {np.round(rho, 3)}")
    print(f"   spin  = {np.round(dphi, 3)} rad")
    check("identical range streams", float(np.abs(rA - rB).max()) < 1e-12,
          f"max |r_A - r_B| = {np.abs(rA - rB).max():.1e} (exact in exact "
          f"arithmetic; float rounding only)")
    ids = [f"A{k}" for k in range(n)]
    dfA = pd.DataFrame({"id": ids, "x": A[:, 0], "y": A[:, 1]})
    dfB = pd.DataFrame({"id": ids, "x": B[:, 0], "y": B[:, 1]})
    _, resid = frames.align(dfA, dfB, dim=2, with_scale=False,
                            allow_reflection=True)
    check("geometries differ", resid.attrs["rmse_m"] > 1.0,
          f"aligned (reflection allowed) anchor RMSE between the two "
          f"geometries = {resid.attrs['rmse_m']:.3f} m")


# ---------------------------------------------------------------- part 2

def part2_exact_recovery() -> float:
    """Noiseless exact recovery through mask + registry + evaluation.
    Returns the fg_slam full-length runtime (s)."""
    print("== part 2: exact recovery through the production path")
    ref = evaluation.surveyed_reference(DATA / "set_b" / "s03")
    traj = synthetic.synthetic_trajectory(ref, duration_s=240.0)
    g = synthetic.synthetic_slam_graph(ref, traj, sigma=0.0)
    gb = g.apply_mask(family_masks.resolve_mask("slam_bipartite"))

    per_epoch = gb.symmetrized().groupby("seq").size()
    print(f"   {len(ref)} anchors, {per_epoch.size} epochs; per-epoch "
          f"anchor visibility min/median/max = {per_epoch.min()}/"
          f"{int(per_epoch.median())}/{per_epoch.max()} (>= 3 needed)")
    check("anchor visibility", per_epoch.min() >= 3,
          f"min {per_epoch.min()} anchors per epoch")

    ctx = evaluation_slam.context_from_truth(g.truth, tf=None)  # truth = enu
    fg_runtime = 0.0
    gates = {"fg_slam": (1e-2, 1e-2), "ekf_slam": (0.1, 0.1)}
    for method, (rmse_max, ate_max) in gates.items():
        for variant in ("gaussian", "huber", "cauchy"):
            t0 = time.perf_counter()
            est, info = REGISTRY[method].run(gb, variant, sigma_m=SIGMA_M)
            dt = time.perf_counter() - t0
            if method == "fg_slam":
                fg_runtime = max(fg_runtime, dt)
            result = evaluation.evaluate_estimate(est, ref)
            rows, metrics = evaluation_slam.evaluate_trajectory(
                info["trajectory"], result, ctx)
            ok = (result["rmse_m"] < rmse_max
                  and metrics["ate_rmse_m"] < ate_max
                  and metrics["n_truth_matched"] == len(traj))
            check(f"{method}:{variant}", ok,
                  f"anchor RMSE {result['rmse_m']:.2e} m (< {rmse_max}), "
                  f"ATE {metrics['ate_rmse_m']:.2e} m (< {ate_max}), "
                  f"RPE {metrics['rpe_rmse_m']:.2e} m, "
                  f"{int(metrics['n_truth_matched'])}/{len(traj)} matched, "
                  f"{dt:.1f} s")
    print(f"   fg_slam full-length (3000 epochs) runtime <= "
          f"{fg_runtime:.1f} s per kernel")
    return fg_runtime


# ---------------------------------------------------------------- part 2b

def part2b_mobile_assisted_dynamic() -> None:
    """Exact recovery of fg_slam on the mobile_assisted view of a DYNAMIC
    synthetic session (the M11 configuration: full anchor mesh + moving
    tag). The same superset graph as part 2 under the mobile_assisted mask
    retains the anchor-anchor pairs, so fg_slam must (a) pick up one robust
    range factor per inter-anchor pair and (b) still recover anchors and
    trajectory exactly at zero noise."""
    print("== part 2b: mobile_assisted dynamic view (fg_slam + E_aa)")
    import slam_common
    from common import default_anchor_order
    ref = evaluation.surveyed_reference(DATA / "set_b" / "s03")
    traj = synthetic.synthetic_trajectory(ref, duration_s=240.0)
    g = synthetic.synthetic_slam_graph(ref, traj, sigma=0.0)
    gma = g.apply_mask(family_masks.resolve_mask("mobile_assisted"))

    n = len(ref)
    aa = slam_common.anchor_pair_arrays(gma, default_anchor_order(gma))
    n_expected = n * (n - 1) // 2
    check("aa factor count", aa is not None and len(aa["r"]) == n_expected,
          f"{0 if aa is None else len(aa['r'])} anchor-anchor pairs "
          f"(complete mesh = {n_expected})")

    ctx = evaluation_slam.context_from_truth(g.truth, tf=None)
    for variant in ("gaussian", "huber", "cauchy"):
        est, info = REGISTRY["fg_slam"].run(gma, variant, sigma_m=SIGMA_M)
        result = evaluation.evaluate_estimate(est, ref)
        rows, metrics = evaluation_slam.evaluate_trajectory(
            info["trajectory"], result, ctx)
        ok = (result["rmse_m"] < 1e-2 and metrics["ate_rmse_m"] < 1e-2
              and info.get("n_aa_pairs") == n_expected)
        check(f"fg_slam+aa:{variant}", ok,
              f"anchor RMSE {result['rmse_m']:.2e} m (< 1e-2), "
              f"ATE {metrics['ate_rmse_m']:.2e} m (< 1e-2), "
              f"n_aa_pairs {info.get('n_aa_pairs')}")


# ---------------------------------------------------------------- part 3

def part3_metric_check() -> None:
    """ATE/RPE arithmetic against analytic expectations on gappy truth."""
    print("== part 3: ATE/RPE code check (gappy coverage, known transform)")
    ref = evaluation.surveyed_reference(DATA / "set_b" / "s03")
    rate = 10.0  # RPE horizon = exactly 10 epochs -> unambiguous pairing
    traj = synthetic.synthetic_trajectory(ref, rate_hz=rate,
                                          duration_s=240.0)
    yaw = np.radians(25.0)
    c, s = np.cos(yaw), np.sin(yaw)
    tf_true = frames.Transform(
        np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]),
        np.array([5.0, -3.0, 0.5]), 1.0, src="vicon", dst="enu")
    coverage = ((0.0, 0.2), (0.35, 0.5), (0.7, 0.9))
    g = synthetic.synthetic_slam_graph(ref, traj, sigma=0.0,
                                       vicon_tf=tf_true, truth_z=1.0,
                                       coverage=coverage)
    ctx = evaluation_slam.context_from_truth(g.truth, tf_true)
    check("segment count", ctx["n_segments"] == len(coverage),
          f"{ctx['n_segments']} segments from {len(coverage)} coverage "
          f"windows")

    n = len(traj)
    frac = np.arange(n) / (n - 1)
    keep = np.zeros(n, dtype=bool)
    for lo, hi in coverage:
        keep |= (frac >= lo) & (frac < hi)
    t = traj["t"].to_numpy()

    ident = {"tf": frames.Transform.identity(dim=2)}

    # undistorted estimate = generating trajectory -> zero error
    rows, m0 = evaluation_slam.evaluate_trajectory(traj, ident, ctx)
    check("undistorted ATE/RPE ~ 0",
          m0["ate_rmse_m"] < 1e-9 and m0["rpe_rmse_m"] < 1e-9,
          f"ATE {m0['ate_rmse_m']:.2e} m, RPE {m0['rpe_rmse_m']:.2e} m")
    check("matched = covered epochs",
          m0["n_truth_matched"] == keep.sum()
          and (rows["segment"].to_numpy()[~keep] == -1).all()
          and np.isnan(rows["err_m"].to_numpy()[~keep]).all(),
          f"{int(m0['n_truth_matched'])} matched == {keep.sum()} covered; "
          f"uncovered epochs unmatched (segment -1, err NaN)")

    # injected linear drift alpha * t along x
    alpha = 0.01  # m/s
    drift = traj.copy()
    drift["x"] = drift["x"] + alpha * t
    rows, m1 = evaluation_slam.evaluate_trajectory(drift, ident, ctx)
    ate_expect = alpha * float(np.sqrt((t[keep] ** 2).mean()))
    check("drift ATE analytic",
          abs(m1["ate_rmse_m"] - ate_expect) < 1e-9,
          f"ATE {m1['ate_rmse_m']:.6f} m vs alpha*rms(t_matched) "
          f"= {ate_expect:.6f} m")

    # analytic RPE from the STATED acceptance rule (not from the code):
    # a drifting estimate errs by alpha*dt per pair. In a window of L
    # matched epochs at 10 Hz, epochs up to L-10 pair at dt = 1.0 s
    # (L - 10 pairs); the acceptance window |dt - 1| < 0.25 additionally
    # admits the two window-end pairs that only reach the last epoch of
    # their own segment, at dt = 0.9 and 0.8 s. A pair crossing a coverage
    # gap would violate this budget and shift the RMSE.
    seg_sizes = [int(keep[(frac >= lo) & (frac < hi)].sum())
                 for lo, hi in coverage]
    horizon = int(rate * evaluation_slam.RPE_DELTA_S)
    dts = []
    for L in seg_sizes:
        dts += [1.0] * max(0, L - horizon) + [0.9, 0.8]
    rpe_expect = alpha * float(np.sqrt(np.mean(np.square(dts))))
    check("drift RPE analytic",
          abs(m1["rpe_rmse_m"] - rpe_expect) < 1e-9,
          f"RPE {m1['rpe_rmse_m']:.8f} m vs analytic {rpe_expect:.8f} m "
          f"({len(dts)} pairs incl. 2 sub-horizon window-end pairs per "
          f"segment; no pair crosses a gap)")

    matched_seg = rows.loc[rows["segment"] >= 0, "segment"].to_numpy()
    check("segments partition matches", len(np.unique(matched_seg))
          == len(coverage),
          f"matched epochs fall into {len(np.unique(matched_seg))} truth "
          f"segments; expected {len(coverage)}")


# ---------------------------------------------------------------- part 4

def part4_transform_selftest() -> None:
    print("== part 4: estimate_vicon_transform synthetic selftest")
    try:
        estimate_vicon_transform.synthetic_selftest()
        check("transform selftest", True, "recovered to < 1e-6")
    except AssertionError as exc:
        check("transform selftest", False, str(exc))


def main() -> int:
    part1_static_degeneracy()
    part2_exact_recovery()
    part2b_mobile_assisted_dynamic()
    part3_metric_check()
    part4_transform_selftest()
    print()
    if FAILURES:
        print(f"VALIDATION FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("VALIDATION PASSED: all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
