"""Validation of the mobile_assisted family evaluation (evaluation_mobile).

Four blocks, all in memory (no benchmark tables are touched); exits 0 only
if every assertion holds:

1. Exact recovery: synthetic superset graph from the surveyed set_a
   anchors with a noise-free static mobile at the real P01 marker, run
   through the production path (mobile_assisted mask -> registry adapters ->
   evaluate_estimate -> evaluate_mobile). Batch solvers (factor graph, all
   SMACOF variants) must recover anchors AND mobile to < 1e-6 m - this pins
   the extra-node estimation order and the anchor-fitted-transform scoring
   of the mobile in one shot. Sequential methods get their family
   tolerances (ekf 1e-2, particle filter 5e-2, grid filter 0.3). The
   closed form and classical MDS are report-only (completion is exact on a
   full superset mesh, so they may come out exact too).

2. 3-D projection systematic: same scene with 3-D slant ranges (tag
   antenna z from groundtruth.parquet, 1.72 m above the marker, against
   anchors ~ +1.3 m). No exactness asserts - the block
   QUANTIFIES the 2-D-projection bias on the mobile and on the anchors,
   read off the otherwise-exact batch solvers. The numbers are geometry,
   not estimator error, and become a documented statement in the family
   narrative.

3. Flag propagation: session_context must surface truth_quality
   "verified" for s01, s05 and s12. The screen of 04_build_set_a.py flags
   s05 and s12, and the direct range test at the marker decides both
   (ruling D10, Data/set_a/truth_adjudication.csv).

4. Mask sanity on the real s03 session: the mobile_assisted view keeps
   anchor-anchor and 8B05-anchor edges, and the estimation order is the
   11 anchors + ["8B05"].

Plus an edge-case block: evaluate_mobile must degrade to (NaN, None) - not
raise - when the mobile is absent from the estimate or has NaN coordinates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from radiocal import load_session

import common
import evaluation
import evaluation_mobile
import family_masks
import synthetic
from registry import METHOD_ORDER_STATIC, REGISTRY

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data" / "set_a"

SIGMA_M = 0.1  # driver-style session scale handed to every adapter

# family tolerances of the exact-recovery block (anchor RMSE and mobile
# error alike); methods not listed are report-only
EXACT_TOL = {
    ("factor_graph", "gaussian"): 1e-6,
    ("factor_graph", "huber"): 1e-6,
    ("factor_graph", "cauchy"): 1e-6,
    ("mds", "smacof_gaussian"): 1e-6,
    ("mds", "smacof_huber"): 1e-6,
    ("mds", "smacof_cauchy"): 1e-6,
    ("ekf", "gaussian"): 1e-2,
    ("ekf", "huber"): 1e-2,
    ("ekf", "cauchy"): 1e-2,
    ("ekf", "gated"): 1e-2,
    ("particle_filter", "gaussian"): 5e-2,
    ("particle_filter", "huber"): 5e-2,
    ("particle_filter", "cauchy"): 5e-2,
    ("grid_filter", "gaussian"): 0.3,
    ("grid_filter", "huber"): 0.3,
    ("grid_filter", "cauchy"): 0.3,
    ("grid_filter", "mixture"): 0.3,
}

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def run_suite(g_view, ref, ctx) -> pd.DataFrame:
    """Full static suite on one masked view, scored through the production
    evaluation path; one row per (method, variant)."""
    rows = []
    for m in METHOD_ORDER_STATIC:
        spec = REGISTRY[m]
        for v in spec.variants:
            est, _ = spec.run(g_view, v, sigma_m=SIGMA_M, sigma_scale=1.0,
                              seed=0)
            result = evaluation.evaluate_estimate(est, ref)
            metrics, mrows = evaluation_mobile.evaluate_mobile(est, result,
                                                               ctx)
            rows.append({"method": m, "variant": v,
                         "rmse_m": result["rmse_m"],
                         "mobile_error_m": metrics["mobile_error_m"],
                         "mobile_row": mrows is not None,
                         "quality": metrics["mobile_truth_quality"]})
    return pd.DataFrame(rows)


def print_suite(df: pd.DataFrame) -> None:
    for r in df.itertuples(index=False):
        print(f"   {r.method}:{r.variant:<16} anchors {r.rmse_m:.3e} m   "
              f"mobile {r.mobile_error_m:.3e} m   row={r.mobile_row}")


def build_scene(ranges_3d: bool):
    """(masked synthetic view, anchor reference, ctx, marker xyz) of the
    noise-free P01 scene."""
    ref = evaluation.surveyed_reference(DATA / "s01")
    g_real = load_session(DATA / "s01")
    t = g_real.truth.iloc[0]
    marker = (float(t["x"]), float(t["y"]), float(t["z"]))
    g_syn = synthetic.synthetic_static_graph(
        ref, sigma=0.0, seed=0, mobile=("8B05", marker), ranges_3d=ranges_3d)
    ctx = evaluation_mobile._context_from_graph(g_syn)
    g_view = g_syn.apply_mask(family_masks.resolve_mask("mobile_assisted"))
    return g_view, ref, ctx, marker


def block_exact_recovery() -> pd.DataFrame:
    print("\n== 1. exact recovery (2-D ranges, sigma = 0, P01 marker)")
    g_view, ref, ctx, _ = build_scene(ranges_3d=False)
    df = run_suite(g_view, ref, ctx)
    print_suite(df)
    for r in df.itertuples(index=False):
        tol = EXACT_TOL.get((r.method, r.variant))
        if tol is None:
            continue  # closed_form / classical MDS: report-only
        worst = max(r.rmse_m, r.mobile_error_m)
        check(f"exact {r.method}:{r.variant} < {tol:g}", worst < tol,
              f"anchors {r.rmse_m:.3e}, mobile {r.mobile_error_m:.3e}")
        check(f"exact {r.method}:{r.variant} mobile row present",
              r.mobile_row)
    return df


def block_projection_3d() -> None:
    print("\n== 2. 3-D slant-range projection systematic (report-only)")
    g_view, ref, ctx, marker = build_scene(ranges_3d=True)
    dz = ref["z"].mean() - marker[2]
    d_xy = np.hypot(ref["x"] - marker[0], ref["y"] - marker[1])
    print(f"   anchor z: mean {ref['z'].mean():+.3f} m, "
          f"spread (std) {ref['z'].std():.3f} m, "
          f"marker z {marker[2]:+.3f} m -> height offset {dz:.3f} m")
    print(f"   mobile-anchor horizontal range: "
          f"min {d_xy.min():.2f} m, median {np.median(d_xy):.2f} m, "
          f"max {d_xy.max():.2f} m")
    df = run_suite(g_view, ref, ctx)
    print_suite(df)
    batch = df[(df["method"] == "factor_graph")
               | ((df["method"] == "mds")
                  & df["variant"].str.startswith("smacof"))]
    print(f"   batch-solver geometry numbers (estimator error < 1e-6 in 2-D):"
          f" mobile bias {batch['mobile_error_m'].min():.3f}-"
          f"{batch['mobile_error_m'].max():.3f} m, anchor RMSE "
          f"{batch['rmse_m'].min():.3f}-{batch['rmse_m'].max():.3f} m")
    check("3-D block ran on all method:variant cells",
          len(df) == len(EXACT_TOL) + 3
          and df["mobile_error_m"].notna().all())


def block_flags() -> None:
    print("\n== 3. truth_quality flag propagation")
    for sid, want in (("s01", "verified"), ("s05", "verified"),
                      ("s12", "verified")):
        ctx = evaluation_mobile.session_context(DATA / sid)
        check(f"{sid} truth_quality == {want!r}",
              ctx["truth_quality"] == want,
              f"got {ctx['truth_quality']!r}, mobile {ctx['mobile_id']}, "
              f"marker {'set' if ctx['marker'] else 'missing'}")


def block_mask_sanity() -> None:
    print("\n== 4. mobile_assisted mask sanity on real s03")
    g = load_session(DATA / "s03")
    gm = g.apply_mask(family_masks.resolve_mask("mobile_assisted"))
    roles = gm.roles()
    fr = gm.edges["from_id"].map(roles)
    to = gm.edges["to_id"].map(roles)
    n_aa = int((fr.eq("anchor") & to.eq("anchor")).sum())
    n_ma = int(((gm.edges["from_id"] == "8B05")
                | (gm.edges["to_id"] == "8B05")).sum())
    check("anchor-anchor edges present", n_aa > 0, f"{n_aa} directed")
    check("8B05-anchor edges present", n_ma > 0, f"{n_ma} directed")
    order = common.estimation_order(gm)
    anchors = common.default_anchor_order(gm)
    check("estimation order = 11 anchors + ['8B05']",
          len(anchors) == 11 and order == anchors + ["8B05"],
          f"{len(anchors)} anchors, tail {order[-1]!r}")


def block_edge_cases() -> None:
    print("\n== edge cases: degrade to (NaN, None), never raise")
    ref = evaluation.surveyed_reference(DATA / "s01")
    est = ref[["id", "x", "y"]].copy()  # anchors only, no mobile
    result = evaluation.evaluate_estimate(est, ref)
    ctx = evaluation_mobile.session_context(DATA / "s01")
    m, rows = evaluation_mobile.evaluate_mobile(est, result, ctx)
    check("mobile absent -> NaN metric, no row",
          np.isnan(m["mobile_error_m"]) and rows is None
          and m["mobile_truth_quality"] == "verified")
    est_nan = pd.concat([est, pd.DataFrame(
        [{"id": "8B05", "x": np.nan, "y": np.nan}])], ignore_index=True)
    m, rows = evaluation_mobile.evaluate_mobile(est_nan, result, ctx)
    check("mobile NaN coords -> NaN metric, no row",
          np.isnan(m["mobile_error_m"]) and rows is None)


def main() -> int:
    block_exact_recovery()
    block_projection_3d()
    block_flags()
    block_mask_sanity()
    block_edge_cases()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
