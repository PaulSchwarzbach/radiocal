"""Exact-recovery validation of the multi_hop family (synthetic).

Per reference geometry (set_b/s03 and set_a/s21 surveyed
anchors) and per k in family_masks.MULTIHOP_K_SWEEP plus the un-thinned
fully_meshed reference, a zero-noise synthetic superset graph
(synthetic.synthetic_static_graph, exact 2-D distances) is pushed through
the REAL production path: family_masks.resolve_mask ->
MeasurementGraph.apply_mask -> the full static method roster via
REGISTRY[...].run -> evaluation.evaluate_estimate.

NOTE on thinning semantics: on exact-range synthetic data the per-pair
median measured range EQUALS the true distance, so the kNN thinning selects
by TRUE distances; on real data it selects by measured medians (NLOS bias
can reorder neighbours). The synthetic sweep therefore certifies the
geometry-induced structure, not the measurement-induced one.

What is checked, per (geometry, k) view:

1. Structural certificate of the thinned anchor pair graph (rigidity.py:
   components, generic/redundant rigidity, 3-connectivity, Jackson-Jordan
   global rigidity) plus pair counts and the gauge diagnostic (below).
2. Completion inflation ON EXACT DATA: where the view is connected,
   mds.complete_distances fills unobserved pairs with shortest-path sums,
   which on a thinned graph strictly overestimate the true straight-line
   distances. mean/max of (D_completed - D_true) over the filled entries
   isolates this mechanism - it is the whole systematic error source of
   classical MDS here, and the reason completion-initialized methods start
   biased.
3. Full static roster (all methods x variants), sigma_m = 0.1 fixed (the
   sigma doctrine: one absolute scale into every method; on exact data it
   only scales kernels, the optimum stays at the truth), seed = 0.
   N_EPOCHS = 200 synthetic epochs: the one-pass filters are not iterated,
   so their accuracy from a completion-inflated init scales with the epoch
   budget (EKF gaussian at 2024 k4: 2.4e-2 m after 50 epochs, 6.6e-3 after
   200, 1.7e-3 after 800 - plain 1/N convergence, no trap); real static
   sessions provide thousands of epochs, 50 would starve the filters.

Family-adapted acceptance (the point: exactness is asserted ONLY where it
is information-theoretically possible AND the estimator's own machinery
survives the view; every deviation below was root-caused by a probe, not
assumed):

- factor_graph, globally rigid views: aligned RMSE < FG_TOL = 1e-6 m, all
  kernels. Observed: 1e-14..1e-15 everywhere (second-order LM from the
  completion init). A miss here is a genuine local-minimum finding.
- mds smacof_*, globally rigid views: RMSE < SMACOF_TOL = 1e-6 m, all
  kernels. SMACOF is a first-order majorization scheme and runs to its
  stopping rule (mds.TOL relative objective change, mds.STEP_TOL stagnation
  guard; mds.MAX_ITER is a safety cap). Observed: <= 3e-11 m on every
  globally rigid view. Until 2026-09-30 the solver stopped at a fixed
  500-iteration budget, which left 1.3-1.8e-4 m on the sparsest globally
  rigid view (2024 k4) and unconverged coordinates on the real thinned
  views.
- mds classical: never asserted on thinned views; its RMSE is reported next
  to the completion inflation (it embeds the inflated matrix directly, so
  the two numbers track each other - the mechanism statement).
- ekf, globally rigid views: RMSE < EKF_TOL = 1e-2, all variants (see the
  epoch-budget note above; at 200 epochs the worst case is the gaussian
  kernel at 2024 k4 with ~7e-3 - the robust kernels converge faster there
  because inflating R on large residuals delays premature covariance
  collapse from the biased init).
- particle_filter: RMSE < PF_TOL = 5e-2 asserted ONLY on fully_meshed. On
  thinned views the per-anchor conditional decomposition (each anchor
  weights its particles against point estimates of its neighbours only)
  keeps an error that global rigidity does not cure within the epoch
  budget: at 2025 k4 the RMSE is 0.23 m at 50 and 200 epochs and 0.12 m at
  800 (probe), so the decomposition is not separated from slow convergence. It
  passes 5e-2 only where the conditioning graph is dense (k >= 6 at 2025,
  k = 8 at 2024, fully_meshed). Reported everywhere, asserted where verified.
- grid_filter: RMSE < GRID_TOL = 0.3 (0.25 m grid resolution) asserted ONLY
  on fully_meshed. Its gauge machinery has a structural precondition the
  thinning destroys: the anchor-1 histogram needs the (0, 1) gauge-pair
  range, and each grid anchor's closed-form init needs ranges to BOTH
  gauge anchors. kNN thinning removes LONG edges first and the gauge
  baseline 5BB6-4A23 is long by design: d01 survives only at k = 8 (2025)
  and k >= 5 (2024), and the both-gauge-ranges count collapses (2025 k4:
  2/8 anchors; 2024 k5: 2/10). Where d01 is missing the filter degrades to
  1 placed anchor; where d01 exists but gauge visibility is sparse it
  places everything but locks uniform-started grids onto wrong modes
  (2024 k5: 12 m). The per-view gauge diagnostic is printed as the
  mechanism. closed_form shares the precondition: no assert (per family
  rules), but note its disambiguated variant is machine-exact (< 1e-13) on
  the subset it CAN place, on every view - coverage collapses, correctness
  does not.
- disconnected views: the completion-calling methods (mds, factor_graph,
  ekf) must raise mds.DisconnectedGraphError - a structured outcome; the
  non-raising methods (closed_form, particle_filter via its internal
  closed-form init fallback, grid_filter) must degrade without crashing.
  That asymmetry is part of the family narrative.
- flexible / rigid-but-not-globally-rigid views: no numeric asserts - the
  ambiguity class itself is the result. Estimates may still be exact
  (factor_graph hits 7e-15 on 2025 k3, rigid_not_global: generic rigidity
  makes the truth locally unique and the completion init lay in its
  basin), but exactness is not certifiable there and is not asserted.
  The per-row fit diagnostic (SMACOF stress / factor-graph cost) shows
  whether a wrong embedding still fits the thinned ranges.
- ANY unhandled exception anywhere is a validation FAILURE.

Additionally, spec-dict-vs-YAML consistency on the REAL set_b/s03
session: resolve_mask("multi_hop_k3") (spec dict) and the canonical
"multi_hop" YAML (k = 3) must produce identical edge and node
tables. A final geometry x k summary matrix closes the report.

Exit code 0 iff every assertion holds. Rerunnable, in-memory only: no
benchmark table is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from radiocal import load_session

import evaluation
import family_masks
import mds
import rigidity
import synthetic
from common import estimation_order, pair_counts, pair_range_matrix
from registry import METHOD_ORDER_STATIC, REGISTRY

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
GEOMETRIES = ("set_b/s03", "set_a/s21")

SIGMA_M = 0.1    # fixed absolute scale into every method (sigma doctrine)
SEED = 0
N_EPOCHS = 200   # filter epoch budget; see module docstring

FG_TOL = 1e-6        # factor_graph exactness where globally rigid
SMACOF_TOL = 1e-6    # smacof exactness where globally rigid (run to mds.TOL)
EKF_TOL = 1e-2       # ekf convergence tolerance (200-epoch budget)
PF_TOL = 5e-2        # particle_filter, asserted on fully_meshed only
GRID_TOL = 0.3       # grid_filter (0.25 m cells), fully_meshed only
RAISES_ON_DISCONNECT = ("mds", "factor_graph", "ekf")

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"   FAIL: {msg}")


def true_matrix(ref: pd.DataFrame, order: list[str]) -> np.ndarray:
    """Exact 2-D distance matrix of the surveyed anchors in ``order``."""
    P = ref.set_index("id").loc[order, ["x", "y"]].to_numpy(float)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(d, np.nan)
    return d


def completion_inflation(M: np.ndarray, D_true: np.ndarray) -> dict | None:
    """Shortest-path completion error on exact data: stats of
    D_completed - D_true over the FILLED (unobserved) entries. None if the
    view is disconnected (completion undefined)."""
    try:
        D, n_filled = mds.complete_distances(M)
    except mds.DisconnectedGraphError:
        return None
    iu, ju = np.triu_indices(M.shape[0], k=1)
    filled = ~np.isfinite(M[iu, ju])
    if not filled.any():
        return {"n_filled": 0, "mean_m": 0.0, "max_m": 0.0}
    err = D[iu, ju][filled] - D_true[iu, ju][filled]
    return {"n_filled": int(n_filled), "mean_m": float(err.mean()),
            "max_m": float(err.max())}


def gauge_diagnostic(M: np.ndarray) -> dict:
    """Survival of the sequential gauge machinery under thinning: the
    (0, 1) baseline range and, per anchor >= 2, ranges to both gauge
    anchors (the closed_form / grid_filter init precondition)."""
    n = M.shape[0]
    both = sum(bool(np.isfinite(M[a, 0]) and np.isfinite(M[a, 1]))
               for a in range(2, n))
    return {"d01": bool(np.isfinite(M[0, 1])), "both_gauge": int(both),
            "n_grid_anchors": n - 2}


def _fit_note(method: str, variant: str, info: dict) -> str:
    """Objective-value diagnostic: does the estimate FIT the thinned
    ranges even when it is far from the truth (ambiguous views)?"""
    if method == "mds" and "stress" in info:
        return f"stress {info['stress']:.1e}"
    if method == "factor_graph" and "cost" in info:
        return f"cost {info['cost']:.1e}"
    return ""


def run_roster(g, ref: pd.DataFrame) -> list[dict]:
    """All static methods x variants on a masked view -> result rows."""
    rows = []
    for m in METHOD_ORDER_STATIC:
        spec = REGISTRY[m]
        for v in spec.variants:
            row = {"method": m, "variant": v, "status": "ok",
                   "rmse_m": np.nan, "max_error_m": np.nan, "n_placed": None,
                   "fit": "", "note": ""}
            try:
                est, info = spec.run(g, v, sigma_m=SIGMA_M, sigma_scale=1.0,
                                     seed=SEED)
                est = est.dropna(subset=["x", "y"]).reset_index(drop=True)
                row["n_placed"] = len(est)
                row["fit"] = _fit_note(m, v, info)
                if len(est) >= 2:
                    res = evaluation.evaluate_estimate(est, ref)
                    row["rmse_m"] = res["rmse_m"]
                    row["max_error_m"] = res["max_error_m"]
                else:
                    row["status"] = "degenerate"
                    row["note"] = f"only {len(est)} placed"
            except mds.DisconnectedGraphError as exc:
                row["status"] = "disconnected"
                row["note"] = str(exc)
            except Exception as exc:  # unhandled = anomaly, never OK
                row["status"] = "error"
                row["note"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def check_roster(rows: list[dict], cls: str, mask: str, tag: str) -> None:
    """Family-adapted acceptance rules of one (geometry, k) view; the
    tolerance derivations live in the module docstring."""
    for r in rows:
        mv = f"{r['method']}:{r['variant']}"
        if r["status"] == "error":
            fail(f"{tag} {mv} unhandled exception: {r['note']}")
    if cls == "globally_rigid":
        for r in rows:
            mv = f"{r['method']}:{r['variant']}"
            if r["method"] == "factor_graph":
                tol = FG_TOL
            elif r["method"] == "mds" and r["variant"].startswith("smacof"):
                tol = SMACOF_TOL
            elif r["method"] == "ekf":
                tol = EKF_TOL
            elif r["method"] == "particle_filter" and mask == "fully_meshed":
                tol = PF_TOL
            elif r["method"] == "grid_filter" and mask == "fully_meshed":
                tol = GRID_TOL
            else:  # closed_form, mds classical, pf/grid on thinned views
                continue
            if r["status"] != "ok" or not r["rmse_m"] < tol:
                fail(f"{tag} {mv}: rmse {r['rmse_m']} not < {tol} "
                     f"on a globally rigid view (status {r['status']})")
    elif cls == "disconnected":
        for r in rows:
            mv = f"{r['method']}:{r['variant']}"
            if r["method"] in RAISES_ON_DISCONNECT:
                if r["status"] != "disconnected":
                    fail(f"{tag} {mv}: expected DisconnectedGraphError, "
                         f"got status {r['status']} rmse {r['rmse_m']}")
            elif r["status"] not in ("ok", "degenerate"):
                fail(f"{tag} {mv}: expected structured degrade, got "
                     f"status {r['status']} ({r['note']})")
    # flexible / rigid_not_global: no numeric asserts - the ambiguity is
    # the finding; only the no-crash rule above applies.


def print_roster(rows: list[dict], n_ref: int) -> None:
    for r in rows:
        mv = f"{r['method']}:{r['variant']}"
        if r["status"] == "ok":
            extra = ""
            if r["n_placed"] is not None and r["n_placed"] < n_ref:
                extra += f"  placed {r['n_placed']}/{n_ref}"
            if r["fit"]:
                extra += f"  {r['fit']}"
            print(f"   {mv:<28} rmse {r['rmse_m']:.3e} m  "
                  f"max {r['max_error_m']:.3e} m{extra}")
        else:
            print(f"   {mv:<28} {r['status'].upper()}  {r['note']}")


def spec_vs_yaml_consistency() -> None:
    """resolve_mask('multi_hop_k3') spec dict vs the canonical
    multi_hop YAML (k = 3) on the real set_b/s03 session:
    identical edge and node tables."""
    print("== spec-dict vs YAML consistency (real set_b/s03)")
    g = load_session(DATA / "set_b" / "s03")
    dirs = g.meta.get("_mask_dirs")
    g_spec = g.apply_mask(family_masks.resolve_mask("multi_hop_k3",
                                                    mask_dirs=dirs))
    g_yaml = g.apply_mask("multi_hop")
    e_spec = g_spec.edges.reset_index(drop=True)
    e_yaml = g_yaml.edges.reset_index(drop=True)
    same_edges = e_spec.equals(e_yaml)
    same_nodes = g_spec.nodes.reset_index(drop=True).equals(
        g_yaml.nodes.reset_index(drop=True))
    print(f"   edges: {len(e_spec)} vs {len(e_yaml)} rows, "
          f"identical: {same_edges}; nodes identical: {same_nodes}")
    if not (same_edges and same_nodes):
        fail("multi_hop_k3 spec dict != canonical multi_hop "
             "YAML on the real session")


def main() -> int:
    summary = []
    for geometry in GEOMETRIES:
        campaign, session = geometry.split("/")
        ref = evaluation.surveyed_reference(DATA / campaign / session)
        g0 = synthetic.synthetic_static_graph(ref, n_epochs=N_EPOCHS,
                                              sigma=0.0, seed=SEED)
        masks = ([f"multi_hop_k{k}" for k
                  in family_masks.MULTIHOP_K_SWEEP] + ["fully_meshed"])
        for mask in masks:
            k = family_masks.thinning_k(mask)
            g = g0.apply_mask(family_masks.resolve_mask(mask))
            order = estimation_order(g)
            M = pair_range_matrix(g, order)
            D_true = true_matrix(ref, order)
            iu, ju = np.triu_indices(len(order), k=1)
            pairs = list(zip(iu[np.isfinite(M[iu, ju])],
                             ju[np.isfinite(M[iu, ju])]))
            coords = ref.set_index("id").loc[order, ["x", "y"]].to_numpy()
            cert = rigidity.global_rigidity_certificate(coords, pairs)
            cls = rigidity.certificate_class(cert)
            avail, total = pair_counts(M)
            gd = gauge_diagnostic(M)
            tag = f"{geometry} {mask}"
            print(f"== {tag}: {avail}/{total} pairs, class {cls}")
            print(f"   certificate: {cert}")
            print(f"   gauge: d01 {'present' if gd['d01'] else 'MISSING'}, "
                  f"both gauge ranges {gd['both_gauge']}/"
                  f"{gd['n_grid_anchors']} anchors")
            infl = completion_inflation(M, D_true)
            if infl is None:
                print("   completion: undefined (disconnected)")
            else:
                print(f"   completion inflation: {infl['n_filled']} filled, "
                      f"mean {infl['mean_m']:.3f} m, "
                      f"max {infl['max_m']:.3f} m")
            rows = run_roster(g, ref)
            print_roster(rows, len(ref))
            check_roster(rows, cls, mask, tag)
            ok = [r for r in rows
                  if r["status"] == "ok" and np.isfinite(r["rmse_m"])]
            best = min(ok, key=lambda r: r["rmse_m"]) if ok else None
            worst = max(ok, key=lambda r: r["rmse_m"]) if ok else None
            summary.append({
                "geometry": geometry, "k": "inf" if k is None else k,
                "pairs": f"{avail}/{total}", "class": cls,
                "infl_max_m": np.nan if infl is None else infl["max_m"],
                "best": ("-" if best is None else
                         f"{best['method']}:{best['variant']}"
                         f"={best['rmse_m']:.2e}"),
                "worst": ("-" if worst is None else
                          f"{worst['method']}:{worst['variant']}"
                          f"={worst['rmse_m']:.2e}")})
            print()
    spec_vs_yaml_consistency()
    print()
    print("== summary: geometry x k -> certificate class + best/worst RMSE")
    with pd.option_context("display.width", 200,
                           "display.max_colwidth", 60):
        print(pd.DataFrame(summary).to_string(index=False))
    print()
    if FAILURES:
        print(f"VALIDATION FAILED ({len(FAILURES)} failure(s)):")
        for f in FAILURES:
            print(f" - {f}")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
