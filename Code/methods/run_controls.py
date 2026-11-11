"""Benchmark controls -> results/oracle_start, results/sigma_sweep,
results/offset_control, results/offset_crlb and results/sigma_terms
(parquet + CSV mirrors).

Diagnostics that re-run benchmark cells with ONE setting changed. The rows
are controls, never benchmark results, so they live in their own tables and
leave benchmark.parquet untouched.

Oracle start (factor graph). The benchmark factor graph starts from
classical MDS on the shortest-path-completed range matrix. On a thinned view
that start can lie outside the basin of the correct embedding, so a large
error on a globally rigid view does not say whether the view or the start
is to blame. The control re-solves the same problem (same view, kernel and
session sigma) from the gauged SURVEYED geometry and records both runs side
by side:

- oracle start far below the MDS start, at a lower robust cost: the
  benchmark cell sits in a wrong local minimum, the initialization decides;
- both starts agree: the estimate is what the view's information gives.

The surveyed start is ground truth and no estimator could use it. Default
views: the random-thinning views, the kNN thinning sweep and the full mesh,
all three kernels. Disconnected views are skipped (no MDS-start counterpart
exists). On every generically rigid view (results/rigidity.parquet,
run_rigidity.py first) the MDS-start run must reproduce the stored
benchmark cell.

Noise-scale sweep. The cauchy cell of the five iterative engines on the
full mesh, re-run through the registry at sigma_mult x the session sigma
(0.25 to 4) with every other setting at its benchmark value (grid filter at
the committed stride, particle filter seed 0). The 1x row must reproduce
the stored benchmark cell. This supersedes the per-method
results/<method>/*_scale_sweep.csv of the legacy runners as the reported
sweep (the legacy grid-filter sweep ran at twice the committed stride).

Offset control. The offset-state factor graph
(factor_graph.solve_nls_offset, roster methods fg_offset and fg_offset_dev)
on the full mesh of every session, cauchy kernel: no offset, one common
offset, and per-device offsets at three prior widths, each at 1, 0.5 and
0.25 x the session sigma. Checks: with the offset fixed at zero the solver
reproduces every stored blind factor_graph cell on the fully meshed and the
quasi-static mobile-assisted views (all kernels), and the 1x common and
canonical-width device rows reproduce the stored roster cells. The
offset-aware bounds (crlb.offset_view_bound) go to results/offset_crlb, and
their zero-offset rows reproduce the blind bounds of results/crlb.

Sigma terms. The session noise scale is the larger of two
terms (kernels.session_sigma): the median over pairs of the per-pair epoch
standard deviation, and 1.4826 x the MAD of the classical-MDS residuals.
Both are non-robust in the sense that matters on the NLOS site: a few range
glitches inflate a pair's standard deviation, and the NLOS pairs distort the
classical embedding. The control records both terms per session, which one
sets sigma, and a robust per-pair temporal scatter (1.4826 x the MAD of the
epoch ranges, median over pairs) for comparison. It must reproduce the
benchmark sigma exactly.

    python run_controls.py
    python run_controls.py --only oracle --mask multi_hop_random_sweep
    python run_controls.py --only sweep
    python run_controls.py --only offset
    python run_controls.py --only sigma
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from radiocal import load_session

import benchmark_io
import crlb
import evaluation
import factor_graph
import family_masks
import kernels
import mds
from common import (default_anchor_order, estimation_order, graph_components,
                    pair_counts, pair_range_matrix)
from registry import REGISTRY, check_sigma
from run_benchmark import DATA, GATE_SESSIONS, select_sessions

DEFAULT_MASKS = "fully_meshed,multi_hop_sweep,multi_hop_random_sweep"
MATCH_TOL = 1e-9  # control run at the benchmark setting vs the stored cell

SWEEP_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
SWEEP_CELLS = (("factor_graph", "cauchy"), ("mds", "smacof_cauchy"),
               ("ekf", "cauchy"), ("particle_filter", "cauchy"),
               ("grid_filter", "cauchy"))
SWEEP_SEED = 0  # the benchmark's particle-filter seed

OFFSET_VARIANT = "cauchy"
OFFSET_WIDTHS = (0.03, 0.10, 0.30)  # per-device prior widths [m]
OFFSET_MULTS = (1.0, 0.5, 0.25)     # x the session sigma
OFFSET_ARMS = ((("zero", float("nan")), ("common", float("nan")))
               + tuple(("device", w) for w in OFFSET_WIDTHS))
OFFSET_ROSTER = {"common": "fg_offset", "device": "fg_offset_dev"}


def _stored(table: pd.DataFrame, key, col: str) -> float:
    """The one stored value of a keyed cell. A non-unique index makes .loc
    return a one-row Series, which float() rejects under pandas 3. The
    .item() call takes the value on pandas 2 and 3 alike and raises unless
    exactly one row matches."""
    return float(np.asarray(table.loc[key, col]).item())


def oracle_start(sessions: list[str], masks: list[str],
                 variants: list[str], run_at: str) -> pd.DataFrame:
    bench = benchmark_io.load()
    bench = bench[(bench.method == "factor_graph") & (bench.status == "ok")]
    bench = bench.set_index(["campaign", "session", "mask", "variant"])
    cert = benchmark_io.load("rigidity")  # run_rigidity.py output
    rigid = dict(zip(zip(cert.campaign, cert.session, cert["mask"]),
                     cert.generically_rigid.astype(bool)))

    rows = []
    for s in sessions:
        campaign, session = s.split("/")
        sdir = DATA / campaign / session
        g0 = load_session(sdir)
        g_full = g0.apply_mask("fully_meshed")
        sigma_m = float(kernels.session_sigma(g_full,
                                              default_anchor_order(g_full)))
        ref = evaluation.surveyed_reference(sdir)
        for mask in masks:
            g = g0.apply_mask(family_masks.resolve_mask(
                mask, mask_dirs=g0.meta.get("_mask_dirs")))
            order = estimation_order(g)
            M = pair_range_matrix(g, order)
            if graph_components(M) > 1:
                print(f"{s}  {mask:<24} disconnected, skipped")
                continue
            available, total = pair_counts(M)
            X_ref = ref.set_index("id").loc[order, ["x", "y"]].to_numpy(float)
            for v in variants:
                out = {}
                for start, X0 in (("mds", None), ("oracle", X_ref)):
                    est, info = factor_graph.factor_graph_anchors(
                        g, variant=v, anchor_order=order, sigma_m=sigma_m,
                        X0=X0)
                    res = evaluation.evaluate_estimate(est, ref)
                    out[start] = (res["rmse_m"], info["cost"],
                                  info["n_iter"], bool(info["converged"]))
                key = (campaign, session, mask, v)
                # on a flexible view the minimizer is a continuum and the LM
                # end point is not reproducible to MATCH_TOL across runs
                if key in bench.index and rigid.get(key[:3], False):
                    stored = _stored(bench, key, "rmse_m")
                    assert abs(stored - out["mds"][0]) < MATCH_TOL, \
                        f"MDS-start run differs from benchmark cell {key}"
                rows.append({
                    "campaign": campaign, "session": session, "mask": mask,
                    "variant": v, "sigma_m": sigma_m,
                    "n_anchors": len(order), "n_pairs": available,
                    "n_pairs_total": total,
                    "rmse_mds_start_m": out["mds"][0],
                    "cost_mds_start": out["mds"][1],
                    "n_iter_mds_start": out["mds"][2],
                    "rmse_oracle_start_m": out["oracle"][0],
                    "cost_oracle_start": out["oracle"][1],
                    "n_iter_oracle_start": out["oracle"][2],
                    "converged_mds_start": out["mds"][3],
                    "converged_oracle_start": out["oracle"][3],
                    "run_at": run_at})
                print(f"{s}  {mask:<24} {v:<8} "
                      f"mds start {out['mds'][0]:7.3f} m "
                      f"(cost {out['mds'][1]:9.2f})   "
                      f"oracle start {out['oracle'][0]:7.3f} m "
                      f"(cost {out['oracle'][1]:9.2f})")
    return pd.DataFrame(rows)


def sigma_sweep(sessions: list[str], run_at: str) -> pd.DataFrame:
    bench = benchmark_io.load()
    bench = bench[(bench["mask"] == "fully_meshed") & (bench.status == "ok")
                  & bench.seed.isin([benchmark_io.DETERMINISTIC_SEED,
                                     SWEEP_SEED])]
    bench = bench.set_index(["campaign", "session", "method", "variant"])

    rows = []
    for s in sessions:
        campaign, session = s.split("/")
        sdir = DATA / campaign / session
        g = load_session(sdir).apply_mask("fully_meshed")
        sigma_m = float(kernels.session_sigma(g, default_anchor_order(g)))
        ref = evaluation.surveyed_reference(sdir)
        for method, variant in SWEEP_CELLS:
            spec = REGISTRY[method]
            for mult in SWEEP_MULTS:
                est, info = spec.run(g, variant, sigma_m=sigma_m,
                                     sigma_scale=mult, seed=SWEEP_SEED)
                check_sigma(info, sigma_m, mult)
                res = evaluation.evaluate_estimate(est, ref)
                if mult == 1.0:
                    stored = _stored(bench, (campaign, session, method,
                                             variant), "rmse_m")
                    assert abs(stored - res["rmse_m"]) < MATCH_TOL, \
                        f"1x sweep run differs from benchmark cell " \
                        f"{(s, method, variant)}"
                rows.append({
                    "campaign": campaign, "session": session,
                    "mask": "fully_meshed", "method": method,
                    "variant": variant, "sigma_mult": mult,
                    "seed": (SWEEP_SEED if spec.seeded
                             else benchmark_io.DETERMINISTIC_SEED),
                    "sigma_m": mult * sigma_m, "rmse_m": res["rmse_m"],
                    "max_error_m": res["max_error_m"],
                    "reflection": res["is_reflection"], "run_at": run_at})
                print(f"{s}  {method}:{variant:<14} x{mult:<4g} "
                      f"{res['rmse_m']:.3f} m")
    return pd.DataFrame(rows)


def _device_stats(info: dict) -> dict:
    dev = np.asarray(list(info.get("offset_device_m", {}).values()), float)
    if not len(dev):
        nan = float("nan")
        return {"offset_device_std_m": nan, "offset_device_min_m": nan,
                "offset_device_max_m": nan}
    return {"offset_device_std_m": float(dev.std()),
            "offset_device_min_m": float(dev.min()),
            "offset_device_max_m": float(dev.max())}


def offset_control(sessions: list[str], run_at: str
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    bench = benchmark_io.load()
    bench = bench[bench.status == "ok"].set_index(
        ["campaign", "session", "mask", "method", "variant"])
    blind = benchmark_io.load("crlb").set_index(["campaign", "session",
                                                 "mask"])
    rows, brows, n_repro = [], [], 0
    for s in sessions:
        campaign, session = s.split("/")
        sdir = DATA / campaign / session
        g0 = load_session(sdir)
        g = g0.apply_mask("fully_meshed")
        sigma_m = float(kernels.session_sigma(g, default_anchor_order(g)))
        ref = evaluation.surveyed_reference(sdir)
        order = estimation_order(g)

        # offset fixed at zero == the stored blind factor-graph cells
        views = [("fully_meshed", g)]
        if (family_masks.session_applies("mobile_assisted", sdir)
                and family_masks.session_motion(sdir) != "dynamic"):
            views.append(("mobile_assisted",
                          g0.apply_mask("mobile_assisted")))
        for mask, gv in views:
            for v in factor_graph.KERNELS:
                est, _ = factor_graph.factor_graph_offset(
                    gv, variant=v, anchor_order=estimation_order(gv),
                    sigma_m=sigma_m, offset="zero")
                got = evaluation.evaluate_estimate(est, ref)["rmse_m"]
                stored = _stored(bench, (campaign, session, mask,
                                         "factor_graph", v), "rmse_m")
                assert abs(stored - got) < MATCH_TOL, \
                    f"zero offset differs from blind cell {(s, mask, v)}"
                n_repro += 1

        for mult in OFFSET_MULTS:
            for mode, width in OFFSET_ARMS:
                est, info = factor_graph.factor_graph_offset(
                    g, variant=OFFSET_VARIANT, anchor_order=order,
                    sigma_m=mult * sigma_m, offset=mode,
                    prior_width=(width if mode == "device"
                                 else factor_graph.OFFSET_PRIOR_WIDTH_M))
                res = evaluation.evaluate_estimate(est, ref)
                canonical = (mode == "common" or
                             width == factor_graph.OFFSET_PRIOR_WIDTH_M)
                if mult == 1.0 and mode in OFFSET_ROSTER and canonical:
                    stored = _stored(bench, (campaign, session, "fully_meshed",
                                             OFFSET_ROSTER[mode],
                                             OFFSET_VARIANT), "rmse_m")
                    assert abs(stored - res["rmse_m"]) < MATCH_TOL, \
                        f"offset control differs from roster cell {(s, mode)}"
                rows.append({
                    "campaign": campaign, "session": session,
                    "mask": "fully_meshed", "variant": OFFSET_VARIANT,
                    "offset": mode, "prior_width_m": width,
                    "sigma_mult": mult, "sigma_m": mult * sigma_m,
                    "n_anchors": len(order),
                    "n_pairs": info["n_pairs_available"],
                    "rmse_m": res["rmse_m"],
                    "max_error_m": res["max_error_m"],
                    "offset_common_m": info["offset_common_m"],
                    **_device_stats(info),
                    "n_iter": info["n_iter"],
                    "converged": bool(info["converged"]),
                    "cost": info["cost"], "run_at": run_at})
                wtag = f"w {width:.2f}" if mode == "device" else ""
                print(f"{s}  x{mult:<4g} {mode:<6} {wtag:<6} "
                      f"{res['rmse_m']:.3f} m  "
                      f"b {info['offset_common_m']:+.3f} m")

        for mode, width in OFFSET_ARMS:
            vb = crlb.offset_view_bound(g, ref, sigma_m, offset=mode,
                                        prior_width=width)
            key = (campaign, session, "fully_meshed")
            if mode == "zero" and key in blind.index:
                want = _stored(blind, key, "crlb_rmse_m")
                assert np.isclose(vb["crlb_rmse_m"], want, rtol=1e-9), \
                    f"zero-offset bound differs from crlb.parquet {key}"
            brows.append({
                "campaign": campaign, "session": session,
                "mask": "fully_meshed", "offset": mode,
                "prior_width_m": width, "sigma_m": sigma_m,
                "n_anchors": vb["n_anchors"], "n_pairs": vb["n_pairs"],
                "fim_rank": vb["fim_rank"], "fim_nullity": vb["fim_nullity"],
                "crlb_rmse_m": vb["crlb_rmse_m"],
                "crlb_weak_mode_m": vb["crlb_weak_mode_m"],
                "run_at": run_at})
    print(f"\nzero-offset reproduction: {n_repro} blind cells within "
          f"{MATCH_TOL:g} m")
    return pd.DataFrame(rows), pd.DataFrame(brows)


def sigma_terms(sessions: list[str], run_at: str) -> pd.DataFrame:
    bench = benchmark_io.load()
    bench = bench[bench["mask"] == "fully_meshed"]
    rows = []
    for s in sessions:
        campaign, session = s.split("/")
        g = load_session(DATA / campaign / session).apply_mask("fully_meshed")
        order = default_anchor_order(g)
        M = pair_range_matrix(g, order)
        grp = g.symmetrized().groupby(["node_a", "node_b"])["range_m"]
        keep = grp.count() >= kernels.MIN_PAIR_EPOCHS
        std = grp.std()[keep]
        mad = grp.apply(lambda r: 1.4826 * np.median(np.abs(r - np.median(r))))[keep]
        scatter = kernels.scatter_scale(g)
        mds_mad = kernels.init_residual_scale(M)
        sigma = kernels.session_sigma(g, order)
        # every stored cell of the session carries this sigma. Cells written
        # in another environment can differ from it in the last digits.
        stored = bench[(bench.campaign == campaign) & (bench.session == session)
                       ].sigma_m.to_numpy(float)
        assert len(stored) and np.all(np.abs(stored - sigma) < 1e-12), \
            f"sigma terms do not reproduce the benchmark sigma of {s}"
        D, _ = mds.complete_distances(M)
        X0, _ = mds.classical_mds(D, dim=2)
        iu, ju = np.triu_indices(len(order), k=1)
        obs = np.isfinite(M[iu, ju])
        res = np.abs(np.linalg.norm(X0[iu[obs]] - X0[ju[obs]], axis=1)
                     - M[iu, ju][obs])
        rows.append({
            "campaign": campaign, "session": session,
            "n_anchors": len(order), "n_pairs": int(keep.sum()),
            "scatter_std_m": scatter, "mds_mad_m": mds_mad, "sigma_m": sigma,
            "sigma_term": "scatter" if scatter >= mds_mad else "mds_mad",
            "robust_scatter_m": float(mad.median()),
            "pair_std_p90_m": float(std.quantile(0.9)),
            "mds_resid_median_abs_m": float(np.median(res)),
            "mds_resid_p90_abs_m": float(np.quantile(res, 0.9)),
            "run_at": run_at})
        print(f"{s}  scatter {scatter:.4f}  mds-MAD {mds_mad:.4f}  "
              f"sigma {sigma:.4f} ({rows[-1]['sigma_term']})  "
              f"robust scatter {rows[-1]['robust_scatter_m']:.4f} m")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=("oracle", "sweep", "offset", "sigma"),
                    default=None,
                    help="run one control (default: all three)")
    ap.add_argument("--sessions", default=None,
                    help="comma-separated campaign/session globs (default: "
                         "the reference sessions, all sessions for offset)")
    ap.add_argument("--mask", default=DEFAULT_MASKS,
                    help="oracle start: comma-separated mask list (aliases "
                         "allowed)")
    ap.add_argument("--variants", default=",".join(factor_graph.KERNELS),
                    help="oracle start: comma-separated kernels")
    args = ap.parse_args()

    picked = (select_sessions([p for p in args.sessions.split(",") if p])
              if args.sessions else None)
    sessions = picked or list(GATE_SESSIONS)
    run_at = datetime.now().isoformat(timespec="seconds")
    if args.only in (None, "oracle"):
        rows = oracle_start(sessions, family_masks.expand_masks(args.mask),
                            [v for v in args.variants.split(",") if v],
                            run_at)
        benchmark_io.upsert(rows, "oracle_start")
        print(f"\nwrote {len(rows)} rows to results/oracle_start.parquet "
              "(+ CSV)\n")
    if args.only in (None, "sweep"):
        rows = sigma_sweep(sessions, run_at)
        benchmark_io.upsert(rows, "sigma_sweep")
        print(f"\nwrote {len(rows)} rows to results/sigma_sweep.parquet "
              "(+ CSV)")
    if args.only in (None, "offset"):
        rows, brows = offset_control(picked or select_sessions(["all"]),
                                     run_at)
        benchmark_io.upsert(rows, "offset_control")
        benchmark_io.upsert(brows, "offset_crlb")
        print(f"wrote {len(rows)} rows to results/offset_control.parquet "
              f"and {len(brows)} rows to results/offset_crlb.parquet (+ CSV)")
    if args.only in (None, "sigma"):
        rows = sigma_terms(picked or select_sessions(["all"]), run_at)
        benchmark_io.upsert(rows, "sigma_terms")
        print(f"wrote {len(rows)} rows to results/sigma_terms.parquet (+ CSV)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
