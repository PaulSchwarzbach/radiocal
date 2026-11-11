"""Benchmark driver: session x mask x method x variant -> long tables.

Supersedes the per-method run_<method>.py runners (kept as provenance of the
frozen gate-session artifacts; scale sweeps and convergence traces stay theirs).
Results go to results/benchmark.parquet (+ CSV mirror),
results/benchmark_estimates.parquet and - for the slam_bipartite family -
results/benchmark_trajectories.parquet via keyed upserts, so reruns and
resumes are safe (default: cells whose key exists are skipped; --force
reruns them).

Masks / families (see family_masks.py): --mask takes a comma-separated list
of canonical names (fully_meshed, multi_hop, mobile_assisted,
slam_bipartite), parameterized multihop names (multi_hop_k2, ...,
multi_hop_random_k3_s0 for the random-thinning ablation) or the aliases
multi_hop_sweep and multi_hop_random_sweep. Each family runs only on the
sessions and methods it is defined for, with a motion-aware roster:
mobile_assisted on a DYNAMIC session runs fg_slam (pose chain consuming the
view's anchor-anchor factors) and carries the trajectory metrics. The full
parameterized mask name is the `mask` key column. Cells whose construction
leaves too few anchors for the alignment metric record status "undefined"
(a documented structural outcome, distinct from status "error" crashes).

Sigma doctrine: the shared session noise scale sigma is computed ONCE per
session from its fully_meshed view (kernels.session_sigma) and passed to every
mask cell of that session - sigma is a hardware/environment property, not a
topology property, so holding it fixed keeps the mask family a pure
connectivity/information axis. Identical to the historical per-(session,
mask) value on fully_meshed itself (the regression gate pins this). The masked
view's own sigma is recorded as extras_json["sigma_masked_m"] where it is
computable.

Regression gate (must pass before any new numbers):

    python run_benchmark.py --check-regression

runs the full variant roster on set_b/s03 + set_a/s21
(fully_meshed) in memory, compares 40 cells against
results/regression_frozen.csv (frozen at three decimals), reruns the
synthetic baseline of run_synthetic.py in memory against
results/synthetic_baseline/frozen.csv and the gated subset of the
frame-choice sweep of run_frame_choice.py (set_b/s03: every defining pair
of closed_form faithful and factor_graph cauchy, every seed triple of
fg_anchored that is not near-collinear) against
results/frame_choice/frozen.csv, and writes all 52 computed benchmark
cells, the 12 offset-model cells included, into the benchmark tables only if
all three pass. The synthetic and frame-choice parts write nothing.

Family sweeps:

    python run_benchmark.py --sessions "set_a/s21,set_b/s03" --mask multi_hop_sweep
    python run_benchmark.py --sessions "set_a/s21,set_b/s03" --mask prior_sweep
    python run_benchmark.py --sessions "set_a/*" --mask mobile_assisted
    python run_benchmark.py --sessions "set_b/*" --mask slam_bipartite

The prior sweep (fully_meshed_p<K> / multi_hop_p<K>, K = 0..3 known
anchors) runs the prior-anchored factor graph only; its cells additionally
record the absolute (non-aligned) error columns abs_rmse_m /
abs_max_error_m / abs_reflection, defined from two known anchors on
(evaluation.evaluate_absolute), and every row records the view's surviving
anchor-prior count as n_priors.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from radiocal import load_session

import benchmark_io
import evaluation
import family_masks
import kernels
import mds
import regression
import reporting
from common import (default_anchor_order, estimation_order, graph_components,
                    known_anchor_priors, pair_counts, pair_range_matrix)
from registry import KERNEL_OF, METHOD_ORDER, REGISTRY, check_sigma

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
CAMPAIGNS = ("set_a", "set_b")
GATE_SESSIONS = ("set_b/s03", "set_a/s21")

# every summary column a run_cell row must fill even when N/A for the family
_NA_METRICS = {"n_components": np.nan,
               "mobile_error_m": np.nan, "mobile_truth_quality": "",
               "ate_rmse_m": np.nan, "rpe_rmse_m": np.nan,
               "n_truth_matched": np.nan,
               "abs_rmse_m": np.nan, "abs_max_error_m": np.nan,
               "abs_reflection": pd.NA}


def discover_sessions() -> list[str]:
    """All session dirs as 'campaign/session', campaign order then name."""
    out = []
    for c in CAMPAIGNS:
        for d in sorted((DATA / c).iterdir()):
            if d.is_dir() and d.name != "masks" and (d / "session.yaml").exists():
                out.append(f"{c}/{d.name}")
    return out


def select_sessions(patterns: list[str]) -> list[str]:
    all_sessions = discover_sessions()
    if not patterns or patterns == ["all"]:
        return all_sessions
    keep = [s for s in all_sessions
            if any(fnmatch.fnmatch(s, p) for p in patterns)]
    missing = [p for p in patterns
               if not any(fnmatch.fnmatch(s, p) for s in all_sessions)]
    if missing:
        raise SystemExit(f"no session matches selector(s): {missing}")
    return keep


def _planned_cells(campaign: str, session: str, mask: str, family: str,
                   motion: str, methods: list[str],
                   variants: list[str] | None, seed: int,
                   skip_keys: set[tuple]) -> list[tuple]:
    """(spec, variant) cells of one (session, mask), roster- and skip-filtered."""
    planned = []
    for m in methods:
        if m not in family_masks.methods_for(family, motion, mask):
            continue
        spec = REGISTRY[m]
        for v in spec.variants:
            if variants and v not in variants:
                continue
            key = (campaign, session, mask, m, v,
                   seed if spec.seeded else benchmark_io.DETERMINISTIC_SEED)
            if key not in skip_keys:
                planned.append((spec, v))
    return planned


def _hook_family(family: str, motion: str) -> str:
    """Evaluation route of a (family, session motion) cell: mobile_assisted
    on a dynamic session carries the trajectory metrics (its estimator is
    the pose-chain factor graph), not the quasi-static marker metric."""
    if family == "slam_bipartite" or (family == "mobile_assisted"
                                      and motion == "dynamic"):
        return "trajectory"
    return family


def _family_hook(hook: str, est: pd.DataFrame, info: dict, result: dict,
                 ctx: dict) -> tuple[dict, pd.DataFrame | None,
                                     pd.DataFrame | None]:
    """Family-specific metrics -> (row updates, extra estimate rows,
    trajectory rows). Imports lazily so the driver runs before the family
    evaluation modules exist."""
    if hook == "mobile_assisted":
        import evaluation_mobile
        metrics, mobile_rows = evaluation_mobile.evaluate_mobile(est, result,
                                                                 ctx)
        return metrics, mobile_rows, None
    if hook == "trajectory":
        import evaluation_slam
        traj, metrics = evaluation_slam.evaluate_trajectory(
            info["trajectory"], result, ctx)
        return metrics, None, traj
    return {}, None, None


def run_cell(g, ref, spec, variant: str, *, sigma_m: float, seed: int,
             session_meta: dict, family_ctx: dict | None
             ) -> tuple[dict, pd.DataFrame | None, pd.DataFrame | None]:
    """One benchmark cell -> (summary row, per-node estimates or None,
    trajectory rows or None)."""
    campaign, session, mask = (session_meta["campaign"],
                               session_meta["session"], session_meta["mask"])
    hook = session_meta["hook"]
    kernel, mechanism = KERNEL_OF[variant]
    row = {"campaign": campaign, "session": session, "mask": mask,
           "method": spec.name, "variant": variant,
           "seed": seed if spec.seeded else benchmark_io.DETERMINISTIC_SEED,
           "kernel": kernel, "mechanism": mechanism,
           "sigma_m": sigma_m, "sigma_mult": 1.0,
           "n_anchors": session_meta["n_anchors"],
           "n_pairs_available": session_meta["n_pairs_available"],
           "n_pairs_total": session_meta["n_pairs_total"],
           "n_epochs": session_meta["n_epochs"],
           "n_priors": session_meta["n_priors"],
           "run_at": datetime.now().isoformat(timespec="seconds"),
           **_NA_METRICS}
    row["n_components"] = session_meta["n_components"]
    anchors, traj = None, None
    t0 = time.perf_counter()
    try:
        est, info = spec.run(g, variant, sigma_m=sigma_m, sigma_scale=1.0,
                             seed=seed)
        check_sigma(info, sigma_m, 1.0)
        result = evaluation.evaluate_estimate(est, ref)
        metrics, mobile_rows, traj = _family_hook(hook, est, info, result,
                                                  family_ctx or {})
        row.update(metrics)
        extras = spec.extras(info)
        extras["sigma_masked_m"] = session_meta["sigma_masked_m"]
        row.update(rmse_m=result["rmse_m"],
                   max_error_m=result["max_error_m"],
                   reflection=result["is_reflection"],
                   n_aligned=result["n_aligned"],
                   status="ok", error="",
                   extras_json=json.dumps(extras, sort_keys=True))
        if len(info.get("prior_ids", [])) >= 2:
            absolute = evaluation.evaluate_absolute(est, ref,
                                                    info["prior_ids"])
            if absolute is not None:
                row.update(absolute)
        anchors = reporting.per_anchor_table(est, ref, result,
                                             campaign, session, variant)
        if mobile_rows is not None and len(mobile_rows):
            mobile_rows = mobile_rows.assign(campaign=campaign,
                                             session=session, variant=variant)
            anchors = pd.concat([anchors, mobile_rows], ignore_index=True)
        anchors["mask"] = mask
        anchors["method"] = spec.name
        anchors["seed"] = row["seed"]
        anchors = anchors[benchmark_io.ESTIMATE_COLUMNS]
        if traj is not None and len(traj):
            traj = traj.assign(campaign=campaign, session=session, mask=mask,
                               method=spec.name, variant=variant,
                               seed=row["seed"])
            traj = traj[benchmark_io.TRAJECTORY_COLUMNS]
    except mds.DisconnectedGraphError as exc:
        row.update(rmse_m=float("nan"), max_error_m=float("nan"),
                   reflection=pd.NA, status="disconnected",
                   error=f"{type(exc).__name__}: {exc}", extras_json="{}")
        anchors, traj = None, None
    except evaluation.UndefinedConstructionError as exc:
        row.update(rmse_m=float("nan"), max_error_m=float("nan"),
                   reflection=pd.NA, status="undefined",
                   error=f"{type(exc).__name__}: {exc}", extras_json="{}")
        anchors, traj = None, None
    except Exception as exc:
        row.update(rmse_m=float("nan"), max_error_m=float("nan"),
                   reflection=pd.NA, status="error",
                   error=f"{type(exc).__name__}: {exc}", extras_json="{}")
        traceback.print_exc()
        anchors, traj = None, None
    row["runtime_s"] = time.perf_counter() - t0
    return row, anchors, traj


def _family_context(hook: str, sdir: Path) -> dict | None:
    """Session-level evaluation context of a hook route (lazy imports)."""
    if hook == "mobile_assisted":
        import evaluation_mobile
        return evaluation_mobile.session_context(sdir)
    if hook == "trajectory":
        import evaluation_slam
        return evaluation_slam.session_context(sdir)
    return None


def run_sessions(sessions: list[str], methods: list[str],
                 variants: list[str] | None, masks: list[str], seed: int,
                 skip_keys: set[tuple], write: bool = True,
                 with_estimates: bool = True
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The main loop. Upserts after every cell when write=True."""
    all_rows, all_anchors = [], []
    for s in sessions:
        campaign, session = s.split("/")
        sdir = DATA / campaign / session
        g0 = None
        sigma_m, ref = None, None
        motion = family_masks.session_motion(sdir)
        for mask in masks:
            family = family_masks.family_of(mask)
            if not family_masks.session_applies(family, sdir):
                continue
            planned = _planned_cells(campaign, session, mask, family, motion,
                                     methods, variants, seed, skip_keys)
            if not planned:
                continue
            if g0 is None:
                g0 = load_session(sdir)
                g_full = g0.apply_mask("fully_meshed")
                sigma_m = kernels.session_sigma(
                    g_full, default_anchor_order(g_full))
                ref = evaluation.surveyed_reference(sdir)
            print(f"== {s} (mask={mask}): {len(planned)} cells")
            g = g0.apply_mask(family_masks.resolve_mask(
                mask, mask_dirs=g0.meta.get("_mask_dirs")))
            order = estimation_order(g)
            M = pair_range_matrix(g, order)
            available, total = pair_counts(M)
            try:
                sigma_masked = float(kernels.session_sigma(g, order))
            except Exception:
                sigma_masked = None  # e.g. disconnected thinned view
            hook = _hook_family(family, motion)
            family_ctx = _family_context(hook, sdir)
            session_meta = {"campaign": campaign, "session": session,
                            "mask": mask, "family": family, "hook": hook,
                            "n_anchors": len(default_anchor_order(g)),
                            "n_pairs_available": available,
                            "n_pairs_total": total,
                            "n_epochs": int(g.edges["seq"].nunique()),
                            "n_components": graph_components(M),
                            "n_priors": len(known_anchor_priors(g, order)[0]),
                            "sigma_masked_m": sigma_masked}
            print(f"   {session_meta['n_anchors']} anchors, "
                  f"{available}/{total} pairs, "
                  f"{session_meta['n_epochs']} epochs, "
                  f"{session_meta['n_components']} component(s), "
                  f"sigma {sigma_m:.3f} m")
            for spec, v in planned:
                row, anchors, traj = run_cell(g, ref, spec, v,
                                              sigma_m=sigma_m, seed=seed,
                                              session_meta=session_meta,
                                              family_ctx=family_ctx)
                tag = (f"{row['rmse_m']:.3f} m" if row["status"] == "ok"
                       else row["status"].upper()
                       + (f" ({row['error']})" if row["error"] else ""))
                print(f"   {spec.name}:{v:<16} {tag}  "
                      f"({row['runtime_s']:.1f} s)")
                all_rows.append(row)
                if anchors is not None:
                    all_anchors.append(anchors)
                if write:
                    benchmark_io.upsert(pd.DataFrame([row]))
                    if with_estimates and anchors is not None:
                        benchmark_io.upsert(anchors, "benchmark_estimates")
                    if traj is not None:
                        benchmark_io.upsert(traj, "benchmark_trajectories")
    rows_df = pd.DataFrame(all_rows, columns=benchmark_io.SUMMARY_COLUMNS)
    anchors_df = (pd.concat(all_anchors, ignore_index=True) if all_anchors
                  else pd.DataFrame(columns=benchmark_io.ESTIMATE_COLUMNS))
    return rows_df, anchors_df


def check_regression(with_estimates: bool) -> int:
    """Gate mode: full roster on the two gate sessions, in memory;
    upsert only on a full pass."""
    rows, anchors = run_sessions(list(GATE_SESSIONS), list(METHOD_ORDER),
                                 None, ["fully_meshed"], seed=0,
                                 skip_keys=set(), write=False)
    rep = regression.compare(rows)
    print()
    print(regression.report(rep))
    import run_synthetic  # the synthetic baseline, rerun in memory
    syn = run_synthetic.check()
    print()
    print(run_synthetic.report(syn))
    import run_frame_choice  # the frame-choice sweep, gated subset in memory
    fc = run_frame_choice.check()
    print()
    print(run_frame_choice.report(fc))
    if not (rep["pass"].all() and len(syn) and syn["pass"].all()
            and len(fc) and fc["pass"].all()):
        print("\nGATE FAILED - tables not written.")
        return 1
    benchmark_io.upsert(rows)
    if with_estimates and len(anchors):
        benchmark_io.upsert(anchors, "benchmark_estimates")
    print("\nGATE PASSED - results upserted into benchmark tables.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", default="all",
                    help="comma-separated campaign/session globs "
                         "(e.g. 'set_a/*,set_b/s03'); default all")
    ap.add_argument("--methods", default=",".join(METHOD_ORDER))
    ap.add_argument("--variants", default="",
                    help="optional comma-separated variant filter")
    ap.add_argument("--mask", default="fully_meshed",
                    help="comma-separated mask list; parameterized "
                         "multi_hop_k<INT> and the alias "
                         "multi_hop_sweep are allowed")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for seeded methods (particle filter)")
    ap.add_argument("--check-regression", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rerun cells whose key already exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-estimates", action="store_true")
    args = ap.parse_args()

    if args.check_regression:
        return check_regression(with_estimates=not args.no_estimates)

    methods = [m for m in METHOD_ORDER if m in args.methods.split(",")]
    unknown = set(args.methods.split(",")) - set(METHOD_ORDER)
    if unknown:
        raise SystemExit(f"unknown method(s): {sorted(unknown)}")
    variants = [v for v in args.variants.split(",") if v] or None
    masks = family_masks.expand_masks(args.mask)
    sessions = select_sessions([p for p in args.sessions.split(",") if p])
    skip_keys = set() if args.force else benchmark_io.existing_keys()

    if args.dry_run:
        n = 0
        for s in sessions:
            campaign, session = s.split("/")
            sdir = DATA / campaign / session
            motion = family_masks.session_motion(sdir)
            for mask in masks:
                family = family_masks.family_of(mask)
                if not family_masks.session_applies(family, sdir):
                    continue
                for m in methods:
                    if m not in family_masks.methods_for(family, motion,
                                                         mask):
                        continue
                    spec = REGISTRY[m]
                    for v in spec.variants:
                        if variants and v not in variants:
                            continue
                        key = (campaign, session, mask, m, v,
                               args.seed if spec.seeded
                               else benchmark_io.DETERMINISTIC_SEED)
                        state = "skip" if key in skip_keys else "run"
                        print(f"{state}  {s}  {mask}  {m}:{v}")
                        n += state == "run"
        print(f"{n} cells to run")
        return 0

    run_sessions(sessions, methods, variants, masks, args.seed,
                 skip_keys, write=True,
                 with_estimates=not args.no_estimates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
