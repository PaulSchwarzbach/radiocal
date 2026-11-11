"""Choice of the frame -> results/frame_choice/ (pairs, triples and line as
parquet + CSV mirror, summary.md, and frozen.csv, the reference of the gate).

Ranges fix a layout only up to a rigid motion and a reflection, so every
method fixes a frame. Either a convention puts the first anchor of the
method's order at the origin and the second on the positive x-axis
(common.GAUGE_TRIPLE heads the order of every method, paper Sec. IV-A), or
seed priors tie the estimate to the survey (fg_anchored and the prior sweep
of run_benchmark.py). The benchmark never varies either choice. This sweep
does, on the full mesh of the two sessions of the paper's sweeps, set_b/s03
and set_a/s21, at the shared session sigma, with every other setting as in
the registry adapters.

Part A, the defining pair (table frame_choice/pairs). The anchor order is
[i, j] followed by the release order without i and j, so the third anchor
is the first remaining one of GAUGE_TRIPLE. closed_form (faithful,
disambiguated) and factor_graph (gaussian, huber, cauchy) run over the
unordered pairs, since the pairs (i, j) and (j, i) give mirror images under
the reflection-allowed alignment. ekf (cauchy) runs over the ordered pairs
of both sessions, grid_filter (cauchy, mixture) over those of set_b/s03
only (about 2 s per run there, about 30 s on set_a/s21). Each row records:

- the aligned RMSE (evaluation.evaluate_estimate, the benchmark metric);
- the anchors left unplaced, with the reason: a missing range to one of the
  two defining anchors, a negative radicand (an anchor too close to their
  line for the measured ranges), or not reached (the axis anchor itself
  unplaced);
- the robust cost of the factor graph at its solution, which does not
  depend on the frame, so two pairs that end at one cost reached one
  minimum;
- the wrong-side count. Over the anchors that lie more than 0.5 m off the
  line of the pair (survey and estimate each taken into the pair's own
  frame by a proper rotation), it is the smaller of the counts of anchors
  whose estimated side agrees or disagrees with the surveyed side, so a
  whole mirror image counts as right;
- for the disambiguated closed form, ref_near_line: the side test of some
  anchor used a reference anchor within 0.5 m of the line;
- the geometry of the pair: its surveyed length, its median range residual
  against the surveyed 3-D distance (outlier above 4 m, the threshold of
  the paper's Sec. VI-A), whether it is an edge of the layout's convex
  hull, and how many anchors lie off its line and on its minority side.

Part B, the seed triple (table frame_choice/triples), every triple of
anchors of each session:

    B1  the stored blind cauchy factor-graph estimate (x_est, y_est of
        benchmark_estimates, fully_meshed) mapped onto the surveyed
        positions of the triple by a rigid Umeyama fit, reflection allowed,
        then scored without further alignment over all anchors
        (b1_abs_rmse_m) and over the other anchors (b1_abs_rmse_other_m);
    B2  fg_anchored (cauchy) with the triple as its seeds, the priors of the
        nodes table (1 cm), through the fully_meshed mask with a prior
        subset (the spec family_masks.resolve_mask builds for
        fully_meshed_p3), scored aligned (b2_rmse_m) and absolute
        (b2_abs_rmse_m, evaluation.evaluate_absolute).

Each row records the triangle's smallest height (2-D survey), whether all
three anchors are vertices of the layout's convex hull, and whether one of
its pairs is an outlier pair or missing. A triple whose smallest height lies
below 0.8 m counts as near-collinear. Its reflection is fragile, so it is
reported apart and not gated.

Line (table frame_choice/line): per anchor, its signed offset from the line
of the release pair (5BB6 at the origin, 4A23 on the positive x-axis),
whether it is a hull vertex, and for the release closed form (faithful)
whether it was placed, why not, and its aligned error.

At the release order every engine must reproduce its stored benchmark cell
(benchmark.parquet, fully_meshed) to 1e-9 m, and B2 at the release triple
the stored fully_meshed_p3 cell, aligned and absolute. The run aborts
otherwise.

    python run_frame_choice.py            # run, write the tables + summary
    python run_frame_choice.py --force    # rewrite existing tables + summary
    python run_frame_choice.py --freeze   # write frozen.csv once
    python run_frame_choice.py --check    # rerun the gated subset in memory
    python run_frame_choice.py --summary  # rewrite summary.md from the tables

The gated subset is set_b/s03: closed_form faithful and factor_graph cauchy
over all pairs, and B2 over the triples that are not near-collinear. A row
passes when its status equals the frozen one and its rmse_m (for B2 also
its abs_rmse_m) lies within 5e-4 m of it, the tolerance of the regression
gate. --check writes nothing, and run_benchmark.py --check-regression runs
it after the synthetic baseline. --freeze refuses to overwrite frozen.csv,
as in run_synthetic.py.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from radiocal import frames, load_session
from radiocal.masks import load_mask

import benchmark_io
import closed_form
import common
import ekf
import evaluation
import factor_graph
import grid_filter
import kernels
import regression
from common import GAUGE_TRIPLE, default_anchor_order, pair_range_matrix
from registry import NO_TRACE, check_sigma

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
SESSIONS = ("set_b/s03", "set_a/s21")
GATE_SESSION = "set_b/s03"
PAIRS = "frame_choice/pairs"
TRIPLES = "frame_choice/triples"
LINE = "frame_choice/line"
OUT_DIR = benchmark_io.RESULTS_DIR / "frame_choice"
FROZEN_CSV = OUT_DIR / "frozen.csv"
SUMMARY_MD = OUT_DIR / "summary.md"

TOL = regression.TOL      # 5e-4 m, the tolerance of the regression gate
REPRO_TOL = 1e-9          # release order against the stored benchmark cell
SIDE_M = 0.5              # anchors closer to a line count as on it
NEAR_LINE_M = 0.8         # "near the line" in the line summary
DEGENERATE_M = 0.8        # smallest height of a near-collinear triple
OUTLIER_M = 4.0           # outlier pair, paper Sec. VI-A
COST_RTOL = 1e-6          # two factor-graph runs at one minimum

# (method, variant, ordered pairs, sessions)
PAIR_RUNS = (
    ("closed_form", "faithful", False, SESSIONS),
    ("closed_form", "disambiguated", False, SESSIONS),
    ("factor_graph", "gaussian", False, SESSIONS),
    ("factor_graph", "huber", False, SESSIONS),
    ("factor_graph", "cauchy", False, SESSIONS),
    ("ekf", "cauchy", True, SESSIONS),
    ("grid_filter", "cauchy", True, ("set_b/s03",)),
    ("grid_filter", "mixture", True, ("set_b/s03",)),
)
GATED_PAIR_RUNS = (("closed_form", "faithful"), ("factor_graph", "cauchy"))
FROZEN_COLUMNS = ["part", "campaign", "session", "method", "variant",
                  "nodes", "status", "rmse_m", "abs_rmse_m"]

MARK_B = "<!-- BEGIN GENERATED (run_frame_choice.py) -->"
MARK_E = "<!-- END GENERATED (run_frame_choice.py) -->"
HEADER = """\
# frame_choice - the choice of the defining pair and of the seed triple

The benchmark fixes the frame of every method by one convention: the anchor
order starts with the triple 5BB6, 4A23, 8B05 (`common.GAUGE_TRIPLE`), so
5BB6 sits at the origin and 4A23 on the positive x-axis, and the prior sweep
seeds nested prefixes of the same triple. `run_frame_choice.py` varies both
choices on the full mesh of set_b/s03 and set_a/s21 (see its docstring for
the definitions). Every RMSE below is in meters. "Aligned" is the benchmark
metric, "absolute" the error in the survey frame without alignment.
"""


# ------------------------------------------------------------ geometry

def _frame_y(xy: dict, i: str, j: str) -> dict:
    """Signed offsets from the line i -> j (left of the direction positive),
    i.e. the y coordinates in the frame with i at the origin and j on the
    positive x-axis, reached by a proper rotation."""
    if i not in xy or j not in xy:
        return {}
    pi, pj = np.asarray(xy[i], float), np.asarray(xy[j], float)
    u = pj - pi
    n = float(np.linalg.norm(u))
    if not np.isfinite(n) or n < 1e-9:
        return {}
    u = u / n
    return {k: float(u[0] * (p[1] - pi[1]) - u[1] * (p[0] - pi[0]))
            for k, p in ((k, np.asarray(v, float)) for k, v in xy.items())}


def _wrong_side(est_xy: dict, ref_xy: dict, i: str, j: str) -> float:
    """Anchors off the line of (i, j) on the wrong side, up to a whole
    mirror image (NaN when i or j has no estimate)."""
    ye = _frame_y(est_xy, i, j)
    yt = _frame_y(ref_xy, i, j)
    if not ye or not yt:
        return np.nan
    agree = disagree = 0
    for k, y in yt.items():
        if k in (i, j) or k not in ye or abs(y) <= SIDE_M:
            continue
        if np.sign(ye[k]) == np.sign(y):
            agree += 1
        else:
            disagree += 1
    return float(min(agree, disagree))


def _hull(ref_xy: dict) -> tuple[set, set]:
    """(hull vertex ids, hull edges as frozensets) of the 2-D survey."""
    ids = list(ref_xy)
    pts = np.array([ref_xy[k] for k in ids], float)
    h = ConvexHull(pts)
    verts = [ids[v] for v in h.vertices]
    edges = {frozenset((verts[k], verts[(k + 1) % len(verts)]))
             for k in range(len(verts))}
    return set(verts), edges


def _min_height(p: np.ndarray) -> float:
    """Smallest height of the triangle of three 2-D points."""
    a, b, c = p
    area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    longest = max(np.linalg.norm(b - a), np.linalg.norm(c - b),
                  np.linalg.norm(a - c))
    return float(area2 / longest) if longest > 0 else 0.0


# ------------------------------------------------------------- context

def session_context(sess: str) -> dict:
    """Everything one session's sweep needs, loaded once."""
    campaign, session = sess.split("/")
    sdir = DATA / campaign / session
    g0 = load_session(sdir)
    g = g0.apply_mask("fully_meshed")
    order = default_anchor_order(g)
    sigma_m = float(kernels.session_sigma(g, order))
    ref = evaluation.surveyed_reference(sdir)
    ref = ref[ref["id"].isin(order)].reset_index(drop=True)
    ref_xy = {r.id: (float(r.x), float(r.y)) for r in ref.itertuples()}
    ref_xyz = {r.id: np.array([r.x, r.y, r.z], float)
               for r in ref.itertuples()}
    M = pair_range_matrix(g, order)
    pos = {a: k for k, a in enumerate(order)}
    resid = {}
    for a, b in itertools.combinations(order, 2):
        r = M[pos[a], pos[b]]
        resid[frozenset((a, b))] = (
            float(r - np.linalg.norm(ref_xyz[a] - ref_xyz[b]))
            if np.isfinite(r) else np.nan)
    hull_v, hull_e = _hull(ref_xy)
    return {"campaign": campaign, "session": session, "sdir": sdir,
            "g0": g0, "g": g, "order": order, "sigma_m": sigma_m,
            "ref": ref, "ref_xy": ref_xy, "resid": resid,
            "hull_v": hull_v, "hull_e": hull_e}


def _pair_order(release: list[str], i: str, j: str) -> list[str]:
    return [i, j] + [a for a in release if a not in (i, j)]


def _pairs(release: list[str], ordered: bool) -> list[tuple[str, str]]:
    return (list(itertools.permutations(release, 2)) if ordered
            else list(itertools.combinations(release, 2)))


# --------------------------------------------------------------- part A

def _run_engine(method: str, variant: str, g, order: list[str],
                sigma_m: float):
    """The registry adapter's call with an explicit anchor order."""
    if method == "closed_form":
        return closed_form.closed_form_anchors(
            g, anchor_order=order, disambiguate=(variant == "disambiguated"))
    if method == "factor_graph":
        return factor_graph.factor_graph_anchors(
            g, variant=variant, anchor_order=order, sigma_m=sigma_m)
    if method == "ekf":
        return ekf.ekf_anchors(g, variant=variant, anchor_order=order,
                               sigma_m=sigma_m, trace_every=NO_TRACE)
    if method == "grid_filter":
        return grid_filter.grid_filter_anchors(
            g, variant=variant, anchor_order=order, sigma=sigma_m)
    raise ValueError(f"unknown method {method!r}")


def _cf_diagnostics(M: np.ndarray, order: list[str], disambiguate: bool,
                    ref_y: dict) -> tuple[dict, bool]:
    """(unplaced id -> reason, whether a side test used a reference anchor
    within SIDE_M of the line), from the closed form's own construction."""
    coords, placed = closed_form.solve_closed_form(M, disambiguate=disambiguate)
    reasons = {}
    for a in range(1, len(order)):
        if placed[a]:
            continue
        if a == 1:
            reasons[order[a]] = "missing range"
        elif not placed[1]:
            reasons[order[a]] = "not reached"
        elif not (np.isfinite(M[a, 0]) and np.isfinite(M[a, 1])):
            reasons[order[a]] = "missing range"
        else:
            reasons[order[a]] = "radicand"
    near = False
    if disambiguate:
        for a in range(3, len(order)):
            if not placed[a]:
                continue
            for r in range(2, a):
                if (placed[r] and abs(coords[r, 1]) > 1e-9
                        and np.isfinite(M[a, r])):
                    near |= abs(ref_y.get(order[r], np.inf)) < SIDE_M
                    break
    return reasons, near


def pair_row(ctx: dict, method: str, variant: str, i: str, j: str,
             ordered: bool) -> tuple[dict, pd.DataFrame | None]:
    """One run of Part A -> (row, per-anchor residuals or None)."""
    release = ctx["order"]
    order = _pair_order(release, i, j)
    ref_y = _frame_y(ctx["ref_xy"], i, j)
    off = [a for a in release if a not in (i, j) and abs(ref_y[a]) > SIDE_M]
    pos_n = sum(ref_y[a] > 0 for a in off)
    row = {"campaign": ctx["campaign"], "session": ctx["session"],
           "method": method, "variant": variant, "origin": i, "axis": j,
           "third": order[2], "ordered": ordered,
           "is_release": order == release,
           "n_anchors": len(release),
           "pair_length_m": float(np.hypot(
               *(np.subtract(ctx["ref_xy"][j], ctx["ref_xy"][i])))),
           "pair_residual_m": ctx["resid"][frozenset((i, j))],
           "pair_hull_edge": frozenset((i, j)) in ctx["hull_e"],
           "n_off_line": len(off),
           "n_minority_side": int(min(pos_n, len(off) - pos_n)),
           "sigma_m": ctx["sigma_m"],
           "cost": np.nan, "n_iter": np.nan, "converged": pd.NA,
           "wrong_side": np.nan, "ref_near_line": pd.NA,
           "unplaced": "", "error": ""}
    row["pair_missing"] = not np.isfinite(row["pair_residual_m"])
    row["pair_outlier"] = bool(abs(row["pair_residual_m"]) > OUTLIER_M) \
        if not row["pair_missing"] else False
    resid = None
    t0 = time.perf_counter()
    try:
        est, info = _run_engine(method, variant, ctx["g"], order,
                                ctx["sigma_m"])
        check_sigma(info, ctx["sigma_m"], 1.0)
        placed_ids = set(est.loc[est[["x", "y"]].notna().all(axis=1), "id"])
        row["n_placed"] = len(placed_ids & set(release))
        reasons = {a: "not placed" for a in release if a not in placed_ids}
        if method == "closed_form":
            M = pair_range_matrix(ctx["g"], order)
            reasons, near = _cf_diagnostics(
                M, order, variant == "disambiguated", ref_y)
            if variant == "disambiguated":
                row["ref_near_line"] = bool(near)
        row["unplaced"] = ";".join(f"{a}:{reasons[a]}" for a in release
                                   if a in reasons)
        if method == "factor_graph":
            row.update(cost=float(info["cost"]), n_iter=int(info["n_iter"]),
                       converged=bool(info["converged"]))
        est_xy = {r.id: (float(r.x), float(r.y)) for r in est.itertuples()
                  if np.isfinite(r.x) and np.isfinite(r.y)}
        row["wrong_side"] = _wrong_side(est_xy, ctx["ref_xy"], i, j)
        result = evaluation.evaluate_estimate(est, ctx["ref"])
        row.update(status="ok", rmse_m=float(result["rmse_m"]),
                   max_error_m=float(result["max_error_m"]),
                   reflection=bool(result["is_reflection"]))
        resid = result["resid"]
    except evaluation.UndefinedConstructionError as exc:
        row.update(status="undefined", error=f"{type(exc).__name__}: {exc}",
                   rmse_m=np.nan, max_error_m=np.nan, reflection=pd.NA)
        row.setdefault("n_placed", np.nan)
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}",
                   rmse_m=np.nan, max_error_m=np.nan, reflection=pd.NA)
        row.setdefault("n_placed", np.nan)
        traceback.print_exc()
    row["runtime_s"] = time.perf_counter() - t0
    return row, resid


def run_pairs(ctx: dict, runs, verbose: bool = False) -> list[dict]:
    rows = []
    for method, variant, ordered in runs:
        t0 = time.perf_counter()
        pairs = _pairs(ctx["order"], ordered)
        for i, j in pairs:
            row, _ = pair_row(ctx, method, variant, i, j, ordered)
            rows.append(row)
        if verbose:
            ok = [r["rmse_m"] for r in rows[-len(pairs):]
                  if r["status"] == "ok"]
            print(f"   {method}:{variant:<14} {len(pairs)} pairs, "
                  f"{len(ok)} ok, rmse {np.min(ok):.3f}-{np.max(ok):.3f} m "
                  f"({time.perf_counter() - t0:.0f} s)")
    return rows


def line_rows(ctx: dict) -> list[dict]:
    """Per anchor: offset from the release line, hull flag, and the release
    closed form (faithful): placed, reason, aligned error."""
    release = ctx["order"]
    i, j = release[0], release[1]
    row, resid = pair_row(ctx, "closed_form", "faithful", i, j, False)
    reasons = dict(p.split(":", 1) for p in row["unplaced"].split(";") if p)
    err = (resid.set_index("id")["error_m"].to_dict()
           if resid is not None else {})
    y = _frame_y(ctx["ref_xy"], i, j)
    return [{"campaign": ctx["campaign"], "session": ctx["session"],
             "id": a, "origin": i, "axis": j, "offset_m": y[a],
             "on_hull": a in ctx["hull_v"], "cf_placed": a not in reasons,
             "cf_reason": reasons.get(a, ""),
             "cf_error_m": float(err.get(a, np.nan))}
            for a in release]


# --------------------------------------------------------------- part B

def _stored_blind(ctx: dict) -> dict:
    est = benchmark_io.load("benchmark_estimates")
    est = est[(est.campaign == ctx["campaign"]) & (est.session == ctx["session"])
              & (est["mask"] == "fully_meshed")
              & (est.method == "factor_graph") & (est.variant == "cauchy")]
    if not len(est):
        raise RuntimeError(f"no stored blind cauchy factor-graph estimate "
                           f"for {ctx['campaign']}/{ctx['session']}")
    return {r.id: (float(r.x_est), float(r.y_est)) for r in est.itertuples()}


def _seeded_view(ctx: dict, triple: tuple[str, ...]):
    spec = copy.deepcopy(load_mask(
        "fully_meshed", search_dirs=ctx["g0"].meta.get("_mask_dirs")))
    spec["name"] = "fully_meshed_seeds_" + "_".join(triple)
    spec["priors"] = {"subset": list(triple)}
    return ctx["g0"].apply_mask(spec)


def triple_row(ctx: dict, triple: tuple[str, ...], blind: dict | None,
               b1: bool = True, b2: bool = True) -> dict:
    """One seed triple of Part B -> row."""
    a, b, c = triple
    p = np.array([ctx["ref_xy"][k] for k in triple], float)
    pairs = [frozenset(q) for q in itertools.combinations(triple, 2)]
    res = [ctx["resid"][q] for q in pairs]
    row = {"campaign": ctx["campaign"], "session": ctx["session"],
           "a": a, "b": b, "c": c,
           "is_release": set(triple) == set(GAUGE_TRIPLE),
           "min_height_m": _min_height(p),
           "all_hull": all(k in ctx["hull_v"] for k in triple),
           "outlier_pair": any(np.isfinite(r) and abs(r) > OUTLIER_M
                               for r in res),
           "missing_pair": any(not np.isfinite(r) for r in res),
           "sigma_m": ctx["sigma_m"],
           "b1_abs_rmse_m": np.nan, "b1_abs_rmse_other_m": np.nan,
           "b1_reflection": pd.NA, "b2_status": "", "b2_error": "",
           "b2_rmse_m": np.nan, "b2_abs_rmse_m": np.nan,
           "b2_abs_reflection": pd.NA, "b2_cost": np.nan,
           "b2_n_iter": np.nan, "b2_prior_max_err_m": np.nan}
    row["near_collinear"] = row["min_height_m"] < DEGENERATE_M
    t0 = time.perf_counter()
    if b1:
        ids = [k for k in ctx["order"] if k in blind]
        E = np.array([blind[k] for k in ids], float)
        R = np.array([ctx["ref_xy"][k] for k in ids], float)
        sel = [ids.index(k) for k in triple]
        tf = frames.umeyama(E[sel], R[sel], with_scale=False,
                            allow_reflection=True)
        err = np.linalg.norm(tf.apply(E) - R, axis=1)
        other = np.array([k not in triple for k in ids])
        row.update(b1_abs_rmse_m=float(np.sqrt((err ** 2).mean())),
                   b1_abs_rmse_other_m=float(np.sqrt((err[other] ** 2).mean())),
                   b1_reflection=bool(np.linalg.det(tf.R) < 0))
    if b2:
        try:
            g = _seeded_view(ctx, triple)
            order = common.estimation_order(g)
            est, info = factor_graph.factor_graph_anchored(
                g, variant="cauchy", anchor_order=order,
                sigma_m=ctx["sigma_m"])
            check_sigma(info, ctx["sigma_m"], 1.0)
            if sorted(info["prior_ids"]) != sorted(triple):
                raise AssertionError(f"seeds {info['prior_ids']} != {triple}")
            result = evaluation.evaluate_estimate(est, ctx["ref"])
            absolute = evaluation.evaluate_absolute(est, ctx["ref"],
                                                    list(info["prior_ids"]))
            row.update(b2_status="ok", b2_rmse_m=float(result["rmse_m"]),
                       b2_abs_rmse_m=float(absolute["abs_rmse_m"]),
                       b2_abs_reflection=bool(absolute["abs_reflection"]),
                       b2_cost=float(info["cost"]),
                       b2_n_iter=int(info["n_iter"]),
                       b2_prior_max_err_m=float(info["prior_max_err_m"]))
        except Exception as exc:
            row.update(b2_status="error",
                       b2_error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
    row["runtime_s"] = time.perf_counter() - t0
    return row


def run_triples(ctx: dict, b1: bool = True, b2: bool = True,
                skip_near_collinear: bool = False,
                verbose: bool = False) -> list[dict]:
    blind = _stored_blind(ctx) if b1 else None
    rows = []
    t0 = time.perf_counter()
    for triple in itertools.combinations(ctx["order"], 3):
        p = np.array([ctx["ref_xy"][k] for k in triple], float)
        if skip_near_collinear and _min_height(p) < DEGENERATE_M:
            continue
        rows.append(triple_row(ctx, triple, blind, b1=b1, b2=b2))
    if verbose:
        print(f"   triples: {len(rows)} ({time.perf_counter() - t0:.0f} s)")
    return rows


# ----------------------------------------------------- release asserts

def assert_release(pairs: pd.DataFrame, triples: pd.DataFrame) -> list[str]:
    """Release order and release triple against the stored benchmark cells;
    returns the lines of the check, raises on a mismatch."""
    bench = benchmark_io.load("benchmark")
    lines, bad = [], []
    rel = pairs[pairs.is_release]
    for r in rel.itertuples():
        s = bench[(bench.campaign == r.campaign) & (bench.session == r.session)
                  & (bench["mask"] == "fully_meshed")
                  & (bench.method == r.method) & (bench.variant == r.variant)
                  & (bench.seed == benchmark_io.DETERMINISTIC_SEED)]
        if len(s) != 1:
            bad.append(f"no stored cell for {r.campaign}/{r.session} "
                       f"{r.method}:{r.variant}")
            continue
        d = abs(float(s.rmse_m.iloc[0]) - r.rmse_m)
        lines.append(f"{r.campaign}/{r.session} {r.method}:{r.variant} "
                     f"stored {float(s.rmse_m.iloc[0]):.9f} got "
                     f"{r.rmse_m:.9f} (d={d:.1e})")
        if not d <= REPRO_TOL:
            bad.append(lines[-1])
    for r in triples[triples.is_release].itertuples():
        s = bench[(bench.campaign == r.campaign) & (bench.session == r.session)
                  & (bench["mask"] == "fully_meshed_p3")
                  & (bench.method == "fg_anchored")
                  & (bench.variant == "cauchy")]
        if len(s) != 1:
            bad.append(f"no stored p3 cell for {r.campaign}/{r.session}")
            continue
        d1 = abs(float(s.rmse_m.iloc[0]) - r.b2_rmse_m)
        d2 = abs(float(s.abs_rmse_m.iloc[0]) - r.b2_abs_rmse_m)
        lines.append(f"{r.campaign}/{r.session} fg_anchored:cauchy p3 "
                     f"aligned d={d1:.1e}, absolute d={d2:.1e}")
        if not (d1 <= REPRO_TOL and d2 <= REPRO_TOL):
            bad.append(lines[-1])
    if bad:
        raise AssertionError("release reproduction failed:\n  "
                             + "\n  ".join(bad))
    return lines


# ----------------------------------------------------------------- main

def run_all(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame,
                                           pd.DataFrame, list[str]]:
    pair_rows, triple_rows, line_tab = [], [], []
    for sess in SESSIONS:
        ctx = session_context(sess)
        if verbose:
            print(f"== {sess}: {len(ctx['order'])} anchors, "
                  f"sigma {ctx['sigma_m']:.3f} m")
        runs = [(m, v, o) for m, v, o, ss in PAIR_RUNS if sess in ss]
        pair_rows += run_pairs(ctx, runs, verbose=verbose)
        triple_rows += run_triples(ctx, verbose=verbose)
        line_tab += line_rows(ctx)
    pairs = pd.DataFrame(pair_rows, columns=benchmark_io.FRAME_PAIR_COLUMNS)
    triples = pd.DataFrame(triple_rows,
                           columns=benchmark_io.FRAME_TRIPLE_COLUMNS)
    line = pd.DataFrame(line_tab, columns=benchmark_io.FRAME_LINE_COLUMNS)
    lines = assert_release(pairs, triples)
    return pairs, triples, line, lines


def _frozen_rows(pairs: pd.DataFrame, triples: pd.DataFrame) -> pd.DataFrame:
    """The gated subset in the frozen format."""
    gp = pairs[(pairs.campaign + "/" + pairs.session == GATE_SESSION)
               & pairs[["method", "variant"]].apply(tuple, axis=1)
               .isin(GATED_PAIR_RUNS)]
    a = pd.DataFrame({"part": "pair", "campaign": gp.campaign,
                      "session": gp.session, "method": gp.method,
                      "variant": gp.variant,
                      "nodes": gp.origin + "-" + gp.axis,
                      "status": gp.status, "rmse_m": gp.rmse_m,
                      "abs_rmse_m": np.nan})
    gt = triples[(triples.campaign + "/" + triples.session == GATE_SESSION)
                 & ~triples.near_collinear.astype(bool)]
    b = pd.DataFrame({"part": "triple", "campaign": gt.campaign,
                      "session": gt.session, "method": "fg_anchored",
                      "variant": "cauchy",
                      "nodes": gt.a + "-" + gt.b + "-" + gt.c,
                      "status": gt.b2_status, "rmse_m": gt.b2_rmse_m,
                      "abs_rmse_m": gt.b2_abs_rmse_m})
    return pd.concat([a, b], ignore_index=True)[FROZEN_COLUMNS]


def check(verbose: bool = False) -> pd.DataFrame:
    """Rerun the gated subset in memory and compare with frozen.csv
    (writes nothing)."""
    key = ["part", "campaign", "session", "method", "variant", "nodes"]
    frozen = (pd.read_csv(FROZEN_CSV, dtype={"nodes": str})
              if FROZEN_CSV.exists()
              else pd.DataFrame(columns=FROZEN_COLUMNS))
    ctx = session_context(GATE_SESSION)
    runs = [(m, v, False) for m, v in GATED_PAIR_RUNS]
    pairs = pd.DataFrame(run_pairs(ctx, runs, verbose=verbose),
                         columns=benchmark_io.FRAME_PAIR_COLUMNS)
    triples = pd.DataFrame(run_triples(ctx, b1=False, b2=True,
                                       skip_near_collinear=True,
                                       verbose=verbose),
                           columns=benchmark_io.FRAME_TRIPLE_COLUMNS)
    got = _frozen_rows(pairs, triples)
    rep = frozen.merge(got, on=key, how="outer", suffixes=("_frozen", ""),
                       indicator=True)
    rep["delta"] = rep["rmse_m"] - rep["rmse_m_frozen"]
    rep["delta_abs"] = rep["abs_rmse_m"] - rep["abs_rmse_m_frozen"]
    same_status = rep["status"] == rep["status_frozen"]
    ok_rmse = ((rep["delta"].abs() < TOL)
               | (rep["rmse_m"].isna() & rep["rmse_m_frozen"].isna()))
    ok_abs = ((rep["delta_abs"].abs() < TOL)
              | (rep["abs_rmse_m"].isna() & rep["abs_rmse_m_frozen"].isna()))
    rep["pass"] = (rep["_merge"] == "both") & same_status & ok_rmse & ok_abs
    return rep


def report(rep: pd.DataFrame) -> str:
    lines = [f"frame choice: {int(rep['pass'].sum())}/{len(rep)} PASS "
             f"(tol {TOL} m on rmse_m and abs_rmse_m against "
             f"{FROZEN_CSV.name}; set_b/s03, closed_form faithful and "
             "factor_graph cauchy over all pairs, fg_anchored over the "
             "triples that are not near-collinear)"]
    if not len(rep):
        lines.append(f"  FAIL  no frozen reference {FROZEN_CSV}")
    for _, r in rep[~rep["pass"]].iterrows():
        cell = f"{r['part']} {r['nodes']} {r['method']}:{r['variant']}"
        if r["_merge"] != "both":
            why = ("missing from the rerun" if r["_merge"] == "left_only"
                   else "not in the frozen reference")
        elif r["status"] != r["status_frozen"]:
            why = f"status {r['status_frozen']} -> {r['status']}"
        else:
            why = (f"frozen {r['rmse_m_frozen']:.6f}/{r['abs_rmse_m_frozen']:.6f}"
                   f" got {r['rmse_m']:.6f}/{r['abs_rmse_m']:.6f}")
        lines.append(f"  FAIL  {cell}: {why}")
    return "\n".join(lines)


def freeze() -> int:
    if FROZEN_CSV.exists():
        print(f"{FROZEN_CSV} exists; delete it by hand to refreeze "
              "(see the module docstring)")
        return 1
    pairs, triples = benchmark_io.load(PAIRS), benchmark_io.load(TRIPLES)
    if not len(pairs) or not len(triples):
        print("no rows to freeze; run run_frame_choice.py first")
        return 1
    rows = _frozen_rows(pairs, triples)
    rows.to_csv(FROZEN_CSV, index=False, float_format="%.6f",
                lineterminator="\n")
    print(f"wrote {FROZEN_CSV} ({len(rows)} rows)")
    return 0


# -------------------------------------------------------------- summary

def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _rank(values: pd.Series, value: float) -> int:
    """1 + the number of values strictly below `value` (1 = best)."""
    return int((values < value - 1e-12).sum()) + 1


_SESSION_RANK = {s: i for i, s in enumerate(sorted(SESSIONS))}
_ENGINE_RANK = {e: i for i, e in enumerate(sorted((m, v) for m, v, _, _
                                                  in PAIR_RUNS))}


def _canonical(df: pd.DataFrame, engines: bool = False) -> pd.DataFrame:
    """Rows in the order of the stored tables, sessions and, with ``engines``,
    methods and variants by name, so that --force (run order) and --summary
    write the same summary.md (code check of 2026-11-10, EXE-02)."""
    keys = {"_s": (df.campaign + "/" + df.session).map(_SESSION_RANK)}
    if engines:
        keys["_e"] = [_ENGINE_RANK.get((m, v), len(_ENGINE_RANK))
                      for m, v in zip(df.method, df.variant)]
    out = df.assign(**keys).sort_values(list(keys), kind="mergesort")
    return out.drop(columns=list(keys))


def _release_key(ln: str) -> tuple[int, int, int]:
    """The checks of the defining pairs first, then those of the seed
    triples, each by session and engine."""
    session, engine = ln.split()[:2]
    m, _, v = engine.partition(":")
    return (0 if (m, v) in _ENGINE_RANK else 1,
            _SESSION_RANK.get(session, len(_SESSION_RANK)),
            _ENGINE_RANK.get((m, v), len(_ENGINE_RANK)))


def summary_block(pairs: pd.DataFrame, triples: pd.DataFrame,
                  line: pd.DataFrame, release_lines: list[str]) -> str:
    pairs, triples, line = (_canonical(pairs, engines=True),
                            _canonical(triples), _canonical(line))
    release_lines = sorted(release_lines, key=_release_key)
    out = []
    rows = []
    for (c, s, m, v), d in pairs.groupby(["campaign", "session", "method",
                                           "variant"], sort=False):
        ok = d[d.status == "ok"]
        full = ok[ok.n_placed == ok.n_anchors]
        rel = d[d.is_release]
        relv = float(rel.rmse_m.iloc[0]) if len(rel) else np.nan
        rel_full = bool(len(rel)) and int(rel.n_placed.iloc[0]) == int(
            rel.n_anchors.iloc[0])
        r = {"session": f"{c}/{s}", "engine": f"{m}:{v}",
             "pairs": len(d), "ok": len(ok), "all placed": len(full),
             "min": full.rmse_m.min(), "median": full.rmse_m.median(),
             "max": full.rmse_m.max(), "release": relv,
             "release rank": (f"{_rank(full.rmse_m, relv)} of {len(full)}"
                              if rel_full and len(full) else
                              "partial" if len(rel) else ""),
             "wrong side > 0": int((ok.wrong_side > 0).sum())}
        if m == "factor_graph" and len(rel):
            cost0 = float(rel.cost.iloc[0])
            dev = (ok.rmse_m - relv).abs()
            far = (dev >= 1e-4) | ((ok.cost - cost0).abs()
                                   > COST_RTOL * abs(cost0))
            r["max |d| to release"] = float(dev[~far].max())
            r["other minimum"] = int(far.sum())
        else:
            r["max |d| to release"] = np.nan
            r["other minimum"] = ""
        rows.append(r)
    out += ["## Part A, the defining pair", "",
            "Aligned RMSE over the pairs that place every anchor; rank 1 is "
            "the best, given only where the release pair places every anchor "
            "too (\"partial\": the release construction leaves an anchor "
            "unplaced and scores fewer anchors). \"Other minimum\" counts "
            "factor-graph pairs whose robust cost differs from the release "
            f"pair's by more than {COST_RTOL:g} relative or whose RMSE moves "
            "by 1e-4 m or more, and \"max |d| to release\" is taken over the "
            "pairs at the release minimum.", "", _md(pd.DataFrame(rows)), ""]
    for (c, s, m, v), d in pairs[pairs.method == "closed_form"].groupby(
            ["campaign", "session", "method", "variant"], sort=False):
        ok = d[d.status == "ok"].sort_values(["rmse_m", "origin", "axis"],
                                             kind="mergesort")
        cols = ["origin", "axis", "rmse_m", "n_placed", "unplaced",
                "wrong_side", "pair_length_m", "pair_residual_m",
                "pair_outlier", "pair_hull_edge", "n_minority_side"]
        out += [f"### {c}/{s} {m}:{v}, best and worst five pairs", "",
                _md(pd.concat([ok.head(5), ok.tail(5)])[cols]), ""]
    trows = []
    for (c, s), d in triples.groupby(["campaign", "session"], sort=False):
        nd = d[~d.near_collinear.astype(bool) & (d.b2_status == "ok")]
        rel = d[d.is_release]
        for col, label in (("b1_abs_rmse_m", "B1 alignment after, absolute"),
                           ("b2_abs_rmse_m", "B2 seeds as priors, absolute"),
                           ("b2_rmse_m", "B2 seeds as priors, aligned")):
            relv = float(rel[col].iloc[0]) if len(rel) else np.nan
            q = nd.sort_values([col, "a", "b", "c"], kind="mergesort")
            nq = max(1, len(q) // 4)
            best, worst = q.head(nq), q.tail(nq)
            trows.append({
                "session": f"{c}/{s}", "measure": label,
                "triples": len(d), "near-collinear": int(
                    d.near_collinear.astype(bool).sum()),
                "min": nd[col].min(), "median": nd[col].median(),
                "max": nd[col].max(), "release": relv,
                "release rank": (f"{_rank(nd[col], relv)} of {len(nd)}"
                                 if np.isfinite(relv) else ""),
                "best q: height": best.min_height_m.median(),
                "worst q: height": worst.min_height_m.median(),
                "best q: all hull": best.all_hull.astype(bool).mean(),
                "worst q: all hull": worst.all_hull.astype(bool).mean(),
                "best q: outlier": best.outlier_pair.astype(bool).mean(),
                "worst q: outlier": worst.outlier_pair.astype(bool).mean()})
    out += ["## Part B, the seed triple", "",
            "Over the triples that are not near-collinear (smallest height "
            f"at least {DEGENERATE_M} m). The quartile columns compare the "
            "best and the worst quarter of the triples by the measure: the "
            "median smallest height (m), the share of triples whose three "
            "anchors are hull vertices, and the share holding an outlier "
            "pair.", "", _md(pd.DataFrame(trows)), ""]
    for (c, s), d in triples.groupby(["campaign", "session"], sort=False):
        nc = d[d.near_collinear.astype(bool)].sort_values(["a", "b", "c"],
                                                          kind="mergesort")
        cols = ["a", "b", "c", "min_height_m", "b1_abs_rmse_m",
                "b2_abs_rmse_m", "b2_rmse_m", "b2_abs_reflection"]
        out += [f"### {c}/{s}, near-collinear triples (reported apart)", "",
                _md(nc[cols]) if len(nc) else "none", ""]
    lrows = []
    for (c, s), d in line.groupby(["campaign", "session"], sort=False):
        oth = d[~d.id.isin([d.origin.iloc[0], d.axis.iloc[0]])]
        lrows.append({"session": f"{c}/{s}",
                      "pair": f"{d.origin.iloc[0]}-{d.axis.iloc[0]}",
                      f"within {NEAR_LINE_M} m": int(
                          (oth.offset_m.abs() < NEAR_LINE_M).sum()),
                      "far side max": float(-oth.offset_m.min())
                      if (oth.offset_m < 0).any() else 0.0,
                      "unplaced": ";".join(
                          f"{r.id}:{r.cf_reason} ({r.offset_m:+.2f} m)"
                          for r in oth[~oth.cf_placed.astype(bool)]
                          .itertuples())})
    out += ["## The line of the release pair", "",
            "Signed offsets of the other anchors from the line of the "
            "release pair (positive on the side of 8B05), and what the "
            "release closed form (faithful) does with them.", "",
            _md(pd.DataFrame(lrows)), ""]
    for (c, s), d in line.groupby(["campaign", "session"], sort=False):
        out += [f"### {c}/{s}", "",
                _md(d.sort_values(["offset_m", "id"], kind="mergesort")[
                    ["id", "offset_m", "on_hull", "cf_placed", "cf_reason",
                     "cf_error_m"]]), ""]
    out += ["## Reproduction of the release cells", ""]
    out += [f"- {ln}" for ln in release_lines]
    return "\n".join(out)


def write_summary(pairs, triples, line, release_lines) -> None:
    block = (f"{MARK_B}\n{summary_block(pairs, triples, line, release_lines)}"
             f"\n{MARK_E}\n")
    if SUMMARY_MD.exists():
        text = SUMMARY_MD.read_text(encoding="utf-8")
        text = re.sub(re.escape(MARK_B) + r".*?" + re.escape(MARK_E) + r"\n?",
                      lambda _: block, text, flags=re.S)
    else:
        text = HEADER + "\n" + block
    SUMMARY_MD.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true",
                      help="rewrite existing tables and summary")
    mode.add_argument("--freeze", action="store_true",
                      help="write frozen.csv once from the stored tables")
    mode.add_argument("--check", action="store_true",
                      help="rerun the gated subset and compare with "
                           "frozen.csv")
    mode.add_argument("--summary", action="store_true",
                      help="rewrite summary.md from the stored tables")
    args = ap.parse_args()
    if args.freeze:
        return freeze()
    if args.summary:
        pairs, triples = benchmark_io.load(PAIRS), benchmark_io.load(TRIPLES)
        line = benchmark_io.load(LINE)
        if not len(pairs) or not len(triples) or not len(line):
            print("no stored tables; run run_frame_choice.py first")
            return 1
        write_summary(pairs, triples, line, assert_release(pairs, triples))
        print(f"rewrote {SUMMARY_MD}")
        return 0
    if args.check:
        rep = check(verbose=True)
        print()
        print(report(rep))
        return 0 if rep["pass"].all() and len(rep) else 1
    if len(benchmark_io.load(PAIRS)) and not args.force:
        print("results/frame_choice/pairs exists; use --force to rewrite it")
        return 1
    t0 = time.perf_counter()
    pairs, triples, line, release_lines = run_all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_io.replace(pairs, PAIRS)
    benchmark_io.replace(triples, TRIPLES)
    benchmark_io.replace(line, LINE)
    write_summary(pairs, triples, line, release_lines)
    print("\n".join(release_lines))
    print(f"\nwrote {len(pairs)} pair rows, {len(triples)} triple rows and "
          f"{len(line)} line rows to results/frame_choice/ and summary.md "
          f"({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
