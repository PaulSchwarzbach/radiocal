"""Synthetic baseline -> results/synthetic_baseline/ (cells.parquet + CSV
mirror, summary.md, and frozen.csv, the reference of the gate).

A controlled check of the lessons of the paper's Sec. VI on ranges whose
errors are known one by one. The range-error generator of range_errors.py
(Schwarzbach et al., IEEE Sensors J. 2022) draws the directed ranges of a
synthetic superset graph on the surveyed anchors of set_b/s03
(synthetic.synthetic_static_graph, 200 epochs, 2-D distances) in five
conditions:

    noise_free      exact ranges, one draw
    gaussian        hardware noise, sigma_hw = 0.33 m per range
    nlos            the multipath state and excess of each anchor pair drawn
                    once (the per-pair extension of range_errors.py), plus
                    hardware noise and outliers (p_out = 0.05) per range
    nlos_per_range  the published per-range model, eq. (5) of the 2022 paper
    biased          hardware noise plus +0.27 m on every range, the set B
                    per-pair median residual (paper, Sec. VI-C)

with 20 draws each (one for noise_free). Draw k uses seed k, with one
generator stream per error
component, so draw k has the same hardware noise in every condition. Each
draw is masked through the production masks to the full mesh and to the
kappa = 3 nearest-neighbor view, and the factor graph (gaussian, cauchy)
runs on both at two kernel scales:

    session   the shared session sigma of the benchmark doctrine
              (kernels.session_sigma on the synthetic full mesh)
    hardware  the generator's sigma_hw

The noise-free condition runs at the hardware scale only, since its session
sigma is zero up to rounding (about 3e-15 m), which no kernel can use as a
scale. The common-offset factor graph (fg_offset) runs on every full
mesh. Every factor-graph cell is also solved from the true geometry, an
oracle start as in run_controls.py, which shows whether the classical start
reached the minimum of lower cost. Each row also records the two terms of
the session sigma, a robust per-pair scatter (1.4826 x MAD over the epochs,
median over the pairs), the rigidity class of the view's pair graph, the
NLOS pairs of the mesh and of the view, the per-pair median residuals of the
full mesh, and a hash of the drawn ranges.

    python run_synthetic.py            # run, write cells + summary
    python run_synthetic.py --force    # rewrite existing cells + summary
    python run_synthetic.py --freeze   # write frozen.csv once from the cells
    python run_synthetic.py --check    # rerun in memory against frozen.csv

--check writes nothing. A cell passes when its status equals the frozen one
and its rmse_m lies within 5e-4 m of it, the tolerance of the regression
gate. A changed hash of the drawn ranges fails as generator drift (numpy does
not guarantee its Generator streams across versions), which is told apart
from an estimator change. Cells on flexible views are reported but not
gated, since their solver end point lies on a continuum of equal cost.
run_benchmark.py --check-regression runs the same check after its 40 frozen
cells. Unlike freeze_expected.py, which overwrites its reference when run,
--freeze refuses to overwrite frozen.csv, since a refreeze from new outputs
would let the gate check them against themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_io
import evaluation
import factor_graph
import family_masks
import kernels
import mds
import range_errors
import regression
import rigidity
import synthetic
from common import default_anchor_order, estimation_order, pair_range_matrix
from registry import REGISTRY, check_sigma

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
TABLE = "synthetic_baseline/cells"
OUT_DIR = benchmark_io.RESULTS_DIR / "synthetic_baseline"
FROZEN_CSV = OUT_DIR / "frozen.csv"
SUMMARY_MD = OUT_DIR / "summary.md"

GEOMETRY = "set_b/s03"
N_EPOCHS = 200   # the epoch budget of validate_multihop.py
DRAWS = 20
BIAS_M = 0.27    # set B per-pair median residual (paper, Sec. VI-C)
MODEL = range_errors.PAPER
VIEWS = ("fully_meshed", "multi_hop_k3")
KERNELS = ("gaussian", "cauchy")
TOL = regression.TOL
KEY = benchmark_io.SYNTHETIC_KEY
FROZEN_COLUMNS = KEY + ["view_class", "status", "rmse_m", "ranges_sha256"]

MARK_B = "<!-- BEGIN GENERATED (run_synthetic.py) -->"
MARK_E = "<!-- END GENERATED (run_synthetic.py) -->"
HEADER = """\
# synthetic_baseline - controlled-truth check of the lessons (paper, Sec. VI)

The range-error generator of `range_errors.py`, a seeded port of the
simulation framework of Schwarzbach, Weber and Michler (IEEE Sensors J.
22(6):4836-4843, 2022, doi 10.1109/JSEN.2021.3121627) with the values of
that paper: hardware noise sigma_hw = 0.33 m, log-normal multipath (0.8 and
1.07 on the log scale) where p_eps > 0.8 - 0.3 d / d_max and the excess is
shorter than d, d_max = 100 m, and uniform outliers U(-d, d_max - d) with
p_out = 0.05, the setting of this baseline. Dropouts are off, since the
topology views set the edges.

Geometry: the surveyed anchors of set_b/s03 (10 anchors, 45 pairs), 200
synthetic epochs, 2-D distances, 20 draws per condition with noise and one
without (seeds 0-19, one generator stream per error component). Conditions:
noise_free, gaussian,
nlos (state and excess per anchor pair, both directions, plus per-range
noise and outliers), nlos_per_range (the published per-range model) and
biased (+0.27 m on every range, plus noise). Views: fully_meshed and
multi_hop_k3 through the production masks. Kernel scales: the session sigma
of the benchmark doctrine and the generator's sigma_hw (the noise-free
condition only at sigma_hw, since its session sigma is zero up to rounding).
Methods: the factor graph (gaussian, cauchy) on both views, also from an
oracle start at the true geometry, and fg_offset (one common range offset)
on the full mesh.

Why the nlos condition holds the multipath per pair: through the per-pair
medians of the batch engines, the per-range model of the 2022 paper becomes
a positive bias of a few decimeters on every pair, growing with the distance,
which acts like a range bias (rows nlos_per_range below), whereas the
measured sessions show single pairs biased by meters (Sec. VI-A of the
paper).

Regenerate with `python run_synthetic.py --force`. `python run_synthetic.py
--check` (also run by `run_benchmark.py --check-regression`) compares a rerun
with `frozen.csv`.
"""


def _sha(r: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(r, dtype="<f8")
                          .tobytes()).hexdigest()[:16]


def robust_scatter(g) -> float:
    """1.4826 x MAD of the epoch ranges per pair, median over the pairs (the
    robust per-pair scatter of run_controls.sigma_terms)."""
    grp = g.symmetrized().groupby(["node_a", "node_b"])["range_m"]
    keep = grp.count() >= kernels.MIN_PAIR_EPOCHS
    mad = grp.apply(lambda r: 1.4826 * np.median(np.abs(r - np.median(r))))
    return float(mad[keep].median())


def solve(g, method: str, variant: str, sigma_m: float, ref: pd.DataFrame,
          X_true: np.ndarray | None) -> dict:
    """One cell from the classical start (the registry adapter, as the
    benchmark runs it) and, for the factor graph, from the true geometry."""
    nan = float("nan")
    out = {"status": "ok", "error": "", "rmse_m": nan, "max_error_m": nan,
           "reflection": pd.NA, "cost": nan, "n_iter": pd.NA,
           "converged": pd.NA, "rmse_oracle_m": nan, "cost_oracle": nan,
           "b_hat_m": nan}
    try:
        est, info = REGISTRY[method].run(g, variant, sigma_m=sigma_m,
                                         sigma_scale=1.0, seed=0)
        check_sigma(info, sigma_m, 1.0)
        res = evaluation.evaluate_estimate(est, ref)
        out.update(rmse_m=res["rmse_m"], max_error_m=res["max_error_m"],
                   reflection=bool(res["is_reflection"]),
                   cost=float(info["cost"]), n_iter=int(info["n_iter"]),
                   converged=bool(info["converged"]),
                   b_hat_m=float(info.get("offset_common_m", nan)))
        if X_true is not None:
            est_o, info_o = factor_graph.factor_graph_anchors(
                g, variant=variant, anchor_order=estimation_order(g),
                sigma_m=sigma_m, X0=X_true)
            out.update(rmse_oracle_m=evaluation.evaluate_estimate(
                est_o, ref)["rmse_m"], cost_oracle=float(info_o["cost"]))
    except mds.DisconnectedGraphError as exc:
        out.update(status="disconnected", error=f"{type(exc).__name__}: {exc}")
    except evaluation.UndefinedConstructionError as exc:
        out.update(status="undefined", error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        out.update(status="error", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    return out


def draw_cells(g0, ref: pd.DataFrame, condition: str, draw: int) -> list[dict]:
    """Every cell of one draw of one condition."""
    xy = ref.set_index("id")[["x", "y"]]
    g, excess = range_errors.perturbed_graph(
        g0, condition, draw, MODEL,
        bias_m=BIAS_M if condition == "biased" else 0.0)
    gf = g.apply_mask(family_masks.resolve_mask("fully_meshed"))
    order_f = default_anchor_order(gf)
    M_f = pair_range_matrix(gf, order_f)
    scatter = kernels.scatter_scale(gf)
    mds_mad = kernels.init_residual_scale(M_f)
    sigma = kernels.session_sigma(gf, order_f)
    assert sigma == max(scatter, mds_mad), "session sigma is not its max term"
    P = xy.loc[order_f].to_numpy(float)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    iu, ju = np.triu_indices(len(order_f), k=1)
    obs = np.isfinite(M_f[iu, ju])
    resid = M_f[iu, ju][obs] - D[iu, ju][obs]
    nlos = excess[excess > 0]
    base = {"condition": condition, "draw": draw,
            "sigma_scatter_m": scatter, "sigma_mds_m": mds_mad,
            "sigma_robust_m": robust_scatter(gf),
            "n_nlos_pairs": len(nlos),
            "nlos_excess_median_m": (float(nlos.median()) if len(nlos)
                                     else float("nan")),
            "pair_resid_median_m": float(np.median(resid)),
            "pair_resid_max_m": float(resid.max()),
            "pair_resid_corr_d": (float(np.corrcoef(D[iu, ju][obs], resid)
                                        [0, 1]) if resid.std() > 1e-9
                                  else float("nan")),
            "ranges_sha256": _sha(g.edges["range_m"].to_numpy(float)),
            "geometry": GEOMETRY, "n_epochs": N_EPOCHS,
            "sigma_hw_m": MODEL.sigma_hw, "mp_mean": MODEL.mp_mean,
            "mp_sigma": MODEL.mp_sigma, "d_max_m": MODEL.d_max,
            "p_out": MODEL.p_out,
            "bias_m": BIAS_M if condition == "biased" else 0.0}
    scales = {"hardware": MODEL.sigma_hw}
    if condition != "noise_free":  # there the session sigma is float residue
        scales = {"session": sigma, **scales}
    rows = []
    for view in VIEWS:
        gv = gf if view == "fully_meshed" else g.apply_mask(
            family_masks.resolve_mask(view))
        order = estimation_order(gv)
        M = pair_range_matrix(gv, order)
        iv, jv = np.triu_indices(len(order), k=1)
        kept = np.isfinite(M[iv, jv])
        pairs = list(zip(iv[kept], jv[kept]))
        X_true = xy.loc[order].to_numpy(float)
        cls = rigidity.certificate_class(
            rigidity.global_rigidity_certificate(X_true, pairs))
        n_kept = sum(excess.get((min(order[i], order[j]),
                                 max(order[i], order[j])), 0.0) > 0
                     for i, j in pairs)
        methods = [("factor_graph", v) for v in KERNELS]
        if view == "fully_meshed":
            methods += [("fg_offset", v) for v in KERNELS]
        for scale, s in scales.items():
            for method, v in methods:
                row = {**base, "view": view, "scale": scale,
                       "method": method, "variant": v, "sigma_m": s,
                       "n_anchors": len(order), "n_pairs": len(pairs),
                       "view_class": cls, "n_nlos_kept": int(n_kept)}
                row.update(solve(gv, method, v, s, ref,
                                 X_true if method == "factor_graph"
                                 else None))
                rows.append(row)
    return rows


def run_all(verbose: bool = True) -> pd.DataFrame:
    """Every cell of the baseline, in memory."""
    ref = evaluation.surveyed_reference(DATA / GEOMETRY)
    g0 = synthetic.synthetic_static_graph(ref, n_epochs=N_EPOCHS, sigma=0.0)
    rows = []
    for condition in range_errors.CONDITIONS:
        for draw in ([0] if condition == "noise_free" else range(DRAWS)):
            cells = draw_cells(g0, ref, condition, draw)
            rows += cells
            if verbose:
                fm = {(c["scale"], c["variant"]): c["rmse_m"] for c in cells
                      if c["view"] == "fully_meshed"
                      and c["method"] == "factor_graph"}
                print(f"{condition:<15} draw {draw:>2}  full mesh "
                      + "  ".join(f"{s}/{v} {r:.3f}"
                                  for (s, v), r in sorted(fm.items()))
                      + f"  ({len(cells)} cells)")
    df = pd.DataFrame(rows, columns=benchmark_io.SYNTHETIC_COLUMNS)
    df["reflection"] = df["reflection"].astype("boolean")
    df["converged"] = df["converged"].astype("boolean")
    df["n_iter"] = df["n_iter"].astype("Int64")
    return df


# ------------------------------------------------------------------ summary

def _span(x: pd.Series) -> str:
    x = x.dropna()
    if not len(x):
        return "-"
    return f"{x.median():.3f} ({x.min():.3f}-{x.max():.3f})"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |"
              for r in df.itertuples(index=False)]
    return "\n".join(lines)


def summary_block(df: pd.DataFrame) -> str:
    """Generated part of summary.md: medians (min-max) over the draws."""
    ok = df[df.status == "ok"]
    out = []
    t = (ok.groupby(["condition", "view", "scale", "method", "variant"],
                    sort=False)
         .agg(draws=("draw", "size"), rmse_m=("rmse_m", _span),
              oracle_start_m=("rmse_oracle_m", _span),
              b_hat_m=("b_hat_m", _span))
         .reset_index())
    out += ["### Aligned anchor RMSE m, median (min-max) over the draws", "",
            _md(t), ""]
    fg = ok[ok.method == "factor_graph"].copy()
    fg["worse_start"] = ((fg.rmse_m > fg.rmse_oracle_m + 0.01)
                         & (fg.cost > fg.cost_oracle))
    w = (fg.groupby(["condition", "view", "scale", "variant"], sort=False)
         .agg(draws=("draw", "size"),
              classical_start_worse=("worse_start", "sum"))
         .reset_index())
    out += ["### Classical start ended more than 0.01 m above the oracle start"
            " at a higher cost (factor graph)", "", _md(w), ""]
    per = df.drop_duplicates(["condition", "draw"])
    cls = (df[df.view == "multi_hop_k3"].drop_duplicates(["condition", "draw"])
           .groupby("condition", sort=False).view_class
           .agg(lambda c: ", ".join(f"{k} {n}" for k, n
                                    in c.value_counts().items())))
    kept = (df[df.view == "multi_hop_k3"].drop_duplicates(["condition", "draw"])
            .groupby("condition", sort=False).n_nlos_kept.median())
    s = (per.groupby("condition", sort=False)
         .agg(sigma_scatter_m=("sigma_scatter_m", _span),
              sigma_mds_m=("sigma_mds_m", _span),
              sigma_robust_m=("sigma_robust_m", _span),
              nlos_pairs=("n_nlos_pairs", "median"),
              pair_resid_median_m=("pair_resid_median_m", _span),
              pair_resid_max_m=("pair_resid_max_m", _span),
              pair_resid_corr_d=("pair_resid_corr_d", _span))
         .reset_index())
    s["nlos_pairs_kept_k3"] = s.condition.map(kept)
    s["k3_view_class"] = s.condition.map(cls)
    out += ["### Per draw: sigma terms, NLOS pairs, per-pair median residuals"
            " of the full mesh, kappa = 3 view classes", "", _md(s)]
    return "\n".join(out)


def write_summary(df: pd.DataFrame) -> None:
    block = f"{MARK_B}\n{summary_block(df)}\n{MARK_E}\n"
    if SUMMARY_MD.exists():
        text = SUMMARY_MD.read_text(encoding="utf-8")
        text = re.sub(re.escape(MARK_B) + r".*?" + re.escape(MARK_E) + r"\n?",
                      lambda _: block, text, flags=re.S)
    else:
        text = HEADER + "\n" + block
    SUMMARY_MD.write_text(text, encoding="utf-8", newline="\n")


# -------------------------------------------------------------------- gate

def check(verbose: bool = False) -> pd.DataFrame:
    """Rerun in memory and compare with frozen.csv (writes nothing)."""
    frozen = (pd.read_csv(FROZEN_CSV, dtype={"ranges_sha256": str})
              if FROZEN_CSV.exists()
              else pd.DataFrame(columns=FROZEN_COLUMNS))
    got = run_all(verbose=verbose)[FROZEN_COLUMNS]
    rep = frozen.merge(got, on=KEY, how="outer", suffixes=("_frozen", ""),
                       indicator=True)
    rep["delta"] = rep["rmse_m"] - rep["rmse_m_frozen"]
    rep["drift"] = rep["ranges_sha256"] != rep["ranges_sha256_frozen"]
    same_status = rep["status"] == rep["status_frozen"]
    value_ok = ((rep["delta"].abs() < TOL)
                | (rep["rmse_m"].isna() & rep["rmse_m_frozen"].isna()))
    rep["gated"] = rep["view_class_frozen"] != "flexible"
    rep["pass"] = ((rep["_merge"] == "both") & ~rep["drift"] & same_status
                   & (value_ok | ~rep["gated"]))
    return rep


def report(rep: pd.DataFrame) -> str:
    n_flex = int((~rep["gated"] & (rep["_merge"] == "both")).sum())
    lines = [f"synthetic baseline: {int(rep['pass'].sum())}/{len(rep)} PASS "
             f"(tol {TOL} m on rmse_m against {FROZEN_CSV.name}; {n_flex} "
             "cells on flexible views reported, not gated)"]
    if not len(rep):
        lines.append(f"  FAIL  no frozen reference {FROZEN_CSV}")
    for _, r in rep[~rep["pass"]].iterrows():
        cell = (f"{r['condition']} draw {r['draw']} {r['view']} {r['scale']} "
                f"{r['method']}:{r['variant']}")
        if r["_merge"] != "both":
            why = ("missing from the rerun" if r["_merge"] == "left_only"
                   else "not in the frozen reference")
        elif r["drift"]:
            why = "generator drift (drawn ranges changed)"
        elif r["status"] != r["status_frozen"]:
            why = f"status {r['status_frozen']} -> {r['status']}"
        else:
            why = (f"frozen {r['rmse_m_frozen']:.6f} got {r['rmse_m']:.6f} "
                   f"(d={r['delta']:+.6f})")
        lines.append(f"  FAIL  {cell}: {why}")
    return "\n".join(lines)


def freeze() -> int:
    if FROZEN_CSV.exists():
        print(f"{FROZEN_CSV} exists; delete it by hand to refreeze "
              "(see the module docstring)")
        return 1
    cells = benchmark_io.load(TABLE)
    if not len(cells):
        print("no cells to freeze; run run_synthetic.py first")
        return 1
    cells[FROZEN_COLUMNS].to_csv(FROZEN_CSV, index=False,
                                 float_format="%.6f", lineterminator="\n")
    print(f"wrote {FROZEN_CSV} ({len(cells)} cells)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true",
                      help="rewrite existing cells and summary")
    mode.add_argument("--freeze", action="store_true",
                      help="write frozen.csv once from the stored cells")
    mode.add_argument("--check", action="store_true",
                      help="rerun in memory and compare with frozen.csv")
    args = ap.parse_args()
    if args.freeze:
        return freeze()
    if args.check:
        rep = check(verbose=True)
        print()
        print(report(rep))
        return 0 if rep["pass"].all() and len(rep) else 1
    if len(benchmark_io.load(TABLE)) and not args.force:
        print("results/synthetic_baseline/cells exists; use --force to "
              "rewrite it")
        return 1
    df = run_all()
    benchmark_io.replace(df, TABLE)
    write_summary(df)
    print(f"\nwrote {len(df)} cells to results/synthetic_baseline/"
          "cells.parquet (+ CSV) and summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
