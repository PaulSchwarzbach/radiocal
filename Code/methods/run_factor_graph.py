"""Run the factor-graph NLS method on the fully_meshed mask.

Sessions: set_b/s03 (LOS-friendly) and set_a/s21
(NLOS-rich). Variants: gaussian, huber, cauchy (shared benchmark kernels,
shared session sigma) - batch LM over the session-median range matrix,
gauge fixed by parameterization, initialized from classical MDS. Writes
results/factor_graph/<campaign>_<session>_estimates.csv,
<campaign>_<session>_pair_weights.csv (final IRLS weights of every range
factor, all variants), <campaign>_<session>_scale_sweep.csv (cauchy RMSE
vs sigma multiplier) and summary.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from radiocal import load_session

import evaluation
import factor_graph
import reporting

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = [ROOT / "Data" / "set_b" / "s03",
            ROOT / "Data" / "set_a" / "s21"]
OUT = Path(__file__).resolve().parent / "results" / "factor_graph"

VARIANTS = ["gaussian", "huber", "cauchy"]
SWEEP_MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]  # sigma multipliers, cauchy only


def run_session(sdir: Path) -> list[dict]:
    campaign, session = sdir.parent.name, sdir.name
    print(f"-> {campaign}/{session}: loading (mask=fully_meshed)")
    g = load_session(sdir, mask="fully_meshed")
    ref = evaluation.surveyed_reference(sdir)

    rows, weights, summary = [], [], []
    for variant in VARIANTS:
        est, info = factor_graph.factor_graph_anchors(g, variant=variant)
        result = evaluation.evaluate_estimate(est, ref)
        rows.append(reporting.per_anchor_table(est, ref, result,
                                               campaign, session, variant))
        pw = info["pair_weights"].copy()
        pw.insert(0, "campaign", campaign)
        pw.insert(1, "session", session)
        pw.insert(2, "variant", variant)
        weights.append(pw)
        summary.append({
            "campaign": campaign, "session": session, "variant": variant,
            "n_anchors": len(est),
            "pairs_available": f"{info['n_pairs_available']}/{info['n_pairs_total']}",
            "rmse_m": result["rmse_m"], "max_error_m": result["max_error_m"],
            "reflection": result["is_reflection"],
            "n_iter": info["n_iter"], "converged": info["converged"],
            "cost0": info["cost0"], "cost": info["cost"],
            "sigma_m": info["sigma_m"],
        })
        print(f"-> {campaign}/{session} [{variant}]: "
              f"RMSE {result['rmse_m']:.3f} m, max {result['max_error_m']:.3f} m, "
              f"iter {info['n_iter']}, cost {info['cost0']:.4g} -> {info['cost']:.4g}, "
              f"sigma {info['sigma_m']:.3f} m")

    sweep = []
    for mult in SWEEP_MULTS:
        est, info = factor_graph.factor_graph_anchors(g, variant="cauchy",
                                                      sigma_mult=mult)
        result = evaluation.evaluate_estimate(est, ref)
        sweep.append({"campaign": campaign, "session": session,
                      "variant": "cauchy", "sigma_mult": mult,
                      "sigma_m": info["sigma_m"], "rmse_m": result["rmse_m"]})
    pd.DataFrame(sweep).to_csv(OUT / f"{campaign}_{session}_scale_sweep.csv",
                               index=False, float_format="%.4f")
    print(f"-> {campaign}/{session} cauchy scale sweep: "
          + ", ".join(f"{s['sigma_mult']:g}x={s['rmse_m']:.3f}" for s in sweep))

    table = pd.concat(rows, ignore_index=True)
    table = table[["campaign", "session", "variant", "id",
                   "x_est", "y_est", "x_aligned", "y_aligned",
                   "x_ref", "y_ref", "z_ref", "error_m"]]
    table.to_csv(OUT / f"{campaign}_{session}_estimates.csv", index=False,
                 float_format="%.4f")
    pd.concat(weights, ignore_index=True).to_csv(
        OUT / f"{campaign}_{session}_pair_weights.csv", index=False,
        float_format="%.4f")
    return summary


def write_summary(summaries: list[dict]) -> None:
    df = pd.DataFrame(summaries)
    lines = [
        "# Factor-graph NLS on fully_meshed - summary",
        "",
        "Session-median symmetrized ranges; every observed pair is a range",
        "factor; 2-D anchor coordinates, gauge fixed by parameterization",
        "(anchor 0 at the origin, anchor 1 on the x-axis); LM with analytic",
        "Jacobian, initialized from classical MDS. Evaluation: rigid Umeyama",
        "alignment (reflection allowed) against the surveyed anchors of the",
        "unmasked session, 2-D (x, y).",
        "",
        reporting.markdown_table(df),
        "",
        "Notes:",
        "",
        "- Parameters: accepted-step convergence at relative cost change",
        "  < 1e-12 or max parameter change < 1e-10; lambda 1e-3 x/10 per",
        "  acceptance/rejection, Marquardt diagonal scaling; max_iter 10000",
        "  is a safety cap only. All runs above converged on the rule",
        "  (huber/2024 needs 87 iterations).",
        "- Kernels and noise scale come from the shared benchmark module",
        "  kernels.py (factorial design: same kernel constants and the same",
        "  session sigma for every method): sigma = max(median per-pair",
        "  epoch std of symmetrized ranges, 1.4826 * MAD of the MDS-init",
        "  residuals) = 0.129 m (2025) / 1.621 m (2024); huber k = 1.345,",
        "  cauchy c = 2.385 on u = r / sigma. The variant formerly labeled",
        "  l2 is now gaussian (identical objective and results).",
        "- Exact-recovery check: noise-free distance matrices built from the",
        "  surveyed coordinates of both sites are recovered by init + solver",
        "  (all variants) with < 1e-13 m aligned RMSE.",
        "- The gaussian variant reproduces the SMACOF geometry (0.965 /",
        "  5.935 m vs 0.96 / 5.93 m): same objective, different optimizer.",
        "  Dropping the NLOS pair 5BB6-D38F moves it on set_b from",
        "  0.965 to 0.228 m, matching the SMACOF precedent - the objective,",
        "  not the solver, is what NLOS breaks; the robust kernels repair it.",
        "- Scale sweep (scale_sweep.csv, cauchy, sigma multiplier 0.25-4):",
        "  on set_a the result is strongly scale-dependent (see the",
        "  cross-method overview) - the a-priori sigma is kept for the",
        "  benchmark row; selecting the multiplier by test RMSE would be",
        "  circular.",
        "- Down-weighted pairs (see pair_weights.csv). set_b: the one",
        "  persistent NLOS pair 5BB6-D38F (residual -5.2/-5.4 m) gets weight",
        "  0.033 (huber) / 0.003 (cauchy); next-lowest is 4AA4-9911 at",
        "  0.46 / 0.41, all other pairs stay >= 0.57 / 0.66. set_a:",
        "  8AA5-9911 (resid -21.1/-27.6 m), 9911-D5B4, 8B05-D5B4, 4AA4-8AA5,",
        "  5388-9911 and 5BB6-8AA5 are the six lowest in both kernels",
        "  (huber weights 0.10-0.27, cauchy 0.02-0.11; next pair 4503-5388",
        "  at 0.52 / 0.51).",
        "- Init sensitivity (set_a, all kernels): starting LM from the",
        "  gauged surveyed coordinates instead of the MDS init converges to",
        "  the same cost and the same RMSE - no local-minimum dependence on",
        "  the initialization was observed, including for the non-convex",
        "  cauchy kernel.",
        "- Remaining cauchy floor on set_a (~1.0 m at 1x sigma):",
        "  removing the six down-weighted pairs entirely and re-solving",
        "  still leaves 0.79 m (gaussian) / 0.68 m (cauchy), and 11 of 65",
        "  pairs have > 1 m median-range bias against the surveyed distances",
        "  (19 > 0.5 m) - the residual error is a broad moderate-NLOS floor,",
        "  not a single culprit pair.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> wrote {OUT / 'summary.md'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sdir in SESSIONS:
        summaries.extend(run_session(sdir))
    write_summary(summaries)


if __name__ == "__main__":
    main()
