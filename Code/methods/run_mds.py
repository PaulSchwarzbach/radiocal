"""Run the MDS baselines on the fully_meshed mask (benchmark view).

Sessions: set_b/s03 (LOS-friendly) and set_a/s21
(NLOS-rich). Variants: classical (MDS-MAP, shortest-path completion) and
SMACOF (stress majorization, missing pairs weighted out), both on the
session-median range matrix. Writes results/mds/<campaign>_<session>_
estimates.csv and summary.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from radiocal import load_session

import evaluation
import mds
import reporting

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = [ROOT / "Data" / "set_b" / "s03",
            ROOT / "Data" / "set_a" / "s21"]
OUT = Path(__file__).resolve().parent / "results" / "mds"

VARIANTS = ["classical", "smacof_gaussian", "smacof_huber", "smacof_cauchy"]
SWEEP_MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]  # sigma multipliers, smacof_cauchy


def run_session(sdir: Path) -> list[dict]:
    campaign, session = sdir.parent.name, sdir.name
    print(f"-> {campaign}/{session}: loading (mask=fully_meshed)")
    g = load_session(sdir, mask="fully_meshed")
    ref = evaluation.surveyed_reference(sdir)

    rows, summary = [], []
    for variant in VARIANTS:
        est, info = mds.mds_anchors(g, variant=variant)
        result = evaluation.evaluate_estimate(est, ref)
        rows.append(reporting.per_anchor_table(est, ref, result,
                                               campaign, session, variant))
        summary.append({
            "campaign": campaign, "session": session, "variant": variant,
            "n_anchors": len(est),
            "pairs_available": f"{info['n_pairs_available']}/{info['n_pairs_total']}",
            "pairs_completed": info["n_pairs_completed"],
            "rmse_m": result["rmse_m"], "max_error_m": result["max_error_m"],
            "reflection": result["is_reflection"],
            "stress1": info["stress"], "neg_eig_frac": info["neg_eig_frac"],
            "sigma_m": info["sigma_m"], "n_iter": info["n_iter"],
        })
        print(f"-> {campaign}/{session} [{variant}]: "
              f"RMSE {result['rmse_m']:.3f} m, max {result['max_error_m']:.3f} m, "
              f"stress1 {info['stress']:.4f}, neg-eig {info['neg_eig_frac']:.3f}")

    sweep = []
    for mult in SWEEP_MULTS:
        est, info = mds.mds_anchors(g, variant="smacof_cauchy", sigma_mult=mult)
        result = evaluation.evaluate_estimate(est, ref)
        sweep.append({"campaign": campaign, "session": session,
                      "variant": "smacof_cauchy", "sigma_mult": mult,
                      "sigma_m": info["sigma_m"], "rmse_m": result["rmse_m"]})
    pd.DataFrame(sweep).to_csv(OUT / f"{campaign}_{session}_scale_sweep.csv",
                               index=False, float_format="%.4f")
    print(f"-> {campaign}/{session} smacof_cauchy scale sweep: "
          + ", ".join(f"{s['sigma_mult']:g}x={s['rmse_m']:.3f}" for s in sweep))

    table = pd.concat(rows, ignore_index=True)
    table = table[["campaign", "session", "variant", "id",
                   "x_est", "y_est", "x_aligned", "y_aligned",
                   "x_ref", "y_ref", "z_ref", "error_m"]]
    table.to_csv(OUT / f"{campaign}_{session}_estimates.csv", index=False,
                 float_format="%.4f")
    return summary


def write_summary(summaries: list[dict]) -> None:
    df = pd.DataFrame(summaries)
    lines = [
        "# MDS baselines on fully_meshed - summary",
        "",
        "Session-median symmetrized ranges; evaluation: rigid Umeyama",
        "alignment (reflection allowed) against the surveyed anchors of the",
        "unmasked session, 2-D (x, y).",
        "",
        reporting.markdown_table(df),
        "",
        "Notes:",
        "",
        "- `classical` = Torgerson double centering on the shortest-path-",
        "  completed matrix (MDS-MAP), kernel-less; `smacof_gaussian` =",
        "  stress majorization over observed pairs only, initialized from",
        "  the classical solution (formerly labeled `smacof`, identical",
        "  results); `smacof_huber` / `smacof_cauchy` = IRLS-robust weighted",
        "  SMACOF with the shared benchmark kernels and session sigma",
        "  (kernels.py; huber k = 1.345, cauchy c = 2.385, sigma 0.129 m on",
        "  2025 / 1.621 m on 2024).",
        "- Scale sweep (scale_sweep.csv, smacof_cauchy, sigma multiplier",
        "  0.25-4): see the cross-method overview; the a-priori sigma is",
        "  kept for the benchmark rows.",
        "- `stress1` is Kruskal stress-1 over observed pairs; `neg_eig_frac`",
        "  is the negative eigenvalue mass of the centered squared-distance",
        "  matrix - both indicate non-Euclidean noise (NLOS, ranging bias).",
        "- All anchors are always placed (connected range graph), and every",
        "  observed pair enters the estimate, unlike the closed-form",
        "  construction.",
        "- The two variants split under NLOS: classical strain spreads a",
        "  biased range across the Gram matrix, while SMACOF fits distances",
        "  directly and adopts the bias - lower stress1 can coincide with",
        "  worse geometry (verified on set_b: dropping the one NLOS",
        "  pair 5BB6-D38F moves SMACOF 0.96 -> 0.23 m, classical stays).",
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
