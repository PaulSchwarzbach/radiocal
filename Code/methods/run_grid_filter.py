"""Run the grid-filter reference method on the fully_meshed mask.

Sessions: set_b/s03 (LOS-friendly) and set_a/s21
(NLOS-rich). Factorial cells: gaussian / huber / cauchy on the textbook
log-domain product update, all at the shared session sigma of
kernels.session_sigma; mechanism cell "mixture(gaussian)" = the original
IOPS sum-of-likelihoods update with Gaussian components at the same sigma.
Also runs the shared scale-sweep diagnostic (cauchy, sigma_mult 0.25..4).
Writes results/grid_filter/<campaign>_<session>_estimates.csv,
<campaign>_<session>_scale_sweep.csv and summary.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from radiocal import load_session

import evaluation
import grid_filter
import kernels
import reporting
from common import default_anchor_order

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = [ROOT / "Data" / "set_b" / "s03",
            ROOT / "Data" / "set_a" / "s21"]
OUT = Path(__file__).resolve().parent / "results" / "grid_filter"

VARIANTS = ["gaussian", "huber", "cauchy", "mixture"]
LABELS = {"mixture": "mixture(gaussian)"}
SWEEP_MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]
SWEEP_STRIDE_FACTOR = 2  # sweep runs at twice the committed stride


def run_session(sdir: Path) -> list[dict]:
    campaign, session = sdir.parent.name, sdir.name
    print(f"-> {campaign}/{session}: loading (mask=fully_meshed)")
    g = load_session(sdir, mask="fully_meshed")
    order = default_anchor_order(g)
    sigma = kernels.session_sigma(g, order)
    sym = g.symmetrized()
    ref = evaluation.surveyed_reference(sdir)
    print(f"-> {campaign}/{session}: shared sigma {sigma:.3f} m")

    rows, summary = [], []
    for variant in VARIANTS:
        label = LABELS.get(variant, variant)
        est, info = grid_filter.run_grid_filter(sym, order, variant, sigma)
        result = evaluation.evaluate_estimate(est, ref)
        rows.append(reporting.per_anchor_table(est, ref, result,
                                               campaign, session, label))
        summary.append({
            "campaign": campaign, "session": session, "variant": label,
            "n_anchors": len(est),
            "rmse_m": result["rmse_m"], "max_error_m": result["max_error_m"],
            "reflection": result["is_reflection"],
            "grid_res_m": info["grid_res_m"],
            "likelihood_sigma_m": info["sigma_m"],
        })
        print(f"-> {campaign}/{session} [{label}]: "
              f"RMSE {result['rmse_m']:.3f} m, max {result['max_error_m']:.3f} m, "
              f"grid {info['n_cells']} cells (half-width {info['grid_half_m']:.1f} m), "
              f"epochs {info['n_epochs_used']}/{info['n_epochs_total']} "
              f"(stride {info['stride']})")

    table = pd.concat(rows, ignore_index=True)
    table = table[["campaign", "session", "variant", "id",
                   "x_est", "y_est", "x_aligned", "y_aligned",
                   "x_ref", "y_ref", "z_ref", "error_m",
                   "posterior_std_m", "n_update_epochs", "map_vs_mean_m"]]
    table.to_csv(OUT / f"{campaign}_{session}_estimates.csv", index=False,
                 float_format="%.4f")

    # scale-sweep diagnostic: cauchy at sigma_mult * shared sigma
    n_seq = sym["seq"].nunique()
    stride = SWEEP_STRIDE_FACTOR * max(
        1, n_seq // grid_filter.TARGET_UPDATE_EPOCHS)
    sweep = []
    for m in SWEEP_MULTS:
        est, info = grid_filter.run_grid_filter(sym, order, "cauchy",
                                                m * sigma, stride=stride)
        result = evaluation.evaluate_estimate(est, ref)
        sweep.append({"campaign": campaign, "session": session,
                      "variant": "cauchy", "sigma_mult": m,
                      "sigma_m": m * sigma, "rmse_m": result["rmse_m"]})
        print(f"-> {campaign}/{session} sweep [cauchy] x{m}: "
              f"sigma {m * sigma:.3f} m, RMSE {result['rmse_m']:.3f} m")
    pd.DataFrame(sweep).to_csv(OUT / f"{campaign}_{session}_scale_sweep.csv",
                               index=False, float_format="%.4f")
    return summary


def write_summary(summaries: list[dict]) -> None:
    df = pd.DataFrame(summaries)
    lines = [
        "# Grid filter (IOPS auto-positioning port) on fully_meshed - summary",
        "",
        "Round 2 (controlled factorial): one shared noise model per session",
        "(kernels.session_sigma: 0.129 m in set_b, 1.621 m in",
        "set_a) and the kernel axis gaussian / huber / cauchy on the",
        "textbook log-domain product update; `mixture(gaussian)` keeps the",
        "original IOPS mechanism (sum of Gaussian likelihood components over",
        "the references) at the same sigma. Round 1 confounded kernel, scale",
        "and fusion rule; its private scatter sigma (0.05 / 1.039 m) and the",
        "reference-variance inflation are gone (grid_filter.py, deviations 1",
        "and 6). Per-epoch symmetrized ranges (grouped by seq); evaluation:",
        "rigid Umeyama alignment (reflection allowed) against the surveyed",
        "anchors, 2-D.",
        "",
        reporting.markdown_table(df),
        "",
        "Notes:",
        "",
        "- Fixed pipeline pieces (historical, outside the kernel axis):",
        "  closed-form init bump (std 0.5 m), 0.25 m grid (half-width = max",
        "  median range around anchor 0 + 5 m: 27.4 m -> 48400 cells in",
        "  2025, 45.4 m -> 132496 cells in 2024), probability floor 6e-3 x",
        "  uniform, estimate = probability-weighted mean within 4 m of the",
        "  MAP cell. Committed subsampling: stride 27 (78 update epochs,",
        "  2025) and 48 (76 epochs, 2024); the 4-variant table runs in",
        "  ~15 s (2025) / ~130 s (2024). Stride is stable: cauchy 2024",
        "  gives 0.609/0.619/0.634 m at stride 36/48/72, mixture 0.385/",
        "  0.397/0.421 m; cauchy 2025 gives 0.251 m at stride 20/27/41.",
        "- Exact-recovery validation (synthetic exact distances from the",
        "  surveyed coordinates, 200 epochs, shared sigma): 2025 RMSE",
        "  0.099 m / max 0.153 m for all four variants; 2024 RMSE 0.075-",
        "  0.098 m / max 0.176 m (kernels) and 0.248 m (mixture), all",
        "  anchors placed, no reflection - within grid quantization (cell",
        "  half-diagonal 0.177 m; the mixture posterior at sigma 1.621 m is",
        "  broad, so its 4-m window mean smooths up to one cell).",
        "- Kernel axis (product update, shared sigma): robustness is",
        "  monotone in the kernel at both sites - see table. The gaussian",
        "  cell fails exactly like round 1's product variant: in 2025,",
        "  dropping the single persistent NLOS pair 5BB6-D38F (+5.82 m",
        "  median bias) moves it from 0.805 to 0.247 m (committed stride) -",
        "  that pair is the root cause; in 2024 the bias is distributed",
        "  (p95 pair bias +16.6 m; in round 1 dropping the worst pair moved",
        "  the product update only from 2.70 to 2.43 m).",
        "- Sum vs product, now at identical kernel and scale: the original",
        "  mixture update beats the gaussian product 0.238 vs 0.805 m",
        "  (2025) and 0.397 vs 3.042 m (2024) - the mechanism, not the",
        "  scale, is what made the historical pipeline NLOS-tolerant. It",
        "  also matches or beats the redescending cauchy product (0.251 m",
        "  in 2025, 0.619 m in 2024): a biased range contributes one",
        "  displaced component that the other references outvote, while",
        "  any product update must down-weight it explicitly.",
        "- Mixture under the shared sigma (vs round 1): 2025 0.266 ->",
        "  0.238 m; 2024 0.342 -> 0.397 m. The 2024 change decomposes",
        "  cleanly at the committed stride: rerunning with the round-1",
        "  sigma 1.039 m gives 0.358 m (+0.039 m from the wider scale),",
        "  and the round-1 sigma at round-1 stride 18 gave 0.342 m",
        "  (+0.016 m from coarser subsampling). No mechanism surprise.",
        "- Cauchy and mixture(gaussian) agree closely on 2025 (both fully",
        "  reject the NLOS pair); at stride 20 they produced cell-identical",
        "  estimates (max difference 2.3 mm), at the committed stride 27",
        "  they differ by up to 0.35 m per anchor at near-equal RMSE -",
        "  grid-cell-level coincidence, not an equivalence.",
        "- Scale sweep (cauchy, 2x committed stride, 39/38 epochs): RMSE",
        "  0.245 / 0.245 / 0.251 / 0.281 / 0.306 m (2025) and 0.319 /",
        "  0.395 / 0.598 / 0.900 / 1.257 m (2024) at sigma_mult 0.25 / 0.5",
        "  / 1 / 2 / 4. At the NLOS site the cauchy filter improves",
        "  monotonically as the scale shrinks below the shared sigma - a",
        "  tighter scale sharpens the redescend and rejects the distributed",
        "  biased pairs harder; in 2025 the sweep is flat below 1x, where",
        "  grid quantization dominates.",
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
