"""Run the sequential EKF on the fully_meshed mask.

Sessions: set_b/s03 (LOS-friendly) and set_a/s21
(NLOS-rich). Variants = kernel axis (gaussian, huber, cauchy; shared
session sigma from kernels.session_sigma, a-priori kernel constants) plus
the mechanism variant "gated" (gaussian kernel + NIS > 9 innovation gate).
State = gauge-reduced 2-D anchor coordinates (2n-3), init = classical MDS
on the first 20 epochs, zero process noise. Writes results/ekf/<campaign>_
<session>_estimates.csv, <campaign>_<session>_convergence.csv (aligned RMSE
every 50 epochs, all variants), <campaign>_<session>_scale_sweep.csv
(cauchy at sigma multiples) and summary.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from radiocal import load_session

import ekf
import evaluation
import kernels
import reporting
from common import default_anchor_order

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = [ROOT / "Data" / "set_b" / "s03",
            ROOT / "Data" / "set_a" / "s21"]
OUT = Path(__file__).resolve().parent / "results" / "ekf"

VARIANTS = ["gaussian", "huber", "cauchy", "gated"]
INIT = "mds"       # default init family; "closed_form" kept in ekf.py
TRACE_EVERY = 50   # epochs between convergence-trace evaluations
TOP_GATED = 3      # gated pairs named in the per-variant printout
SWEEP_VARIANT = "cauchy"
SWEEP_MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]


def convergence_rows(info: dict, ref: pd.DataFrame, variant: str) -> list[dict]:
    """Aligned RMSE of every coordinate snapshot (one row per trace point)."""
    rows = []
    for epoch_index, coords in info["snapshots"]:
        est = pd.DataFrame({"id": info["anchor_order"],
                            "x": coords[:, 0], "y": coords[:, 1]})
        res = evaluation.evaluate_estimate(est, ref)
        rows.append({"variant": variant, "epoch_index": epoch_index,
                     "rmse_m": res["rmse_m"]})
    return rows


def run_session(sdir: Path) -> list[dict]:
    campaign, session = sdir.parent.name, sdir.name
    print(f"-> {campaign}/{session}: loading (mask=fully_meshed)")
    g = load_session(sdir, mask="fully_meshed")
    ref = evaluation.surveyed_reference(sdir)
    order = default_anchor_order(g)
    sigma = kernels.session_sigma(g, order)
    print(f"-> {campaign}/{session}: shared sigma {sigma:.3f} m")

    rows, conv, summary = [], [], []
    for variant in VARIANTS:
        est, info = ekf.ekf_anchors(g, variant=variant, sigma_m=sigma,
                                    init=INIT, trace_every=TRACE_EVERY)
        result = evaluation.evaluate_estimate(est, ref)
        rows.append(reporting.per_anchor_table(est, ref, result,
                                               campaign, session, variant))
        conv.extend(convergence_rows(info, ref, variant))
        init_label = f"{info['init']['init']}{info['init']['init_epochs']}"
        summary.append({
            "campaign": campaign, "session": session, "variant": variant,
            "init": init_label, "n_anchors": len(est), "sigma_m": sigma,
            "rmse_m": result["rmse_m"], "max_error_m": result["max_error_m"],
            "reflection": result["is_reflection"],
            "gated_frac": info["gated_frac"] if variant == "gated" else "-",
        })
        print(f"-> {campaign}/{session} [{variant}]: "
              f"RMSE {result['rmse_m']:.3f} m, max {result['max_error_m']:.3f} m, "
              f"gated {info['gated_frac']:.4f} "
              f"({info['n_gated']}/{info['n_updates']})")
        if variant == "gated" and info["n_gated"] > 0:
            top = info["gate_stats"].head(TOP_GATED)
            pairs = ", ".join(f"{r.node_a}-{r.node_b} {r.gated_frac:.2f}"
                              for r in top.itertuples())
            print(f"-> {campaign}/{session} [{variant}]: top gated pairs: {pairs}")

    sweep = []
    for mult in SWEEP_MULTS:
        est, _ = ekf.ekf_anchors(g, variant=SWEEP_VARIANT,
                                 sigma_m=mult * sigma, init=INIT,
                                 trace_every=10 ** 9)
        result = evaluation.evaluate_estimate(est, ref)
        sweep.append({"campaign": campaign, "session": session,
                      "variant": SWEEP_VARIANT, "sigma_mult": mult,
                      "sigma_m": mult * sigma, "rmse_m": result["rmse_m"]})
        print(f"-> {campaign}/{session} [sweep {SWEEP_VARIANT} m={mult:g}]: "
              f"sigma {mult * sigma:.3f} m, RMSE {result['rmse_m']:.3f} m")

    table = pd.concat(rows, ignore_index=True)
    table = table[["campaign", "session", "variant", "id",
                   "x_est", "y_est", "x_aligned", "y_aligned",
                   "x_ref", "y_ref", "z_ref", "error_m",
                   "sigma_x_m", "sigma_y_m"]]
    table.to_csv(OUT / f"{campaign}_{session}_estimates.csv", index=False,
                 float_format="%.4f")
    pd.DataFrame(conv).to_csv(OUT / f"{campaign}_{session}_convergence.csv",
                              index=False, float_format="%.4f")
    pd.DataFrame(sweep).to_csv(OUT / f"{campaign}_{session}_scale_sweep.csv",
                               index=False, float_format="%.4f")
    return summary


def write_summary(summaries: list[dict]) -> None:
    df = pd.DataFrame(summaries)
    lines = [
        "# Sequential EKF on fully_meshed - summary (factorial round 2)",
        "",
        "Per-epoch symmetrized ranges processed sequentially (state updated",
        "between pairs of an epoch); evaluation: rigid Umeyama alignment",
        "(reflection allowed) against the surveyed anchors of the unmasked",
        "session, 2-D (x, y).",
        "",
        reporting.markdown_table(df),
        "",
        "Parameters: state = 2-D anchor coordinates minus gauge (anchor 5BB6",
        "fixed at the origin, 4A23 y = 0; dim 2n-3); init = classical MDS,",
        "shortest-path completed, on the median matrix of the first 20",
        "epochs, re-gauged (mds20; a closed-form init remains available in",
        "ekf.py); initial sigma 2 m per state coordinate; process noise zero",
        "(static anchors). Noise model shared across all C4 methods:",
        "sigma_m = kernels.session_sigma (scatter / init-residual scale),",
        "kernel constants a-priori (huber k = 1.345, cauchy c = 2.385).",
        "Robust variants use IRLS-style R inflation per scalar update:",
        "u = innovation / sigma, R_eff = sigma^2 / kernels.weight(u, kernel).",
        "gated = gaussian kernel + NIS > 9 innovation gate (3-sigma, 1 DOF).",
        "",
        "Notes (all statements verified by targeted experiments):",
        "",
        "- Exact-recovery check: on synthetic streams built from the",
        "  surveyed coordinates (200 epochs, all pairs, exact distances and",
        "  1 mm noise) the full pipeline (init + filter) recovers both",
        "  sites to < 0.1 mm RMSE, all four variants.",
        "- Changes vs round 1: the shared session sigma replaces the",
        "  per-method scatter estimate (0.050 -> 0.129 m on 2025, 1.039 ->",
        "  1.621 m on 2024) and the kernel axis was added. The gaussian",
        "  results match the round-1 plain filter (1.07 / 3.28-3.29 m):",
        "  a uniform rescaling of R changes gains, not the fixed point.",
        "- Kernel axis: gaussian adopts the NLOS biases (2025: 1.07 m,",
        "  dominated by 5BB6-D38F at +5.8 m - the round-1 drop experiment",
        "  gives 0.23 m without that pair; 2024: 3.29 m). huber (0.240 /",
        "  0.900 m) improves but keeps the linear tail influence of large",
        "  outliers; cauchy redescends and gives 0.224 / 0.649 m.",
        "- Gate behavior change on 2025 (sigma 0.050 -> 0.129 m): the",
        "  3-sigma gate is now ~0.39 m instead of ~0.15 m, wider than the",
        "  +0.2..0.5 m systematic LOS biases, so the round-1 pathology of",
        "  wholesale LOS rejection is gone: 3.1% of updates gated (round 1:",
        "  17.4%), and only the true NLOS pair 5BB6-D38F is > 99% gated",
        "  (100%, from epoch 1); next is 4AA4-9911 (+0.99 m) at 49%. The",
        "  gated filter now re-estimates the geometry (0.220 m) instead of",
        "  freezing the 0.264 m init as in round 1 (0.238 m).",
        "- 2024 gate: the six worst NLOS pairs (+14.1 to +30.3 m median",
        "  bias) are 100% gated from their first observation, 3.8% of all",
        "  updates. gated (0.767 m) still trails cauchy (0.649 m): pairs",
        "  with moderate bias that pass the gate enter at full gaussian",
        "  weight, while cauchy downweights them continuously.",
        "- Cauchy scale sweep (*_scale_sweep.csv): on 2025 the result is",
        "  flat (0.218-0.251 m over sigma multiples 0.25-4) - the one",
        "  dominant outlier is rejected at every scale. On 2024 it is",
        "  monotone (0.278 m at 0.25x to 2.475 m at 4x): the shared sigma",
        "  of the NLOS-rich session is itself inflated by NLOS scatter, so",
        "  the a-priori scale under-rejects there; reported as scale",
        "  sensitivity, not retuned (constants stay a-priori).",
        "- Init dependence persists in this configuration: from the",
        "  closed-form init (reflected, 17.8 m wrong on 2024) gaussian ends",
        "  at 13.0 m and cauchy at 18.3 m - the robust kernel downweights",
        "  exactly the measurements that contradict the wrong geometry, so",
        "  robustness amplifies init lock-in. On 2025 both are init-",
        "  independent (1.071 / 0.224 m from the closed-form init). All",
        "  table rows use the mds20 init (0.264 m on 2025, 3.097 m on",
        "  2024, no reflection).",
        "- The filter remains optimistic: final per-coordinate posterior",
        "  sigma is 0.002-0.004 m (2025) and 0.015-0.034 m (2024) while",
        "  actual per-anchor errors reach 2.73 m / 7.52 m (gaussian) and",
        "  0.32-0.39 m / 1.2-1.6 m (robust variants); R inflation does not",
        "  repair the unmodeled systematic bias in the covariance.",
        "- Convergence (*_convergence.csv): on 2025 the robust and gated",
        "  traces are flat at their final level from epoch 50; gaussian",
        "  drifts 0.78 m (epoch 400) -> 1.07 m as biased epochs accumulate.",
        "  On 2024 the robust traces improve slowly (cauchy 0.80 m at epoch",
        "  50 -> 0.65 m at the end).",
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
