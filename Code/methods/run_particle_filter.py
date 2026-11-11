"""Run the particle filter baseline on the fully_meshed mask.

Sessions: set_b/s03 (LOS-friendly) and set_a/s21
(NLOS-rich). Variants: the shared factorial-benchmark kernels gaussian /
huber / cauchy (kernels.py, one session sigma for every method), each as
a per-anchor conditional bootstrap filter over the epoch stream (N = 2000
particles, classical-MDS-on-first-20-epochs init, 2 sweeps, seed 0; see
particle_filter.py). Writes results/particle_filter/
<campaign>_<session>_estimates.csv, <campaign>_<session>_convergence.csv
(aligned RMSE snapshot every 50 epochs, cumulative over both sweeps),
<campaign>_<session>_scale_sweep.csv (cauchy only, sigma_used = m * sigma
for m in SCALE_MULTS) and summary.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from radiocal import load_session

import evaluation
import particle_filter
import reporting

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = [ROOT / "Data" / "set_b" / "s03",
            ROOT / "Data" / "set_a" / "s21"]
OUT = Path(__file__).resolve().parent / "results" / "particle_filter"

VARIANTS = ["gaussian", "huber", "cauchy"]
SWEEPS = particle_filter.SWEEPS      # 2 (see summary notes for the 1-vs-2 comparison)
SEED = 0                             # committed results use seed 0 only
SCALE_MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]  # scale-sweep diagnostic (cauchy)


def convergence_rows(info: dict, ref: pd.DataFrame, variant: str) -> list[dict]:
    """Aligned RMSE for every stored posterior-mean snapshot."""
    rows = []
    for epoch_index, coords in info["trace"]:
        snap = pd.DataFrame({"id": info["anchor_order"],
                             "x": coords[:, 0], "y": coords[:, 1]})
        r = evaluation.evaluate_estimate(snap, ref)
        rows.append({"variant": variant, "epoch_index": epoch_index,
                     "rmse_m": r["rmse_m"]})
    return rows


def run_session(sdir: Path) -> list[dict]:
    campaign, session = sdir.parent.name, sdir.name
    print(f"-> {campaign}/{session}: loading (mask=fully_meshed)")
    g = load_session(sdir, mask="fully_meshed")
    ref = evaluation.surveyed_reference(sdir)

    rows, summary, conv = [], [], []
    for variant in VARIANTS:
        est, info = particle_filter.particle_filter_anchors(
            g, variant=variant, sweeps=SWEEPS, seed=SEED)
        result = evaluation.evaluate_estimate(est, ref)
        rows.append(reporting.per_anchor_table(est, ref, result,
                                               campaign, session, variant))
        conv.extend(convergence_rows(info, ref, variant))
        summary.append({
            "campaign": campaign, "session": session, "variant": variant,
            "init": info["init"], "n_anchors": len(est),
            "rmse_m": result["rmse_m"], "max_error_m": result["max_error_m"],
            "reflection": result["is_reflection"],
            "sigma_m": info["sigma_m"],
            "n_particles": info["n_particles"], "sweeps": info["sweeps"],
        })
        n_res = sum(info["n_resample"].values())
        n_dep = sum(info["n_depletion"].values())
        print(f"-> {campaign}/{session} [{variant}]: "
              f"RMSE {result['rmse_m']:.3f} m, max {result['max_error_m']:.3f} m, "
              f"sigma {info['sigma_m']:.3f} m, "
              f"{n_res} resamplings, {n_dep} depletion events")

    table = pd.concat(rows, ignore_index=True)
    table = table[["campaign", "session", "variant", "id",
                   "x_est", "y_est", "x_aligned", "y_aligned",
                   "x_ref", "y_ref", "z_ref", "error_m", "posterior_std_m"]]
    table.to_csv(OUT / f"{campaign}_{session}_estimates.csv", index=False,
                 float_format="%.4f")
    pd.DataFrame(conv).to_csv(OUT / f"{campaign}_{session}_convergence.csv",
                              index=False, float_format="%.4f")

    sweep_rows = []
    for m in SCALE_MULTS:
        est, info = particle_filter.particle_filter_anchors(
            g, variant="cauchy", sweeps=SWEEPS, seed=SEED, sigma_mult=m)
        result = evaluation.evaluate_estimate(est, ref)
        sweep_rows.append({"campaign": campaign, "session": session,
                           "variant": "cauchy", "sigma_mult": m,
                           "sigma_m": info["sigma_m"],
                           "rmse_m": result["rmse_m"]})
        print(f"-> {campaign}/{session} [cauchy scale x{m}]: "
              f"sigma {info['sigma_m']:.3f} m, RMSE {result['rmse_m']:.3f} m")
    pd.DataFrame(sweep_rows).to_csv(OUT / f"{campaign}_{session}_scale_sweep.csv",
                                    index=False, float_format="%.4f")
    return summary, sweep_rows


def scale_sweep_lines(sweeps: list[dict]) -> list[str]:
    """One Notes line per session with the measured scale-sweep RMSEs."""
    df = pd.DataFrame(sweeps)
    lines = []
    for campaign, d in df.groupby("campaign", sort=False):
        d = d.sort_values("sigma_mult")
        vals = " / ".join(f"{r:.2f}" for r in d["rmse_m"])
        mults = " / ".join(f"x{m:g}" for m in d["sigma_mult"])
        lines.append(f"  {campaign}: {vals} m at {mults}.")
    return lines


def write_summary(summaries: list[dict], sweeps: list[dict]) -> None:
    df = pd.DataFrame(summaries)
    lines = [
        "# Particle filter baseline on fully_meshed - summary (round 2)",
        "",
        "Per-anchor conditional bootstrap particle filter over the epoch",
        "stream (grouped by seq); evaluation: rigid Umeyama alignment",
        "(reflection allowed) against the surveyed anchors of the unmasked",
        "session, 2-D (x, y).",
        "",
        reporting.markdown_table(df),
        "",
        "Notes:",
        "",
        "- Round 2 (factorial benchmark): the likelihood is the shared",
        "  kernel module kernels.py - per-range log-weight",
        "  -rho((r - d) / sigma, kernel), kernels gaussian / huber",
        "  (k = 1.345) / cauchy (c = 2.385), one session scale sigma =",
        "  kernels.session_sigma: 0.129 m (set_b), 1.621 m",
        "  (set_a). Round 1 used a per-method scale (median per-pair",
        "  epoch std floored at 0.05 m; cauchy gamma = that sigma, 1.04 m",
        "  at 2024). The 2024 cauchy change 0.28 -> 0.96 m is this scale",
        "  unification (effective half-width 2.385 * 1.621 = 3.87 m admits",
        "  more NLOS influence), not an estimator change.",
        "- Parameters unchanged from round 1: N = 2000 particles per",
        "  anchor, init = mds20 (classical MDS on the median range matrix",
        "  of the first 20 epochs, shortest-path completed, gauge-fixed;",
        "  the same init recipe as the EKF and factor-graph methods) + 2 m",
        "  Gaussian scatter, annulus/disk fallback for anchors without a",
        "  finite init position, systematic resampling at ESS < N/2 with",
        "  0.02 m jitter, 2 sweeps (sweep 1: condition on earlier anchors",
        "  only; sweep 2: re-scatter 2 m around the sweep-1 means,",
        "  condition on all anchors), seed 0. mds20 init RMSE after",
        "  alignment: 0.26 m (2025), 3.10 m (2024, worst anchor 8AA5 at",
        "  5.2 m).",
        "- Exact-recovery check (synthetic ranges from the surveyed",
        "  coordinates, 1 mm noise, 200 epochs; the session sigma resolves",
        "  to the noise floor): 0.06-0.22 cm RMSE on both site geometries,",
        "  all three kernels, sweeps 1 and 2.",
        "- Seed spread (seeds 0/1/2, committed config): <= 0.006 m on all",
        "  set_b cells; set_a gaussian 2.940-2.969, huber",
        "  1.049-1.100, cauchy 0.902-1.274. The 2024 cauchy spread is",
        "  dominated by 8AA5 (2.08 m at seed 0 vs 3.33 m at seed 1): under",
        "  the wide shared scale the redescending posterior at the most",
        "  NLOS-loaded anchor is broad, so the selected mode varies with",
        "  the sampling stream.",
        "- Sweeps 1 vs 2 (seed 0): gaussian 0.30 -> 0.76 m (2025) and",
        "  3.94 -> 2.94 m (2024); huber 0.23 -> 0.24 m and 3.96 -> 1.10 m;",
        "  cauchy 0.23 -> 0.23 m and 3.95 -> 0.96 m. Two sweeps stay the",
        "  default (the 2024 robust cells improve by ~3 m; sweep 1 is",
        "  limited by the conditioning order, not the kernel). The",
        "  gaussian/2025 regression is the NLOS pair 5BB6-D38F (+5.8 m",
        "  bias): in sweep 2 all anchors condition on the displaced D38F",
        "  mean and the bias propagates. Dropping that pair gives 0.23 m",
        "  for both sweep counts (re-verified under the round-2",
        "  likelihood; same pair that moved SMACOF 0.96 -> 0.23 m). It",
        "  also collapses ESS at D38F: 1111 of 4160 updates below 1% of N",
        "  under the 0.129 m Gaussian; huber and cauchy have <= 2 such",
        "  events per anchor.",
        "- Init-mode capture, re-verified under the round-2 likelihood:",
        "  with init='closed_form' (8AA5 seeded 16.2 m off by the +14.1 m",
        "  NLOS pair 8AA5-5BB6) the 2024 cauchy cell degrades to 1.51 m",
        "  RMSE with 8AA5 at 4.28 m, vs 0.96 m / 2.08 m under mds20 - the",
        "  effect persists, attenuated by the wider scale. Round-1",
        "  experiments at the tight per-method scale (gamma = 1.04 m)",
        "  showed the full mechanism: closed-form init 3.88 m RMSE / 8AA5",
        "  at 12.5 m; clean init + full NLOS stream 0.28 m; biased init +",
        "  cleaned stream 14.9 m - a redescending kernel that rejects NLOS",
        "  votes equally defends a wrong init mode, while gaussian is",
        "  init-insensitive but dragged by the stream bias.",
        "- Scale-sweep diagnostic (cauchy, sigma_used = m * sigma, seed 0;",
        "  RMSE per multiplier, see the scale_sweep CSVs):",
        *scale_sweep_lines(sweeps),
        "- posterior_std_m is the spread of the per-anchor conditional",
        "  filter given point estimates of the other anchors; it does not",
        "  include the uncertainty of the conditioning set and understates",
        "  the joint uncertainty.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> wrote {OUT / 'summary.md'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, sweeps = [], []
    for sdir in SESSIONS:
        summary, sweep_rows = run_session(sdir)
        summaries.extend(summary)
        sweeps.extend(sweep_rows)
    write_summary(summaries, sweeps)


if __name__ == "__main__":
    main()
