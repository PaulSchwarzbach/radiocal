"""Run the closed-form baseline on the fully_meshed mask (benchmark view).

Sessions: set_b/s03 (LOS-friendly) and set_a/s21
(NLOS-rich). Both y-sign variants are run; the session-median range matrix
gives the primary estimate, per-epoch placement success is a diagnostic.
Writes results/closed_form/<campaign>_<session>_estimates.csv and summary.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from radiocal import load_session

import closed_form
import evaluation
import reporting
from common import default_anchor_order

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = [ROOT / "Data" / "set_b" / "s03",
            ROOT / "Data" / "set_a" / "s21"]
OUT = Path(__file__).resolve().parent / "results" / "closed_form"

VARIANTS = [("faithful", False), ("disambiguated", True)]


def run_session(sdir: Path) -> tuple[pd.DataFrame, list[dict]]:
    campaign, session = sdir.parent.name, sdir.name
    print(f"-> {campaign}/{session}: loading (mask=fully_meshed)")
    g = load_session(sdir, mask="fully_meshed")
    ref = evaluation.surveyed_reference(sdir)
    order = default_anchor_order(g)

    print(f"-> {campaign}/{session}: per-epoch success diagnostic")
    success = closed_form.epoch_success_rate(g, order).set_index("id")

    rows, summary = [], []
    for variant, disambiguate in VARIANTS:
        est, info = closed_form.closed_form_anchors(g, anchor_order=order,
                                                    disambiguate=disambiguate)
        result = evaluation.evaluate_estimate(est, ref)
        per_anchor = reporting.per_anchor_table(est, ref, result,
                                                campaign, session, variant)
        per_anchor["epoch_success_rate"] = per_anchor["id"].map(success["success_rate"])
        rows.append(per_anchor)

        summary.append({
            "campaign": campaign, "session": session, "variant": variant,
            "n_placed": len(est), "n_anchors": len(order),
            "pairs_available": f"{info['n_pairs_available']}/{info['n_pairs_total']}",
            "rmse_m": result["rmse_m"], "max_error_m": result["max_error_m"],
            "reflection": result["is_reflection"],
            "z_spread_ref_m": float(ref["z"].max() - ref["z"].min()),
        })
        print(f"-> {campaign}/{session} [{variant}]: "
              f"{len(est)}/{len(order)} anchors placed, "
              f"RMSE {result['rmse_m']:.3f} m, max {result['max_error_m']:.3f} m")

    table = pd.concat(rows, ignore_index=True)
    table = table[["campaign", "session", "variant", "id",
                   "x_est", "y_est", "x_aligned", "y_aligned",
                   "x_ref", "y_ref", "z_ref", "error_m", "epoch_success_rate"]]
    table.to_csv(OUT / f"{campaign}_{session}_estimates.csv", index=False,
                 float_format="%.4f")
    return table, summary


def write_summary(summaries: list[dict]) -> None:
    df = pd.DataFrame(summaries)
    lines = [
        "# Closed-form baseline on fully_meshed - summary",
        "",
        "Session-median symmetrized ranges, one construction per session/variant;",
        "evaluation: rigid Umeyama alignment (reflection allowed) against the",
        "surveyed anchors of the unmasked session, 2-D (x, y).",
        "",
        reporting.markdown_table(df),
        "",
        "Notes:",
        "",
        "- `faithful` reproduces the original positioning.py (always +y);",
        "  `disambiguated` resolves the y sign against a third placed anchor.",
        "  The sign test relies on a single reference range, so at an NLOS-rich",
        "  site it can itself be fooled and flip anchors the wrong way.",
        "- The construction is planar while ranges are slope distances; the",
        "  surveyed z-spread per site (last column) bounds that approximation.",
        "- set_a is the NLOS-rich site: degradation vs set_b is",
        "  expected and reported as-is, not tuned away.",
        "- `epoch_success_rate` in the estimate CSVs is the fraction of epochs",
        "  in which an anchor could be placed from that epoch's matrix alone",
        "  (faithful placement rule), i.e. the original per-epoch diagnostic.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> wrote {OUT / 'summary.md'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sdir in SESSIONS:
        _, s = run_session(sdir)
        summaries.extend(s)
    write_summary(summaries)


if __name__ == "__main__":
    main()
