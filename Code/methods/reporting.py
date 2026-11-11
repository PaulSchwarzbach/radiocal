"""Shared result assembly and formatting for the per-method runners."""

from __future__ import annotations

import pandas as pd


def per_anchor_table(est: pd.DataFrame, ref: pd.DataFrame, result: dict,
                     campaign: str, session: str, variant: str) -> pd.DataFrame:
    """Per-anchor result rows: estimate, aligned estimate, surveyed reference,
    residual. `result` is the dict from evaluation.evaluate_estimate."""
    aligned = result["tf"].apply(est[["x", "y"]].to_numpy())
    t = est.assign(x_aligned=aligned[:, 0], y_aligned=aligned[:, 1])
    t = t.merge(result["resid"], on="id")
    t = t.merge(ref.rename(columns={"x": "x_ref", "y": "y_ref", "z": "z_ref"}),
                on="id")
    t.insert(0, "campaign", campaign)
    t.insert(1, "session", session)
    t.insert(2, "variant", variant)
    return t.rename(columns={"x": "x_est", "y": "y_est"})


def markdown_table(df: pd.DataFrame) -> str:
    cells = df.map(lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    widths = [max(len(c), *(len(v) for v in cells[c])) for c in cells.columns]
    def row(vals):
        return "| " + " | ".join(v.ljust(w) for v, w in zip(vals, widths)) + " |"
    return "\n".join([row(cells.columns), row(["-" * w for w in widths]),
                      *(row(r) for r in cells.itertuples(index=False))])
