"""Per-session QA: data-quality report (markdown + figures) for a canonical session.

Answers the campaign's verification questions: does the mobile node move as
expected, which node pairs range reliably, do measured anchor-anchor ranges
agree with the surveyed geometry, is the ground truth internally consistent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .load import load_session


def _heatmap(ax, mat: pd.DataFrame, title: str, fmt: str = "{:.0f}", cmap="viridis"):
    im = ax.imshow(mat.to_numpy(dtype=float), cmap=cmap)
    ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(mat.index)), mat.index, fontsize=7)
    ax.set_title(title, fontsize=9)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            v = mat.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.5,
                        color="white")
    return im


def run_qa(session_dir: str | Path, make_plots: bool = True,
           out_dir: str | Path | None = None) -> Path:
    """Write report.md (+ figures) for one session directory; returns the path.

    `out_dir` receives the files. The default is the session's own qa/
    folder, which overwrites the shipped report. Text files are written with
    LF line endings on every platform."""
    sdir = Path(session_dir)
    g = load_session(sdir)
    qa_dir = Path(out_dir) if out_dir is not None else sdir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    meta = g.meta
    ids = sorted(set(g.edges["from_id"]) | set(g.edges["to_id"]))
    mobile_ids = set(g.nodes.loc[g.nodes["role"] == "mobile", "id"])
    has_time = meta.get("t_start") is not None
    span = (meta["t_end"] - meta["t_start"]) if has_time else None

    lines = [f"# QA report - {meta['session_id']} ({meta['motion']})", ""]
    lines += [f"- epochs: {meta['counts']['range_epochs']}, directed ranges: "
              f"{meta['counts']['directed_ranges']} "
              f"({meta['counts']['dropped_invalid']} invalid dropped at conversion)"]
    if has_time:
        lines += [f"- span: {span:.1f} s, mean epoch rate: "
                  f"{meta['counts']['range_epochs'] / span:.2f} Hz",
                  f"- clock: offset {meta['clock']['offset_s']:.3f} s "
                  f"+/- {meta['clock']['uncertainty_s']} s (method {meta['clock']['method']}, "
                  f"n={meta['clock']['n_matched']}; logger queue lag reached "
                  f"{meta['clock']['queue_lag_max_s']:.1f} s -> broker times unusable directly)"]
    else:
        lines += ["- no timestamps in this campaign's logs; epoch ordering via seq"]
    if "truth_quality" in meta:
        v = meta.get("verification", {})
        scr, rt = v.get("screen", {}), v.get("range_test", {})
        detail = f"screen: multilateration deviation {scr.get('deviation_m', '?')} m"
        if rt:
            lo, hi = rt["reference_range_m"]
            detail += (f", range test: median line-of-sight residual "
                       f"{rt['median_los_residual_m']:+.3f} m, reference "
                       f"{lo:+.3f} to {hi:+.3f} m")
        lines += [f"- ground-truth quality: **{meta['truth_quality']}** ({detail})"]
    lines += [""]

    # --- per-pair availability -------------------------------------------------
    n_epochs = meta["counts"]["range_epochs"]
    counts = (g.edges.groupby(["from_id", "to_id"]).size()
              .unstack(fill_value=0).reindex(index=ids, columns=ids))
    avail = counts / n_epochs * 100.0
    worst = avail.stack().sort_values()
    worst = worst[~worst.index.map(lambda p: p[0] == p[1])]
    lines += ["## Per-pair availability",
              f"- directed pairs observed: {(counts.to_numpy() > 0).sum()} "
              f"of {len(ids) * (len(ids) - 1)} possible",
              "- weakest 5 directed pairs (availability % of epochs): " +
              ", ".join(f"{a}->{b} {v:.0f}%" for (a, b), v in worst.head(5).items()), ""]

    # --- mobility ---------------------------------------------------------------
    tr = g.truth
    if mobile_ids:
        mv = g.edges[g.edges["from_id"].isin(mobile_ids) | g.edges["to_id"].isin(mobile_ids)]
        pair_std = mv.groupby(["from_id", "to_id"])["range_m"].std()
        lines += [f"## Mobile node ({', '.join(sorted(mobile_ids))})",
                  f"- mobile<->anchor range std: median {pair_std.median():.2f} m, "
                  f"max {pair_std.max():.2f} m"]
    if len(tr) == 1:
        t0 = tr.iloc[0]
        lines += [f"- quasi-static truth: {t0['node_id']} at "
                  f"({t0['x']:.2f}, {t0['y']:.2f}, {t0['z']:.2f}) (surveyed marker, "
                  f"z at the tag antenna)"]
    elif len(tr) > 1 and tr["t"].notna().all():
        xyz = tr[["x", "y", "z"]].to_numpy()
        step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        path_len = float(step.sum())
        bbox = xyz.max(axis=0) - xyz.min(axis=0)
        dt = np.diff(tr["t"].to_numpy())
        gaps = dt[dt > 1.0]
        # physically implausible motion between CLOSE poses = labeling anomaly;
        # large steps across gaps are just missing coverage
        with np.errstate(divide="ignore"):
            speed = step / dt
        n_anom = int(((speed > 5.0) & (dt < 0.5)).sum())
        mean_rate = (len(tr) - 1) / (tr["t"].iloc[-1] - tr["t"].iloc[0])
        lines += [f"- Vicon: {len(tr)} poses, burst rate {1 / np.median(dt):.0f} Hz, "
                  f"mean rate {mean_rate:.1f} Hz, path length {path_len:.1f} m, "
                  f"bbox {bbox[0]:.1f} x {bbox[1]:.1f} x {bbox[2]:.2f} m",
                  f"- Vicon coverage: {len(gaps)} gaps > 1 s (total {gaps.sum():.1f} s of "
                  f"{tr['t'].iloc[-1] - tr['t'].iloc[0]:.1f} s), max gap {dt.max():.2f} s",
                  f"- Vicon labeling: {n_anom} implausible-speed anomalies (>5 m/s between "
                  f"close poses), frames strictly increasing: "
                  f"{bool(tr['frame'].is_monotonic_increasing)}"]
    lines += [""]

    # --- geometry sanity ---------------------------------------------------------
    anchors = g.nodes[(g.nodes["role"] == "anchor") & (g.nodes["prior_type"] == "surveyed")]
    pos = {r.id: np.array([r.x, r.y, r.z]) for r in anchors.itertuples()}
    sym = g.symmetrized()
    aa = sym[sym["node_a"].isin(pos) & sym["node_b"].isin(pos)]
    med = aa.groupby(["node_a", "node_b"])["range_m"].median().reset_index()
    med["surveyed_m"] = [np.linalg.norm(pos[a] - pos[b])
                         for a, b in zip(med["node_a"], med["node_b"])]
    med["residual_m"] = med["range_m"] - med["surveyed_m"]
    lines += ["## Geometry sanity (median measured vs surveyed anchor-anchor distance)",
              f"- {len(med)} anchor pairs; residual mean {med['residual_m'].mean():+.3f} m, "
              f"std {med['residual_m'].std():.3f} m, max |.| {med['residual_m'].abs().max():.3f} m",
              "- worst 5 pairs: " + ", ".join(
                  f"{r.node_a}-{r.node_b} {r.residual_m:+.2f} m"
                  for r in med.reindex(med["residual_m"].abs()
                                       .sort_values(ascending=False).index).head(5).itertuples()), ""]

    # --- figures ------------------------------------------------------------------
    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        _heatmap(axes[0], counts, "directed range count per pair")
        _heatmap(axes[1], avail, "availability [% of epochs]", fmt="{:.0f}")
        fig.tight_layout()
        fig.savefig(qa_dir / "pair_availability.png", dpi=150)
        plt.close(fig)

        has_traj = len(tr) > 1 and tr["t"].notna().all()
        if has_time or has_traj:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            if has_time:
                t0 = g.edges["t"].min()
                axes[0].hist(g.edges["t"] - t0, bins=int(span), color="#356a9c")
                axes[0].set_xlabel("t - t0 [s]"), axes[0].set_ylabel("directed ranges / s")
                axes[0].set_title("range rate over time", fontsize=9)
            if has_traj:
                sc = axes[1].scatter(tr["x"], tr["y"], c=tr["t"] - tr["t"].min(), s=2, cmap="plasma")
                fig.colorbar(sc, ax=axes[1], label="t - t0 [s]")
                axes[1].set_aspect("equal")
                axes[1].set_xlabel("x_vicon [m]"), axes[1].set_ylabel("y_vicon [m]")
                axes[1].set_title("mobile trajectory (Vicon frame)", fontsize=9)
            fig.tight_layout()
            fig.savefig(qa_dir / "rate_and_trajectory.png", dpi=150)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4))
        resid = med.sort_values("surveyed_m")
        ax.stem(resid["surveyed_m"], resid["residual_m"])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("surveyed distance [m]"), ax.set_ylabel("median measured - surveyed [m]")
        ax.set_title("anchor-anchor range residuals vs Total Station", fontsize=9)
        fig.tight_layout()
        fig.savefig(qa_dir / "geometry_residuals.png", dpi=150)
        plt.close(fig)

        lines += ["## Figures", "![pairs](pair_availability.png)"]
        if has_time or has_traj:
            lines += ["![rate](rate_and_trajectory.png)"]
        lines += ["![geometry](geometry_residuals.png)", ""]

    med.to_csv(qa_dir / "geometry_residuals.csv", index=False, lineterminator="\n")
    report = qa_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return report
