"""Generated views over the benchmark tables (results/benchmark.parquet).

The per-method results/<method>/summary.md keep their hand-written title,
methodology preamble, parameter paragraphs and verified narrative notes;
only the stats table between the marker comments

    <!-- BEGIN GENERATED (summarize.py) -->
    <!-- END GENERATED (summarize.py) -->

is a view rendered from benchmark.parquet (filtered to the benchmark
reference sessions set_b/s03 and set_a/s21, fully_meshed,
particle filter seed 0). results/overview.md keeps its hand-written frozen
factorial matrix and findings; the fully_meshed replication section and one
section per mask family (multi_hop, mobile_assisted, slam_bipartite,
each between its own marker pair) are generated here, as are the tables in
the family results/<family>/summary.md files. The generator refuses to
touch a file whose markers are missing.

Usage:
    python summarize.py                  # regenerate all views
    python summarize.py --init-markers   # one-off: wrap the existing tables
    python summarize.py --flag-anomalies # validate-and-narrate worklist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_io
import evaluation
import family_masks
from registry import METHOD_ORDER, METHOD_ORDER_STATIC, REGISTRY
from reporting import markdown_table

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA = HERE.parents[1] / "Data"

MARK_B = "<!-- BEGIN GENERATED (summarize.py) -->"
MARK_E = "<!-- END GENERATED (summarize.py) -->"
REP_B = "<!-- BEGIN GENERATED: replication (summarize.py) -->"
REP_E = "<!-- END GENERATED: replication (summarize.py) -->"

FAMILIES = ("multi_hop", "mobile_assisted", "slam_bipartite",
            "prior_sweep")  # prior_sweep: pseudo-family of the p-masks


def _fam_marks(family: str) -> tuple[str, str]:
    return (f"<!-- BEGIN GENERATED: {family} (summarize.py) -->",
            f"<!-- END GENERATED: {family} (summarize.py) -->")


VIEW_SESSIONS = (("set_b", "s03"), ("set_a", "s21"))
ANOMALY_FACTOR = 3.0
CONSISTENCY_TOL = 5e-4  # canonical multi_hop vs multi_hop_k3


def load_bench(mask: str | None = "fully_meshed") -> pd.DataFrame:
    """Benchmark rows of one mask (None = all masks, with `family` and
    multihop `k` columns attached). Seeded methods enter with seed 0 only:
    the summaries are seed-0 views, while the generated paper tables
    (Paper/tables/gen_benchmark_tables.py) average the particle filter over
    its seeds 0 to 4."""
    df = benchmark_io.load()
    df = df[df["seed"].isin([benchmark_io.DETERMINISTIC_SEED, 0])]
    if mask is not None:
        df = df[df["mask"] == mask].copy()
    else:
        df = df.copy()
        df["family"] = df["mask"].map(family_masks.family_of)
        df["k"] = df["mask"].map(family_masks.thinning_k)
    df["extras"] = df["extras_json"].map(json.loads)
    return df


def _assert_unique(df: pd.DataFrame, context: str) -> None:
    """Exactly one row per benchmark key in a view selection."""
    dup = df.groupby(benchmark_io.KEY).size()
    dup = dup[dup > 1]
    if len(dup):
        raise ValueError(f"duplicate benchmark rows in {context}: "
                         f"{dup.index.tolist()[:5]}")


def _rows(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Benchmark rows of one method in committed view order (2025 first,
    variants in registry order, seed 0 for seeded methods)."""
    spec = REGISTRY[method]
    out = []
    for campaign, session in VIEW_SESSIONS:
        for v in spec.variants:
            sel = df[(df["method"] == method) & (df["campaign"] == campaign)
                     & (df["session"] == session) & (df["variant"] == v)
                     & (df["seed"].isin([benchmark_io.DETERMINISTIC_SEED, 0]))]
            if len(sel) != 1:
                raise ValueError(f"expected exactly one row for "
                                 f"{campaign}/{session} {method}:{v}, "
                                 f"got {len(sel)}")
            out.append(sel.iloc[0])
    return pd.DataFrame(out).reset_index(drop=True)


def _pairs(r) -> str:
    return f"{int(r['n_pairs_available'])}/{int(r['n_pairs_total'])}"


def _z_spread(campaign: str, session: str) -> float:
    ref = evaluation.surveyed_reference(DATA / campaign / session)
    return float(ref["z"].max() - ref["z"].min())


# ------------------------------------------------- per-method view builders

def _view_closed_form(df):
    z = {cs: _z_spread(*cs) for cs in VIEW_SESSIONS}
    return pd.DataFrame([{
        "campaign": r["campaign"], "session": r["session"],
        "variant": r["variant"], "n_placed": r["extras"]["n_placed"],
        "n_anchors": int(r["n_anchors"]), "pairs_available": _pairs(r),
        "rmse_m": r["rmse_m"], "max_error_m": r["max_error_m"],
        "reflection": bool(r["reflection"]),
        "z_spread_ref_m": z[(r["campaign"], r["session"])],
    } for _, r in _rows(df, "closed_form").iterrows()])


def _view_mds(df):
    return pd.DataFrame([{
        "campaign": r["campaign"], "session": r["session"],
        "variant": r["variant"], "n_anchors": int(r["n_anchors"]),
        "pairs_available": _pairs(r),
        "pairs_completed": r["extras"]["n_pairs_completed"],
        "rmse_m": r["rmse_m"], "max_error_m": r["max_error_m"],
        "reflection": bool(r["reflection"]),
        "stress1": r["extras"]["stress"],
        "neg_eig_frac": r["extras"]["neg_eig_frac"],
        "sigma_m": (r["sigma_m"] if r["variant"] in ("smacof_huber",
                                                     "smacof_cauchy")
                    else float("nan")),
        "n_iter": r["extras"]["n_iter"],
    } for _, r in _rows(df, "mds").iterrows()])


def _view_factor_graph(df):
    return pd.DataFrame([{
        "campaign": r["campaign"], "session": r["session"],
        "variant": r["variant"], "n_anchors": int(r["n_anchors"]),
        "pairs_available": _pairs(r),
        "rmse_m": r["rmse_m"], "max_error_m": r["max_error_m"],
        "reflection": bool(r["reflection"]),
        "n_iter": r["extras"]["n_iter"],
        "converged": bool(r["extras"]["converged"]),
        "cost0": r["extras"]["cost0"], "cost": r["extras"]["cost"],
        "sigma_m": r["sigma_m"],
    } for _, r in _rows(df, "factor_graph").iterrows()])


def _view_ekf(df):
    return pd.DataFrame([{
        "campaign": r["campaign"], "session": r["session"],
        "variant": r["variant"],
        "init": f"{r['extras']['init']}{r['extras']['init_epochs']}",
        "n_anchors": int(r["n_anchors"]), "sigma_m": r["sigma_m"],
        "rmse_m": r["rmse_m"], "max_error_m": r["max_error_m"],
        "reflection": bool(r["reflection"]),
        "gated_frac": (f"{r['extras']['gated_frac']:.3f}"
                       if r["variant"] == "gated" else "-"),
    } for _, r in _rows(df, "ekf").iterrows()])


def _view_particle_filter(df):
    return pd.DataFrame([{
        "campaign": r["campaign"], "session": r["session"],
        "variant": r["variant"], "init": r["extras"]["init"],
        "n_anchors": int(r["n_anchors"]),
        "rmse_m": r["rmse_m"], "max_error_m": r["max_error_m"],
        "reflection": bool(r["reflection"]), "sigma_m": r["sigma_m"],
        "n_particles": r["extras"]["n_particles"],
        "sweeps": r["extras"]["sweeps"],
    } for _, r in _rows(df, "particle_filter").iterrows()])


def _view_grid_filter(df):
    return pd.DataFrame([{
        "campaign": r["campaign"], "session": r["session"],
        "variant": ("mixture(gaussian)" if r["variant"] == "mixture"
                    else r["variant"]),
        "n_anchors": int(r["n_anchors"]),
        "rmse_m": r["rmse_m"], "max_error_m": r["max_error_m"],
        "reflection": bool(r["reflection"]),
        "grid_res_m": r["extras"]["grid_res_m"],
        "likelihood_sigma_m": r["sigma_m"],
    } for _, r in _rows(df, "grid_filter").iterrows()])


VIEW_BUILDERS = {
    "closed_form": _view_closed_form,
    "mds": _view_mds,
    "factor_graph": _view_factor_graph,
    "ekf": _view_ekf,
    "particle_filter": _view_particle_filter,
    "grid_filter": _view_grid_filter,
}


# --------------------------------------------------------- marker plumbing

def replace_between(text: str, begin: str, end: str, payload: str) -> str:
    ib, ie = text.find(begin), text.find(end)
    if ib < 0 or ie < 0 or ie < ib:
        raise ValueError(f"markers {begin!r} / {end!r} not found - refusing "
                         "to write (run --init-markers first)")
    return text[:ib + len(begin)] + "\n" + payload + "\n" + text[ie:]


def _wrap_first_table(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(MARK_B in ln for ln in lines):
        print(f"   markers already present: {path.name}")
        return
    start = next(i for i, ln in enumerate(lines) if ln.startswith("|"))
    stop = start
    while stop < len(lines) and lines[stop].startswith("|"):
        stop += 1
    out = lines[:start] + [MARK_B] + lines[start:stop] + [MARK_E] + lines[stop:]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"   wrapped stats table: {path}")


def init_markers() -> None:
    for method in METHOD_ORDER_STATIC:
        _wrap_first_table(RESULTS / method / "summary.md")
    ov = RESULTS / "overview.md"
    text = ov.read_text(encoding="utf-8")
    changed = False
    if REP_B not in text:
        text = (text.rstrip("\n")
                + f"\n\n## fully_meshed replication over all sessions\n\n{REP_B}\n{REP_E}\n")
        changed = True
        print(f"   appended replication markers: {ov}")
    for family in FAMILIES:
        fb, fe = _fam_marks(family)
        if fb not in text:
            text = (text.rstrip("\n")
                    + f"\n\n## {family} family\n\n{fb}\n{fe}\n")
            changed = True
            print(f"   appended {family} markers: {ov}")
    if changed:
        ov.write_text(text, encoding="utf-8")
    else:
        print("   all overview markers already present: overview.md")


# ------------------------------------------------------- replication views

def _fmt(v: float) -> str:
    return f"{v:.3f}" if np.isfinite(v) else "-"


def _text(v) -> str:
    """A label cell, '-' when missing (None under pandas 2, NaN under 3)."""
    return v if isinstance(v, str) and v else "-"


def _variant_rows(df: pd.DataFrame):
    for method in METHOD_ORDER:
        for v in REGISTRY[method].variants:
            sel = df[(df["method"] == method) & (df["variant"] == v)]
            if len(sel):
                yield method, v, sel


def replication_tables(df: pd.DataFrame) -> str:
    lines = []

    c24 = df[df["campaign"] == "set_a"]
    n_sessions = c24["session"].nunique()
    rows = []
    for method, v, sel in _variant_rows(c24):
        ok = sel[sel["status"] == "ok"]
        r = ok["rmse_m"]
        worst = ok.loc[r.idxmax()] if len(ok) else None
        rows.append({
            "method": method, "variant": v,
            "kernel": _text(sel.iloc[0]["kernel"]),
            "median_rmse_m": _fmt(r.median()),
            "iqr_m": f"{r.quantile(0.25):.3f}-{r.quantile(0.75):.3f}"
                     if len(ok) else "-",
            "worst_session": (f"{worst['session']} ({worst['rmse_m']:.3f})"
                              if worst is not None else "-"),
            "n_ok": f"{len(ok)}/{n_sessions}",
        })
    lines += [f"### set_a ({n_sessions} sessions: s21, s22, "
              "s01-s20) - RMSE m", "", markdown_table(pd.DataFrame(rows)), ""]

    c25 = df[df["campaign"] == "set_b"]
    sess25 = [s for s in ("s03", "s01", "s02")
              if s in set(c25["session"])]
    rows = []
    for method, v, sel in _variant_rows(c25):
        row = {"method": method, "variant": v,
               "kernel": _text(sel.iloc[0]["kernel"])}
        for s in sess25:
            cell = sel[sel["session"] == s]
            row[s] = (_fmt(cell.iloc[0]["rmse_m"])
                      if len(cell) and cell.iloc[0]["status"] == "ok" else "-")
        rows.append(row)
    lines += [f"### set_b ({len(sess25)} sessions) - RMSE m", "",
              markdown_table(pd.DataFrame(rows)), ""]

    err = df[df["status"] != "ok"]
    grid = df[(df["method"] == "grid_filter") & (df["status"] == "ok")]
    strides = sorted({json.loads(j).get("stride") for j in grid["extras_json"]}
                     ) if len(grid) else []
    lines += ["Generated from results/benchmark.parquet (fully_meshed, shared "
              "session sigma, 1x scale,",
              "particle filter seed 0). Grid filter epoch subsampling: "
              "stride = n_epochs // 75",
              f"(session-derived; strides used {strides}), total grid runtime "
              f"{grid['runtime_s'].sum():.0f} s.",
              f"Cells with status != ok: "
              + (", ".join(f"{r['campaign']}/{r['session']} "
                           f"{r['method']}:{r['variant']} ({r['error']})"
                           for _, r in err.iterrows()) if len(err) else "none")
              + "."]
    return "\n".join(lines)


# -------------------------------------------------------- family sections

def _cell(r) -> str:
    """RMSE cell of a family pivot: value, or the non-ok status short."""
    if r is None:
        return "-"
    if r["status"] == "ok":
        return _fmt(r["rmse_m"])
    return {"disconnected": "disc.", "undefined": "undef."}.get(
        r["status"], "ERR")


def _one(sel: pd.DataFrame, context: str):
    """The single row of a cell selection, or None if absent."""
    if len(sel) > 1:
        raise ValueError(f"expected at most one row for {context}, "
                         f"got {len(sel)}")
    return sel.iloc[0] if len(sel) else None


def check_multihop_consistency(fam: pd.DataFrame) -> None:
    """Canonical multi_hop must equal multi_hop_k3 (the
    standing spec-dict-vs-YAML consistency check)."""
    canon = fam[fam["mask"] == "multi_hop"]
    k3 = fam[fam["mask"] == "multi_hop_k3"]
    on = ["campaign", "session", "method", "variant", "seed"]
    both = canon.merge(k3, on=on, suffixes=("_c", "_k3"))
    for _, r in both.iterrows():
        if r["status_c"] != r["status_k3"]:
            raise ValueError(f"multihop k3 consistency: status differs at "
                             f"{tuple(r[c] for c in on)}: "
                             f"{r['status_c']} vs {r['status_k3']}")
        if r["status_c"] == "ok" and \
                abs(r["rmse_m_c"] - r["rmse_m_k3"]) > CONSISTENCY_TOL:
            raise ValueError(f"multihop k3 consistency: rmse differs at "
                             f"{tuple(r[c] for c in on)}: "
                             f"{r['rmse_m_c']:.4f} vs {r['rmse_m_k3']:.4f}")


def multihop_section(df_all: pd.DataFrame) -> str:
    fam_all = df_all[df_all["family"] == "multi_hop"]
    # the prior-sweep masks are a separate arm (prior_section), not part of
    # the thinning narrative
    fam_all = fam_all[fam_all["mask"].map(family_masks.prior_count).isna()]
    # the range-kNN arm carries this section's sweep narrative; the seeded
    # random-thinning ablation arm is summarized separately below
    rand = fam_all[fam_all["mask"].str.startswith("multi_hop_random")]
    fam = fam_all[~fam_all["mask"].str.startswith("multi_hop_random")]
    if not len(fam):
        return "(no results yet)"
    _assert_unique(fam, "multihop view")
    check_multihop_consistency(fam)
    full = df_all[df_all["mask"] == "fully_meshed"]
    lines = []
    swept = fam[fam["mask"] != "multi_hop"]
    ks = sorted(swept["k"].dropna().unique(), reverse=True)
    for campaign, session in VIEW_SESSIONS:
        sub = swept[(swept["campaign"] == campaign)
                    & (swept["session"] == session)]
        if not len(sub):
            continue
        rows = []
        for method in METHOD_ORDER_STATIC:
            for v in REGISTRY[method].variants:
                mv = sub[(sub["method"] == method) & (sub["variant"] == v)]
                if not len(mv):
                    continue
                fr = _one(full[(full["campaign"] == campaign)
                               & (full["session"] == session)
                               & (full["method"] == method)
                               & (full["variant"] == v)],
                          f"fully_meshed ref {method}:{v}")
                row = {"method": method, "variant": v,
                       "full": _cell(fr)}
                for k in ks:
                    row[f"k{int(k)}"] = _cell(
                        _one(mv[mv["k"] == k], f"k{k} {method}:{v}"))
                rows.append(row)
        lines += [f"### {campaign}/{session} - RMSE m by thinning k "
                  "(full = fully_meshed reference; disc. = disconnected; "
                  "undef. = undefined construction)", "",
                  markdown_table(pd.DataFrame(rows)), ""]
    canon = fam[(fam["mask"] == "multi_hop")]
    if len(canon):
        rows = []
        for (campaign, method, v), grp in canon.groupby(
                ["campaign", "method", "variant"], sort=False):
            ok = grp[grp["status"] == "ok"]
            rows.append({
                "campaign": campaign, "method": method, "variant": v,
                "median_rmse_m": _fmt(ok["rmse_m"].median()),
                "worst_session": (f"{ok.loc[ok['rmse_m'].idxmax(), 'session']}"
                                  f" ({ok['rmse_m'].max():.3f})"
                                  if len(ok) else "-"),
                "n_ok": f"{len(ok)}/{len(grp)}",
                "n_disconnected": int((grp["status"] == "disconnected").sum()),
            })
        lines += ["### canonical multi_hop (k=3) over all sessions",
                  "", markdown_table(pd.DataFrame(rows)), ""]
    if len(rand):
        robust = rand[((rand["method"] == "factor_graph")
                       & rand["variant"].isin(["huber", "cauchy"]))
                      | ((rand["method"] == "mds")
                         & rand["variant"].isin(["smacof_huber",
                                                 "smacof_cauchy"]))]
        rows = []
        for campaign, session in VIEW_SESSIONS:
            sub = robust[(robust["campaign"] == campaign)
                         & (robust["session"] == session)]
            if not len(sub):
                continue
            row = {"campaign/session": f"{campaign}/{session}"}
            for k in sorted(sub["k"].dropna().unique()):
                cell = sub[(sub["k"] == k) & (sub["status"] == "ok")]
                row[f"k{int(k)}"] = (_fmt(cell["rmse_m"].median())
                                     if len(cell) else "-")
            rows.append(row)
        lines += ["", "### random-thinning ablation - robust-batch median "
                  "RMSE m (seeded-uniform selection, 3 seeds pooled)", "",
                  markdown_table(pd.DataFrame(rows)), "",
                  "Same union-of-per-node construction as the k-NN sweep "
                  "with the selection criterion",
                  "changed from nearest-by-median-range to seeded-uniform; "
                  "the arm isolates the",
                  "NLOS-filtering side effect of range-based thinning from "
                  "the rigidity effect.", ""]
    lines += ["Generated from results/benchmark.parquet (multihop family; "
              "shared per-session fully_meshed sigma).",
              "Canonical multi_hop verified equal to "
              f"multi_hop_k3 within {CONSISTENCY_TOL:g} m."]
    return "\n".join(lines)


def mobile_section(df_all: pd.DataFrame) -> str:
    # quasi-static arm (set_a quasi-static sessions, marker metric); the 2025 dynamic
    # MA cells (fg_slam + E_aa, trajectory metrics) ride with the slam
    # comparison and are not part of this section's marker narrative
    fam = df_all[(df_all["family"] == "mobile_assisted")
                 & (df_all["campaign"] == "set_a")]
    if not len(fam):
        return "(no results yet)"
    _assert_unique(fam, "mobile view")
    full = df_all[df_all["mask"] == "fully_meshed"]
    sessions = sorted(fam["session"].unique())
    full = full[full["campaign"].eq("set_a")
                & full["session"].isin(sessions)]
    rows = []
    for method in METHOD_ORDER_STATIC:
        for v in REGISTRY[method].variants:
            mv = fam[(fam["method"] == method) & (fam["variant"] == v)]
            if not len(mv):
                continue
            ok = mv[mv["status"] == "ok"]
            ver = ok[ok["mobile_truth_quality"] == "verified"]
            fm = full[(full["method"] == method) & (full["variant"] == v)]
            fm_ok = fm[fm["status"] == "ok"]
            rows.append({
                "method": method, "variant": v,
                "anchor_median_rmse_m": _fmt(ok["rmse_m"].median()),
                "fully_meshed_median_rmse_m": _fmt(fm_ok["rmse_m"].median()),
                "mobile_median_err_m": _fmt(ver["mobile_error_m"].median()),
                "mobile_worst": (f"{ver.loc[ver['mobile_error_m'].idxmax(), 'session']}"
                                 f" ({ver['mobile_error_m'].max():.3f})"
                                 if len(ver) and
                                 np.isfinite(ver["mobile_error_m"]).any()
                                 else "-"),
                "n_ok": f"{len(ok)}/{len(mv)}",
            })
    lines = ["### set_a quasi-static sessions - anchor RMSE (primary) and mobile "
             "error vs marker (secondary)", "",
             markdown_table(pd.DataFrame(rows)), ""]
    unrel = fam[(fam["mobile_truth_quality"] != "verified")
                & (fam["status"] == "ok")]
    if len(unrel):
        rows = [{"session": s,
                 "mobile_err_range_m": f"{g['mobile_error_m'].min():.2f}-"
                                       f"{g['mobile_error_m'].max():.2f}",
                 "note": "truth_quality unreliable - excluded from all "
                         "mobile aggregates"}
                for s, g in unrel.groupby("session")]
        lines += ["### sessions with unreliable marker truth", "",
                  markdown_table(pd.DataFrame(rows)), ""]
    lines += ["Generated from results/benchmark.parquet (mobile_assisted; "
              "mobile error = xy distance to the surveyed marker after the "
              "anchor-fitted alignment;",
              "sessions flagged truth_quality: unreliable are excluded from "
              "every mobile aggregate; fully_meshed medians cover the same "
              "quasi-static sessions)."]
    return "\n".join(lines)


def slam_section(df_all: pd.DataFrame) -> str:
    fam = df_all[df_all["family"] == "slam_bipartite"]
    if not len(fam):
        return "(no results yet)"
    _assert_unique(fam, "slam view")
    rows = []
    for method in ("ekf_slam", "fg_slam"):
        for v in REGISTRY[method].variants:
            for session in sorted(fam["session"].unique()):
                r = _one(fam[(fam["method"] == method)
                             & (fam["variant"] == v)
                             & (fam["session"] == session)],
                         f"slam {method}:{v} {session}")
                if r is None:
                    continue
                ok = r["status"] == "ok"
                rows.append({
                    "method": method, "variant": v, "session": session,
                    "anchor_rmse_m": _cell(r),
                    "ate_rmse_m": _fmt(r["ate_rmse_m"]) if ok else "-",
                    "rpe_rmse_m": _fmt(r["rpe_rmse_m"]) if ok else "-",
                    "n_truth_matched": (int(r["n_truth_matched"])
                                        if ok and
                                        np.isfinite(r["n_truth_matched"])
                                        else "-"),
                })
    lines = ["### set_b dyn sessions - anchor RMSE (primary), "
             "checkpoint ATE and within-segment RPE (secondary)", "",
             markdown_table(pd.DataFrame(rows)), ""]

    # edge-set contrast: the same fg_slam engine on the mobile_assisted view
    # (bipartite factors + per-pair-median anchor-anchor factors)
    ma_dyn = df_all[(df_all["mask"] == "mobile_assisted")
                    & (df_all["campaign"] == "set_b")]
    if len(ma_dyn):
        rows = []
        for v in REGISTRY["fg_slam"].variants:
            for session in sorted(ma_dyn["session"].unique()):
                r = _one(ma_dyn[(ma_dyn["variant"] == v)
                                & (ma_dyn["session"] == session)],
                         f"ma-dyn fg_slam:{v} {session}")
                b = _one(fam[(fam["method"] == "fg_slam")
                             & (fam["variant"] == v)
                             & (fam["session"] == session)],
                         f"bipartite ref fg_slam:{v} {session}")
                if r is None:
                    continue
                rows.append({
                    "variant": v, "session": session,
                    "anchor_rmse_m (+E_aa)": _cell(r),
                    "anchor_rmse_m (bipartite)": _cell(b),
                    "ate_rmse_m (+E_aa)": (_fmt(r["ate_rmse_m"])
                                           if r["status"] == "ok" else "-"),
                    "ate_rmse_m (bipartite)": (_fmt(b["ate_rmse_m"])
                                               if b is not None
                                               and b["status"] == "ok"
                                               else "-"),
                })
        lines += ["### mobile_assisted dynamic views - fg_slam with "
                  "anchor-anchor factors vs the bipartite run", "",
                  markdown_table(pd.DataFrame(rows)), ""]

    lines += ["Generated from results/benchmark.parquet (slam_bipartite "
              "and the mobile_assisted dynamic arm; trajectory metrics "
              "only at Vicon-covered epochs, never interpolated across "
              "gaps)."]
    return "\n".join(lines)


def check_prior_consistency(pr: pd.DataFrame, df_all: pd.DataFrame,
                            cb_all: pd.DataFrame) -> None:
    """fully_meshed_p0 / multi_hop_p0 must equal the blind factor graph on
    the base mask (the standing blind-consistency arm of the sweep).

    Asserted only on views the FIM certifies as rigid: on a flexible view
    (crlb status "flexible") the minimum is a continuous family, and the
    reduced (2n-3) and full (2n, damped) parameterizations legitimately
    select different flex configurations - that disagreement is the
    identifiability failure made visible, not a solver defect."""
    blind = df_all[df_all["mask"].isin(("fully_meshed", "multi_hop"))
                   & (df_all["method"] == "factor_graph")]
    for _, r in pr[pr["p"] == 0].iterrows():
        c = _one(cb_all[(cb_all["campaign"] == r["campaign"])
                        & (cb_all["session"] == r["session"])
                        & (cb_all["mask"] == r["mask"])],
                 f"crlb row {r['mask']}")
        if c is not None and c["status"] == "flexible":
            continue
        b = _one(blind[(blind["campaign"] == r["campaign"])
                       & (blind["session"] == r["session"])
                       & (blind["mask"] == r["base"])
                       & (blind["variant"] == r["variant"])],
                 f"blind ref {r['base']} {r['variant']}")
        if b is None or abs(r["rmse_m"] - b["rmse_m"]) > CONSISTENCY_TOL:
            got = "missing" if b is None else f"{b['rmse_m']:.4f}"
            raise ValueError(f"prior-sweep p0 consistency: "
                             f"{r['campaign']}/{r['session']} {r['base']} "
                             f"{r['variant']}: {r['rmse_m']:.4f} vs blind "
                             f"factor_graph {got}")


def prior_section(df_all: pd.DataFrame) -> str:
    pr = df_all[df_all["mask"].map(family_masks.prior_count).notna()].copy()
    if not len(pr):
        return "(no results yet)"
    _assert_unique(pr, "prior view")
    pr["p"] = pr["mask"].map(family_masks.prior_count)
    pr["base"] = pr["mask"].map(family_masks.family_of)
    if (pr["method"] != "fg_anchored").any():
        raise ValueError("non-fg_anchored rows on prior-sweep masks")
    cb_all = benchmark_io.load("crlb")
    check_prior_consistency(pr, df_all, cb_all)
    bad = pr[(pr["p"] < 2) & pr["abs_rmse_m"].notna()]
    if len(bad):
        raise ValueError("absolute metric present below |K| = 2")
    bad = pr[(pr["p"] >= 2) & (pr["status"] == "ok")
             & pr["abs_rmse_m"].isna()]
    if len(bad):
        raise ValueError("absolute metric missing on an ok |K| >= 2 cell")
    cols = (["p0", "p1", "p2", "p3", "p2_abs", "p3_abs"])
    lines = []
    for campaign, session in VIEW_SESSIONS:
        for base in ("fully_meshed", "multi_hop"):
            sub = pr[(pr["campaign"] == campaign)
                     & (pr["session"] == session) & (pr["base"] == base)]
            if not len(sub):
                continue
            rows = []
            for v in REGISTRY["fg_anchored"].variants:
                row = {"variant": v, **{c: "-" for c in cols}}
                for p in range(4):
                    r = _one(sub[(sub["variant"] == v) & (sub["p"] == p)],
                             f"{base}_p{p} {v}")
                    row[f"p{p}"] = _cell(r)
                    if p >= 2 and r is not None and r["status"] == "ok":
                        row[f"p{p}_abs"] = _fmt(r["abs_rmse_m"])
                rows.append(row)
            cb = cb_all[(cb_all["campaign"] == campaign)
                        & (cb_all["session"] == session)]
            brow = {"variant": "CRLB aligned", **{c: "-" for c in cols}}
            arow = {"variant": "CRLB absolute", **{c: "-" for c in cols}}
            for p in range(4):
                c = _one(cb[cb["mask"] == f"{base}_p{p}"],
                         f"crlb {base}_p{p}")
                if c is None:
                    continue
                flex = " (flex)" if c["status"] == "flexible" else ""
                brow[f"p{p}"] = _fmt(c["crlb_rmse_m"]) + flex
                if p >= 2:
                    arow[f"p{p}_abs"] = _fmt(c["crlb_abs_rmse_m"]) + flex
            rows += [brow, arow]
            lines += [f"### {campaign}/{session} {base} - fg_anchored "
                      "RMSE m by |K| (aligned p0-p3; absolute from |K| = 2; "
                      "CRLB rows = Gaussian floors at the session sigma)",
                      "", markdown_table(pd.DataFrame(rows)), ""]
    lines += ["Generated from results/benchmark.parquet (prior-sweep masks, "
              "fg_anchored roster, shared per-session fully_meshed sigma) "
              "and results/crlb.parquet",
              "(bounds consume the mask's subset; (flex) marks views whose "
              "FIM nullity exceeds the gauge",
              "expectation, where the absolute bound is undefined). On "
              "rigid views p0 is verified equal",
              f"to the blind factor_graph within {CONSISTENCY_TOL:g} m; on "
              "flexible views the reduced and full",
              "parameterizations may select different flex configurations "
              "and the check is not asserted.",
              "The absolute metric is defined exactly from |K| = 2 on."]
    return "\n".join(lines)


FAMILY_SECTIONS = {"multi_hop": multihop_section,
                   "mobile_assisted": mobile_section,
                   "slam_bipartite": slam_section,
                   "prior_sweep": prior_section}


# ------------------------------------------------------------ anomaly flags

def flag_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(anomaly worklist, expected structural outcomes). `df` covers all
    masks (load_bench(mask=None)); grouping is per (campaign, family,
    method, variant), so each family's cells are judged against their own
    scale. `disconnected` and `undefined` cells are structural results of
    low-k thinning, listed separately, not anomalies."""
    flags, structural = [], []
    for (campaign, family, method, variant), grp in df.groupby(
            ["campaign", "family", "method", "variant"]):
        ok = grp[grp["status"] == "ok"]
        med = ok["rmse_m"].median()
        ver = ok[ok["mobile_truth_quality"] == "verified"] \
            if family == "mobile_assisted" else ok.iloc[0:0]
        med_mob = ver["mobile_error_m"].median() if len(ver) else np.nan
        for _, r in grp.iterrows():
            if r["status"] in ("disconnected", "undefined"):
                structural.append({"campaign": campaign, "family": family,
                                   "session": r["session"], "mask": r["mask"],
                                   "method": method, "variant": variant})
                continue
            why = []
            if r["status"] != "ok":
                why.append(f"status={r['status']}: {r['error']}")
            else:
                if np.isfinite(med) and med > 0 and \
                        r["rmse_m"] > ANOMALY_FACTOR * med:
                    why.append(f"rmse {r['rmse_m']:.3f} > {ANOMALY_FACTOR:g}x "
                               f"family median {med:.3f}")
                if bool(r["reflection"]):
                    why.append("reflection=True")
                if (r["mobile_truth_quality"] == "verified"
                        and np.isfinite(med_mob) and med_mob > 0
                        and r["mobile_error_m"] > ANOMALY_FACTOR * med_mob):
                    why.append(f"mobile_error {r['mobile_error_m']:.3f} > "
                               f"{ANOMALY_FACTOR:g}x median {med_mob:.3f}")
            if why:
                flags.append({"campaign": campaign, "session": r["session"],
                              "mask": r["mask"], "method": method,
                              "variant": variant, "rmse_m": r["rmse_m"],
                              "why": "; ".join(why)})
    out = pd.DataFrame(flags)
    if len(out):
        out = out.sort_values(["campaign", "mask", "method", "variant",
                               "session"], kind="stable").reset_index(drop=True)
    return out, pd.DataFrame(structural)


# --------------------------------------------------------------------- main

def regenerate() -> None:
    df_all = load_bench(mask=None)
    df = df_all[df_all["mask"] == "fully_meshed"]
    for method in METHOD_ORDER_STATIC:
        path = RESULTS / method / "summary.md"
        table = markdown_table(VIEW_BUILDERS[method](df))
        text = replace_between(path.read_text(encoding="utf-8"),
                               MARK_B, MARK_E, table)
        path.write_text(text, encoding="utf-8")
        print(f"-> regenerated view: {path}")
    ov = RESULTS / "overview.md"
    text = replace_between(ov.read_text(encoding="utf-8"),
                           REP_B, REP_E, replication_tables(df))
    for family in FAMILIES:
        payload = FAMILY_SECTIONS[family](df_all)
        fb, fe = _fam_marks(family)
        text = replace_between(text, fb, fe, payload)
        fpath = RESULTS / family / "summary.md"
        ftext = replace_between(fpath.read_text(encoding="utf-8"),
                                MARK_B, MARK_E, payload)
        fpath.write_text(ftext, encoding="utf-8")
        print(f"-> regenerated family view: {fpath}")
    ov.write_text(text, encoding="utf-8")
    print(f"-> regenerated overview sections: {ov}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--init-markers", action="store_true")
    ap.add_argument("--flag-anomalies", action="store_true")
    args = ap.parse_args()
    if args.init_markers:
        init_markers()
        return
    if args.flag_anomalies:
        flags, structural = flag_anomalies(load_bench(mask=None))
        if len(structural):
            print("expected structural outcomes (disconnected thinned "
                  "views):")
            print(structural.to_string(index=False))
            print()
        if len(flags):
            print(flags.to_string(index=False))
            print(f"{len(flags)} flagged rows - root-cause before reporting.")
        else:
            print("no anomalies flagged.")
        return
    regenerate()


if __name__ == "__main__":
    main()
