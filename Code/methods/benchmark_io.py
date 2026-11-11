"""Long-format benchmark tables with idempotent keyed upserts.

Tables under results/, each as parquet (canonical) + CSV mirror (diff-
friendly), always rewritten together from one in-memory frame (the three
cell tables below, plus the per-view tables crlb and rigidity and the
oracle_start and sigma_sweep controls):

- benchmark.parquet / .csv: one row per benchmark cell
  (campaign, session, mask, method, variant, seed). Family-specific metric
  columns (mobile_*, ate_*, rpe_*, n_truth_matched) are NaN / "" where a
  family does not define them; `mask` carries the full parameterized mask
  name (e.g. multi_hop_k2).
- benchmark_estimates.parquet / .csv: per-node estimates of every cell
  (same key + node id; in the mobile_assisted family the mobile's row rides
  here next to the anchors, with the marker as its reference).
- benchmark_trajectories.parquet / .csv: per-epoch mobile trajectory of the
  trajectory-metric cells (slam_bipartite, and mobile_assisted on dynamic
  sessions; same key + seq), estimate in the solver frame and in
  ENU, matched truth where Vicon covers the epoch (segment = -1, err_m = NaN
  where unmatched).

`seed` is -1 for deterministic methods (a real int keeps the merge key
NA-free); the particle filter records its actual seed. Upserting replaces
rows whose key already exists, so partial reruns and resumes are safe;
frames are reindexed to the schema on write, which NaN-fills columns added
after rows were first stored.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"

KEY = ["campaign", "session", "mask", "method", "variant", "seed"]

SUMMARY_COLUMNS = KEY + [
    "kernel", "mechanism", "sigma_m", "sigma_mult",
    "rmse_m", "max_error_m", "reflection",
    "abs_rmse_m", "abs_max_error_m", "abs_reflection", "n_priors",
    "n_anchors", "n_aligned", "n_pairs_available", "n_pairs_total",
    "n_epochs", "n_components",
    "mobile_error_m", "mobile_truth_quality",
    "ate_rmse_m", "rpe_rmse_m", "n_truth_matched",
    "runtime_s", "status", "error", "run_at", "extras_json",
]

ESTIMATE_COLUMNS = KEY + [
    "id", "x_est", "y_est", "x_aligned", "y_aligned",
    "x_ref", "y_ref", "z_ref", "error_m",
]

TRAJECTORY_COLUMNS = KEY + [
    "seq", "t", "x_est", "y_est", "x_enu", "y_enu",
    "x_ref", "y_ref", "segment", "err_m",
]

# Gaussian (CRLB) bounds (run_crlb.py): one row per VIEW, not per method
# cell - the bound is a property of (campaign, session, mask) alone.
CRLB_KEY = ["campaign", "session", "mask"]
CRLB_COLUMNS = CRLB_KEY + [
    "n_anchors", "n_pairs", "n_pairs_total", "n_priors", "sigma_m",
    "fim_rank", "fim_nullity", "crlb_rmse_m", "crlb_abs_rmse_m",
    "crlb_weak_mode_m", "status", "run_at",
]

# Rigidity certificates (run_rigidity.py): one row per VIEW, the structural
# class of the view's measured anchor pair graph (rigidity.py).
RIGIDITY_KEY = ["campaign", "session", "mask"]
RIGIDITY_COLUMNS = RIGIDITY_KEY + [
    "thinning", "k", "thin_seed", "n_anchors", "n_pairs", "n_pairs_total",
    "n_components", "generically_rigid", "redundantly_rigid",
    "three_connected", "globally_rigid", "cert_class", "run_at",
]

# Oracle-start control (run_controls.py): the factor graph started from the
# surveyed geometry instead of the classical-MDS initialization, next to the
# MDS-start run of the same view and kernel. A control, not a roster method.
ORACLE_KEY = ["campaign", "session", "mask", "variant"]
ORACLE_COLUMNS = ORACLE_KEY + [
    "sigma_m", "n_anchors", "n_pairs", "n_pairs_total",
    "rmse_mds_start_m", "cost_mds_start", "n_iter_mds_start",
    "rmse_oracle_start_m", "cost_oracle_start", "n_iter_oracle_start",
    "converged_mds_start", "converged_oracle_start", "run_at",
]

# Noise-scale sweep control (run_controls.py): the benchmark cell re-run at
# sigma_mult x the session sigma, everything else at the benchmark setting.
SWEEP_KEY = ["campaign", "session", "mask", "method", "variant", "sigma_mult"]
SWEEP_COLUMNS = SWEEP_KEY + [
    "seed", "sigma_m", "rmse_m", "max_error_m", "reflection", "run_at",
]

# Offset control (run_controls.py): the offset-state factor
# graph (factor_graph.solve_nls_offset) on the full mesh at several prior
# widths and noise scales, and the offset-aware bounds (crlb.
# offset_view_bound). prior_width_m is NaN for the zero and common modes.
OFFSET_KEY = ["campaign", "session", "mask", "variant", "offset",
              "prior_width_m", "sigma_mult"]
OFFSET_COLUMNS = OFFSET_KEY + [
    "sigma_m", "n_anchors", "n_pairs", "rmse_m", "max_error_m",
    "offset_common_m", "offset_device_std_m", "offset_device_min_m",
    "offset_device_max_m", "n_iter", "converged", "cost", "run_at",
]
OFFSET_CRLB_KEY = ["campaign", "session", "mask", "offset", "prior_width_m"]
OFFSET_CRLB_COLUMNS = OFFSET_CRLB_KEY + [
    "sigma_m", "n_anchors", "n_pairs", "fim_rank", "fim_nullity",
    "crlb_rmse_m", "crlb_weak_mode_m", "run_at",
]

# Sigma terms (run_controls.py): the two terms of the session
# noise-scale recipe (kernels.session_sigma) on the fully meshed view, next
# to a robust per-pair temporal scatter. One row per session.
SIGMA_TERMS_KEY = ["campaign", "session"]
SIGMA_TERMS_COLUMNS = SIGMA_TERMS_KEY + [
    "n_anchors", "n_pairs", "scatter_std_m", "mds_mad_m", "sigma_m",
    "sigma_term", "robust_scatter_m", "pair_std_p90_m",
    "mds_resid_median_abs_m", "mds_resid_p90_abs_m", "run_at",
]

# synthetic baseline of run_synthetic.py, results/synthetic_baseline/cells:
# one row per condition x draw x view x kernel scale x method x variant
SYNTHETIC_KEY = ["condition", "draw", "view", "scale", "method", "variant"]
SYNTHETIC_COLUMNS = SYNTHETIC_KEY + [
    "status", "error", "rmse_m", "max_error_m", "reflection", "cost",
    "n_iter", "converged", "rmse_oracle_m", "cost_oracle", "b_hat_m",
    "sigma_m", "sigma_scatter_m", "sigma_mds_m", "sigma_robust_m",
    "n_anchors", "n_pairs", "view_class", "n_nlos_pairs", "n_nlos_kept",
    "nlos_excess_median_m", "pair_resid_median_m", "pair_resid_max_m",
    "pair_resid_corr_d", "ranges_sha256", "geometry", "n_epochs",
    "sigma_hw_m", "mp_mean", "mp_sigma", "d_max_m", "p_out", "bias_m",
]

# frame-choice sweep of run_frame_choice.py, results/frame_choice/: one row
# per defining pair x engine variant (pairs), per seed triple (triples), and
# per anchor of the release pair's line (line)
FRAME_PAIR_KEY = ["campaign", "session", "method", "variant", "origin",
                  "axis"]
FRAME_PAIR_COLUMNS = FRAME_PAIR_KEY + [
    "third", "ordered", "is_release", "status", "error", "rmse_m",
    "max_error_m", "reflection", "n_anchors", "n_placed", "unplaced",
    "cost", "n_iter", "converged", "wrong_side", "ref_near_line",
    "pair_length_m", "pair_residual_m", "pair_outlier", "pair_missing",
    "pair_hull_edge", "n_off_line", "n_minority_side", "sigma_m",
    "runtime_s",
]
FRAME_TRIPLE_KEY = ["campaign", "session", "a", "b", "c"]
FRAME_TRIPLE_COLUMNS = FRAME_TRIPLE_KEY + [
    "is_release", "min_height_m", "all_hull", "outlier_pair",
    "missing_pair", "near_collinear", "b1_abs_rmse_m",
    "b1_abs_rmse_other_m", "b1_reflection", "b2_status", "b2_error",
    "b2_rmse_m", "b2_abs_rmse_m", "b2_abs_reflection", "b2_cost",
    "b2_n_iter", "b2_prior_max_err_m", "sigma_m", "runtime_s",
]
FRAME_LINE_KEY = ["campaign", "session", "id"]
FRAME_LINE_COLUMNS = FRAME_LINE_KEY + [
    "origin", "axis", "offset_m", "on_hull", "cf_placed", "cf_reason",
    "cf_error_m",
]

_SCHEMAS = {"benchmark": SUMMARY_COLUMNS,
            "benchmark_estimates": ESTIMATE_COLUMNS,
            "benchmark_trajectories": TRAJECTORY_COLUMNS,
            "crlb": CRLB_COLUMNS,
            "rigidity": RIGIDITY_COLUMNS,
            "oracle_start": ORACLE_COLUMNS,
            "sigma_sweep": SWEEP_COLUMNS,
            "offset_control": OFFSET_COLUMNS,
            "offset_crlb": OFFSET_CRLB_COLUMNS,
            "sigma_terms": SIGMA_TERMS_COLUMNS,
            "synthetic_baseline/cells": SYNTHETIC_COLUMNS,
            "frame_choice/pairs": FRAME_PAIR_COLUMNS,
            "frame_choice/triples": FRAME_TRIPLE_COLUMNS,
            "frame_choice/line": FRAME_LINE_COLUMNS}

_KEYS = {"crlb": CRLB_KEY, "rigidity": RIGIDITY_KEY,
         "oracle_start": ORACLE_KEY, "sigma_sweep": SWEEP_KEY,
         "offset_control": OFFSET_KEY, "offset_crlb": OFFSET_CRLB_KEY,
         "sigma_terms": SIGMA_TERMS_KEY,
         "synthetic_baseline/cells": SYNTHETIC_KEY,
         "frame_choice/pairs": FRAME_PAIR_KEY,
         "frame_choice/triples": FRAME_TRIPLE_KEY,
         "frame_choice/line": FRAME_LINE_KEY}

DETERMINISTIC_SEED = -1


def table_key(name: str) -> list[str]:
    """Upsert key of a results table (the cell KEY unless overridden)."""
    return _KEYS.get(name, KEY)


def _paths(name: str) -> tuple[Path, Path]:
    return RESULTS_DIR / f"{name}.parquet", RESULTS_DIR / f"{name}.csv"


def load(name: str = "benchmark") -> pd.DataFrame:
    """Load a results table; empty frame with the right schema if absent."""
    pq, _ = _paths(name)
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.DataFrame(columns=_SCHEMAS[name])


def _replace_with_retry(src: Path, dst: Path, attempts: int = 9) -> None:
    """os.replace with backoff: the results dir is cloud-synced and the
    sync client transiently locks files on Windows (WinError 5/32)."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.5 * 2 ** i)


def _write_with_retry(write, attempts: int = 9) -> None:
    """Run a file write with the same backoff: the sync client can also
    hold the temporary file of the previous write."""
    for i in range(attempts):
        try:
            write()
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.5 * 2 ** i)


def _write(df: pd.DataFrame, name: str) -> None:
    df = df.reindex(columns=_SCHEMAS[name])  # order + NaN-fill added columns
    pq, csv = _paths(name)
    pq.parent.mkdir(parents=True, exist_ok=True)
    tmp = pq.with_suffix(".parquet.tmp")
    _write_with_retry(lambda: df.to_parquet(tmp, index=False))
    _replace_with_retry(tmp, pq)
    tmp_csv = csv.with_suffix(".csv.tmp")  # same sync-lock exposure
    _write_with_retry(lambda: df.to_csv(tmp_csv, index=False,
                                        float_format="%.6f"))
    _replace_with_retry(tmp_csv, csv)


def upsert(new_rows: pd.DataFrame, name: str = "benchmark") -> pd.DataFrame:
    """Replace-or-append rows by the table's key; rewrites parquet + CSV."""
    key = table_key(name)
    old = load(name)
    if len(old):
        # estimates replace whole cells too: keying on KEY (without id) drops
        # every per-anchor row of a re-run cell before appending the new ones
        new_keys = set(map(tuple, new_rows[key].itertuples(index=False)))
        keep = ~old[key].apply(tuple, axis=1).isin(new_keys)
        old = old[keep]
    out = pd.concat([old, new_rows], ignore_index=True)
    sort_cols = key + (["id"] if "id" in out.columns else [])
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    _write(out, name)
    return out


def replace(df: pd.DataFrame, name: str) -> None:
    """Rewrite a whole results table (parquet + CSV), sorted by its key, for
    tables that one run produces in full."""
    _write(df.sort_values(table_key(name), kind="stable")
           .reset_index(drop=True), name)


def existing_keys(name: str = "benchmark") -> set[tuple]:
    """Set of key tuples already stored in a results table."""
    df = load(name)
    if not len(df):
        return set()
    return set(map(tuple, df[table_key(name)].itertuples(index=False)))
