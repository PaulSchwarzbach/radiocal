"""Regression gate: compare driver results against the frozen factorial matrix.

The frozen expectation (results/regression_frozen.csv, written once by the
maintenance script freeze_expected.py from the per-method summary tables at
three decimals) covers the full benchmark scope on set_b/s03 and set_a/s21,
fully_meshed: 5 estimators x 3 kernels (30 rows), the mechanism variants
gated and mixture (4) and the kernel-less references classical MDS and
closed form faithful and disambiguated (6), 40 rows in all.

These fully meshed cells are deterministic (the particle filter runs with
seed 0) and agree across the tested environments far below the tolerance of
5e-4 m, the acceptance bound at the three published decimals. A failure is
therefore a behavioural regression in the driver or an adapter, not float
noise. The repository README states which results agree to float precision
and which can move with the library versions (cells on flexible views).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FROZEN_CSV = Path(__file__).resolve().parent / "results" / "regression_frozen.csv"
TOL = 5e-4
MATCH = ["campaign", "session", "method", "variant"]


def load_frozen() -> pd.DataFrame:
    return pd.read_csv(FROZEN_CSV)


def compare(results: pd.DataFrame, frozen: pd.DataFrame | None = None,
            tol: float = TOL) -> pd.DataFrame:
    """Per frozen row: frozen vs obtained rmse_m; hard-fail on rmse, soft
    checks (warn columns) on max_error_m and reflection."""
    if frozen is None:
        frozen = load_frozen()
    got = results[MATCH + ["rmse_m", "max_error_m", "reflection", "status"]]
    rep = frozen.merge(got, on=MATCH, how="left",
                       suffixes=("_frozen", ""))
    rep["delta"] = rep["rmse_m"] - rep["rmse_m_frozen"]
    rep["pass"] = (rep["status"] == "ok") & (rep["delta"].abs() < tol)
    rep["warn_max_error"] = (rep["max_error_m"] - rep["max_error_m_frozen"]).abs() >= tol
    rep["warn_reflection"] = rep["reflection"].astype("boolean") != \
        rep["reflection_frozen"].astype("boolean")
    return rep


def report(rep: pd.DataFrame) -> str:
    lines = [f"regression gate: {int(rep['pass'].sum())}/{len(rep)} PASS "
             f"(tol {TOL} m on rmse_m)"]
    for _, r in rep.iterrows():
        flag = "PASS" if r["pass"] else "FAIL"
        warns = []
        if r["warn_max_error"]:
            warns.append(f"max_error {r['max_error_m_frozen']:.3f}->"
                         f"{r['max_error_m']:.3f}")
        if r["warn_reflection"]:
            warns.append(f"reflection {r['reflection_frozen']}->{r['reflection']}")
        w = f"  [warn: {', '.join(warns)}]" if warns else ""
        lines.append(f"  {flag}  {r['campaign']}/{r['session']} "
                     f"{r['method']}:{r['variant']:<16} "
                     f"frozen {r['rmse_m_frozen']:.3f}  got "
                     + (f"{r['rmse_m']:.3f}" if pd.notna(r["rmse_m"]) else "MISSING")
                     + (f"  (d={r['delta']:+.4f})" if pd.notna(r["delta"]) else "")
                     + w)
    return "\n".join(lines)
