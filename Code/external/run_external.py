"""Benchmark runs on external campaigns, through the unchanged benchmark driver.

    python -B -P run_external.py fm <campaign>
    python -B -P run_external.py sens <campaign>     (0.25 sigma sensitivity)

`fm`: the fully meshed static roster (seed 0, particle filter seeds 0 to 4 as
in the benchmark), the cauchy/huber/gaussian factor graph from the surveyed
start, which separates the effect of the start from that of the kernel, and a
summary of the kernel comparison per engine. The driver is always called with
write=False; the frames go to EXT_ROOT/results/<campaign>_fm.* only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ext_context  # noqa: E402  (puts Code/methods on sys.path)

import common  # noqa: E402
import evaluation  # noqa: E402
import factor_graph  # noqa: E402
import run_benchmark  # noqa: E402
from radiocal import load_session  # noqa: E402
from registry import METHOD_ORDER  # noqa: E402

H1_ENGINES = {"factor_graph": ("gaussian", "cauchy"),
              "mds": ("smacof_gaussian", "smacof_cauchy"),
              "ekf": ("gaussian", "cauchy"),
              "particle_filter": ("gaussian", "cauchy")}
PF_SEEDS = (0, 1, 2, 3, 4)
GUARD_TOL_M = 1e-3


def _cell(rows: pd.DataFrame, method: str, variant: str) -> float:
    sel = rows[(rows["method"] == method) & (rows["variant"] == variant)
               & (rows["status"] == "ok")]
    if sel.empty:
        return float("nan")
    return float(sel["rmse_m"].mean()) if method == "particle_filter" else float(sel["rmse_m"].iloc[0])


def h1(rows: pd.DataFrame, guard_gap: float) -> dict:
    g = {m: _cell(rows, m, v[0]) for m, v in H1_ENGINES.items()}
    c = {m: _cell(rows, m, v[1]) for m, v in H1_ENGINES.items()}
    ok = [m for m in H1_ENGINES if np.isfinite(g[m]) and np.isfinite(c[m])]
    K = float(np.median([g[m] - c[m] for m in ok]))
    S = float(max(c[m] for m in ok) - min(c[m] for m in ok))
    R = float(np.median([g[m] / c[m] for m in ok]))
    if guard_gap > GUARD_TOL_M:
        verdict = "initialization-confounded"
    elif K >= 2 * S and R >= 1.5:
        verdict = "holds"
    elif K <= S:
        verdict = "fails"
    else:
        verdict = "mixed"
    return {"K": K, "S": S, "R": R, "guard_gap_m": guard_gap, "verdict": verdict,
            **{f"gauss_{m}": g[m] for m in H1_ENGINES},
            **{f"cauchy_{m}": c[m] for m in H1_ENGINES}}


def run_fm(ext_root: Path, campaign: str) -> int:
    cdir = ext_root / "data" / campaign
    bench = yaml.safe_load((cdir / "campaign.yaml").read_text(encoding="utf-8"))["benchmark"]
    session = f"{campaign}/r01"
    with ext_context.external_campaign(ext_root / "data", campaign, bench["gauge_triple"]):
        rows, est = run_benchmark.run_sessions([session], list(METHOD_ORDER), None,
                                               ["fully_meshed"], seed=0,
                                               skip_keys=set(), write=False)
        extra = [run_benchmark.run_sessions([session], ["particle_filter"], None,
                                            ["fully_meshed"], seed=s, skip_keys=set(),
                                            write=False)[0] for s in PF_SEEDS[1:]]
        rows = pd.concat([rows, *extra], ignore_index=True)
        sdir = cdir / "r01"
        g = load_session(sdir).apply_mask("fully_meshed")
        order = common.default_anchor_order(g)
        ref = evaluation.surveyed_reference(sdir)
        sigma = float(rows["sigma_m"].iloc[0])
        X_ref = ref.set_index("id").loc[order, ["x", "y"]].to_numpy(float)
        oracle = []
        for v in ("gaussian", "huber", "cauchy"):
            e_o, info = factor_graph.factor_graph_anchors(g, variant=v, anchor_order=order,
                                                         sigma_m=sigma, X0=X_ref)
            oracle.append({"variant": v, "rmse_surveyed_start_m":
                           float(evaluation.evaluate_estimate(e_o, ref)["rmse_m"]),
                           "rmse_mds_start_m": _cell(rows, "factor_graph", v)})
    oracle = pd.DataFrame(oracle)
    gap = float(abs(oracle.loc[oracle["variant"] == "cauchy", "rmse_surveyed_start_m"].iloc[0]
                    - oracle.loc[oracle["variant"] == "cauchy", "rmse_mds_start_m"].iloc[0]))
    stats = h1(rows, gap)
    res = ext_root / "results"
    res.mkdir(parents=True, exist_ok=True)
    rows.to_csv(res / f"{campaign}_fm.csv", index=False)
    est.to_csv(res / f"{campaign}_fm_estimates.csv", index=False)
    oracle.to_csv(res / f"{campaign}_fm_oracle_start.csv", index=False)
    pd.DataFrame([stats]).to_csv(res / f"{campaign}_fm_h1.csv", index=False)
    pd.set_option("display.width", 200)
    view = rows[["method", "variant", "seed", "rmse_m", "status", "runtime_s"]]
    print(view.to_string(index=False))
    print("\nfactor graph, surveyed start vs shared MDS start:")
    print(oracle.to_string(index=False))
    print("\nH1:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()})
    return 0


def run_sensitivity(ext_root: Path, campaign: str, scale: float = 0.25) -> int:
    """The 0.25 sigma sensitivity of H1 (reported, not decisive): the H1 engines
    at sigma_scale=scale from the shared start, plus the factor graph from the
    surveyed start, as in the run_controls sigma sweep."""
    from registry import REGISTRY
    cdir = ext_root / "data" / campaign
    bench = yaml.safe_load((cdir / "campaign.yaml").read_text(encoding="utf-8"))["benchmark"]
    out = []
    with ext_context.external_campaign(ext_root / "data", campaign, bench["gauge_triple"]):
        sdir = cdir / "r01"
        g = load_session(sdir).apply_mask("fully_meshed")
        order = common.default_anchor_order(g)
        ref = evaluation.surveyed_reference(sdir)
        import kernels
        sigma = float(kernels.session_sigma(g, order))
        for m, (vg, vc) in H1_ENGINES.items():
            seeds = PF_SEEDS if m == "particle_filter" else (0,)
            for v in (vg, vc):
                r = [float(evaluation.evaluate_estimate(
                    REGISTRY[m].run(g, v, sigma_m=sigma, sigma_scale=scale, seed=s)[0],
                    ref)["rmse_m"]) for s in seeds]
                out.append({"method": m, "variant": v, "start": "shared",
                            "sigma_scale": scale, "rmse_m": float(np.mean(r))})
        X_ref = ref.set_index("id").loc[order, ["x", "y"]].to_numpy(float)
        for v in ("gaussian", "cauchy"):
            e_o, _ = factor_graph.factor_graph_anchors(g, variant=v, anchor_order=order,
                                                       sigma_m=sigma * scale, X0=X_ref)
            out.append({"method": "factor_graph", "variant": v, "start": "surveyed",
                        "sigma_scale": scale,
                        "rmse_m": float(evaluation.evaluate_estimate(e_o, ref)["rmse_m"])})
    df = pd.DataFrame(out)
    df.to_csv(ext_root / "results" / f"{campaign}_fm_sigma{scale:g}.csv", index=False)
    print(df.to_string(index=False))
    return 0


def main() -> int:
    cfg = yaml.safe_load((HERE / "sources.yaml").read_text(encoding="utf-8"))
    ext_root = Path(cfg["ext_root"]).expanduser()
    if len(sys.argv) != 3 or sys.argv[1] not in ("fm", "sens") or not sys.argv[2].startswith("ext_"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "sens":
        return run_sensitivity(ext_root, sys.argv[2])
    return run_fm(ext_root, sys.argv[2])


if __name__ == "__main__":
    sys.exit(main())
