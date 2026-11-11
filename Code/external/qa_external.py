"""Data checks of the converted external campaigns.

    python -B -P qa_external.py

Per external campaign (EXT_ROOT/data/<campaign>/r01): units, repeats per pair,
connectivity and the fully meshed rigidity certificate, truth, planarity
(projection floor), the noise-scale rule, exact recovery from noise-free
planar ranges, one sanity cell (cauchy factor graph, fully meshed) from both
the shared and the surveyed start, and the counts the kernel comparison needs.
The projection floor is also computed on set_b/s03 and set_a/s21 for comparison
(in memory). Writes EXT_ROOT/results/ext_sessions.csv and day1_checks.md,
nothing in the repository.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ext_context  # noqa: E402  (puts Code/methods on sys.path)

import common  # noqa: E402
import crlb  # noqa: E402
import evaluation  # noqa: E402
import factor_graph  # noqa: E402
import kernels  # noqa: E402
import rigidity  # noqa: E402
from radiocal import load_session  # noqa: E402
from radiocal.masks import load_mask  # noqa: E402
from registry import REGISTRY  # noqa: E402
from synthetic import synthetic_static_graph  # noqa: E402

CAMPAIGNS = ("ext_bern_a40", "ext_selfloc_department", "ext_selfloc_reception")
PUBLISHED_MEAN_M = {"ext_selfloc_reception": 0.24}  # Corbalan et al. 2023, 4 known anchors
FM = load_mask("fully_meshed")


def _pairs(M: np.ndarray) -> list[tuple[int, int]]:
    iu, ju = np.triu_indices(M.shape[0], k=1)
    obs = np.isfinite(M[iu, ju])
    return list(zip(iu[obs].tolist(), ju[obs].tolist()))


def _restrict(g, order: list[str], pair_idx: list[tuple[int, int]]):
    """Keep only edges whose undirected pair is in the measured pair set."""
    keep = {tuple(sorted((order[i], order[j]))) for i, j in pair_idx}
    e = g.edges
    k = [tuple(sorted(t)) in keep for t in zip(e["from_id"], e["to_id"])]
    g.edges = e[np.array(k, dtype=bool)].reset_index(drop=True)
    return g


def _fg(g, variant: str, sigma_m: float):
    est, info = REGISTRY["factor_graph"].run(g, variant, sigma_m=sigma_m,
                                             sigma_scale=1.0, seed=0)
    return est, info


def _noise_free(ref: pd.DataFrame, order: list[str], pair_idx, ranges_3d: bool):
    g = synthetic_static_graph(ref[["id", "x", "y", "z"]], n_epochs=2,
                               ranges_3d=ranges_3d).apply_mask(FM)
    return _restrict(g, order, pair_idx)


def _fg_from(g, ref, order, start: str) -> float:
    """Gaussian factor graph from the shared MDS start or the surveyed start."""
    X0 = None
    if start == "surveyed":
        X0 = ref.set_index("id").loc[order, ["x", "y"]].to_numpy(float)
    est, _ = factor_graph.factor_graph_anchors(g, variant="gaussian", anchor_order=order,
                                               sigma_m=0.1, X0=X0)
    return float(evaluation.evaluate_estimate(est, ref)["rmse_m"])


def projection_floor(ref: pd.DataFrame, order: list[str], pair_idx) -> float:
    """Aligned RMSE of the gaussian factor graph on noise-free 3D slant ranges
    of the measured pair set, started at the surveyed geometry: the error the
    2D model makes on this geometry. (Changed on day 1: from the shared MDS
    start the value measured initialization failure on sparse graphs.)"""
    return _fg_from(_noise_free(ref, order, pair_idx, True), ref, order, "surveyed")


def exact_recovery(ref: pd.DataFrame, order: list[str], pair_idx) -> tuple[float, float]:
    """Noise-free planar ranges of the measured pair set: (from the shared MDS
    start, from the surveyed start). The second checks the pipeline, the first
    whether the shared initialization reaches the solution at all."""
    g = _noise_free(ref, order, pair_idx, False)
    return _fg_from(g, ref, order, "mds"), _fg_from(g, ref, order, "surveyed")


def tilt_deg(ref: pd.DataFrame) -> float:
    A = np.column_stack([ref["x"], ref["y"], np.ones(len(ref))])
    (a, b, _), *_ = np.linalg.lstsq(A, ref["z"].to_numpy(float), rcond=None)
    return float(np.degrees(np.arctan(np.hypot(a, b))))


def planarity_class(floor: float) -> str:
    return "planar" if floor <= 0.02 else ("near-planar" if floor <= 0.10 else "reject")


def check_campaign(ext_root: Path, campaign: str) -> dict:
    cdir = ext_root / "data" / campaign
    meta = yaml.safe_load((cdir / "campaign.yaml").read_text(encoding="utf-8"))
    bench = meta["benchmark"]
    sdir = cdir / "r01"
    out = {"campaign": campaign, "gauge_triple": " ".join(bench["gauge_triple"])}
    with ext_context.external_campaign(ext_root / "data", campaign, bench["gauge_triple"]):
        g0 = load_session(sdir)
        g = g0.apply_mask("fully_meshed")
        order = common.default_anchor_order(g)
        out["triple_heads_order"] = order[:3] == list(bench["gauge_triple"])
        ref = evaluation.surveyed_reference(sdir)
        out["truth_all_nodes"] = set(order) <= set(ref["id"])
        R = ref.set_index("id").loc[order, ["x", "y", "z"]].to_numpy(float)
        M = common.pair_range_matrix(g, order)
        pidx = _pairs(M)
        D3 = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=2)
        meas = np.array([M[i, j] for i, j in pidx])
        true = np.array([D3[i, j] for i, j in pidx])
        res = meas - true
        out.update(n_anchors=len(order), n_pairs=len(pidx),
                   n_pairs_total=len(order) * (len(order) - 1) // 2,
                   unit_ratio=float(np.median(meas / true)),
                   median_residual_m=float(np.median(res)),
                   pairs_resid_gt_1m=int((res > 1.0).sum()),
                   share_resid_gt_1m=float((res > 1.0).mean()))
        sym = g.symmetrized()
        cnt = sym.groupby(["node_a", "node_b"])["seq"].nunique()
        out["share_pairs_ge10_epochs"] = float((cnt >= kernels.MIN_PAIR_EPOCHS).mean())
        out["median_epochs_per_pair"] = float(cnt.median())
        cert = rigidity.global_rigidity_certificate(R[:, :2], pidx)
        out["fm_certificate"] = rigidity.certificate_class(cert)
        floor = projection_floor(ref, order, pidx)
        out.update(projection_floor_m=floor, planarity=planarity_class(floor),
                   z_range_m=float(R[:, 2].max() - R[:, 2].min()),
                   tilt_deg=tilt_deg(ref.set_index("id").loc[order].reset_index()))
        scatter = kernels.scatter_scale(g)
        init = kernels.init_residual_scale(M)
        rule = "recipe" if np.isfinite(scatter) else "init_only"
        sigma = max(scatter, init) if rule == "recipe" else init
        out.update(sigma_scatter_m=scatter, sigma_init_m=init, sigma_m=sigma,
                   sigma_rule=rule, sigma_term="scatter" if scatter > init else "mds_mad")
        out["exact_recovery_shared_start_m"], out["exact_recovery_surveyed_start_m"] = \
            exact_recovery(ref, order, pidx)
        t0 = time.perf_counter()
        est, info = _fg(g, "cauchy", sigma)
        out["fg_cauchy_runtime_s"] = time.perf_counter() - t0
        out["fg_cauchy_rmse_m"] = float(evaluation.evaluate_estimate(est, ref)["rmse_m"])
        out["fg_cauchy_converged"] = bool(info.get("converged", True))
        X_ref = ref.set_index("id").loc[order, ["x", "y"]].to_numpy(float)
        est_o, _ = factor_graph.factor_graph_anchors(g, variant="cauchy", anchor_order=order,
                                                     sigma_m=sigma, X0=X_ref)
        out["fg_cauchy_oracle_start_rmse_m"] = float(
            evaluation.evaluate_estimate(est_o, ref)["rmse_m"])
        out["confound_gap_m"] = abs(out["fg_cauchy_rmse_m"] - out["fg_cauchy_oracle_start_rmse_m"])
        b = crlb.view_bound(g, ref, sigma, use_priors=False)
        out["crlb_blind_m"] = float(b["crlb_rmse_m"])
    lo, hi = out["crlb_blind_m"], max(1.0, 5 * sigma)
    out["sanity_ok"] = bool(out["fg_cauchy_converged"] and lo <= out["fg_cauchy_rmse_m"] <= hi)
    if campaign in PUBLISHED_MEAN_M:
        pub = PUBLISHED_MEAN_M[campaign]
        out["published_mean_m"] = pub
        out["sanity_ok"] = out["sanity_ok"] and pub / 3 <= out["fg_cauchy_rmse_m"] <= 3 * pub
    out["units_ok"] = 0.95 <= out["unit_ratio"] <= 1.05 and abs(out["median_residual_m"]) <= 0.5
    out["repeats_ok"] = out["share_pairs_ge10_epochs"] >= 0.90
    out["h1_precondition"] = out["fm_certificate"] == "globally_rigid" and out["pairs_resid_gt_1m"] >= 1
    out["h3_precondition_pairs"] = out["share_resid_gt_1m"] <= 0.05
    return out


def torgau_floors() -> list[dict]:
    rows = []
    for s in ("set_b/s03", "set_a/s21"):
        sdir = ext_context.ROOT / "Data" / s
        g = load_session(sdir).apply_mask("fully_meshed")
        order = common.default_anchor_order(g)
        ref = evaluation.surveyed_reference(sdir)
        M = common.pair_range_matrix(g, order)
        floor = projection_floor(ref, order, _pairs(M))
        rows.append({"session": s, "projection_floor_m": floor,
                     "planarity": planarity_class(floor),
                     "z_range_m": float(ref["z"].max() - ref["z"].min())})
    return rows


def main() -> int:
    cfg = yaml.safe_load((HERE / "sources.yaml").read_text(encoding="utf-8"))
    ext_root = Path(cfg["ext_root"]).expanduser()
    rows = [check_campaign(ext_root, c) for c in CAMPAIGNS]
    df = pd.DataFrame(rows)
    tf = pd.DataFrame(torgau_floors())
    res = ext_root / "results"
    res.mkdir(parents=True, exist_ok=True)
    df.to_csv(res / "ext_sessions.csv", index=False)
    pd.set_option("display.width", 200)
    text = ["# Data checks of the external campaigns", "",
            df.set_index("campaign").T.to_string(), "",
            "## Projection floor on the Torgau gate sessions (in memory)", "",
            tf.to_string(index=False)]
    (res / "day1_checks.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
