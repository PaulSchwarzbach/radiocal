"""Smoke test: data loads, masks hold their invariants, fast methods
reproduce the frozen regression values. Takes a few seconds on a
laptop-class CPU with the pinned environment (README, "Runtimes"). The
full benchmark, including the sigma recomputation, is exercised by
Code/methods/run_benchmark.py --check-regression, not here.

Run as plain python (python tests/test_smoke.py) or via pytest.
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Code" / "methods"))

import radiocal  # noqa: E402

RANGE_COLUMNS = ["t", "t_raw", "seq", "from_id", "to_id", "range_m"]
MASKS = ["fully_meshed", "slam_bipartite", "mobile_assisted", "multi_hop"]
PRIOR_SUBSET = {"5BB6", "4A23", "8B05"}

# fast deterministic cells checked against results/regression_frozen.csv
# session is part of the key: s01..s03 exist in both sets, so a bare
# session literal is no longer unique (it was, when these were dyn*/p*).
# The cauchy factor graph is the one cell that uses the noise scale and a
# robust kernel.
FROZEN_CELLS = [("set_b", "s03", "closed_form", "disambiguated"),
                ("set_a", "s21", "closed_form", "disambiguated"),
                ("set_b", "s03", "mds", "classical"),
                ("set_a", "s21", "mds", "classical"),
                ("set_b", "s03", "factor_graph", "cauchy")]
FROZEN_TOL = 5.1e-4  # frozen values are rounded to 3 decimals


def test_sessions_load():
    g25 = radiocal.load_session(REPO / "Data/set_b/s03")
    g24 = radiocal.load_session(REPO / "Data/set_a/s21")
    assert list(g25.edges.columns[:6]) == RANGE_COLUMNS, g25.edges.columns
    assert list(g24.edges.columns[:6]) == RANGE_COLUMNS, g24.edges.columns
    assert len(g25.edges) > 0 and len(g24.edges) > 0
    assert len(g25.nodes) == 11, f"set_b nodes: {len(g25.nodes)}"
    assert len(g24.nodes) == 12, f"set_a nodes: {len(g24.nodes)}"
    print(f"ok  sessions load ({len(g25.edges)} + {len(g24.edges)} ranges)")


def test_mask_invariants():
    for sdir in ["Data/set_b/s01", "Data/set_a/s07"]:
        g = radiocal.load_session(REPO / sdir)
        roles = dict(zip(g.nodes["id"], g.nodes["role"]))
        for name in MASKS:
            v = g.apply_mask(name)
            assert len(v.edges) > 0, f"{sdir}/{name}: empty view"
            aa = ((v.edges["from_id"].map(roles) == "anchor")
                  & (v.edges["to_id"].map(roles) == "anchor")).sum()
            if name == "slam_bipartite":
                assert aa == 0, f"{sdir}: anchor-anchor edges in bipartite view"
            if name == "fully_meshed":
                assert len(v.priors()) == 0, f"{sdir}: fully_meshed exposes priors"
                assert aa == len(v.edges), f"{sdir}: non-anchor edges in fully_meshed"
            if name in ("mobile_assisted", "multi_hop"):
                # the spec subset, minus nodes whose prior a session override
                # removed (set_a quasi-static sessions: 8B05 becomes the mobile)
                anchors = set(g.nodes.loc[g.nodes["role"] == "anchor", "id"])
                assert set(v.priors()["id"]) == PRIOR_SUBSET & anchors, \
                    f"{sdir}/{name}: prior subset {set(v.priors()['id'])}"
        print(f"ok  mask invariants hold on {sdir}")


def test_frozen_cells():
    import evaluation, registry

    # The session noise scale is taken from the shipped benchmark table
    # (the driver's stored sigma_m) instead of being recomputed, since
    # run_benchmark.py --check-regression covers the sigma recipe itself.
    # The stored values of set_b/s03 and set_a/s21 are pinned against the
    # documented doctrine values here.
    documented_sigma = {"set_b": 0.129, "set_a": 1.621}
    bench = pd.read_parquet(REPO / "Code/methods/results/benchmark.parquet")
    frozen = pd.read_csv(REPO / "Code/methods/results/regression_frozen.csv")
    for campaign, session, method, variant in FROZEN_CELLS:
        sdir = REPO / "Data" / campaign / session
        g_full = radiocal.load_session(sdir).apply_mask("fully_meshed")
        ref = evaluation.surveyed_reference(sdir)
        row = bench.query(
            "campaign == @campaign and session == @session and "
            "mask == 'fully_meshed' and method == @method and "
            "variant == @variant").iloc[0]
        sigma = float(row["sigma_m"])
        assert abs(sigma - documented_sigma[campaign]) < 5.1e-4, \
            f"{campaign}: stored sigma {sigma} vs documented"
        est, _ = registry.REGISTRY[method].run(g_full, variant, sigma_m=sigma,
                                               sigma_scale=1.0, seed=None)
        rmse = evaluation.evaluate_estimate(est, ref)["rmse_m"]
        want = frozen.query(
            "campaign == @campaign and session == @session and "
            "mask == 'fully_meshed' and method == @method and "
            "variant == @variant")["rmse_m"].iloc[0]
        assert abs(rmse - want) < FROZEN_TOL, \
            f"{campaign}/{method}/{variant}: got {rmse:.6f}, frozen {want}"
        print(f"ok  {campaign}/{method}/{variant}: rmse {rmse:.3f} m == frozen")


def test_results_tables():
    df = pd.read_parquet(REPO / "Code/methods/results/benchmark.parquet")
    for col in ["campaign", "session", "mask", "method", "variant", "seed",
                "rmse_m", "runtime_s", "status"]:
        assert col in df.columns, f"benchmark table misses column {col}"
    assert len(df) > 0
    print(f"ok  benchmark table readable ({len(df)} rows)")


if __name__ == "__main__":
    test_sessions_load()
    test_mask_invariants()
    test_frozen_cells()
    test_results_tables()
    print("\nsmoke test passed")
