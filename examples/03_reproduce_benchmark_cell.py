"""Reproduce one benchmark cell and check it against the shipped results.

Runs robust SMACOF (mds / smacof_huber) on set_b/s03 under the
fully_meshed view, exactly as the benchmark driver does: shared session noise
scale, surveyed reference, rigid alignment with reflection allowed. Expected
output: an anchor RMSE around 0.24 m that matches the shipped
Code/methods/results/benchmark.parquet row to numerical precision.
Runtime: a few seconds with the pinned environment, although the session
noise scale is recomputed from scratch here, exactly as the benchmark driver
does it.
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Code" / "methods"))

from radiocal import load_session          # noqa: E402
import evaluation, kernels, registry       # noqa: E402
from common import default_anchor_order    # noqa: E402

sdir = REPO / "Data/set_b/s03"
g_full = load_session(sdir).apply_mask("fully_meshed")
sigma = kernels.session_sigma(g_full, default_anchor_order(g_full))
ref = evaluation.surveyed_reference(sdir)

est, info = registry.REGISTRY["mds"].run(g_full, "smacof_huber",
                                         sigma_m=sigma, sigma_scale=1.0,
                                         seed=None)
result = evaluation.evaluate_estimate(est, ref)
print(f"sigma = {sigma:.3f} m, anchor RMSE = {result['rmse_m']:.4f} m "
      f"({result['n_aligned']} anchors)")

bench = pd.read_parquet(REPO / "Code/methods/results/benchmark.parquet")
row = bench.query("campaign=='set_b' and session=='s03' and "
                  "mask=='fully_meshed' and method=='mds' and "
                  "variant=='smacof_huber'").iloc[0]
diff = abs(result["rmse_m"] - row["rmse_m"])
print(f"shipped benchmark row: RMSE = {row['rmse_m']:.4f} m "
      f"(|diff| = {diff:.2e} m)")
assert diff < 5e-4, "reproduced cell deviates from the shipped results"
print("OK: the shipped results are reproducible on this machine.")
