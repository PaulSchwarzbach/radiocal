"""What if the ranges were noisier, more often NLOS, or biased?

Draws synthetic ranges with known truth on the surveyed anchors of a session,
using the range-error generator of Code/methods/range_errors.py (Schwarzbach
et al., IEEE Sensors J. 2022). It then runs the factor graph on one topology
view, exactly as the benchmark driver does, and prints the aligned anchor
RMSE of every draw. It writes nothing. Change the arguments to try another
deployment:

    python examples/04_what_if_ranges.py
    python examples/04_what_if_ranges.py --condition biased --bias 0.1
    python examples/04_what_if_ranges.py --condition nlos --p-out 0.01
    python examples/04_what_if_ranges.py --session set_a/s21 --view multi_hop_k4

The defaults reproduce draws 0 to 4 of the nlos rows of
Code/methods/results/synthetic_baseline/ (cauchy, full mesh, session scale).
Runtime: a few seconds.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "Code" / "methods").is_dir())
sys.path.insert(0, str(REPO / "Code" / "methods"))

import evaluation, family_masks, kernels, range_errors, synthetic  # noqa: E402
from common import default_anchor_order                            # noqa: E402
from registry import REGISTRY                                      # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--session", default="set_b/s03",
                help="session whose surveyed anchors set the geometry")
ap.add_argument("--condition", default="nlos",
                choices=range_errors.CONDITIONS)
ap.add_argument("--draws", type=int, default=5)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--sigma-hw", type=float,
                default=range_errors.PAPER.sigma_hw,
                help="hardware noise per range [m]")
ap.add_argument("--p-out", type=float, default=range_errors.PAPER.p_out,
                help="outlier probability per range")
ap.add_argument("--bias", type=float, default=0.27,
                help="common range offset of the biased condition [m]")
ap.add_argument("--view", default="fully_meshed",
                help="fully_meshed, multi_hop_k<k>, ...")
ap.add_argument("--kernel", default="cauchy",
                choices=("gaussian", "huber", "cauchy"))
ap.add_argument("--scale", default="session", choices=("session", "hardware"),
                help="kernel scale: the session sigma of the benchmark, or "
                     "--sigma-hw")
args = ap.parse_args()

model = replace(range_errors.PAPER, sigma_hw=args.sigma_hw, p_out=args.p_out)
ref = evaluation.surveyed_reference(REPO / "Data" / args.session)
g0 = synthetic.synthetic_static_graph(ref, n_epochs=args.epochs, sigma=0.0)
errors = []
for k in range(args.draws):
    g, excess = range_errors.perturbed_graph(g0, args.condition, k, model,
                                             bias_m=args.bias)
    gf = g.apply_mask(family_masks.resolve_mask("fully_meshed"))
    sigma = (kernels.session_sigma(gf, default_anchor_order(gf))
             if args.scale == "session" and args.condition != "noise_free"
             else model.sigma_hw)  # exact ranges have no session sigma
    gv = g.apply_mask(family_masks.resolve_mask(args.view))
    est, _ = REGISTRY["factor_graph"].run(gv, args.kernel, sigma_m=sigma,
                                          sigma_scale=1.0, seed=0)
    errors.append(evaluation.evaluate_estimate(est, ref)["rmse_m"])
    print(f"draw {k}: {int((excess > 0).sum())} NLOS pairs, kernel scale "
          f"{sigma:.3f} m, anchor RMSE {errors[-1]:.3f} m")
print(f"median over {args.draws} draws: {np.median(errors):.3f} m "
      f"({args.condition}, {args.view}, factor_graph {args.kernel})")
