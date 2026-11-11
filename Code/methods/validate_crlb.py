"""Validation of the CRLB machinery and the prior-anchored solver.

Structural checks need no data; the Monte-Carlo closure uses the surveyed
set_b/s03 geometry (read-only) with synthetic Gaussian ranges, so the
Gaussian model of the bound holds exactly and the ML factor graph should
approach the bound from above:

  (i)   the aligned bound is linear in sigma;
  (ii)  adding an edge never increases the bound at fixed nullity;
  (iii) complete blind graph -> nullity exactly 3 (d = 2);
  (iv)  |K| >= 2 -> nullity 0 and a finite absolute bound;
  (v)   MC closure, blind: aligned RMSE of the gaussian factor graph
        straddles the aligned bound from above (efficiency ~ 1);
  (vi)  MC closure, anchored |K| = 3: absolute RMSE vs the absolute bound;
  (vii) evaluate_absolute releases exactly the |K| = 2 mirror branch;
  (viii) prior-sweep mask plumbing (names, subsets, roster gating).

    python validate_crlb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import crlb
import evaluation
import factor_graph as fg
import family_masks
from common import GAUGE_TRIPLE

DATA = Path(__file__).resolve().parents[2] / "Data"
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


def complete_pairs(n: int) -> np.ndarray:
    iu, ju = np.triu_indices(n, k=1)
    return np.column_stack([iu, ju])


def surveyed_xy() -> tuple[np.ndarray, list[str]]:
    ref = evaluation.surveyed_reference(DATA / "set_b" / "s03")
    return ref[["x", "y"]].to_numpy(float), list(ref["id"])


def main() -> int:
    X, ids = surveyed_xy()
    n = len(X)
    pairs = complete_pairs(n)

    # (i) linearity in sigma
    b1 = crlb.bound(crlb.range_fim(X, pairs, 0.1))
    b2 = crlb.bound(crlb.range_fim(X, pairs, 0.2))
    check(np.isclose(b2["crlb_rmse_m"], 2 * b1["crlb_rmse_m"], rtol=1e-9),
          "aligned bound linear in sigma")

    # (ii) edge monotonicity at fixed nullity
    sub = pairs[:-1]
    bs = crlb.bound(crlb.range_fim(X, sub, 0.1))
    check(bs["fim_nullity"] == 3 and b1["fim_nullity"] == 3,
          "complete graph minus one edge keeps nullity 3")
    check(b1["crlb_rmse_m"] <= bs["crlb_rmse_m"] + 1e-12,
          "adding an edge never increases the aligned bound")

    # (iii) blind nullity
    check(b1["fim_nullity"] == 3, "complete blind graph: nullity exactly 3")
    check(np.isnan(b1["crlb_abs_rmse_m"]),
          "blind absolute bound undefined (NaN)")
    check(0 < b1["crlb_weak_mode_m"] <= b1["crlb_rmse_m"],
          "weak-mode contribution positive and within the bound")

    # (iv) two seeds -> nonsingular
    idx2 = np.array([0, 1])
    sig2 = np.array([0.01, 0.01])
    F2 = crlb.range_fim(X, pairs, 0.1) + crlb.prior_fim(n, idx2, sig2)
    b2s = crlb.bound(F2)
    check(b2s["fim_nullity"] == 0 and np.isfinite(b2s["crlb_abs_rmse_m"]),
          "|K| = 2: nullity 0, absolute bound finite")
    idx1 = np.array([0])
    F1 = crlb.range_fim(X, pairs, 0.1) + crlb.prior_fim(n, idx1,
                                                        sig2[:1])
    check(crlb.bound(F1)["fim_nullity"] == 1,
          "|K| = 1: rotation about the seed survives (nullity 1)")

    # aligned (relative) bound: projected form equals the pinv form blind,
    # and is weakly decreasing in |K| (priors cannot hurt the shape)
    b_rel = crlb.bound(crlb.range_fim(X, pairs, 0.1), X=X)
    check(np.isclose(b_rel["crlb_rmse_m"], b1["crlb_rmse_m"], rtol=1e-6),
          "blind aligned bound: projected form equals the pinv form")
    al = []
    for k in range(4):
        Fk = crlb.range_fim(X, pairs, 0.1) + crlb.prior_fim(
            n, np.arange(k), np.full(k, 0.01))
        al.append(crlb.bound(Fk, X=X)["crlb_rmse_m"])
    check(all(al[i + 1] <= al[i] + 1e-12 for i in range(3)),
          f"aligned bound weakly decreasing in |K|: "
          + " >= ".join(f"{v:.4f}" for v in al))

    # (v) MC closure, blind gaussian factor graph vs aligned bound
    rng = np.random.default_rng(0)
    sigma = 0.05
    d_true = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
    bound_al = crlb.bound(crlb.range_fim(X, pairs, sigma))["crlb_rmse_m"]
    ref_df = pd.DataFrame({"id": ids, "x": X[:, 0], "y": X[:, 1],
                           "z": 0.0})
    trials, sq = 200, []
    for _ in range(trials):
        M = np.full((n, n), np.nan)
        noisy = d_true + rng.normal(scale=sigma, size=len(d_true))
        M[pairs[:, 0], pairs[:, 1]] = M[pairs[:, 1], pairs[:, 0]] = noisy
        Xs, info = fg.solve_nls(M, fg.initial_coords(M), kernel="gaussian",
                                sigma=sigma)
        est = pd.DataFrame({"id": ids, "x": Xs[:, 0], "y": Xs[:, 1]})
        sq.append(evaluation.evaluate_estimate(est, ref_df)["rmse_m"] ** 2)
    mc = float(np.sqrt(np.mean(sq)))
    check(mc >= 0.95 * bound_al,
          f"blind MC aligned RMSE {mc:.4f} >= 0.95 x bound {bound_al:.4f}")
    check(mc <= 1.3 * bound_al,
          f"blind MC aligned RMSE {mc:.4f} <= 1.3 x bound {bound_al:.4f}")

    # (vi) MC closure, anchored |K| = 3 vs absolute bound
    idx3 = np.array([0, 1, 2])
    sig3 = np.full(3, 0.01)
    Fa = crlb.range_fim(X, pairs, sigma) + crlb.prior_fim(n, idx3, sig3)
    bound_abs = crlb.bound(Fa)["crlb_abs_rmse_m"]
    sq = []
    for _ in range(trials):
        M = np.full((n, n), np.nan)
        noisy = d_true + rng.normal(scale=sigma, size=len(d_true))
        M[pairs[:, 0], pairs[:, 1]] = M[pairs[:, 1], pairs[:, 0]] = noisy
        prior_xy = X[idx3] + rng.normal(scale=0.01, size=(3, 2))
        X0 = fg.initial_coords(M)
        from radiocal import frames
        X0 = frames.umeyama(X0[idx3], prior_xy,
                            allow_reflection=True).apply(X0)
        Xs, info = fg.solve_nls_anchored(M, X0, kernel="gaussian",
                                         sigma=sigma, prior_idx=idx3,
                                         prior_xy=prior_xy,
                                         prior_sigma=sig3)
        sq.append(np.mean(np.sum((Xs - X) ** 2, axis=1)))
    mc_abs = float(np.sqrt(np.mean(sq)))
    check(mc_abs >= 0.95 * bound_abs,
          f"anchored MC absolute RMSE {mc_abs:.4f} >= 0.95 x bound "
          f"{bound_abs:.4f}")
    check(mc_abs <= 1.3 * bound_abs,
          f"anchored MC absolute RMSE {mc_abs:.4f} <= 1.3 x bound "
          f"{bound_abs:.4f}")

    # (vii) evaluate_absolute releases exactly the |K| = 2 mirror
    a, b = X[0], X[1]
    u = (b - a) / np.linalg.norm(b - a)
    H = 2.0 * np.outer(u, u) - np.eye(2)
    mirrored = (X - a) @ H + a
    est_m = pd.DataFrame({"id": ids, "x": mirrored[:, 0],
                          "y": mirrored[:, 1]})
    res = evaluation.evaluate_absolute(est_m, ref_df, [ids[0], ids[1]])
    check(res is not None and res["abs_reflection"]
          and res["abs_rmse_m"] < 1e-9,
          "|K| = 2 mirror branch released (abs_reflection, zero error)")
    res3 = evaluation.evaluate_absolute(est_m, ref_df, ids[:3])
    check(res3 is not None and not res3["abs_reflection"]
          and res3["abs_rmse_m"] > 1.0,
          "|K| = 3 keeps the mirror as error")
    check(evaluation.evaluate_absolute(est_m, ref_df, ids[:1]) is None,
          "|K| = 1 absolute metric undefined (None)")

    # (viii) prior-sweep plumbing
    names = family_masks.expand_masks("prior_sweep")
    check(len(names) == 8 and names[0] == "fully_meshed_p0"
          and names[-1] == "multi_hop_p3", "prior_sweep expands to 8 masks")
    spec = family_masks.resolve_mask("fully_meshed_p2")
    check(spec["priors"] == {"subset": list(GAUGE_TRIPLE[:2])},
          "fully_meshed_p2 spec overrides only the prior subset")
    check(family_masks.resolve_mask("fully_meshed_p0")["priors"] == "none",
          "fully_meshed_p0 spec equals the blind canonical priors")
    check(family_masks.methods_for("fully_meshed", "static",
                                   "fully_meshed_p1") == ("fg_anchored",),
          "p-masks gate the roster to fg_anchored")
    check(family_masks.methods_for("fully_meshed", "static",
                                   "fully_meshed") != ("fg_anchored",),
          "canonical masks keep the static roster")
    check(family_masks.prior_count("multi_hop_p2") == 2
          and family_masks.prior_count("multi_hop_k2") is None,
          "prior_count reads p-masks only")

    if FAILURES:
        print(f"\nvalidate_crlb: {len(FAILURES)} FAILURE(S)")
        return 1
    print("\nvalidate_crlb: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
