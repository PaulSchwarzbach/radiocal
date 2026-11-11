"""Seeded UWB range-error generator, ported from the author's simulation
framework.

A port of create_noisy_measurements() of that framework (not imported: its
module also needs gnss_ins_sim and a local package) with the values of the
published model:

    P. Schwarzbach, R. Weber, O. Michler, "Statistical Evaluation and
    Synthetic Generation of Ultra-Wideband Distance Measurements for Indoor
    Positioning Systems", IEEE Sensors J. 22(6):4836-4843, 2022,
    doi:10.1109/JSEN.2021.3121627.

The model of that paper (Sec. IV) for a true distance d:

    (4)  r = d + eps if p > d / d_max, else no range (dropout); p ~ U(0, 1)
    (5)  eps = eps_mp   if p_eps > 0.8 - 0.3 d / d_max and eps_mp < d
               eps_out  if p_eps < p_out
               eps_hw   otherwise
         with one p_eps ~ U(0, 1) per range, eps_hw ~ N(0, sigma_hw^2) and
         eps_out ~ U(-d, d_max - d)
    (6)  eps_mp = exp(R_mp + sigma_mp z), z ~ N(0, 1), the log-normal of (3)

Values (Sec. III-D and IV): sigma_hw = 0.33 m, R_mp = 0.8 and sigma_mp =
1.07 on the log scale (the paper prints both in m), d_max = 100 m.

Where this module differs from the paper or the legacy function:

    item                  2022 paper            legacy function   here
    sigma_hw              0.33 m (III-D)        0.1 m             0.33 m
    d_max                 100 m (IV)            120 m             100 m
    p_out                 left free (IV)        0.05              0.05
    case of eq. (5)       one p_eps per range   the same          the same
    noise on multipath    none                  none              none in
      and outlier ranges                                          range_errors()
    non-positive ranges   positive values only, clamped to 0      clamped to 0
                          as an option
    dropouts, eq. (4)     no range              inf               rows removed
    random numbers        -                     global np.random  one seeded
                                                                  stream per
                                                                  component

The legacy function returns r - d, as range_errors() does. Where both rules
of eq. (5) hold, which needs d > 2.5 d_max at p_out = 0.05, the outlier
wins, as in the legacy function. The clamp never acts on the released
geometries: their anchors are meters apart, and eps_out >= -d.

The per-pair extension (a step beyond the 2022 paper). The paper draws
every range independently and names, as a possible extension, "partial
anchor- or direction-dependent biases of the multipath probability", which
"would simulate local stochastic dependencies of temporally successive
measurements and thus slightly increase the error potential" (Sec. IV).
pair_excess() takes the limiting case for the state, which it draws once
per unordered anchor pair by rule (5), and goes beyond the paper in drawing
the excess of eq. (6) once per pair as well. Both hold for the whole session
in both directions, while hardware noise and outliers stay per range. The
effect is not slight. Through the per-pair medians of the batch engines the
per-range model gives every pair a positive bias of a few decimeters that
grows with the distance, whereas the measured sessions of this release show
single pairs biased by meters (the obstructed pairs of set_a, the blocked
pair 5BB6-D38F of set_b). run_synthetic.py runs both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from radiocal import MeasurementGraph

CONDITIONS = ("noise_free", "gaussian", "nlos", "nlos_per_range", "biased")
STREAMS = ("noise", "multipath", "outlier", "dropout")


@dataclass(frozen=True)
class RangeErrorModel:
    """Parameters of the range-error model (defaults: the 2022 paper, and
    p_out the legacy value)."""
    sigma_hw: float = 0.33  # m, hardware noise eps_hw (Sec. III-D)
    mp_mean: float = 0.8    # R_mp, log scale (Sec. III-D, eqs. 3 and 6)
    mp_sigma: float = 1.07  # sigma_mp, log scale
    d_max: float = 100.0    # m, maximum range (eq. 4)
    p_out: float = 0.05     # outlier probability (left free in eq. 5)
    mp_base: float = 0.8    # multipath above mp_base - mp_slope d / d_max
    mp_slope: float = 0.3   # (eq. 5)


PAPER = RangeErrorModel()


def streams(seed: int) -> dict[str, np.random.Generator]:
    """One generator per error component, spawned from one seed, so that a
    seed gives the same hardware noise in every condition and switching one
    component on or off moves no other draw."""
    kids = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(k) for name, k in zip(STREAMS, kids)}


def hardware_noise(n: int, rng: np.random.Generator,
                   model: RangeErrorModel = PAPER) -> np.ndarray:
    """eps_hw ~ N(0, sigma_hw^2) of eq. (5)."""
    return model.sigma_hw * rng.standard_normal(n)


def multipath_excess(n: int, rng: np.random.Generator,
                     model: RangeErrorModel = PAPER) -> np.ndarray:
    """eps_mp = exp(R_mp + sigma_mp z) of eq. (6)."""
    return np.exp(model.mp_mean + model.mp_sigma * rng.standard_normal(n))


def multipath_threshold(d, model: RangeErrorModel = PAPER) -> np.ndarray:
    """The p_eps above which eq. (5) selects multipath."""
    return model.mp_base - model.mp_slope * np.asarray(d, float) / model.d_max


def outlier_error(d, rng: np.random.Generator,
                  model: RangeErrorModel = PAPER) -> np.ndarray:
    """eps_out ~ U(-d, d_max - d) of eq. (5)."""
    d = np.asarray(d, float)
    return model.d_max * rng.random(d.size) - d


def dropout_mask(d, rng: np.random.Generator,
                 model: RangeErrorModel = PAPER) -> np.ndarray:
    """True where a range survives eq. (4), p > d / d_max."""
    d = np.asarray(d, float)
    return rng.random(d.size) > d / model.d_max


def range_errors(d, gens: dict[str, np.random.Generator],
                 model: RangeErrorModel = PAPER, *,
                 multipath: bool = True) -> np.ndarray:
    """Errors r - d of the published per-range model, eq. (5): one p_eps per
    range selects multipath, outlier or hardware noise, and the ranges are
    clamped at zero. ``gens`` comes from streams()."""
    d = np.asarray(d, float)
    n = d.size
    eps_hw = hardware_noise(n, gens["noise"], model)
    p_eps = gens["multipath"].random(n)
    eps_mp = multipath_excess(n, gens["multipath"], model)
    eps_out = outlier_error(d, gens["outlier"], model)
    is_mp = ((p_eps > multipath_threshold(d, model)) & (eps_mp < d)
             if multipath else np.zeros(n, dtype=bool))
    is_out = p_eps < model.p_out
    eps = np.where(is_out, eps_out, np.where(is_mp, eps_mp, eps_hw))
    return np.maximum(d + eps, 0.0) - d


def pair_excess(d_pair, rng: np.random.Generator,
                model: RangeErrorModel = PAPER) -> np.ndarray:
    """Multipath excess of each anchor pair, drawn once (the per-pair
    extension of the module docstring): rule (5) with one p_eps per pair and
    the excess of eq. (6), zero where the pair stays line-of-sight."""
    d_pair = np.asarray(d_pair, float)
    p_eps = rng.random(d_pair.size)
    eps_mp = multipath_excess(d_pair.size, rng, model)
    is_mp = (p_eps > multipath_threshold(d_pair, model)) & (eps_mp < d_pair)
    return np.where(is_mp, eps_mp, 0.0)


def perturbed_graph(graph: MeasurementGraph, condition: str, seed: int,
                    model: RangeErrorModel = PAPER, *, bias_m: float = 0.0,
                    dropouts: bool = False
                    ) -> tuple[MeasurementGraph, pd.Series]:
    """``graph`` with every directed range drawn under one condition.

    ``graph`` must carry exact distances, as the superset graphs of
    synthetic.synthetic_static_graph(..., sigma=0.0) do. Conditions:

        noise_free      r = d
        gaussian        r = d + eps_hw
        biased          r = d + bias_m + eps_hw
        nlos            r = d + e_ab + eps_hw with the excess e_ab of the
                        unordered pair drawn once (pair_excess), and each
                        range replaced by an outlier d + eps_out with p_out
        nlos_per_range  the published model, r = d + range_errors()

    Pairs come in sorted (node_a, node_b) order and ranges in edge order.
    Every condition draws its hardware noise from the same stream, so one
    seed gives the same eps_hw everywhere. With ``dropouts`` the ranges that
    eq. (4) drops are removed from the edges. Returns the graph and the
    excess per pair, which is zero outside the nlos condition.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    gens = streams(seed)
    e = graph.edges
    d = e["range_m"].to_numpy(float)
    node_a = e[["from_id", "to_id"]].min(axis=1).to_numpy()
    node_b = e[["from_id", "to_id"]].max(axis=1).to_numpy()
    index = pd.MultiIndex.from_tuples(sorted(set(zip(node_a, node_b))),
                                      names=["node_a", "node_b"])
    code = index.get_indexer(pd.MultiIndex.from_arrays([node_a, node_b]))
    excess = pd.Series(0.0, index=index, name="excess_m")
    if condition == "noise_free":
        eps = np.zeros(d.size)
    elif condition in ("gaussian", "biased"):
        eps = hardware_noise(d.size, gens["noise"], model)
        if condition == "biased":
            eps = eps + bias_m
    elif condition == "nlos":
        d_pair = pd.Series(d).groupby(code).first().to_numpy()
        excess[:] = pair_excess(d_pair, gens["multipath"], model)
        eps = (excess.to_numpy()[code]
               + hardware_noise(d.size, gens["noise"], model))
        is_out = gens["outlier"].random(d.size) < model.p_out
        eps = np.where(is_out, outlier_error(d, gens["outlier"], model), eps)
    else:
        eps = range_errors(d, gens, model)
    r = np.maximum(d + eps, 0.0)
    keep = (dropout_mask(d, gens["dropout"], model) if dropouts
            else np.ones(d.size, dtype=bool))
    edges = e.assign(range_m=r)[keep].reset_index(drop=True)
    meta = {**graph.meta, "condition": condition, "seed": int(seed)}
    return (MeasurementGraph(nodes=graph.nodes, edges=edges,
                             truth=graph.truth, meta=meta), excess)
