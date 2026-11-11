"""Method registry: one uniform entry point per method (benchmark driver).

Each method module keeps its own public function (`<method>_anchors`); this
module wraps them behind a single signature so the driver can loop
session x mask x method x variant without per-method special cases:

    spec.run(graph, variant, *, sigma_m, sigma_scale=1.0, seed=None)
        -> (est_df, info)

`sigma_m` is the shared session noise scale (kernels.session_sigma of the
session's fully_meshed view), computed once per SESSION by the driver and shared
across every mask of that session - sigma is a property of the radio
hardware and environment, not of the topology view, and holding it fixed
keeps the mask family a pure connectivity/information axis. Every adapter
passes the absolute scale `sigma_scale * sigma_m` into its method (the
methods' own legacy sigma computation stays available via sigma_m=None for
the historical runners). `check_sigma` asserts the method used exactly the
driver's value where it reports one. `seed` only reaches the particle
filter.

The estimation order is `common.estimation_order`: the anchor order plus any
mobile nodes present in the view's edges (mobile_assisted / slam_bipartite);
on fully_meshed views it equals the committed anchor order exactly, so the
regression gate pins the equivalence.

Variant names stay exactly as the per-method modules define them; the
`KERNEL_OF` map derives the shared factorial axes (kernel, mechanism) for
the long-format results table. Adapters change no behaviour: they pass
through to the committed entry points with the committed defaults
(convergence traces disabled - snapshots never affect the estimate).

The slam_bipartite methods (ekf_slam, fg_slam) live in their own modules,
imported lazily so the registry loads before they exist. Their frozen
contract: `*_slam_anchors(graph, variant, sigma_m=...) -> (est_df, info)`
with est_df holding the ANCHOR estimates only (id, x, y), and info carrying
"trajectory" (DataFrame seq, t, x, y - the per-epoch mobile estimate in the
solver frame) plus "sigma_m" for check_sigma.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from radiocal import MeasurementGraph

import closed_form
import common
import ekf
import factor_graph
import grid_filter
import mds
import particle_filter

NO_TRACE = 10 ** 9  # trace_every that suppresses intermediate snapshots

# variant -> (kernel | None, mechanism | None); shared factorial axes
KERNEL_OF: dict[str, tuple[str | None, str | None]] = {
    "faithful": (None, None), "disambiguated": (None, None),  # closed_form
    "classical": (None, None),                                # mds
    "smacof_gaussian": ("gaussian", None),
    "smacof_huber": ("huber", None),
    "smacof_cauchy": ("cauchy", None),
    "gaussian": ("gaussian", None),
    "huber": ("huber", None),
    "cauchy": ("cauchy", None),
    "gated": ("gaussian", "gate"),                            # ekf
    "mixture": ("gaussian", "mixture"),                       # grid_filter
}


def _py(v):
    """numpy scalar -> plain python (JSON-able); passthrough otherwise."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _pick(info: dict, keys: tuple[str, ...]) -> dict:
    return {k: _py(info[k]) for k in keys if k in info}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    variants: tuple[str, ...]
    seeded: bool
    run: Callable  # (graph, variant, *, sigma_m, sigma_scale, seed) -> (est, info)
    extras: Callable[[dict], dict]  # info -> flat JSON-able diagnostics


def check_sigma(info: dict, sigma_m: float, sigma_scale: float) -> None:
    """Assert a method's internally computed scale equals the driver's."""
    got = info.get("sigma_m")
    if got is None or not np.isfinite(got):
        return
    want = sigma_m * sigma_scale
    if not np.isclose(got, want, rtol=1e-9, atol=0.0):
        raise AssertionError(f"sigma mismatch: method used {got!r}, "
                             f"driver expects {want!r}")


# ----------------------------------------------------------------- adapters

def _run_closed_form(g: MeasurementGraph, variant: str, *, sigma_m: float,
                     sigma_scale: float = 1.0, seed: int | None = None):
    return closed_form.closed_form_anchors(
        g, anchor_order=common.estimation_order(g),
        disambiguate=(variant == "disambiguated"))


def _run_mds(g: MeasurementGraph, variant: str, *, sigma_m: float,
             sigma_scale: float = 1.0, seed: int | None = None):
    return mds.mds_anchors(g, variant=variant,
                           anchor_order=common.estimation_order(g),
                           sigma_m=sigma_scale * sigma_m)


def _run_factor_graph(g: MeasurementGraph, variant: str, *, sigma_m: float,
                      sigma_scale: float = 1.0, seed: int | None = None):
    return factor_graph.factor_graph_anchors(
        g, variant=variant, anchor_order=common.estimation_order(g),
        sigma_m=sigma_scale * sigma_m)


def _run_fg_anchored(g: MeasurementGraph, variant: str, *, sigma_m: float,
                     sigma_scale: float = 1.0, seed: int | None = None):
    return factor_graph.factor_graph_anchored(
        g, variant=variant, anchor_order=common.estimation_order(g),
        sigma_m=sigma_scale * sigma_m)


def _run_fg_offset(g: MeasurementGraph, variant: str, *, sigma_m: float,
                   sigma_scale: float = 1.0, seed: int | None = None):
    return factor_graph.factor_graph_offset(
        g, variant=variant, anchor_order=common.estimation_order(g),
        sigma_m=sigma_scale * sigma_m, offset="common")


def _run_fg_offset_dev(g: MeasurementGraph, variant: str, *, sigma_m: float,
                       sigma_scale: float = 1.0, seed: int | None = None):
    return factor_graph.factor_graph_offset(
        g, variant=variant, anchor_order=common.estimation_order(g),
        sigma_m=sigma_scale * sigma_m, offset="device",
        prior_width=factor_graph.OFFSET_PRIOR_WIDTH_M)


def _run_ekf(g: MeasurementGraph, variant: str, *, sigma_m: float,
             sigma_scale: float = 1.0, seed: int | None = None):
    return ekf.ekf_anchors(g, variant=variant,
                           anchor_order=common.estimation_order(g),
                           sigma_m=sigma_scale * sigma_m,
                           trace_every=NO_TRACE)


def _run_particle_filter(g: MeasurementGraph, variant: str, *, sigma_m: float,
                         sigma_scale: float = 1.0, seed: int | None = None):
    return particle_filter.particle_filter_anchors(
        g, variant=variant, anchor_order=common.estimation_order(g),
        seed=0 if seed is None else seed,
        sigma_m=sigma_scale * sigma_m, trace_every=NO_TRACE)


def _run_grid_filter(g: MeasurementGraph, variant: str, *, sigma_m: float,
                     sigma_scale: float = 1.0, seed: int | None = None):
    return grid_filter.grid_filter_anchors(
        g, variant=variant, anchor_order=common.estimation_order(g),
        sigma=sigma_scale * sigma_m)


def _run_ekf_slam(g: MeasurementGraph, variant: str, *, sigma_m: float,
                  sigma_scale: float = 1.0, seed: int | None = None):
    import ekf_slam  # lazy import of the SLAM family
    return ekf_slam.ekf_slam_anchors(g, variant=variant,
                                     sigma_m=sigma_scale * sigma_m)


def _run_fg_slam(g: MeasurementGraph, variant: str, *, sigma_m: float,
                 sigma_scale: float = 1.0, seed: int | None = None):
    import fg_slam  # lazy import of the SLAM family
    return fg_slam.fg_slam_anchors(g, variant=variant,
                                   sigma_m=sigma_scale * sigma_m)


# ------------------------------------------------------------------- extras

def _extras_closed_form(info: dict) -> dict:
    placed = info["placed"]
    return {"n_placed": int(sum(placed.values())),
            "unplaced": sorted(a for a, p in placed.items() if not p)}


def _extras_mds(info: dict) -> dict:
    return _pick(info, ("stress", "neg_eig_frac", "n_pairs_completed",
                        "n_iter", "converged"))


def _extras_factor_graph(info: dict) -> dict:
    return _pick(info, ("n_iter", "converged", "cost0", "cost"))


def _extras_fg_anchored(info: dict) -> dict:
    out = _pick(info, ("n_iter", "converged", "cost0", "cost",
                       "n_priors_used", "prior_max_err_m"))
    out["prior_ids"] = list(info.get("prior_ids", []))
    return out


def _extras_fg_offset(info: dict) -> dict:
    out = _pick(info, ("n_iter", "converged", "cost0", "cost",
                       "offset_common_m", "offset_prior_width_m",
                       "offset_device_std_m"))
    if "offset_device_m" in info:
        out["offset_device_m"] = {k: float(v) for k, v
                                  in info["offset_device_m"].items()}
    return out


def _extras_ekf(info: dict) -> dict:
    out = _pick(info, ("n_updates", "n_gated", "gated_frac", "gate_nis",
                       "sigma0_m"))
    out["init"] = info["init"]["init"]
    out["init_epochs"] = _py(info["init"]["init_epochs"])
    return out


def _extras_particle_filter(info: dict) -> dict:
    out = _pick(info, ("n_particles", "sweeps", "init"))
    out["n_resample_total"] = int(sum(info["n_resample"].values()))
    out["n_depletion_total"] = int(sum(info["n_depletion"].values()))
    ess = [v for v in info["ess_min"].values() if np.isfinite(v)]
    out["ess_min"] = float(min(ess)) if ess else None
    return out


def _extras_grid_filter(info: dict) -> dict:
    out = _pick(info, ("grid_res_m", "grid_half_m", "n_cells", "stride",
                       "n_epochs_total", "n_epochs_used", "n_placed"))
    out["n_zero_updates_total"] = int(sum(info["n_zero_updates"].values()))
    out["init_mode"] = dict(info["init_mode"])
    return out


def _extras_slam(info: dict) -> dict:
    # scalar diagnostics only; info["trajectory"] goes to the trajectory
    # table via the driver's family hook, never into extras_json
    return _pick(info, ("n_poses", "n_epochs_used", "n_updates",
                        "n_aa_pairs", "n_iter",
                        "cost0", "cost", "converged", "init",
                        "init_pose_stride", "q_step_m"))


REGISTRY: dict[str, MethodSpec] = {
    "closed_form": MethodSpec(
        "closed_form", ("faithful", "disambiguated"), False,
        _run_closed_form, _extras_closed_form),
    "mds": MethodSpec(
        "mds", ("classical", "smacof_gaussian", "smacof_huber",
                "smacof_cauchy"), False,
        _run_mds, _extras_mds),
    "factor_graph": MethodSpec(
        "factor_graph", ("gaussian", "huber", "cauchy"), False,
        _run_factor_graph, _extras_factor_graph),
    "fg_anchored": MethodSpec(
        "fg_anchored", ("gaussian", "huber", "cauchy"), False,
        _run_fg_anchored, _extras_fg_anchored),
    "fg_offset": MethodSpec(
        "fg_offset", ("gaussian", "huber", "cauchy"), False,
        _run_fg_offset, _extras_fg_offset),
    "fg_offset_dev": MethodSpec(
        "fg_offset_dev", ("gaussian", "huber", "cauchy"), False,
        _run_fg_offset_dev, _extras_fg_offset),
    "ekf": MethodSpec(
        "ekf", ("gaussian", "huber", "cauchy", "gated"), False,
        _run_ekf, _extras_ekf),
    "particle_filter": MethodSpec(
        "particle_filter", ("gaussian", "huber", "cauchy"), True,
        _run_particle_filter, _extras_particle_filter),
    "grid_filter": MethodSpec(
        "grid_filter", ("gaussian", "huber", "cauchy", "mixture"), False,
        _run_grid_filter, _extras_grid_filter),
    "ekf_slam": MethodSpec(
        "ekf_slam", ("gaussian", "huber", "cauchy"), False,
        _run_ekf_slam, _extras_slam),
    "fg_slam": MethodSpec(
        "fg_slam", ("gaussian", "huber", "cauchy"), False,
        _run_fg_slam, _extras_slam),
}

# the static estimator suite (fully_meshed / multi_hop / mobile_assisted
# rosters); fast batch methods first, the grid filter last
METHOD_ORDER_STATIC = ("closed_form", "mds", "factor_graph", "ekf",
                       "particle_filter", "grid_filter")

# driver run order over all registered methods; fg_anchored runs only on
# the prior-sweep masks (family_masks gates it there), so appending it here
# leaves every existing roster, and the regression gate, untouched. The
# offset-state variants (fg_offset: one common offset, fg_offset_dev: plus
# per-device offsets under a prior) run on the fully meshed and the
# quasi-static mobile-assisted views only; the gate runs them but no frozen
# cell refers to them
METHOD_ORDER = METHOD_ORDER_STATIC + ("ekf_slam", "fg_slam", "fg_anchored",
                                      "fg_offset", "fg_offset_dev")
