"""Run the unchanged benchmark code on an external campaign.

The driver and the methods read four module attributes at call time. This
context sets them for one external campaign and restores them in `finally`:

    run_benchmark.DATA          campaign data root (EXT_ROOT/data)
    common.GAUGE_TRIPLE         the campaign's gauge triple (anchor order)
    family_masks.PRIOR_SUBSETS  nested first-K subsets of that triple
    kernels.session_sigma       only with sigma_rule="init_only"

Set A and set B never pass through this context. Results are never written
by the driver here: callers use run_sessions(..., write=False) and save the
returned frames under EXT_ROOT/results themselves.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ROOT / "Code" / "methods"
if str(METHODS) not in sys.path:
    sys.path.insert(0, str(METHODS))

import common  # noqa: E402
import family_masks  # noqa: E402
import kernels  # noqa: E402
import run_benchmark  # noqa: E402

_RECIPE = kernels.session_sigma


def init_only_sigma(graph, anchor_order, agg: str = "median") -> float:
    """The MDS-residual term of the recipe alone (fallback when no pair has
    10 epochs; this term sets sigma in all 25 Torgau sessions)."""
    M = common.pair_range_matrix(graph, anchor_order, agg=agg)
    return kernels.init_residual_scale(M)


def _check_root(p: Path) -> Path:
    p = Path(p).resolve()
    if ROOT in p.parents or p == ROOT or "nextcloud" in str(p).lower():
        raise ValueError(f"external data root must lie outside the repository "
                         f"and outside Nextcloud: {p}")
    return p


@contextmanager
def external_campaign(data_root: Path, campaign: str, gauge_triple,
                      sigma_rule: str = "recipe"):
    if not campaign.startswith("ext_"):
        raise ValueError("external campaign ids start with 'ext_'")
    if sigma_rule not in ("recipe", "init_only"):
        raise ValueError(f"unknown sigma rule {sigma_rule!r}")
    root = _check_root(data_root)
    triple = tuple(str(a) for a in gauge_triple)
    saved = (run_benchmark.DATA, common.GAUGE_TRIPLE,
             family_masks.PRIOR_SUBSETS, kernels.session_sigma)
    try:
        run_benchmark.DATA = root
        common.GAUGE_TRIPLE = triple
        family_masks.PRIOR_SUBSETS = {k: triple[:k] for k in range(4)}
        kernels.session_sigma = _RECIPE if sigma_rule == "recipe" else init_only_sigma
        yield
    finally:
        (run_benchmark.DATA, common.GAUGE_TRIPLE,
         family_masks.PRIOR_SUBSETS, kernels.session_sigma) = saved
