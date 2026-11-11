"""Mask families for the benchmark driver: names, parameterization, rosters.

The four canonical masks live as declarative YAML in the radiocal package and
in each campaign's Data/<campaign>/masks/ dir. This module adds the driver-side
family logic on top, without touching radiocal:

- parameterized multihop masks ``multi_hop_k<INT>``: the canonical
  multi_hop.yaml with ``edges.thin.anchor_anchor_knn`` overridden.
  ``radiocal.masks.apply_mask`` accepts spec dicts directly, so the sweep needs
  no new YAML files. The full parameterized name goes into the ``mask`` key
  column of the benchmark tables; the canonical ``multi_hop`` (k = 3)
  stays in the sweep as a standing spec-dict-vs-YAML consistency check, and
  fully_meshed is the k = infinity reference.
- random-thinning ablation masks ``multi_hop_random_k<INT>_s<INT>``: the same
  union-of-per-node construction with the selection criterion changed from
  nearest-by-median-range to seeded-uniform. Thinning by measured range
  doubles as an NLOS filter (paper Sec. VI-B), so the random arm separates the
  rigidity effect of sparsity from the NLOS-filtering side effect at a
  matched per-node degree. Family: multi_hop (same roster/motions).
- the family -> method roster (the static suite does not apply to bipartite
  views: without anchor-anchor pairs the aggregated range matrix is empty).
  The roster is motion-aware via ``methods_for``: mobile_assisted on a
  DYNAMIC session runs the pose-chain factor graph (fg_slam, which consumes
  anchor-anchor factors where the view provides them) instead of the static
  suite - a truly dynamic mobile under the static-unknown model would be a
  category error (its per-pair medians correspond to no position).
- prior-sweep masks ``fully_meshed_p<K>`` / ``multi_hop_p<K>`` (K = 0..3):
  the base YAML with only the ``priors`` field overridden to the first K
  anchors of the gauge triple - PRIOR_SUBSETS below is the single
  definition site of the nested subset convention. fully_meshed_p0 equals
  the canonical fully_meshed spec and multi_hop_p3 the canonical multi_hop
  spec, standing consistency anchors of the sweep. Roster: fg_anchored
  only, since a blind method on a p-mask would replicate its base-mask
  cell bit for bit. Alias ``prior_sweep`` expands to all eight.
- the family -> session applicability rule, keyed on session.yaml ``motion``:
  mobile_assisted runs on quasi-static mobiles (set_a quasi-static sessions, static
  suite) AND on dynamic mobiles (2025 dyn sessions, pose-chain roster; the
  taxonomy's mobile-assisted concept with real motion), and slam_bipartite
  needs a dynamic mobile (a static tag leaves bipartite anchor geometry
  unidentifiable - demonstrated in validate_slam.py).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

from radiocal.masks import load_mask

from common import GAUGE_TRIPLE

CANONICAL = ("fully_meshed", "multi_hop", "mobile_assisted",
             "slam_bipartite")

# k-NN thinning sweep: 10-12 anchors per session, so k = 8 is the near-full
# upper point and fully_meshed the k = infinity reference; k = 1 may disconnect
# (union semantics), which is a recorded structural outcome, not an error.
MULTIHOP_K_SWEEP = (1, 2, 3, 4, 5, 6, 8)

# random-thinning ablation: k = 1 excluded (random unions disconnect too
# often to be informative), three seeds document seed sensitivity
RANDOM_K_SWEEP = (2, 3, 4, 5, 6, 8)
RANDOM_SEEDS = (0, 1, 2)

_K_RE = re.compile(r"^multi_hop_k(\d+)$")
_RAND_RE = re.compile(r"^multi_hop_random_k(\d+)_s(\d+)$")

# reference-information sweep: nested known-anchor subsets, |K| = 0..3
PRIOR_SUBSETS = {k: GAUGE_TRIPLE[:k] for k in range(4)}
_P_RE = re.compile(r"^(fully_meshed|multi_hop)_p([0-3])$")

# family -> methods that are defined on its views (registry names). The slam
# roster is bipartite-only; everything else runs the static suite.
STATIC_METHODS = ("closed_form", "mds", "factor_graph", "ekf",
                  "particle_filter", "grid_filter")
SLAM_METHODS = ("ekf_slam", "fg_slam")
# offset-state factor-graph variants: fully meshed and
# quasi-static mobile-assisted views only
OFFSET_METHODS = ("fg_offset", "fg_offset_dev")

FAMILY_METHODS = {
    "fully_meshed": STATIC_METHODS + OFFSET_METHODS,
    "multi_hop": STATIC_METHODS,
    "mobile_assisted": STATIC_METHODS + OFFSET_METHODS,
    "slam_bipartite": SLAM_METHODS,
}

# family -> session.yaml ``motion`` values it applies to
FAMILY_MOTIONS = {
    "fully_meshed": ("static", "quasi_static_point", "dynamic"),
    "multi_hop": ("static", "quasi_static_point", "dynamic"),
    "mobile_assisted": ("quasi_static_point", "dynamic"),
    "slam_bipartite": ("dynamic",),
}


def methods_for(family: str, motion: str,
                mask: str | None = None) -> tuple[str, ...]:
    """Mask- and motion-aware method roster (see module docstring):
    prior-sweep masks run the prior-anchored factor graph only,
    mobile_assisted on a dynamic session runs the pose-chain factor graph
    with anchor-anchor factors; every other combination keeps the family
    roster."""
    if mask is not None and _P_RE.match(mask):
        return ("fg_anchored",)
    if family == "mobile_assisted" and motion == "dynamic":
        return ("fg_slam",)
    return FAMILY_METHODS[family]


def family_of(mask_name: str) -> str:
    """Canonical family of a (possibly parameterized) mask name."""
    if _K_RE.match(mask_name) or _RAND_RE.match(mask_name):
        return "multi_hop"
    p = _P_RE.match(mask_name)
    if p:
        return p.group(1)
    if mask_name in CANONICAL:
        return mask_name
    raise ValueError(f"unknown mask {mask_name!r}")


def thinning_k(mask_name: str) -> int | None:
    """k of a parameterized multihop mask; 3 for the canonical multihop
    mask (its YAML value, shared by the multi_hop prior-sweep masks);
    None for non-multihop masks."""
    m = _K_RE.match(mask_name) or _RAND_RE.match(mask_name)
    if m:
        return int(m.group(1))
    if mask_name == "multi_hop" or (_P_RE.match(mask_name)
                                    and mask_name.startswith("multi_hop")):
        return 3
    return None


def prior_count(mask_name: str) -> int | None:
    """|K| of a prior-sweep mask; None for every other mask name."""
    p = _P_RE.match(mask_name)
    return int(p.group(2)) if p else None


def resolve_mask(mask_name: str, mask_dirs=None) -> str | dict:
    """What to pass to MeasurementGraph.apply_mask for a mask name.

    Canonical names pass through as strings (radiocal resolves them against
    the campaign's masks/ dir, then the packaged set). Parameterized names
    load the canonical multihop YAML and override the single thin field.
    """
    p = _P_RE.match(mask_name)
    if p:
        spec = copy.deepcopy(load_mask(p.group(1), search_dirs=mask_dirs))
        spec["name"] = mask_name
        k = int(p.group(2))
        spec["priors"] = ("none" if k == 0
                          else {"subset": list(PRIOR_SUBSETS[k])})
        return spec
    m = _K_RE.match(mask_name)
    r = _RAND_RE.match(mask_name)
    if not m and not r:
        family_of(mask_name)  # raises on unknown names
        return mask_name
    spec = copy.deepcopy(load_mask("multi_hop", search_dirs=mask_dirs))
    spec["name"] = mask_name
    if m:
        spec["edges"]["thin"] = {"anchor_anchor_knn": int(m.group(1))}
    else:
        spec["edges"]["thin"] = {"anchor_anchor_knn_random":
                                 {"k": int(r.group(1)),
                                  "seed": int(r.group(2))}}
    return spec


def expand_masks(arg: str) -> list[str]:
    """Comma-separated mask list with the ``multi_hop_sweep`` and
    ``multi_hop_random_sweep`` aliases expanded; order-preserving,
    deduplicated, names validated."""
    out: list[str] = []
    for tok in (t.strip() for t in arg.split(",") if t.strip()):
        if tok == "multi_hop_sweep":
            names = [f"multi_hop_k{k}" for k in MULTIHOP_K_SWEEP]
        elif tok == "multi_hop_random_sweep":
            names = [f"multi_hop_random_k{k}_s{s}"
                     for k in RANDOM_K_SWEEP for s in RANDOM_SEEDS]
        elif tok == "prior_sweep":
            names = [f"{base}_p{k}" for base in ("fully_meshed", "multi_hop")
                     for k in sorted(PRIOR_SUBSETS)]
        else:
            names = [tok]
        for name in names:
            family_of(name)  # raises on unknown names
            if name not in out:
                out.append(name)
    return out


def session_motion(session_dir: str | Path) -> str:
    """The session.yaml ``motion`` value of a session directory."""
    import yaml
    meta = yaml.safe_load(
        (Path(session_dir) / "session.yaml").read_text(encoding="utf-8"))
    return meta.get("motion")


def session_applies(family: str, session_dir: str | Path) -> bool:
    """Whether a family is defined on a session, by session.yaml ``motion``."""
    return session_motion(session_dir) in FAMILY_MOTIONS[family]
