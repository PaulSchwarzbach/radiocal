"""Topology masks: declarative edge/prior filters realizing the C5 benchmark views.

A mask spec (YAML) has:

  name: slam_bipartite
  description: ...
  edges:
    keep:                      # directed role-pair whitelist
      - {from_role: mobile, to_role: anchor}
      - {from_role: anchor, to_role: mobile}
    thin: null                 # or {anchor_anchor_knn: 3}
                               # or {anchor_anchor_knn_random: {k: 3, seed: 0}}
  priors: none                 # none | surveyed | {subset: [ids]}
  groundtruth: motion_only     # full | motion_only | none

Masking always operates on the unmasked superset graph; evaluation compares
estimates against the superset's surveyed coordinates / Vicon truth.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# The four canonical masks ship inside the package as package data
# (pyproject.toml), so a wheel or a non-editable install finds them too.
_PACKAGED_MASKS = Path(__file__).resolve().parent / "mask_specs"


def load_mask(name_or_path: str | Path, search_dirs=None) -> dict:
    """Load a mask spec by file path, or by name searched in `search_dirs`
    (e.g. the campaign's masks/ dir) and the masks shipped with the package."""
    p = Path(name_or_path)
    candidates = [p] if p.suffix in (".yaml", ".yml") else []
    for d in list(search_dirs or []) + [_PACKAGED_MASKS]:
        candidates.append(Path(d) / f"{name_or_path}.yaml")
    for c in candidates:
        if c.is_file():
            spec = yaml.safe_load(c.read_text(encoding="utf-8"))
            spec.setdefault("name", c.stem)
            return spec
    raise FileNotFoundError(f"mask spec '{name_or_path}' not found (searched {candidates})")


def _aa_pairs(edges: pd.DataFrame, roles: dict) -> tuple[pd.Series, pd.DataFrame]:
    """(anchor-anchor row mask, undirected pair table a/b/median r)."""
    is_aa = edges["from_id"].map(roles).eq("anchor") & edges["to_id"].map(roles).eq("anchor")
    aa = edges[is_aa]
    pairs = pd.DataFrame({
        "a": aa[["from_id", "to_id"]].min(axis=1),
        "b": aa[["from_id", "to_id"]].max(axis=1),
        "r": aa["range_m"],
    }).groupby(["a", "b"], as_index=False)["r"].median()
    return is_aa, pairs


def _apply_keep_pairs(edges: pd.DataFrame, is_aa: pd.Series,
                      keep_pairs: set[tuple[str, str]]) -> pd.DataFrame:
    aa = edges[is_aa]
    pair_key = [tuple(sorted(t)) for t in zip(aa["from_id"], aa["to_id"])]
    aa_kept = aa[np.array([pk in keep_pairs for pk in pair_key], dtype=bool)]
    return pd.concat([edges[~is_aa], aa_kept]).sort_index()


def _thin_anchor_edges(edges: pd.DataFrame, roles: dict, k: int) -> pd.DataFrame:
    """Keep anchor-anchor edges only between k-nearest-neighbour anchor pairs
    (by median measured range); all non anchor-anchor edges pass through."""
    is_aa, pairs = _aa_pairs(edges, roles)
    keep_pairs: set[tuple[str, str]] = set()
    anchors = sorted({*pairs["a"], *pairs["b"]})
    for node in anchors:
        nbrs = pairs[(pairs["a"] == node) | (pairs["b"] == node)].nsmallest(k, "r")
        keep_pairs.update(tuple(sorted((row.a, row.b))) for row in nbrs.itertuples())
    return _apply_keep_pairs(edges, is_aa, keep_pairs)


def _thin_anchor_edges_random(edges: pd.DataFrame, roles: dict, k: int,
                              seed: int) -> pd.DataFrame:
    """Random-thinning ablation control of ``_thin_anchor_edges``: the same
    union-of-per-node-choices construction with the selection criterion
    changed from nearest-by-median-range to seeded-uniform. Each node makes
    the same number k of choices as in the k-NN thinning, without its
    range-based (NLOS-filtering) selection. The resulting graph is denser:
    random choices are seldom mutual, so the union holds more pairs than the
    k-NN view of the same k and can differ in rigidity class. Deterministic
    for a given seed: nodes are visited in sorted order on one seeded
    generator."""
    is_aa, pairs = _aa_pairs(edges, roles)
    rng = np.random.default_rng(seed)
    keep_pairs: set[tuple[str, str]] = set()
    anchors = sorted({*pairs["a"], *pairs["b"]})
    for node in anchors:
        nbrs = pairs[(pairs["a"] == node) | (pairs["b"] == node)]
        take = min(k, len(nbrs))
        sel = nbrs.iloc[rng.choice(len(nbrs), size=take, replace=False)]
        keep_pairs.update(tuple(sorted((row.a, row.b))) for row in sel.itertuples())
    return _apply_keep_pairs(edges, is_aa, keep_pairs)


def apply_mask(graph, spec: dict):
    """Return a new MeasurementGraph filtered per the mask spec."""
    roles = graph.roles()
    edges = graph.edges

    keep_rules = (spec.get("edges") or {}).get("keep")
    if keep_rules:
        m = pd.Series(False, index=edges.index)
        fr, to = edges["from_id"].map(roles), edges["to_id"].map(roles)
        for rule in keep_rules:
            m |= fr.eq(rule["from_role"]) & to.eq(rule["to_role"])
        edges = edges[m]

    thin = (spec.get("edges") or {}).get("thin")
    if thin and "anchor_anchor_knn" in thin:
        edges = _thin_anchor_edges(edges, roles, int(thin["anchor_anchor_knn"]))
    elif thin and "anchor_anchor_knn_random" in thin:
        cfg = thin["anchor_anchor_knn_random"]
        edges = _thin_anchor_edges_random(edges, roles, int(cfg["k"]),
                                          int(cfg["seed"]))

    nodes = graph.nodes.copy()
    priors = spec.get("priors", "surveyed")
    if priors == "none":
        strip = nodes["role"] == "anchor"
    elif isinstance(priors, dict) and "subset" in priors:
        strip = (nodes["role"] == "anchor") & ~nodes["id"].isin(priors["subset"])
    elif priors == "surveyed":
        strip = pd.Series(False, index=nodes.index)
    else:
        raise ValueError(f"unknown priors spec: {priors!r}")
    nodes.loc[strip, ["x", "y", "z", "sigma_m"]] = np.nan
    nodes.loc[strip, "prior_type"] = "none"

    truth = graph.truth
    if spec.get("groundtruth", "full") == "none":
        truth = truth.iloc[0:0]

    used = set(edges["from_id"]) | set(edges["to_id"])
    nodes = nodes[nodes["id"].isin(used) | (nodes["prior_type"] != "none")]

    meta = dict(graph.meta, mask=spec.get("name"), mask_spec=spec)
    return replace(graph, nodes=nodes.reset_index(drop=True),
                   edges=edges.reset_index(drop=True),
                   truth=truth.reset_index(drop=True), meta=meta)
