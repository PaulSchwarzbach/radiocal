"""Loaders: canonical session directories -> MeasurementGraph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .graph import MeasurementGraph, TRUTH_COLUMNS


def _registry_to_nodes(registry: dict) -> pd.DataFrame:
    rows = []
    for n in registry["nodes"]:
        if "status" in n:  # e.g. surveyed_never_ranging -> excluded from graphs
            continue
        prior = n.get("prior", {"type": "none"})
        if prior["type"] == "none":
            xyz, sigma = (np.nan, np.nan, np.nan), np.nan
        else:
            xyz, sigma = prior["xyz_local"], prior["sigma_m"]
        rows.append((n["id"], n["role"], prior["type"], *xyz, sigma))
    return pd.DataFrame(rows, columns=["id", "role", "prior_type", "x", "y", "z", "sigma_m"])


def _apply_node_overrides(nodes: pd.DataFrame, overrides: list[dict]) -> pd.DataFrame:
    """Session-level role/prior overrides (e.g. an anchor roving in one session)."""
    nodes = nodes.copy()
    for ov in overrides:
        m = nodes["id"] == ov["id"]
        if not m.any():
            raise ValueError(f"node_overrides names unknown node {ov['id']!r}")
        if "role" in ov:
            nodes.loc[m, "role"] = ov["role"]
        prior = ov.get("prior")
        if prior is not None:
            if prior["type"] == "none":
                nodes.loc[m, ["x", "y", "z", "sigma_m"]] = np.nan
                nodes.loc[m, "prior_type"] = "none"
            else:
                nodes.loc[m, ["x", "y", "z"]] = prior["xyz_local"]
                nodes.loc[m, "sigma_m"] = prior["sigma_m"]
                nodes.loc[m, "prior_type"] = prior["type"]
    return nodes


def load_session(session_dir: str | Path, mask=None) -> MeasurementGraph:
    """Load one session (e.g. Data/set_b/s01) as a MeasurementGraph.

    Campaign files (campaign.yaml, nodes.yaml, masks/) are read from the
    session's parent directory. `mask` may be a mask name, path, or spec dict.
    """
    sdir = Path(session_dir)
    cdir = sdir.parent
    session = yaml.safe_load((sdir / "session.yaml").read_text(encoding="utf-8"))
    campaign = yaml.safe_load((cdir / "campaign.yaml").read_text(encoding="utf-8"))
    registry = yaml.safe_load((cdir / "nodes.yaml").read_text(encoding="utf-8"))

    edges = pd.read_parquet(sdir / "ranges.parquet")
    gt_path = sdir / "groundtruth.parquet"
    truth = pd.read_parquet(gt_path) if gt_path.exists() else pd.DataFrame(columns=TRUTH_COLUMNS)

    meta = {**campaign, **session, "registry": registry,
            "_mask_dirs": [cdir / "masks"], "session_dir": str(sdir)}
    nodes = _registry_to_nodes(registry)
    if session.get("node_overrides"):
        nodes = _apply_node_overrides(nodes, session["node_overrides"])
    graph = MeasurementGraph(nodes, edges, truth, meta)
    return graph.apply_mask(mask) if mask is not None else graph


def load_campaign(campaign_dir: str | Path, mask=None) -> dict[str, MeasurementGraph]:
    """Load every session directory (contains session.yaml) of a campaign."""
    cdir = Path(campaign_dir)
    return {p.parent.name: load_session(p.parent, mask=mask)
            for p in sorted(cdir.glob("*/session.yaml"))}
