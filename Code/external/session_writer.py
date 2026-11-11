"""Generic writer: DataFrames -> one radiocal campaign directory.

Writes the layout radiocal.load_session reads (Data/README.md):

    <campaign>/campaign.yaml      metadata, raw-source SHA-256, benchmark block
    <campaign>/nodes.yaml         anchors with surveyed priors; excluded nodes
                                  carry a status key, which the loader skips
    <campaign>/masks/*.yaml       packaged masks, prior subset = gauge triple
    <campaign>/<session>/session.yaml
    <campaign>/<session>/ranges.parquet   EDGE_COLUMNS, set A dtypes

Nothing here touches Data/ or the radiocal package; convert.py stays unchanged.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from radiocal.graph import EDGE_COLUMNS
from radiocal.masks import load_mask

STATIC_MASKS = ("fully_meshed", "multi_hop")


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")


def write_registry(cdir: Path, nodes: pd.DataFrame, *, crs_note: str,
                   prior_source: str, prior_sigma_m: float,
                   status: dict[str, str] | None = None) -> None:
    """nodes.yaml. nodes: id, x, y, z [m]; status: id -> reason (excluded)."""
    status = status or {}
    out = []
    for r in nodes.itertuples(index=False):
        n = {"id": str(r.id), "role": "anchor"}
        if str(r.id) in status:
            n["status"] = status[str(r.id)]
        n["prior"] = {"type": "surveyed", "source": prior_source,
                      "xyz_local": [round(float(r.x), 6), round(float(r.y), 6),
                                    round(float(r.z), 6)],
                      "sigma_m": float(prior_sigma_m)}
        out.append(n)
    _dump({"crs": {"epsg": None, "note": crs_note}, "nodes": out}, cdir / "nodes.yaml")


def write_masks(cdir: Path, gauge_triple: tuple[str, ...]) -> None:
    """Packaged static masks; the multi_hop prior subset becomes the triple."""
    for name in STATIC_MASKS:
        spec = copy.deepcopy(load_mask(name))
        if isinstance(spec.get("priors"), dict) and "subset" in spec["priors"]:
            spec["priors"] = {"subset": [str(a) for a in gauge_triple]}
            spec["description"] = (spec.get("description", "").rstrip()
                                   + " External run: prior subset = this campaign's gauge triple.")
        _dump(spec, cdir / "masks" / f"{name}.yaml")


def write_campaign(cdir: Path, meta: dict) -> None:
    _dump(meta, cdir / "campaign.yaml")


def write_session(sdir: Path, *, session_id: str, campaign: str,
                  edges: pd.DataFrame, dropped_invalid: int,
                  sources: list[dict], clock_note: str) -> dict:
    """ranges.parquet + session.yaml (key order as set_a/s21). Returns counts."""
    e = edges[EDGE_COLUMNS].copy()
    e["t"] = e["t"].astype("float64")
    e["t_raw"] = e["t_raw"].astype("float64")
    e["seq"] = e["seq"].astype("int64")
    e["from_id"] = e["from_id"].astype(str).astype(object)
    e["to_id"] = e["to_id"].astype(str).astype(object)
    e["range_m"] = e["range_m"].astype("float64")
    sdir.mkdir(parents=True, exist_ok=True)
    e.to_parquet(sdir / "ranges.parquet", index=False)
    t = e["t"].dropna()
    counts = {"range_epochs": int(e["seq"].nunique()),
              "directed_ranges": int(len(e)),
              "dropped_invalid": int(dropped_invalid),
              "truth_poses": 0}
    meta = {"session_id": session_id, "campaign": campaign, "motion": "static",
            "t_start": float(t.min()) if len(t) else None,
            "t_end": float(t.max()) if len(t) else None,
            "sources": sources,
            "clock": {"reference": None, "note": clock_note},
            "counts": counts}
    _dump(meta, sdir / "session.yaml")
    return counts


def pair_table(edges: pd.DataFrame) -> pd.DataFrame:
    """Undirected pairs with median range and sample count (all directions)."""
    a = edges[["from_id", "to_id"]].min(axis=1)
    b = edges[["from_id", "to_id"]].max(axis=1)
    g = pd.DataFrame({"a": a, "b": b, "r": edges["range_m"]}).groupby(["a", "b"])["r"]
    return pd.DataFrame({"median_m": g.median(), "n": g.size()}).reset_index()


def largest_component(ids: list[str], pairs: pd.DataFrame) -> set[str]:
    """Node ids of the largest connected component of the pair graph."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["a"], pairs["b"]):
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups: dict[str, set[str]] = {}
    for i in ids:
        groups.setdefault(find(i), set()).add(i)
    return max(groups.values(), key=lambda s: (len(s), sorted(s)))


def gauge_triple(pairs: pd.DataFrame, keep: set[str]) -> tuple[str, str, str]:
    """Pre-registered rule (conversion rule 4 of Code/external/README.md):
    anchor 0 = highest degree; anchor 1 = its neighbour with the most shared
    neighbours (ties: longest median range); anchor 2 = the common neighbour
    maximizing the triangle area from median ranges. Ties fall back to id."""
    p = pairs[pairs["a"].isin(keep) & pairs["b"].isin(keep)]
    nbr: dict[str, dict[str, float]] = {i: {} for i in keep}
    for a, b, r in zip(p["a"], p["b"], p["median_m"]):
        nbr[a][b] = r
        nbr[b][a] = r
    a0 = sorted(keep, key=lambda i: (-len(nbr[i]), i))[0]

    def shared(x):
        return len(set(nbr[a0]) & set(nbr[x]))

    cands1 = sorted(nbr[a0], key=lambda x: (-shared(x), -nbr[a0][x], x))
    for a1 in cands1:
        common = sorted(set(nbr[a0]) & set(nbr[a1]))
        if not common:
            continue

        def area(x):
            d01, d0x, d1x = nbr[a0][a1], nbr[a0][x], nbr[a1][x]
            s = 0.5 * (d01 + d0x + d1x)
            return float(np.sqrt(max(s * (s - d01) * (s - d0x) * (s - d1x), 0.0)))

        a2 = sorted(common, key=lambda x: (-area(x), x))[0]
        return (a0, a1, a2)
    raise ValueError("no anchor triangle with measured sides")
