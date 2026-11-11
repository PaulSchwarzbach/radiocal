"""Reader for the selfloc data of Corbalan et al. 2023 (IEEE Access 11:29711).

Source: github.com/d3s-trento/selfloc (no licence file: analysis only, nothing
is redistributed). Files used:

- data/<testbed>/sstwr.csv: tab-separated single-sided two-way-ranging log,
  columns seqn, ts (Unix s), node (initiator), anchor (responder),
  ppm_offset, dist, dist_bias, dist_drift, dist_drift_bias (m), fppl, rssi;
- sw/deployments/<testbed>.csv: tab-separated node, x, y, z [m], addr.

The epoch rule of the conversion (Code/external/README.md): seq is the 0-based rank
of an exchange within its directed pair, ordered by timestamp, so one epoch
holds one sample of every pair, like a Torgau frame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from radiocal.graph import EDGE_COLUMNS

SSTWR_COLUMNS = ["seqn", "ts", "node", "anchor", "ppm_offset", "dist",
                 "dist_bias", "dist_drift", "dist_drift_bias", "fppl", "rssi"]
RANGE_COLUMNS = ("dist", "dist_bias", "dist_drift", "dist_drift_bias")


def read_deployment(path: str | Path) -> pd.DataFrame:
    """Node table: id (str), x, y, z [m]."""
    d = pd.read_csv(path, sep="\t", skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    return pd.DataFrame({"id": d["node"].astype(int).astype(str),
                         "x": d["x"].astype(float),
                         "y": d["y"].astype(float),
                         "z": d["z"].astype(float)})


def read_sstwr(path: str | Path, range_column: str = "dist_drift") -> pd.DataFrame:
    """Directed range edges in the radiocal schema (EDGE_COLUMNS)."""
    if range_column not in RANGE_COLUMNS:
        raise ValueError(f"range_column must be one of {RANGE_COLUMNS}")
    raw = pd.read_csv(path, sep="\t")
    if list(raw.columns) != SSTWR_COLUMNS:
        raise ValueError(f"unexpected header {list(raw.columns)}")
    e = pd.DataFrame({"t": raw["ts"].astype(float),
                      "t_raw": raw["ts"].astype(float),
                      "from_id": raw["node"].astype(int).astype(str),
                      "to_id": raw["anchor"].astype(int).astype(str),
                      "range_m": raw[range_column].astype(float)})
    e = e.sort_values(["from_id", "to_id", "t"], kind="mergesort")
    e["seq"] = e.groupby(["from_id", "to_id"]).cumcount().astype("int64")
    e = e.sort_values(["seq", "from_id", "to_id"], kind="mergesort")
    return e[EDGE_COLUMNS].reset_index(drop=True)
