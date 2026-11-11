"""Total-station coordinate file parser (Torgau campaigns).

File format: ``name,E,N,h,`` per line (ETRS89/UTM zone 33N, EPSG:25833;
heights DE_DHHN2016). Names may repeat (repeated survey shots); repeats are
averaged unless they disagree by more than `conflict_tol_m`, in which case the
point is flagged and left un-averaged for manual resolution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def parse_totalstation(path: str | Path) -> pd.DataFrame:
    """Raw survey shots. Columns: name, E, N, h (one row per shot)."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().rstrip(",").split(",")]
            if len(parts) < 4 or not parts[0]:
                continue
            rows.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return pd.DataFrame(rows, columns=["name", "E", "N", "h"])


def dedupe_shots(shots: pd.DataFrame, conflict_tol_m: float = 0.10) -> pd.DataFrame:
    """One row per point name. Columns: name, E, N, h, n_shots, spread_m, conflict.

    spread_m is the max pairwise 3D distance among the shots of a name.
    conflict=True when spread_m > conflict_tol_m; coordinates are then the
    FIRST shot (not the average) and must be resolved manually.
    """
    out = []
    for name, g in shots.groupby("name", sort=False):
        xyz = g[["E", "N", "h"]].to_numpy()
        spread = 0.0
        if len(xyz) > 1:
            d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
            spread = float(d.max())
        conflict = spread > conflict_tol_m
        e, n, h = (xyz[0] if conflict else xyz.mean(axis=0))
        out.append((name, e, n, h, len(xyz), spread, conflict))
    return pd.DataFrame(out, columns=["name", "E", "N", "h", "n_shots", "spread_m", "conflict"])


def to_local_enu(df: pd.DataFrame, origin: tuple[float, float, float]) -> pd.DataFrame:
    """Add x, y, z columns: UTM E/N/h minus the campaign-local origin."""
    df = df.copy()
    df["x"] = df["E"] - origin[0]
    df["y"] = df["N"] - origin[1]
    df["z"] = df["h"] - origin[2]
    return df
