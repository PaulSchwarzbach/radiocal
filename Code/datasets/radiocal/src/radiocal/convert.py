"""Builders: raw campaign files -> canonical Parquet + YAML session directories."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .io import coords as coords_io
from .io import mqtt_db, qorvo_log


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------- clock

@dataclass
class ClockOffset:
    """vicon_clock_time ~= t_raw + offset_s.

    Broker receive times (UNIX_TIME) are unusable as timestamps: the logging
    chain queued up during each run (lag growing from ~0.1 s to tens of
    seconds, for BOTH the Qorvo and the Vicon stream). Only the lower envelope
    of (receive time - embedded time), reached at run start with an empty
    queue, reflects clock offset + minimal transport latency. The minimal
    latencies of the two streams cancel approximately, so

        offset = min(t_broker - t_raw_qorvo) - min(t_broker - t_vicon)

    puts the Qorvo ranges on the Vicon (tracking PC) clock, which is the
    ground-truth clock. Uncertainty ~0.1 s; finer alignment is a downstream
    motion-correlation task.
    """
    offset_s: float
    uncertainty_s: float
    n_matched: int
    min_env_qorvo_s: float
    min_env_vicon_s: float
    queue_lag_median_s: float
    queue_lag_max_s: float
    method: str = "min_envelope"

    def apply(self, t_raw: np.ndarray) -> np.ndarray:
        return t_raw + self.offset_s


def estimate_clock_offset(db_path: str | Path, log_df: pd.DataFrame,
                          vicon_topic: str) -> ClockOffset:
    """Per-run offset between the embedded Qorvo clock and the Vicon clock."""
    heads = mqtt_db.read_autopos_heads(db_path)
    if heads.empty:
        raise ValueError(f"no auto_pos/data messages in {db_path}")
    matched = heads[heads["seq"].isin(set(log_df["seq"].unique()))]
    if len(matched) < 10:
        raise ValueError(f"only {len(matched)} seq matches between {db_path} and log")
    diff_q = matched["t_broker"].to_numpy() - matched["t_raw"].to_numpy()

    vic = mqtt_db.read_vicon(db_path, vicon_topic, node_id="_")
    if vic.empty:
        raise ValueError(f"no Vicon messages on {vicon_topic} in {db_path}")
    lag_v = vic["t_broker"].to_numpy() - vic["t"].to_numpy()

    return ClockOffset(
        offset_s=float(diff_q.min() - lag_v.min()),
        uncertainty_s=0.1,
        n_matched=int(len(matched)),
        min_env_qorvo_s=float(diff_q.min()),
        min_env_vicon_s=float(lag_v.min()),
        queue_lag_median_s=float(np.median(diff_q)),
        queue_lag_max_s=float(diff_q.max()),
    )


# ------------------------------------------------------------------ registry

def build_registry(coord_path: str | Path,
                   anchors: list[str],
                   mobiles: dict[str, dict],
                   excluded: dict[str, str],
                   origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
                   sigma_m: float = 0.01,
                   crs: dict | None = None,
                   out_path: str | Path | None = None) -> dict:
    """Build the nodes.yaml registry from the surveyed coordinate file.

    anchors  : node ids with surveyed positions that actively range
    mobiles  : {id: {"groundtruth": {...}}} - nodes with no positional prior
    excluded : {id: reason} - surveyed nodes excluded from graphs (e.g. never range)
    origin   : campaign-local ENU origin subtracted from the file coordinates
    crs      : coordinate-system block for nodes.yaml; defaults to UTM33N/DHHN2016
    """
    shots = coords_io.parse_totalstation(coord_path)
    points = coords_io.to_local_enu(coords_io.dedupe_shots(shots), origin)
    by_name = points.set_index("name")

    def surveyed_entry(nid: str) -> dict:
        p = by_name.loc[nid]
        if bool(p["conflict"]):
            raise ValueError(f"conflicting survey shots for node {nid}")
        return {
            "type": "surveyed", "source": "totalstation",
            "xyz_local": [round(float(p["x"]), 4), round(float(p["y"]), 4), round(float(p["z"]), 4)],
            "sigma_m": sigma_m,
        }

    nodes = []
    for nid in anchors:
        nodes.append({"id": nid, "role": "anchor", "prior": surveyed_entry(nid)})
    for nid, extra in mobiles.items():
        nodes.append({"id": nid, "role": "mobile", "prior": {"type": "none"}, **extra})
    for nid, reason in excluded.items():
        entry = {"id": nid, "role": "anchor", "status": reason}
        if nid in by_name.index:
            entry["prior"] = surveyed_entry(nid)
        nodes.append(entry)

    controls = points[~points["name"].isin([n["id"] for n in nodes])]
    if crs is None:
        crs = {
            "epsg": 25833, "height_system": "DE_DHHN2016",
            "local_origin_utm": [float(origin[0]), float(origin[1]), float(origin[2])],
            "axes": "x=East, y=North, z=Up (metres, local origin)",
        }
    registry = {
        "crs": crs,
        "nodes": nodes,
        "control_points": [
            {"name": r.name, "xyz_local": [round(r.x, 4), round(r.y, 4), round(r.z, 4)],
             "n_shots": int(r.n_shots), "spread_m": round(float(r.spread_m), 4),
             **({"conflict": True} if r.conflict else {})}
            for r in controls.itertuples()
        ],
    }
    if out_path:
        Path(out_path).write_text(yaml.safe_dump(registry, sort_keys=False),
                                  encoding="utf-8", newline="\n")
    return registry


# ------------------------------------------------------------------- session

# failed TWR exchanges / counter wrap-arounds show up as negative values or
# absurd distances (observed: -2.6e9 and +1.9e9 m); genuine long-multipath
# outliers (< ~100 m) are measurements and stay in
VALID_RANGE_M = (0.0, 500.0)


def _drop_invalid_ranges(ranges: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    ok = ranges["range_m"].between(*VALID_RANGE_M, inclusive="neither")
    return ranges[ok].reset_index(drop=True), int((~ok).sum())


def build_session_ranges_only(log_path: str | Path,
                              out_dir: str | Path,
                              session_id: str,
                              motion: str,
                              campaign: str,
                              truth: pd.DataFrame | None = None,
                              notes: list[str] | None = None,
                              extra_meta: dict | None = None) -> Path:
    """Convert a backup log WITHOUT an accompanying MQTT DB (no Vicon, no clock).

    Used for set A (2025-03-13): records carry no timestamps (t = t_raw = NaN);
    epoch ordering is given by seq. `truth` (optional) is a ground-truth table
    in the groundtruth.parquet schema - e.g. one surveyed quasi-static position
    of the tag node.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ranges = qorvo_log.parse_qorvo_log(log_path)
    ranges, n_invalid = _drop_invalid_ranges(ranges)
    ranges.insert(0, "t", np.nan)
    ranges.to_parquet(out / "ranges.parquet", index=False)
    if truth is not None:
        truth.to_parquet(out / "groundtruth.parquet", index=False)

    meta = {
        "session_id": session_id,
        "campaign": campaign,
        "motion": motion,
        "t_start": None, "t_end": None,
        "sources": [{"path": Path(log_path).name, "role": "ranges", "sha256": sha256(log_path)}],
        "clock": {"reference": None,
                  "note": "records carry no timestamps; use seq for epoch ordering"},
        "counts": {
            "range_epochs": int(ranges["seq"].nunique()),
            "directed_ranges": int(len(ranges)),
            "dropped_invalid": n_invalid,
            "truth_poses": 0 if truth is None else int(len(truth)),
        },
    }
    if notes:
        meta["notes"] = notes
    if extra_meta:
        meta.update(extra_meta)
    (out / "session.yaml").write_text(yaml.safe_dump(meta, sort_keys=False),
                                      encoding="utf-8", newline="\n")
    return out


def build_session(log_path: str | Path,
                  db_path: str | Path,
                  out_dir: str | Path,
                  session_id: str,
                  motion: str,
                  vicon_topic: str,
                  vicon_node_id: str,
                  campaign: str = "set_b",
                  db_source: dict | None = None) -> Path:
    """Convert one (backup log, mqtt DB) pair into a canonical session directory.

    Writes ranges.parquet (full-rate log data on the unified broker clock),
    groundtruth.parquet (Vicon poses) and session.yaml. `db_source` is the
    `sources` entry of the database as the caller verified it (for example
    both hashes of a database whose archived copy is stripped); the default
    records the hash of the file read.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ranges = qorvo_log.parse_qorvo_log(log_path)
    clock = estimate_clock_offset(db_path, ranges, vicon_topic)
    ranges, n_invalid = _drop_invalid_ranges(ranges)
    ranges.insert(0, "t", clock.apply(ranges["t_raw"].to_numpy()))
    ranges.to_parquet(out / "ranges.parquet", index=False)

    truth = mqtt_db.read_vicon(db_path, vicon_topic, vicon_node_id)
    truth.to_parquet(out / "groundtruth.parquet", index=False)

    vicon_rate = float((len(truth) - 1) / (truth["t"].iloc[-1] - truth["t"].iloc[0])) if len(truth) > 1 else None
    n_epochs = int(ranges["seq"].nunique())
    db_entry = {"path": Path(db_path).name, "role": "groundtruth+clock",
                "sha256": sha256(db_path)}
    if db_source is not None:
        db_entry.update({k: v for k, v in db_source.items() if k != "path"})
    meta = {
        "session_id": session_id,
        "campaign": campaign,
        "motion": motion,
        "t_start": float(ranges["t"].min()),
        "t_end": float(ranges["t"].max()),
        "sources": [
            {"path": Path(log_path).name, "role": "ranges", "sha256": sha256(log_path)},
            db_entry,
        ],
        "clock": {"reference": "vicon_timestamp", **asdict(clock)},
        "counts": {
            "range_epochs": n_epochs,
            "directed_ranges": int(len(ranges)),
            "dropped_invalid": n_invalid,
            "truth_poses": int(len(truth)),
        },
        "vicon": {
            "topic": vicon_topic, "object": vicon_topic.rsplit("/", 1)[-1],
            "node_id": vicon_node_id, "frame": "vicon_local",
            "rate_hz": round(vicon_rate, 2) if vicon_rate else None,
        },
        "transform_vicon_to_local": None,  # Umeyama fit is a downstream task
    }
    (out / "session.yaml").write_text(yaml.safe_dump(meta, sort_keys=False),
                                      encoding="utf-8", newline="\n")
    return out
