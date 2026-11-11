"""Readers for the mqtt-logger SQLite databases (Torgau, set B, 2025-10-22).

Schema: LOG(ID, RUN_ID, UNIX_TIME, TOPIC, MESSAGE BLOB) + RUN(ID, START/END).
UNIX_TIME is the broker-side receive time and serves as the campaign
reference clock. Relevant topics:

  tracking/objects/<name>   Vicon 6-DoF ground truth, JSON
  auto_pos/data             the Qorvo stream (subsampled duplicate of the
                            backup .log files) - used for clock matching only
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

VICON_TOPIC_PREFIX = "tracking/objects/"

# start of a Qorvo data record: "[<float>, <int>," ; idle records have "None" as 2nd field
_AUTOPOS_HEAD = re.compile(rb"^\[([0-9.]+),\s*(\d+),")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)


def list_topics(db_path: str | Path) -> pd.DataFrame:
    with _connect(db_path) as db:
        return pd.read_sql_query(
            "SELECT TOPIC AS topic, COUNT(*) AS n FROM LOG GROUP BY TOPIC", db)


def read_vicon(db_path: str | Path, topic: str, node_id: str) -> pd.DataFrame:
    """Vicon poses for one tracked object as a ground-truth table.

    Columns: t (Vicon message timestamp), t_broker (mqtt-logger receive time),
    frame, node_id, x, y, z, rot_x, rot_y, rot_z. Coordinates are in the local
    Vicon frame; rotations are the Vicon XYZ values (radians).
    """
    rows = []
    with _connect(db_path) as db:
        cur = db.execute(
            "SELECT UNIX_TIME, MESSAGE FROM LOG WHERE TOPIC=? ORDER BY UNIX_TIME", (topic,))
        for t_broker, msg in cur:
            m = json.loads(msg)
            rows.append((m["timestamp"], t_broker, m["frame_number"], node_id,
                         m["position"]["x"], m["position"]["y"], m["position"]["z"],
                         m["rotation"]["x"], m["rotation"]["y"], m["rotation"]["z"]))
    return pd.DataFrame(rows, columns=["t", "t_broker", "frame", "node_id",
                                       "x", "y", "z", "rot_x", "rot_y", "rot_z"])


def read_autopos_heads(db_path: str | Path, topic: str = "auto_pos/data") -> pd.DataFrame:
    """Header fields of the Qorvo messages captured in the DB.

    Columns: t_broker (broker receive time), t_raw (embedded Qorvo clock), seq.
    Idle records (seq is None) are skipped. Used to estimate the per-run clock
    offset by matching seq against the full-rate backup log.
    """
    rows = []
    with _connect(db_path) as db:
        cur = db.execute("SELECT UNIX_TIME, MESSAGE FROM LOG WHERE TOPIC=?", (topic,))
        for t_broker, msg in cur:
            m = _AUTOPOS_HEAD.match(msg)
            if m:
                rows.append((t_broker, float(m.group(1)), int(m.group(2))))
    return (pd.DataFrame(rows, columns=["t_broker", "t_raw", "seq"])
            .sort_values("t_broker").reset_index(drop=True))


def run_span(db_path: str | Path) -> tuple[float, float]:
    """(min, max) broker time over all LOG rows."""
    with _connect(db_path) as db:
        lo, hi = db.execute("SELECT MIN(UNIX_TIME), MAX(UNIX_TIME) FROM LOG").fetchone()
    return float(lo), float(hi)
