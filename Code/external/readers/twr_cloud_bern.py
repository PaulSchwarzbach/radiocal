"""Reader for the TWR-CLOUD-Bern evaluation scan (Xenakis, Di Maio, Braun).

Source: Zenodo 10.5281/zenodo.20277908 (v5, MIT). Only `40-agents.json` is
read, with json, which executes nothing. It is the anchor-free example over all
40 positions of the evaluation scan. Its layout (from the authors'
EvalScenarioParser.py and reproduce_twr_cloud_bern_release.py, read as text):

- "True Positions of Agents": {position id: {x, y, z}} in centimetres
  (laser-scan registration, Leica BLK360);
- "Iterations": two random draws, each with "Ranging samples between Agents":
  {"<a>-<b>": {"1": tof, "10": [tof, ...]}}, time of flight in ns, at most
  ten samples per directed link drawn without replacement from the full scan
  table (about 2,500 records per link).

Conversion (rules in Code/external/README.md): range = ToF times the
speed of light in air the authors use (299,702,547 m/s). Both iterations'
ten-sample sets are pooled. Equal values are kept, because ToF is quantized and
equal values are distinct measurements. seq is the rank of a sample within its
directed link (iteration, then sample index). No timestamps exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from radiocal.graph import EDGE_COLUMNS

C_AIR_M_PER_NS = 299_702_547 / 1e9


def read_scenario(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(nodes: id, x, y, z [m]; edges: EDGE_COLUMNS; stats)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    pos = {**d.get("True Positions of Anchors", {}),
           **d["True Positions of Agents"]}
    nodes = pd.DataFrame([(str(k), v["x"] / 100.0, v["y"] / 100.0, v["z"] / 100.0)
                          for k, v in pos.items()], columns=["id", "x", "y", "z"])
    nodes = nodes.sort_values("id", key=lambda s: s.astype(int)).reset_index(drop=True)
    rows = []
    links, with_samples = 0, 0
    for it in sorted(d["Iterations"], key=lambda i: i["ID"]):
        for key, s in it["Ranging samples between Agents"].items():
            links += 1
            ten = (s or {}).get("10") or []
            if ten:
                with_samples += 1
            a, b = key.split("-")
            rows += [(int(it["ID"]), j, a, b, float(tof) * C_AIR_M_PER_NS)
                     for j, tof in enumerate(ten)]
    e = pd.DataFrame(rows, columns=["iteration", "sample", "from_id", "to_id", "range_m"])
    e = e.sort_values(["from_id", "to_id", "iteration", "sample"], kind="mergesort")
    e["seq"] = e.groupby(["from_id", "to_id"]).cumcount().astype("int64")
    e["t"] = np.nan
    e["t_raw"] = np.nan
    e = e.sort_values(["seq", "from_id", "to_id"], kind="mergesort")
    stats = {"iterations": len(d["Iterations"]),
             "directed_links_listed": links,
             "directed_links_with_samples": with_samples,
             "anchors_in_file": len(d.get("Anchors", []))}
    return nodes, e[EDGE_COLUMNS].reset_index(drop=True), stats
