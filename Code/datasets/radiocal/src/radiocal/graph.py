"""MeasurementGraph: the single data object all calibration methods consume.

A measurement graph is
  nodes  : one row per node   (id, role, prior_type, x, y, z, sigma_m)
  edges  : one row per directed range measurement (t, t_raw, seq, from_id, to_id, range_m)
  truth  : ground-truth poses of mobile nodes (t, frame, node_id, x, y, z, rot_x, rot_y, rot_z)
  meta   : merged campaign + session metadata (dict)

Coordinates in `nodes` are campaign-local ENU; `truth` is in the frame named by
meta["vicon"]["frame"] until a transform is estimated (see session.yaml).
Nodes without a positional prior carry NaN coordinates and prior_type "none".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

NODE_COLUMNS = ["id", "role", "prior_type", "x", "y", "z", "sigma_m"]
EDGE_COLUMNS = ["t", "t_raw", "seq", "from_id", "to_id", "range_m"]
TRUTH_COLUMNS = ["t", "t_broker", "frame", "node_id",
                 "x", "y", "z", "rot_x", "rot_y", "rot_z"]


@dataclass
class MeasurementGraph:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    truth: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TRUTH_COLUMNS))
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- basics

    def __post_init__(self):
        missing = [c for c in NODE_COLUMNS if c not in self.nodes.columns]
        if missing:
            raise ValueError(f"nodes is missing columns {missing}")
        missing = [c for c in EDGE_COLUMNS if c not in self.edges.columns]
        if missing:
            raise ValueError(f"edges is missing columns {missing}")

    @property
    def node_ids(self) -> list[str]:
        return list(self.nodes["id"])

    def roles(self) -> dict[str, str]:
        return dict(zip(self.nodes["id"], self.nodes["role"]))

    def __repr__(self) -> str:  # noqa: D105
        return (f"MeasurementGraph({len(self.nodes)} nodes, "
                f"{len(self.edges)} directed edges, {len(self.truth)} truth poses, "
                f"session={self.meta.get('session_id', '?')})")

    # ----------------------------------------------------------------- views

    def symmetrized(self, agg: str = "mean") -> pd.DataFrame:
        """Undirected edge list: reciprocal measurements of one (seq, pair)
        aggregated with `agg`. Columns: t, seq, node_a, node_b, range_m, n_dir."""
        e = self.edges.copy()
        pair = pd.DataFrame({
            "node_a": e[["from_id", "to_id"]].min(axis=1),
            "node_b": e[["from_id", "to_id"]].max(axis=1),
        })
        e = pd.concat([e[["t", "seq", "range_m"]], pair], axis=1)
        g = e.groupby(["seq", "node_a", "node_b"], as_index=False).agg(
            t=("t", "min"), range_m=("range_m", agg), n_dir=("range_m", "size"))
        return g[["t", "seq", "node_a", "node_b", "range_m", "n_dir"]]

    def edges_between(self, role_a: str, role_b: str) -> pd.DataFrame:
        """Directed edges whose endpoint roles are (role_a -> role_b)."""
        roles = self.roles()
        m = (self.edges["from_id"].map(roles) == role_a) & \
            (self.edges["to_id"].map(roles) == role_b)
        return self.edges[m]

    def time_window(self, t0: float, t1: float) -> "MeasurementGraph":
        e = self.edges[(self.edges["t"] >= t0) & (self.edges["t"] <= t1)]
        tr = self.truth
        if len(tr):
            tr = tr[(tr["t"] >= t0) & (tr["t"] <= t1)]
        return replace(self, edges=e.reset_index(drop=True), truth=tr.reset_index(drop=True))

    def priors(self) -> pd.DataFrame:
        """Nodes carrying a positional unary prior (C1: reference info)."""
        return self.nodes[self.nodes["prior_type"] != "none"]

    def epochs(self):
        """Iterate (t, edges_of_epoch) grouped by the Qorvo frame counter."""
        for seq, df in self.edges.groupby("seq", sort=True):
            yield float(df["t"].min()), df

    def apply_mask(self, mask) -> "MeasurementGraph":
        """Apply a topology mask (name, path, or loaded spec dict). See radiocal.masks."""
        from . import masks
        if isinstance(mask, (str,)):
            mask = masks.load_mask(mask, search_dirs=self.meta.get("_mask_dirs"))
        return masks.apply_mask(self, mask)
