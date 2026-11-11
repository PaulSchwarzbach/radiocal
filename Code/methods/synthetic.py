"""Synthetic MeasurementGraphs from surveyed geometry (validation harness).

Every family's exact-recovery validation builds its data here so that the
identical production path - real mask specs via ``MeasurementGraph.apply_mask``,
the registry adapters, and ``evaluation.evaluate_estimate`` - runs on ranges
whose generating geometry is known exactly. The graphs are supersets like the
real sessions (anchor-anchor AND mobile-anchor directed edges, surveyed anchor
priors, mobile without prior), so masks strip exactly what they strip on data.

Deliberately NOT synthesized: the parquet/YAML loader path (exercised by every
real run) and campaign meta beyond what masking needs.

Range convention: by default ranges are 2-D (xy-plane) distances, matching the
2-D estimators, so exact recovery means aligned RMSE at solver precision.
``ranges_3d=True`` generates 3-D slant ranges instead, to quantify the
2-D-projection systematic where node heights differ (2024 mobile marker
z = -0.43 m vs anchors at about +1.3 m) - a geometry effect, not estimator
error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radiocal import MeasurementGraph
from radiocal.graph import EDGE_COLUMNS, NODE_COLUMNS, TRUTH_COLUMNS
from radiocal import frames

RATE_HZ = 12.5  # nominal Qorvo epoch rate; only sets the synthetic t axis


def _node_rows(ref: pd.DataFrame, mobile_id: str | None) -> pd.DataFrame:
    """Node table: surveyed anchor priors + optional prior-less mobile."""
    rows = [{"id": r["id"], "role": "anchor", "prior_type": "surveyed",
             "x": r["x"], "y": r["y"], "z": r.get("z", 0.0), "sigma_m": 0.005}
            for _, r in ref.iterrows()]
    if mobile_id is not None:
        rows.append({"id": mobile_id, "role": "mobile", "prior_type": "none",
                     "x": np.nan, "y": np.nan, "z": np.nan,
                     "sigma_m": np.nan})
    return pd.DataFrame(rows, columns=NODE_COLUMNS)


def _distances(p: np.ndarray, q: np.ndarray, ranges_3d: bool) -> np.ndarray:
    d = 3 if ranges_3d else 2
    return np.linalg.norm(p[..., :d] - q[..., :d], axis=-1)


def _edge_frame(recs: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(recs, columns=["t", "seq", "from_id", "to_id", "range_m"])
    df.insert(1, "t_raw", df["t"])
    return df[EDGE_COLUMNS]


def synthetic_static_graph(ref: pd.DataFrame, *, n_epochs: int = 50,
                           sigma: float = 0.0, seed: int = 0,
                           mobile: tuple[str, tuple] | None = None,
                           ranges_3d: bool = False) -> MeasurementGraph:
    """Superset graph of a static scene from surveyed anchors ``ref``
    (columns id, x, y[, z]) plus an optional static mobile
    ``(id, (x, y, z))`` - e.g. a set_a surveyed marker position.

    Every epoch carries both directed measurements of every node pair
    (anchor-anchor and mobile-anchor), exact distances + N(0, sigma).
    The mobile's truth is one row, as in the real quasi-static sessions.
    """
    rng = np.random.default_rng(seed)
    ids = list(ref["id"])
    P = ref.reindex(columns=["x", "y", "z"]).fillna(0.0).to_numpy(float)
    mobile_id = None
    if mobile is not None:
        mobile_id, mpos = mobile
        ids = ids + [mobile_id]
        P = np.vstack([P, np.asarray(mpos, float)])
    n = len(ids)
    ii, jj = np.where(~np.eye(n, dtype=bool))
    d = _distances(P[ii], P[jj], ranges_3d)
    recs = []
    for e in range(n_epochs):
        t = e / RATE_HZ
        r = d + (rng.normal(0.0, sigma, d.shape) if sigma > 0 else 0.0)
        recs += [(t, e, ids[i], ids[j], rij)
                 for i, j, rij in zip(ii, jj, r)]
    truth = pd.DataFrame(columns=TRUTH_COLUMNS)
    if mobile is not None:
        truth = pd.DataFrame([{
            "t": 0.0, "t_broker": np.nan, "frame": "enu",
            "node_id": mobile_id, "x": P[-1, 0], "y": P[-1, 1],
            "z": P[-1, 2], "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0,
        }], columns=TRUTH_COLUMNS)
    meta = {"campaign": "synthetic", "session_id": "static",
            "motion": "static" if mobile is None else "quasi_static_point",
            "truth_quality": "verified"}
    return MeasurementGraph(nodes=_node_rows(ref, mobile_id),
                            edges=_edge_frame(recs), truth=truth, meta=meta)


def synthetic_trajectory(ref: pd.DataFrame, *, rate_hz: float = RATE_HZ,
                         duration_s: float = 240.0, kind: str = "lissajous",
                         seed: int = 0) -> pd.DataFrame:
    """Smooth planar trajectory inside the anchor hull (columns seq, t, x, y).

    ``lissajous`` covers the area with bounded speed; ``line`` is a slow
    diagonal sweep (degenerate-geometry probe). No randomness is used unless
    a future kind needs it (seed reserved for that).
    """
    n = int(duration_s * rate_hz)
    t = np.arange(n) / rate_hz
    x0, x1 = ref["x"].min(), ref["x"].max()
    y0, y1 = ref["y"].min(), ref["y"].max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ax, ay = 0.35 * (x1 - x0), 0.35 * (y1 - y0)
    if kind == "lissajous":
        w = 2.0 * np.pi / duration_s
        x = cx + ax * np.sin(3.0 * w * t)
        y = cy + ay * np.sin(2.0 * w * t + 0.5)
    elif kind == "line":
        x = np.linspace(cx - ax, cx + ax, n)
        y = np.linspace(cy - ay, cy + ay, n)
    else:
        raise ValueError(f"unknown trajectory kind {kind!r}")
    return pd.DataFrame({"seq": np.arange(n), "t": t, "x": x, "y": y})


def synthetic_slam_graph(ref: pd.DataFrame, traj: pd.DataFrame, *,
                         mobile_id: str = "DD98", sigma: float = 0.0,
                         seed: int = 0,
                         vicon_tf: frames.Transform | None = None,
                         coverage: tuple[tuple[float, float], ...] | None = None,
                         truth_z: float = 0.0,
                         ranges_3d: bool = False) -> MeasurementGraph:
    """Superset graph of a dynamic scene: mobile on ``traj`` (seq, t, x, y)
    among surveyed anchors ``ref``.

    Per epoch: both directed mobile-anchor measurements of every anchor plus
    both directed anchor-anchor measurements (so fully_meshed/multihop views of
    the same graph stay meaningful), exact distances + N(0, sigma).

    Truth mimics the bursty Vicon stream: rows only inside the ``coverage``
    windows (fractions of the run, e.g. ((0.0, 0.2), (0.35, 0.5))), positions
    expressed in the frame reached by ``vicon_tf.inverse()`` if a
    vicon->target transform is given (frame label "vicon_local"), else in the
    generating frame ("enu"). ``truth_z`` sets the constant truth height.
    """
    rng = np.random.default_rng(seed)
    aid = list(ref["id"])
    A = ref.reindex(columns=["x", "y", "z"]).fillna(0.0).to_numpy(float)
    na = len(aid)
    ai, aj = np.where(~np.eye(na, dtype=bool))
    d_aa = _distances(A[ai], A[aj], ranges_3d)

    recs = []
    for row in traj.itertuples(index=False):
        p = np.array([row.x, row.y, truth_z])
        d_ma = _distances(np.broadcast_to(p, (na, 3)), A, ranges_3d)
        noise = rng.normal(0.0, sigma, 2 * na + len(d_aa)) if sigma > 0 \
            else np.zeros(2 * na + len(d_aa))
        for k, a in enumerate(aid):
            recs.append((row.t, row.seq, mobile_id, a, d_ma[k] + noise[k]))
            recs.append((row.t, row.seq, a, mobile_id, d_ma[k] + noise[na + k]))
        r_aa = d_aa + noise[2 * na:]
        recs += [(row.t, row.seq, aid[i], aid[j], r)
                 for i, j, r in zip(ai, aj, r_aa)]

    keep = np.ones(len(traj), dtype=bool)
    if coverage is not None:
        frac = np.arange(len(traj)) / max(len(traj) - 1, 1)
        keep = np.zeros(len(traj), dtype=bool)
        for lo, hi in coverage:
            keep |= (frac >= lo) & (frac < hi)
    tr = traj[keep]
    pts = np.column_stack([tr["x"], tr["y"], np.full(len(tr), truth_z)])
    frame = "enu"
    if vicon_tf is not None:
        pts = vicon_tf.inverse().apply(pts)
        frame = "vicon_local"
    truth = pd.DataFrame({
        "t": tr["t"].to_numpy(), "t_broker": np.nan, "frame": frame,
        "node_id": mobile_id, "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
        "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0,
    }, columns=TRUTH_COLUMNS)
    meta = {"campaign": "synthetic", "session_id": "dyn", "motion": "dynamic"}
    return MeasurementGraph(nodes=_node_rows(ref, mobile_id),
                            edges=_edge_frame(recs), truth=truth, meta=meta)
