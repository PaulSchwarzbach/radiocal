"""Shared machinery of the slam_bipartite family.

The slam_bipartite mask leaves only the per-epoch mobile-anchor ranges (no
anchor-anchor pairs, no anchor priors), so both family methods (fg_slam,
ekf_slam) estimate the anchor geometry jointly with the mobile trajectory.
This module holds everything they share, each constant stated once and
a-priori (nothing is tuned against the evaluation metric):

- V_MAX_MS = 1.5 m/s: pedestrian speed bound. The shared motion model is a
  random walk on position (no velocity state): between consecutive ranging
  epochs the mobile moves with per-axis process std

      q_step = V_MAX_MS * dt,   dt = median epoch spacing of the edge stream.

  The motion prior is always Gaussian; the robust kernels of the factorial
  kernel axis apply to the RANGE factors only (the kernel is the factorial
  axis, the motion prior is shared infrastructure).
- POSE_STRIDE = 25: subsampling of the epochs entering the joint
  classical-MDS initialization (init only - every epoch enters the solves).

Initialization (``bipartite_init``): joint classical MDS over the anchors
plus every POSE_STRIDE-th epoch pose, on the observed pose-anchor ranges,
with the missing anchor-anchor / pose-pose entries filled by shortest-path
completion (``mds.complete_distances``; 2-hop pose-anchor-pose paths connect
the bipartite graph). This is the same init family as the static suite
(classical MDS on completed distances). The full pose track is linearly
interpolated across epoch index between the subsampled poses - init only;
the no-interpolation rule for the Vicon truth applies to evaluation, not to
the solver's starting point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radiocal import MeasurementGraph

import mds
from common import default_anchor_order, pair_range_matrix

V_MAX_MS = 1.5     # a-priori pedestrian speed bound (m/s), stated once
POSE_STRIDE = 25   # init subsampling of epoch poses, stated once


def resolve_mobile_id(graph: MeasurementGraph) -> str:
    """The single mobile node present in the view's edges."""
    mobiles = sorted(graph.nodes.loc[graph.nodes["role"] == "mobile", "id"])
    used = set(graph.edges["from_id"]) | set(graph.edges["to_id"])
    mobiles = [m for m in mobiles if m in used]
    if len(mobiles) != 1:
        raise ValueError(f"slam_bipartite needs exactly one ranging mobile, "
                         f"found {mobiles}")
    return mobiles[0]


def epoch_stream(graph: MeasurementGraph, anchor_order: list[str],
                 mobile_id: str) -> dict:
    """Per-epoch mobile-anchor measurement arrays of a (masked) graph.

    Symmetrizes the directed edges, keeps mobile-anchor pairs whose anchor
    is in ``anchor_order``, and returns a dict of parallel arrays sorted by
    (epoch, anchor index):

        seqs (E,)   sorted unique epoch counters
        t    (E,)   per-epoch time (min t of the epoch's kept pairs)
        e    (m,)   epoch index into seqs of each measurement
        k    (m,)   anchor index into anchor_order
        r    (m,)   symmetrized range (m)
    """
    sym = graph.symmetrized()
    idx = {a: i for i, a in enumerate(anchor_order)}
    is_ma = (sym["node_a"] == mobile_id) & sym["node_b"].isin(idx)
    is_am = (sym["node_b"] == mobile_id) & sym["node_a"].isin(idx)
    sym = sym[is_ma | is_am].copy()
    if not len(sym):
        raise ValueError("no mobile-anchor measurements in this view")
    sym["anchor"] = np.where(sym["node_a"] == mobile_id,
                             sym["node_b"], sym["node_a"])
    sym["k"] = sym["anchor"].map(idx)
    sym = sym.sort_values(["seq", "k"], kind="stable")

    seqs, e = np.unique(sym["seq"].to_numpy(), return_inverse=True)
    t = sym.groupby("seq")["t"].min().reindex(seqs).to_numpy(dtype=float)
    return {"seqs": seqs, "t": t, "e": e.astype(int),
            "k": sym["k"].to_numpy(dtype=int),
            "r": sym["range_m"].to_numpy(dtype=float)}


def median_epoch_dt(t_epochs: np.ndarray) -> float:
    """Median spacing of the (sorted) per-epoch times."""
    dt = np.diff(np.sort(np.asarray(t_epochs, dtype=float)))
    if not len(dt):
        raise ValueError("need at least two epochs for a motion model")
    return float(np.median(dt))


def q_step_m(t_epochs: np.ndarray) -> float:
    """Per-epoch random-walk process std (m): V_MAX_MS * median epoch dt."""
    return V_MAX_MS * median_epoch_dt(t_epochs)


def anchor_pair_arrays(graph: MeasurementGraph, anchor_order: list[str]
                       ) -> dict | None:
    """Per-pair anchor-anchor median ranges of a (masked) view, as parallel
    arrays i (m,), j (m,), r (m,) over ``anchor_order`` indices (upper
    triangle, finite pairs only). None when the view carries no
    anchor-anchor pairs (the slam_bipartite case). Aggregation is the
    session median, identical to the static suite's ``pair_range_matrix``.
    """
    M = pair_range_matrix(graph, anchor_order)
    iu, ju = np.triu_indices(len(anchor_order), k=1)
    fin = np.isfinite(M[iu, ju])
    if not fin.any():
        return None
    return {"i": iu[fin].astype(int), "j": ju[fin].astype(int),
            "r": M[iu[fin], ju[fin]].astype(float), "matrix": M}


def bipartite_init(stream: dict, n_anchors: int,
                   pose_stride: int = POSE_STRIDE,
                   aa_matrix: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Joint classical-MDS init of anchors and poses from a bipartite stream.

    ``stream`` is the dict of ``epoch_stream``. Builds the joint distance
    matrix over [anchors, every pose_stride-th epoch pose] from the observed
    pose-anchor median ranges, completes it by shortest paths and embeds it
    with classical MDS (Torgerson). The full pose track is linearly
    interpolated across epoch index between the subsampled poses (init
    only). ``aa_matrix`` (n x n median anchor-anchor ranges) fills the
    anchor block where the view provides inter-anchor pairs (the
    mobile_assisted dynamic case); the bipartite case leaves it to the
    shortest-path completion. Returns (anchors0 (n, 2), poses0 (E, 2), info).
    """
    n = int(n_anchors)
    E = len(stream["seqs"])
    sel = np.arange(0, E, int(pose_stride))
    m = len(sel)
    pose_of = np.full(E, -1)
    pose_of[sel] = np.arange(m)

    M = np.full((n + m, n + m), np.nan)
    if aa_matrix is not None:
        M[:n, :n] = np.asarray(aa_matrix, dtype=float)[:n, :n]
    use = pose_of[stream["e"]] >= 0
    # median over duplicates of one (epoch, anchor) cell
    df = pd.DataFrame({"p": pose_of[stream["e"][use]],
                       "k": stream["k"][use], "r": stream["r"][use]})
    med = df.groupby(["p", "k"])["r"].median()
    for (p, k), r in med.items():
        M[n + p, k] = M[k, n + p] = r

    D, n_completed = mds.complete_distances(M)
    X, _ = mds.classical_mds(D, dim=2)
    anchors0 = X[:n]
    poses_sel = X[n:]

    poses0 = np.column_stack([
        np.interp(np.arange(E), sel, poses_sel[:, 0]),
        np.interp(np.arange(E), sel, poses_sel[:, 1])])
    info = {"init": "bipartite_mds", "init_pose_stride": int(pose_stride),
            "n_init_poses": m, "n_completed_pairs": int(n_completed)}
    return anchors0, poses0, info


def order_and_stream(graph: MeasurementGraph
                     ) -> tuple[list[str], str, dict]:
    """(anchor_order, mobile_id, epoch stream) of a slam_bipartite view."""
    order = default_anchor_order(graph)
    mobile_id = resolve_mobile_id(graph)
    return order, mobile_id, epoch_stream(graph, order, mobile_id)


def fallback_sigma(graph: MeasurementGraph, order: list[str],
                   mobile_id: str) -> float:
    """Standalone-use noise scale when the caller passes sigma_m=None.

    The benchmark driver ALWAYS passes the session's fully_meshed sigma; this
    fallback exists only so the modules run standalone on a bipartite view.
    It is kernels.session_sigma over anchors + mobile, whose per-pair
    scatter is motion-contaminated on a dynamic session - do not use it for
    benchmark numbers.
    """
    import kernels
    return kernels.session_sigma(graph, list(order) + [mobile_id])


def trajectory_frame(stream: dict, P: np.ndarray) -> pd.DataFrame:
    """The frozen trajectory contract: one row per epoch (seq, t, x, y)."""
    return pd.DataFrame({"seq": stream["seqs"], "t": stream["t"],
                         "x": P[:, 0], "y": P[:, 1]})
