"""Coordinate frames and transformations for self-calibration.

Self-calibration estimates node positions from ranges alone, so the estimate
lives in a frame that is only defined up to a similarity transform (the gauge
freedom): translation (3), rotation (3), reflection (1), and - for some
methods - scale (1). Every campaign therefore juggles several frames:

  enu         campaign-local East/North/Up, from the Total Station survey
              (nodes.yaml). The publication frame; all C5 metrics are reported
              here after alignment.
  vicon       the motion-capture system's own local frame (groundtruth.parquet
              of set_b). Metric, but position/orientation relative to
              enu unknown until estimated.
  radio_free  the frame a gauge-free estimator converges in. Arbitrary datum;
              only inter-node geometry is meaningful.
  radio_conv  conventional gauge fix used by the closed-form / grid-based
              approaches (Jung & Schwarzbach): a declared origin anchor at
              (0,0), a second anchor on the positive x-axis, a third with
              positive y. Removes the gauge freedom by convention, not by
              measurement.

The tools here convert between these ideas:

  Transform          similarity transform y = s * R @ x + t, composable,
                     invertible, serializable to YAML (session.yaml slots).
  umeyama(...)       least-squares similarity fit between corresponding point
                     sets (Umeyama 1991) - THE alignment for evaluation
                     ("anchor RMSE after Procrustes") and for estimating
                     vicon->enu once corresponding points exist.
  fix_gauge(...)     build the radio_free -> radio_conv transform from three
                     declared node ids (paper convention).
  align(...)         id-matched Umeyama between two node tables + residuals;
                     use with with_scale=False for rigid (metric-preserving)
                     evaluation, True to also absorb a scale error.

Conventions: points are (n, d) arrays or DataFrames with x, y(, z) columns;
d = 2 or 3. Rotations are proper unless allow_reflection=True - reflections
matter because range-only geometry is chirality-blind (a mirrored solution
fits the ranges equally well; 2D methods resolve it by the positive-y
convention, evaluation resolves it by allowing reflection in the alignment).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

XYZ = ["x", "y", "z"]


def _as_points(obj, dim: int | None = None) -> np.ndarray:
    if isinstance(obj, pd.DataFrame):
        cols = XYZ[: dim or (3 if "z" in obj.columns else 2)]
        obj = obj[cols].to_numpy(dtype=float)
    pts = np.atleast_2d(np.asarray(obj, dtype=float))
    return pts[:, :dim] if dim else pts


@dataclass
class Transform:
    """Similarity transform y = s * R @ x + t between two named frames."""
    R: np.ndarray
    t: np.ndarray
    s: float = 1.0
    src: str = ""
    dst: str = ""

    @property
    def dim(self) -> int:
        return self.R.shape[0]

    def apply(self, points) -> np.ndarray:
        pts = _as_points(points, self.dim)
        return self.s * pts @ self.R.T + self.t

    def apply_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[XYZ[: self.dim]] = self.apply(df)
        return out

    def inverse(self) -> "Transform":
        Rinv = self.R.T
        return Transform(Rinv, -Rinv @ self.t / self.s, 1.0 / self.s,
                         src=self.dst, dst=self.src)

    def compose(self, first: "Transform") -> "Transform":
        """Transform doing `first`, then self (self @ first)."""
        return Transform(self.R @ first.R, self.s * self.R @ first.t + self.t,
                         self.s * first.s, src=first.src, dst=self.dst)

    def is_reflection(self) -> bool:
        return bool(np.linalg.det(self.R) < 0)

    def to_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst, "scale": float(self.s),
                "rotation": [[float(v) for v in row] for row in self.R],
                "translation": [float(v) for v in self.t],
                "model": "y = scale * R @ x + t"}

    @classmethod
    def from_dict(cls, d: dict) -> "Transform":
        return cls(np.asarray(d["rotation"], dtype=float),
                   np.asarray(d["translation"], dtype=float),
                   float(d.get("scale", 1.0)), src=d.get("src", ""), dst=d.get("dst", ""))

    @classmethod
    def identity(cls, dim: int = 3, src: str = "", dst: str = "") -> "Transform":
        return cls(np.eye(dim), np.zeros(dim), 1.0, src=src, dst=dst)


def umeyama(src_pts, dst_pts, with_scale: bool = False,
            allow_reflection: bool = False, src: str = "", dst: str = "") -> Transform:
    """Least-squares similarity transform mapping src_pts onto dst_pts.

    Umeyama (1991), IEEE TPAMI 13(4). allow_reflection lets det(R) = -1 win
    when it fits better (range-only estimates are chirality-blind).
    """
    A, B = _as_points(src_pts), _as_points(dst_pts)
    if A.shape != B.shape or len(A) < A.shape[1]:
        raise ValueError(f"need matching point sets with >= dim points, got {A.shape}, {B.shape}")
    mu_a, mu_b = A.mean(axis=0), B.mean(axis=0)
    Ac, Bc = A - mu_a, B - mu_b
    cov = Bc.T @ Ac / len(A)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(A.shape[1])
    if not allow_reflection and np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() * len(A) / (Ac ** 2).sum()) if with_scale else 1.0
    t = mu_b - s * R @ mu_a
    return Transform(R, t, s, src=src, dst=dst)


def align(est: pd.DataFrame, ref: pd.DataFrame, ids: list[str] | None = None,
          dim: int = 3, with_scale: bool = False,
          allow_reflection: bool = True) -> tuple[Transform, pd.DataFrame]:
    """Id-matched Umeyama alignment of two node tables (columns id, x, y[, z]).

    Returns (transform est->ref, residual table with per-node error_m) - the
    canonical C5 evaluation: fit on the common ids, report residuals in the
    reference frame. Defaults: rigid (no scale), reflection allowed.
    """
    common = sorted(set(est["id"]) & set(ref["id"])) if ids is None else list(ids)
    if len(common) < dim:
        raise ValueError(f"only {len(common)} common ids for a {dim}-D alignment")
    e = est.set_index("id").loc[common, XYZ[:dim]]
    r = ref.set_index("id").loc[common, XYZ[:dim]]
    tf = umeyama(e, r, with_scale=with_scale, allow_reflection=allow_reflection,
                 src="estimate", dst="reference")
    resid = pd.DataFrame({"id": common,
                          "error_m": np.linalg.norm(tf.apply(e) - r.to_numpy(), axis=1)})
    resid.attrs["rmse_m"] = float(np.sqrt((resid["error_m"] ** 2).mean()))
    return tf, resid


def fix_gauge(nodes: pd.DataFrame, origin_id: str, x_axis_id: str,
              positive_y_id: str | None = None, dim: int = 2) -> Transform:
    """Transform into the conventional radio frame of the closed-form / grid
    papers: origin_id at (0,0), x_axis_id on the positive x-axis, and (if
    given) positive_y_id with y > 0 - resolving the reflection ambiguity.

    2-D convention (dim=2): rotation acts in the x-y plane, z (if present in
    later use) is untouched. nodes needs columns id, x, y.
    """
    p = nodes.set_index("id")
    o = p.loc[origin_id, XYZ[:dim]].to_numpy(dtype=float)
    ax = p.loc[x_axis_id, XYZ[:dim]].to_numpy(dtype=float) - o
    if dim != 2:
        raise NotImplementedError("only the planar (dim=2) paper convention is defined")
    theta = np.arctan2(ax[1], ax[0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    tf = Transform(R, -R @ o, 1.0, src="radio_free", dst="radio_conv")
    if positive_y_id is not None and tf.apply(p.loc[[positive_y_id], XYZ[:dim]])[0, 1] < 0:
        F = np.diag([1.0, -1.0])
        tf = Transform(F @ tf.R, F @ tf.t, 1.0, src=tf.src, dst=tf.dst)
    return tf
