r"""Data facts of the testing ground, computed from the released files only.

The data descriptor (UWB-ASC) and its companion tutorial print numbers about
the data: counts, durations, rates, coverage, residual medians, the truth
chain, the synthetic setup, the extent of the surveyed layout and the
recording years. This script computes each of them from the canonical files
under Data/ (set_a, set_b, synthetic) and writes eight files:

  data_macros.tex   one \newcommand per fact
  tab_sessions.tex  the session table (Session, Motion, Epochs, Ranges, Truth)
  sessions.csv      one row per measured session: counts, ranges per pair,
                    the recording window, durations, rates, the truth and the
                    directed pairs without ranges
  data_facts.csv    one row per fact: macro, value, rule, source files
  pairs.csv         one row per node pair of the residual sessions: residual,
                    spread, availability, field label (pair_table)
  labels.csv        the field labels against the residuals (label_table)
  residual_quantiles.csv  the quantiles of the per-epoch range residuals
                    (residual_quantiles)
  synthetic_quantiles.csv  the quantiles of the residuals of the frozen
                    synthetic draws, one column per condition
                    (synthetic_residuals)

    python -B Code/datasets/radiocal/scripts/06_data_facts.py --out facts

No estimator runs. Facts that both papers print keep the tutorial's macro
names (WE...), and the others are named DF... . Four rule constants are
restated from the tutorial's evaluation code, as RULES lists them, and are
written as macros too, each with its source in data_facts.csv. When main()
runs, pair_table checks pairs.csv against the residual macros it reproduces.
compute() also checks the frozen synthetic draws against their model
(synthetic_residuals) and stops when a draw misses a tolerance of SYN_*.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import radiocal  # noqa: E402
from radiocal.frames import Transform  # noqa: E402

DATA = HERE.parents[4] / "Data"
MEASURED = ("set_a", "set_b")

# Rule constants restated from the tutorial's evaluation code.
MATCH_TOL_S = 0.06            # a Vicon pose matches an epoch within this
MAX_GAP_S = 0.5               # a longer step between poses is a gap
SPEED_HORIZON_S = 1.0         # displacement horizon of the Vicon speed
SURVEY_POINT_SIGMA_M = 0.002  # nominal survey point accuracy (TS16 data sheet)
RULES = {
    "MATCH_TOL_S": (MATCH_TOL_S, "Code/methods/evaluation_slam.py, MATCH_TOL_S"),
    "MAX_GAP_S": (MAX_GAP_S, "Code/methods/evaluation_slam.py, MAX_GAP_S"),
    "SPEED_HORIZON_S": (SPEED_HORIZON_S, "Code/methods/gt_error_budget.py, SPEED_HORIZON_S"),
    "SURVEY_POINT_SIGMA_M": (SURVEY_POINT_SIGMA_M,
                             "Code/methods/gt_error_budget.py, SURVEY_POINT_SIGMA_M"),
}
QA_SPEED_MPS, QA_DT_S = 5.0, 0.5   # implausible-speed rule of radiocal/qa.py
JITTER_STEP_M = 0.01               # a flagged step below this is time-stamp jitter
BIAS_FLAG_M = 5.0                  # an anchor pair of set_a counts as biased from this residual
SET_A_RULES = ("SCREEN_DEV_M", "LOS_TOL_M")  # constants of 04_build_set_a.py, read from its source
FLAGGED_TAGS = {"s05": "Afive", "s12": "Atwelve"}  # the sessions the screen flags
MOTION = {"static": "static", "quasi_static_point": "quasi-static", "dynamic": "dynamic"}
# The residual sessions of F04: every set_b session, and the static set_a
# sessions, in which all twelve nodes stand on their surveyed positions.
AA_SESSIONS = {"set_b": ("s03", "s01", "s02"), "set_a": ("s21", "s22")}
PAIR_STATUS = ("ranges", "no ranges", "node never ranges")
LABEL_ORDER = {"LOS": 0, "OLOS": 1, "NLOS": 2}
QUANTILE_LEVELS = 1000                 # residual_quantiles.csv holds the levels k/1000
QUANTILE_COLUMNS = ("set_b_anchor", "set_b_tag", "set_a_anchor")
# Decimals of the written CSV files, as the data descriptor prints them.
PAIR_DECIMALS = {"surveyed_m": 2, "residual_median_m": 2, "residual_iqr_m": 3,
                 "residual_session_min_m": 2, "residual_session_max_m": 2,
                 "availability_ab_pct": 2, "availability_ba_pct": 2, "availability_pct": 2}
LABEL_DECIMALS = {"residual_median_m": 2, "residual_max_m": 2}
QUANTILE_DECIMALS = {"p": 3, **{k: 4 for k in QUANTILE_COLUMNS}}
# The frozen synthetic draws against their model (F21, session G1 of the data
# descriptor). Each tolerance tests one model value and lies far above the
# sampling error of the ranges that one noisy condition pools.
SYN_TOL_M = 0.01           # gaussian mean and sd, biased median and sd, against the model, m
SYN_NLOS_TOL_SIGMA = 0.5   # nlos: every pair median minus its stored excess, in sigma_hw
SYN_TAIL_SIGMA = 3.0       # nlos_per_range: the tail of the residuals begins here, in sigma_hw
SYN_TAIL_TOL = 0.005       # nlos_per_range: the tail share against the share eq. (5) predicts
SYN_QUANTILE_DECIMALS = 4  # synthetic_quantiles.csv, m


def fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


def span(values, nd: int = 2) -> str:
    return f"{min(values):.{nd}f}--{max(values):.{nd}f}"


class Facts:
    def __init__(self):
        self.rows: list[dict] = []
        self.tables: dict[str, pd.DataFrame] = {}  # computed tables that main() writes

    def add(self, name: str, value, rule: str, sources: str) -> None:
        if not re.fullmatch(r"[A-Za-z]+", name) or any(r["macro"] == name for r in self.rows):
            raise SystemExit(f"06_data_facts: bad or duplicate macro name {name}")
        self.rows.append({"macro": name, "value": str(value), "rule": rule,
                          "sources": sources})


# --------------------------------------------------------------- helpers

def load(data: Path, campaign: str) -> dict:
    return {p.parent.name: radiocal.load_session(p.parent)
            for p in sorted((data / campaign).glob("*/session.yaml"))}


def surveyed(g) -> pd.DataFrame:
    n = g.nodes
    return n[(n["role"] == "anchor") & (n["prior_type"] == "surveyed")] \
        .set_index("id")[["x", "y", "z"]].astype(float)


def mobile_of(g) -> str | None:
    ids = list(g.nodes.loc[g.nodes["role"] == "mobile", "id"])
    return ids[0] if ids else None


def anchors_ranging(g) -> list[str]:
    roles = g.roles()
    e = g.edges
    aa = e[e["from_id"].map(roles).eq("anchor") & e["to_id"].map(roles).eq("anchor")]
    return sorted(set(aa["from_id"]) | set(aa["to_id"]))


def missing_pairs(e: pd.DataFrame) -> str:
    """The directed pairs among the ranging nodes of a session that carry no
    range ("A>B", space separated), the complement of the QA report's
    "directed pairs observed"."""
    ids = sorted(set(e["from_id"]) | set(e["to_id"]))
    seen = set(zip(e["from_id"], e["to_id"]))
    return " ".join(f"{a}>{b}" for a in ids for b in ids if a != b and (a, b) not in seen)


def nearest(tt: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest pose of each time and whether it lies within MATCH_TOL_S."""
    j = np.clip(np.searchsorted(tt, t), 1, len(tt) - 1)
    j = np.where(np.abs(tt[j - 1] - t) <= np.abs(tt[j] - t), j - 1, j)
    return j, np.abs(tt[j] - t) <= MATCH_TOL_S


def aa_residuals(g) -> pd.DataFrame:
    """Per undirected anchor pair: median symmetrized range minus the surveyed
    distance (3-D)."""
    sym, anc = g.symmetrized(), surveyed(g)
    aa = sym[sym["node_a"].isin(anc.index) & sym["node_b"].isin(anc.index)]
    st = aa.groupby(["node_a", "node_b"], as_index=False).agg(range_m=("range_m", "median"))
    st["surveyed_m"] = np.linalg.norm(anc.loc[st["node_a"]].to_numpy()
                                      - anc.loc[st["node_b"]].to_numpy(), axis=1)
    st["residual_m"] = st["range_m"] - st["surveyed_m"]
    return st


def aa_epoch_residuals(g) -> pd.DataFrame:
    """Per epoch and undirected anchor pair: the symmetrized range minus the
    surveyed distance (3-D). aa_residuals gives its median per pair."""
    sym, anc = g.symmetrized(), surveyed(g)
    aa = sym[sym["node_a"].isin(anc.index) & sym["node_b"].isin(anc.index)].copy()
    aa["residual_m"] = aa["range_m"] - np.linalg.norm(anc.loc[aa["node_a"]].to_numpy()
                                                      - anc.loc[aa["node_b"]].to_numpy(), axis=1)
    return aa[["seq", "node_a", "node_b", "residual_m"]].reset_index(drop=True)


def mobile_pairs(sym, mobile, anc) -> pd.DataFrame:
    m = (((sym["node_a"] == mobile) & sym["node_b"].isin(anc.index))
         | ((sym["node_b"] == mobile) & sym["node_a"].isin(anc.index)))
    ma = sym[m].copy()
    ma["anchor"] = np.where(ma["node_a"] == mobile, ma["node_b"], ma["node_a"])
    return ma


def mobile_residuals(g) -> pd.DataFrame:
    """Per epoch and anchor: the mobile-anchor range minus the distance to its
    truth. Set B: the Vicon pose matched within MATCH_TOL_S under the wand tie
    of session.yaml (sessions without it give none). Set A: the surveyed marker
    of a verified quasi-static session."""
    mobile, anc = mobile_of(g), surveyed(g)
    if mobile is None:
        return pd.DataFrame()
    sym = g.symmetrized()
    ma = mobile_pairs(sym, mobile, anc)
    if g.meta.get("vicon"):
        slot = g.meta.get("transform_vicon_to_local")
        if not slot:
            return pd.DataFrame()
        truth = g.truth[g.truth["node_id"] == mobile].sort_values("t").reset_index(drop=True)
        enu = Transform.from_dict(slot).apply(truth[["x", "y", "z"]].to_numpy(float))
        j, ok = nearest(truth["t"].to_numpy(float), ma["t"].to_numpy(float))
        ma = ma[ok].copy()
        pos = enu[j[ok]]
    else:
        if g.meta.get("truth_quality") != "verified" or len(g.truth) != 1:
            return pd.DataFrame()
        pos = g.truth.iloc[0][["x", "y", "z"]].to_numpy(float)
    ma["true_m"] = np.linalg.norm(anc.loc[ma["anchor"]].to_numpy() - pos, axis=1)
    ma["residual_m"] = ma["range_m"] - ma["true_m"]
    return ma[["seq", "anchor", "residual_m"]]


def tag_epochs(g) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """The epochs with a mobile-anchor range, each at its earliest range, and
    the Vicon pose matched to it (the coverage rule of the walks)."""
    mobile, anc = mobile_of(g), set(surveyed(g).index)
    e = g.edges
    tag = e[((e["from_id"] == mobile) & e["to_id"].isin(anc))
            | ((e["to_id"] == mobile) & e["from_id"].isin(anc))]
    t_ep = tag.groupby("seq")["t"].min().sort_index().to_numpy(float)
    truth = g.truth[g.truth["node_id"] == mobile].sort_values("t").reset_index(drop=True)
    j, ok = nearest(truth["t"].to_numpy(float), t_ep)
    return t_ep, truth, j, ok


def bursts(truth: pd.DataFrame) -> dict:
    """Vicon bursts under the MAX_GAP_S rule, and the frame-counter rate
    across each gap."""
    t, f = truth["t"].to_numpy(float), truth["frame"].to_numpy(float)
    dt, df = np.diff(t), np.diff(f)
    gap = dt > MAX_GAP_S
    seg = np.r_[0, np.cumsum(gap)]
    dur = np.array([t[seg == s][-1] - t[seg == s][0] for s in range(seg[-1] + 1)])
    rate = df[gap] / dt[gap]
    return {"bursts": int(seg[-1] + 1), "burst_median_s": round(float(np.median(dur)), 2),
            "burst_max_s": round(float(dur.max()), 2),
            "gap_max_s": round(float(dt[gap].max()), 1),
            "rate_min": round(float(rate.min()), 1), "rate_max": round(float(rate.max()), 1)}


def speeds(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    half = SPEED_HORIZON_S / 2
    x0 = np.interp(t - half, t, xy[:, 0]), np.interp(t - half, t, xy[:, 1])
    x1 = np.interp(t + half, t, xy[:, 0]), np.interp(t + half, t, xy[:, 1])
    v = np.hypot(x1[0] - x0[0], x1[1] - x0[1]) / SPEED_HORIZON_S
    v[(t - half < t[0]) | (t + half > t[-1])] = np.nan
    return v


def speed_flags(truth: pd.DataFrame) -> tuple[int, int, list[float]]:
    """The QA report's implausible speeds (radiocal/qa.py), split into
    time-stamp jitter (steps below JITTER_STEP_M) and pose jumps."""
    xyz = truth[["x", "y", "z"]].to_numpy(float)
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    dt = np.diff(truth["t"].to_numpy(float))
    with np.errstate(divide="ignore"):
        flag = (step / dt > QA_SPEED_MPS) & (dt < QA_DT_S)
    jumps = sorted(float(s) for s in step[flag] if s >= JITTER_STEP_M)
    return int(flag.sum()), int((step[flag] < JITTER_STEP_M).sum()), jumps


def min_rectangle(xy: np.ndarray) -> tuple[float, float]:
    """Sides (longer, shorter) of the smallest-area rectangle around the points.

    One side of that rectangle lies along an edge of the convex hull, and every
    hull edge joins two of the points, so trying the direction of every pair
    of points finds it exactly. The result does not depend on the frame."""
    best = None
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            d = xy[j] - xy[i]
            n = float(np.hypot(d[0], d[1]))
            if n == 0.0:
                continue
            u = d / n
            a, b = xy @ u, xy @ np.array([-u[1], u[0]])
            w, h = float(a.max() - a.min()), float(b.max() - b.min())
            if best is None or w * h < best[0] * best[1]:
                best = (w, h)
    return max(best), min(best)


def utc(t: float, up: bool = False) -> str:
    """HH:MM in UTC, rounded down, or up with up=True (an enclosing window)."""
    t = 60.0 * (np.ceil(t / 60.0) if up else np.floor(t / 60.0))
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%H:%M")


def truth_cell(m: dict) -> str:
    """The Truth column of the session table."""
    c = m["counts"]
    if m["motion"] == "dynamic":
        return "motion capture"
    if m["motion"] == "static":
        has_poses = c.get("truth_poses", 0) > 0 and "vicon" in m
        registered = m.get("transform_vicon_to_local") is not None
        return "motion capture (static, unregistered)" if has_poses and not registered else "--"
    return f"marker, {m['truth_quality']}"


# ------------------------------------------------------------------ facts

def compute(data: Path):
    F = Facts()
    S = {c: load(data, c) for c in MEASURED}
    meta = {c: {s: yaml.safe_load((data / c / s / "session.yaml").read_text(encoding="utf-8"))
                for s in S[c]} for c in MEASURED}
    rows = []

    # F02 anchors per session and node sets
    geo = {(c, s): len(anchors_ranging(g)) for c in MEASURED for s, g in S[c].items()}
    by_motion = {}
    for (c, s), n in geo.items():
        by_motion.setdefault((c, meta[c][s]["motion"]), set()).add(n)
    for key, vals in by_motion.items():
        if len(vals) != 1:
            raise SystemExit(f"06_data_facts: {key} sessions differ in anchor count {vals}")
    src = "nodes.yaml, session.yaml (node_overrides), ranges.parquet"
    rule = "anchors (after node_overrides) that range with another anchor"
    F.add("WEnodeSets", len({(c, n) for (c, _), n in geo.items()}),
          "distinct (deployment, anchor count) over the sessions", src)
    F.add("WEnodeAnchorsQsSetA", by_motion[("set_a", "quasi_static_point")].pop(), rule, src)
    F.add("WEnodeAnchorsStaticSetA", by_motion[("set_a", "static")].pop(), rule, src)
    F.add("WEnodeAnchorsSetB", len(anchors_ranging(S["set_b"]["s03"])), rule, src)

    # F03 sessions
    for c in MEASURED:
        for s, g in S[c].items():
            m = meta[c][s]
            e = g.edges
            row = {"set": c, "session": s, "motion": m["motion"],
                   "epochs": m["counts"]["range_epochs"],
                   "directed_ranges": m["counts"]["directed_ranges"],
                   "dropped_invalid": m["counts"]["dropped_invalid"],
                   "truth_poses": m["counts"]["truth_poses"],
                   "ranging_nodes": int(pd.concat([e["from_id"], e["to_id"]]).nunique()),
                   "directed_pairs": int(e.groupby(["from_id", "to_id"]).ngroups),
                   "ranges_per_pair_median":
                       round(float(e.groupby(["from_id", "to_id"]).size().median()), 1),
                   "t_start_utc": "", "t_end_utc": "", "duration_s": "",
                   "epoch_rate_hz": "", "vicon_mean_rate_hz": "", "truth": truth_cell(m),
                   "missing_directed_pairs": missing_pairs(e)}
            if m.get("t_start") is not None:
                dur = m["t_end"] - m["t_start"]
                row.update(t_start_utc=datetime.fromtimestamp(m["t_start"], tz=timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%S"),
                           t_end_utc=datetime.fromtimestamp(m["t_end"], tz=timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%S"),
                           duration_s=round(dur, 1),
                           epoch_rate_hz=round(m["counts"]["range_epochs"] / dur, 2))
            if m.get("vicon"):
                row["vicon_mean_rate_hz"] = m["vicon"]["rate_hz"]
            rows.append(row)
    sess = pd.DataFrame(rows)
    for c, tag in (("set_a", "SetA"), ("set_b", "SetB")):
        x = sess[sess["set"] == c]
        F.add(f"DFsessions{tag}", len(x), "sessions of the deployment", "session.yaml")
        F.add(f"DFrangingNodes{tag}", int(x["ranging_nodes"].max()),
              "nodes that occur in a range", "ranges.parquet")
        F.add(f"DFdirectedRanges{tag}", int(x["directed_ranges"].sum()),
              "directed ranges, sum over the sessions", "session.yaml counts")
        F.add(f"DFepochs{tag}", int(x["epochs"].sum()),
              "ranging epochs, sum over the sessions", "session.yaml counts")
    xb = sess[sess["set"] == "set_b"]
    F.add("DFdirectedPairsSetB", int(xb["directed_pairs"].min()),
          "directed pairs observed in every set_b session (of n(n-1) possible)", "ranges.parquet")
    F.add("DFdirectedPairsPossibleSetB", int(xb["ranging_nodes"].max() * (xb["ranging_nodes"].max() - 1)),
          "n(n-1) for the ranging nodes", "ranges.parquet")
    F.add("DFepochRateSpanSetB", span(xb["epoch_rate_hz"].astype(float), 1),
          "epochs over the session duration (t_end - t_start)", "session.yaml")

    # F04 range residual medians
    for c, tag, aa_sessions in (("set_b", "SetB", AA_SESSIONS["set_b"]),
                                ("set_a", "SetA", AA_SESSIONS["set_a"])):
        aa = pd.concat([aa_residuals(S[c][s]).assign(session=s) for s in aa_sessions])
        per_pair = aa.groupby(["node_a", "node_b"])["residual_m"].median()
        F.add(f"WEresAaMed{tag}", fmt(float(per_pair.median()), 2),
              "anchor pairs: median over sessions of the median symmetrized range minus the "
              f"surveyed distance, then median over the pairs ({', '.join(aa_sessions)})",
              "ranges.parquet, nodes.yaml")
        mob = pd.concat([mobile_residuals(g).assign(session=s) for s, g in S[c].items()])
        F.add(f"WEresMaMed{tag}",
              fmt(float(mob.groupby(["session", "anchor"])["residual_m"].median().median()), 2),
              "mobile-anchor pairs: median per (session, anchor) of the per-epoch residual "
              "against the truth, then median" + (" (Vicon pose within MATCH_TOL_S, wand tie)"
                                                  if c == "set_b" else " (verified markers)"),
              "ranges.parquet, groundtruth.parquet, session.yaml")
        if c == "set_a":
            F.add("DFresAaPairsSetA", len(per_pair), "anchor pairs observed in s21 and s22",
                  "ranges.parquet")
            F.add("DFresAaPairsPossibleSetA", len(surveyed(S[c]["s21"])) * (len(surveyed(S[c]["s21"])) - 1) // 2,
                  "anchor pairs of the 12 surveyed nodes", "nodes.yaml")
            big = per_pair[per_pair >= BIAS_FLAG_M]
            F.add("DFresAaBiasedPairsSetA", len(big), "anchor pairs with a residual of 5 m or more",
                  "ranges.parquet, nodes.yaml")
            F.add("DFresAaBiasMaxSetA", fmt(float(per_pair.max()), 1),
                  "largest anchor-pair residual", "ranges.parquet, nodes.yaml")
        else:
            pair = aa[(aa["node_a"] == "5BB6") & (aa["node_b"] == "D38F")]["residual_m"]
            F.add("DFresObstructedSetB", fmt(float(per_pair[("5BB6", "D38F")]), 2),
                  "residual of the anchor pair 5BB6-D38F (as WEresAaMedSetB), m",
                  "ranges.parquet, nodes.yaml")
            F.add("DFresObstructedSpanSetB", span(pair, 1),
                  "the same residual per session, s01 to s03, m", "ranges.parquet, nodes.yaml")

    # F05, F06, F07, F09a: the Vicon truth of set B
    b = {}
    for s, g in S["set_b"].items():
        t_ep, truth, j, ok = tag_epochs(g)
        b[s] = {"epochs": len(t_ep), "covered": int(ok.sum()), **bursts(truth)}
        if s in ("s01", "s02"):
            xy = truth[["x", "y"]].to_numpy(float)[j[ok]]  # raw Vicon frame
            r95 = float(np.percentile(np.hypot(xy[:, 0], xy[:, 1]), 95))
            prov = meta["set_b"][s]["transform_provenance"]
            baseline = float(prov["baseline_m"])
            sig_yaw = float(np.sqrt(2.0) * SURVEY_POINT_SIGMA_M / baseline)
            sig_t = float(meta["set_b"][s]["clock"]["uncertainty_s"])
            v = speeds(truth["t"].to_numpy(float), truth[["x", "y"]].to_numpy(float))
            b[s].update(baseline_m=round(baseline, 3),
                        yaw_sigma_deg=round(float(np.degrees(sig_yaw)), 3),
                        radius_p95_m=round(r95, 2), effect_r95_m=round(r95 * sig_yaw, 3),
                        clock_sigma_s=sig_t,
                        clock_effect_m=round(float(np.nanmedian(v)) * sig_t, 3))
        n, jit, jumps = speed_flags(g.truth)
        b[s].update(flags=n, flags_jitter=jit, jumps=jumps)
    walks = ("s01", "s02")
    vsrc = "groundtruth.parquet, ranges.parquet, session.yaml"
    shares = [100 * b[s]["covered"] / b[s]["epochs"] for s in walks]
    F.add("WEgtCoverShareSpan", f"{min(shares):.0f}--{max(shares):.0f}",
          "share of the epochs with a DD98 range whose earliest range has a Vicon pose "
          "within MATCH_TOL_S, walks s01 and s02", vsrc)
    F.add("WEgtCoverShareStatic", f"{100 * b['s03']['covered'] / b['s03']['epochs']:.0f}",
          "the same share for the static s03", vsrc)
    for s, tag in (("s01", "Bone"), ("s02", "Btwo"), ("s03", "Bthree")):
        F.add(f"DFcovered{tag}", b[s]["covered"], "epochs with a matched Vicon pose", vsrc)
        F.add(f"DFtagEpochs{tag}", b[s]["epochs"], "epochs with a DD98 range", vsrc)
        F.add(f"DFbursts{tag}", b[s]["bursts"], "Vicon bursts (gaps longer than MAX_GAP_S)", vsrc)
        F.add(f"DFburstMedian{tag}", fmt(b[s]["burst_median_s"], 2), "median burst length, s", vsrc)
        F.add(f"DFburstMax{tag}", fmt(b[s]["burst_max_s"], 2), "longest burst, s", vsrc)
        F.add(f"DFgapMax{tag}", fmt(b[s]["gap_max_s"], 1), "longest gap, s", vsrc)
        F.add(f"DFspeedFlags{tag}", b[s]["flags"],
              "QA implausible speeds (> 5 m/s between poses < 0.5 s apart)", "groundtruth.parquet")
        F.add(f"DFspeedJitter{tag}", b[s]["flags_jitter"],
              f"of those, steps below {JITTER_STEP_M} m (time-stamp jitter)", "groundtruth.parquet")
    jumps = [x for s in b for x in b[s]["jumps"]]
    F.add("DFspeedJumpSpanCm", f"{100 * min(jumps):.0f}--{100 * max(jumps):.0f}",
          "the other flagged steps (pose jumps), cm", "groundtruth.parquet")
    rates = [b[s][k] for s in ("s01", "s02", "s03") for k in ("rate_min", "rate_max")]
    F.add("WEgtViconRateSpan", f"{min(rates):.1f}--{max(rates):.1f}",
          "Vicon frame-counter rate across each gap, s01 to s03", "groundtruth.parquet")
    F.add("DFviconMeanRateSpan",
          span([meta["set_b"][s]["vicon"]["rate_hz"] for s in S["set_b"]], 1),
          "logged poses per second over the session (vicon.rate_hz)", "session.yaml")
    F.add("WEgtClockSigma", f"{b['s01']['clock_sigma_s']:g}", "clock offset uncertainty",
          "session.yaml clock")
    F.add("WEgtClockSpan", span([b[s]["clock_effect_m"] for s in walks]),
          "median Vicon speed (SPEED_HORIZON_S) times the clock uncertainty, s01 and s02",
          "groundtruth.parquet, session.yaml")
    F.add("DFclockOffsetSpanSetB",
          span([meta["set_b"][s]["clock"]["offset_s"] for s in S["set_b"]]),
          "clock offset of the ranges onto the Vicon clock, s", "session.yaml clock")
    F.add("DFqueueLagMaxSetB",
          fmt(max(meta["set_b"][s]["clock"]["queue_lag_max_s"] for s in S["set_b"]), 1),
          "largest MQTT logger queue lag, s", "session.yaml clock")
    F.add("WEgtWandBaseline", fmt(b["s01"]["baseline_m"], 2), "wand-tie baseline, m",
          "session.yaml transform_provenance")
    F.add("WEgtYawSigma", fmt(b["s01"]["yaw_sigma_deg"], 2),
          "yaw uncertainty sqrt(2) SURVEY_POINT_SIGMA_M / baseline, deg",
          "session.yaml transform_provenance")
    F.add("WEgtYawRadius", f"{max(b[s]['radius_p95_m'] for s in walks):.0f}",
          "95th percentile of the distance of the matched poses from the Vicon origin, m",
          vsrc)
    F.add("WEgtYawEffect", fmt(max(b[s]["effect_r95_m"] for s in walks), 2),
          "that radius times the yaw uncertainty, m", vsrc)

    # F08 invalid ranges
    for c, tag in (("set_b", "SetB"), ("set_a", "SetA")):
        x = sess[sess["set"] == c]
        pct = 100 * x["dropped_invalid"] / (x["directed_ranges"] + x["dropped_invalid"])
        F.add(f"DFinvalidPct{tag}",
              fmt(100 * x["dropped_invalid"].sum()
                  / (x["directed_ranges"].sum() + x["dropped_invalid"].sum()), 3),
              "invalid ranges dropped at conversion, % of all ranges", "session.yaml counts")
        F.add(f"DFinvalidPctSpan{tag}", span(pct, 3), "the same per session, %",
              "session.yaml counts")

    # F10 the set A truth chain
    adj = pd.read_csv(data / "set_a" / "truth_adjudication.csv")
    flagged = set(adj.loc[~adj["screen_passed"], "session"])
    if flagged != set(FLAGGED_TAGS):
        raise SystemExit(f"06_data_facts: the screen flags {sorted(flagged)}")
    asrc = "set_a/truth_adjudication.csv"
    F.add("WEgtVerifiedSetA", int((adj["truth_quality"] == "verified").sum()),
          "quasi-static sessions with verified truth", asrc)
    F.add("DFquasiStaticSetA", len(adj), "quasi-static sessions", asrc)
    F.add("DFscreenAgreeSetA", int((adj["screen_deviation_m"] < 0.5).sum()),
          "sessions whose screen lies within 0.5 m of the marker", asrc)
    F.add("DFscreenPassSetA", int(adj["screen_passed"].sum()), "sessions the screen passes", asrc)
    F.add("WEgtTagHeight", fmt(float(adj["tag_height_above_marker_m"].iloc[0]), 2),
          "fitted tag height above the marker, m", asrc)
    nodes_a = yaml.safe_load((data / "set_a" / "nodes.yaml").read_text(encoding="utf-8"))
    z8 = next(n["prior"]["xyz_local"][2] for n in nodes_a["nodes"] if n["id"] == "8B05")
    zp = next(cp["xyz_local"][2] for cp in nodes_a["control_points"] if cp["name"] == "P01")
    F.add("WEgtTagHeightSurvey", fmt(round(z8 - zp, 3), 3),
          "surveyed 8B05 antenna above marker P01, m", "set_a/nodes.yaml")
    F.add("WEgtRangeBiasSetA", fmt(float(adj["common_range_bias_m"].iloc[0]), 2),
          "fitted common range bias, m", asrc)
    F.add("WEgtRangeTestSpan", f"{adj['reference_min_m'].iloc[0]:.2f}--"
          f"{adj['reference_max_m'].iloc[0]:.2f}",
          "reference range of the median line-of-sight residual, m", asrc)
    for s, tag in FLAGGED_TAGS.items():
        r = adj[adj["session"] == s].iloc[0]
        F.add(f"WEgtRangeTest{tag}", fmt(float(r["median_los_residual_m"]), 2),
              f"median line-of-sight residual of {s}, m", asrc)
        F.add(f"WEgtScreenDev{tag}", fmt(float(r["screen_deviation_m"]), 1),
              f"screen deviation of {s}, m", asrc)
        F.add(f"DFlosLinks{tag}", int(r["n_los_links"]), f"line-of-sight links of {s}", asrc)
        F.add(f"DFlinks{tag}", int(r["n_links"]), f"tag-anchor links of {s}", asrc)

    # DX7 field labels (a field judgement, not truth)
    for c, tag in (("set_a", "SetA"), ("set_b", "SetB")):
        lab = pd.read_csv(data / c / "los_nlos_field_labels.csv", dtype=str)
        lsrc = f"{c}/los_nlos_field_labels.csv"
        F.add(f"DFlabelPairs{tag}", len(lab), "node pairs with a field label", lsrc)
        for value, name in (("LOS", "Los"), ("OLOS", "Olos"), ("NLOS", "Nlos")):
            if c == "set_b" and value == "OLOS":
                continue
            F.add(f"DFlabel{name}{tag}", int((lab["label"] == value).sum()),
                  f"pairs labelled {value} in the field", lsrc)
        F.add(f"DFlabelMixed{tag}", int(lab["label"].str.contains("/").sum()),
              "pairs with two values in the field record", lsrc)

    # F13 dates and windows
    camp_b = yaml.safe_load((data / "set_b" / "campaign.yaml").read_text(encoding="utf-8"))
    F.add("DFdateSetB", camp_b["date"], "recording date", "set_b/campaign.yaml")
    F.add("DFwindowSetB", f"{utc(min(m['t_start'] for m in meta['set_b'].values()))}--"
          f"{utc(max(m['t_end'] for m in meta['set_b'].values()), up=True)}",
          "whole minutes enclosing the three sessions, UTC", "session.yaml t_start, t_end")

    # F15 the synthetic counterpart
    syn = yaml.safe_load((data / "synthetic" / "campaign.yaml").read_text(encoding="utf-8"))
    gen = syn["generator"]
    ssrc = "synthetic/campaign.yaml"
    syn_s = sorted((data / "synthetic").glob("syn_*/session.yaml"))
    counts = {yaml.safe_load(p.read_text(encoding="utf-8"))["counts"]["directed_ranges"]
              for p in syn_s}
    noisy = {v for k, v in gen["draws_per_condition"].items() if k != "noise_free"}
    if len(noisy) != 1:
        raise SystemExit(f"06_data_facts: the noisy conditions differ in draws {noisy}")
    n_draws = noisy.pop()
    n_anchors = len(syn["nodes"]["anchors"])
    F.add("WEsynSigmaHw", fmt(gen["model"]["sigma_hw_m"], 2), "hardware noise sigma, m", ssrc)
    F.add("WEsynPoutPct", f"{100 * gen['model']['p_out']:.0f}", "outlier probability, %", ssrc)
    F.add("WEsynDraws", n_draws, "draws per noisy condition", ssrc)
    F.add("WEsynEpochs", gen["epochs"], "synthetic epochs per draw", ssrc)
    F.add("DFsynSessions", len(syn_s), "synthetic sessions (draws)", "synthetic/*/session.yaml")
    F.add("DFsynConditions", len(gen["conditions"]), "conditions", ssrc)
    F.add("DFsynBias", fmt(gen["bias_m"], 2), "bias of the biased condition, m", ssrc)
    F.add("DFsynAnchors", n_anchors, "anchors of the synthetic geometry", ssrc)
    F.add("DFsynPairs", n_anchors * (n_anchors - 1) // 2, "anchor pairs", ssrc)
    F.add("DFsynRangesPerDraw", counts.pop() if len(counts) == 1 else "differs",
          "directed ranges per draw", "synthetic/*/session.yaml")
    F.add("DFsynSeedMax", n_draws - 1, "largest seed (draw k uses seed k)", ssrc)
    F.add("DFsynMpMean", fmt(gen["model"]["mp_mean"], 1), "multipath log-normal R_mp", ssrc)
    F.add("DFsynMpSigma", fmt(gen["model"]["mp_sigma"], 2), "multipath log-normal sigma_mp", ssrc)
    F.add("DFsynDmax", f"{gen['model']['d_max_m']:.0f}", "maximum range d_max, m", ssrc)

    # F19 the surveyed layout (I14 of the data descriptor): its extent in the
    # horizontal plane and its height spread, over every node with a surveyed
    # prior (set_b: D5B4 included, which is surveyed but never ranges)
    for c, tag in (("set_a", "SetA"), ("set_b", "SetB")):
        reg = yaml.safe_load((data / c / "nodes.yaml").read_text(encoding="utf-8"))
        xyz = np.array([n["prior"]["xyz_local"] for n in reg["nodes"]
                        if n["prior"]["type"] == "surveyed"], dtype=float)
        long_m, short_m = min_rectangle(xyz[:, :2])
        nsrc = f"{c}/nodes.yaml"
        F.add(f"DFextentLong{tag}", f"{long_m:.0f}",
              "longer side of the smallest rectangle around the surveyed nodes in the "
              "horizontal plane, m", nsrc)
        F.add(f"DFextentShort{tag}", f"{short_m:.0f}", "shorter side of that rectangle, m", nsrc)
        F.add(f"DFheightSpanCm{tag}", f"{100 * float(np.ptp(xyz[:, 2])):.0f}",
              "height spread of the surveyed nodes (highest minus lowest), cm", nsrc)
    sig = {float(n["prior"]["sigma_m"]) for c in MEASURED
           for n in yaml.safe_load((data / c / "nodes.yaml").read_text(encoding="utf-8"))["nodes"]
           if n["prior"]["type"] == "surveyed"}
    if len(sig) != 1:
        raise SystemExit(f"06_data_facts: the surveyed nodes differ in sigma_m {sorted(sig)}")
    F.add("DFsigmaSurveyCm", f"{100 * sig.pop():.0f}",
          "prior uncertainty sigma_m of the surveyed nodes of set_a and set_b, cm",
          "set_a/nodes.yaml, set_b/nodes.yaml")

    # F13 the recording years, from the date of each campaign file. The set_a
    # day, 2025-03-13, is the author's record (the descriptor's session P,
    # 2026-11-08), since its logs carry no timestamps, and its title names it.
    camp_a = yaml.safe_load((data / "set_a" / "campaign.yaml").read_text(encoding="utf-8"))
    if str(camp_a["date"]) not in camp_a["title"]:
        raise SystemExit("06_data_facts: the set_a title does not name its date")
    F.add("DFyearSetA", str(camp_a["date"])[:4], "recording year", "set_a/campaign.yaml")
    F.add("DFyearSetB", str(camp_b["date"])[:4], "recording year", "set_b/campaign.yaml")

    # F18 the rule constants as macros, each with its source
    for macro, name, value, rule in (
            ("DFmatchTolS", "MATCH_TOL_S", f"{MATCH_TOL_S:g}",
             "a Vicon pose matches an epoch within this, s"),
            ("DFmaxGapS", "MAX_GAP_S", f"{MAX_GAP_S:g}",
             "a longer step between Vicon poses is a gap, s"),
            ("DFspeedHorizonS", "SPEED_HORIZON_S", f"{SPEED_HORIZON_S:g}",
             "displacement horizon of the Vicon speed, s"),
            ("DFsurveyPointSigmaMm", "SURVEY_POINT_SIGMA_M", f"{1000 * SURVEY_POINT_SIGMA_M:g}",
             "nominal accuracy of one survey point (TS16 data sheet), mm")):
        F.add(macro, value, f"rule constant {name}: {rule}", f"restated from {RULES[name][1]}")

    # the rule constants of the set_a truth adjudication, read from the source of
    # its released builder, and the bias flag of DFresAaBiasedPairsSetA
    builder = (HERE.parent / "04_build_set_a.py").read_text(encoding="utf-8")
    for macro, name, rule in (
            ("DFscreenDevM", "SCREEN_DEV_M",
             "the screen passes a session whose multilaterated tag lies within this of the marker, m"),
            ("DFlosTolM", "LOS_TOL_M",
             "a tag-anchor link counts as line of sight when its residual lies within this of "
             "the common range bias, m")):
        m = re.search(rf"^{name}\s*=\s*([0-9.]+)", builder, re.M)
        if m is None:
            raise SystemExit(f"06_data_facts: {name} not found in 04_build_set_a.py")
        F.add(macro, f"{float(m.group(1)):g}", f"rule constant {name} of the set_a builder: {rule}",
              f"Code/datasets/radiocal/scripts/04_build_set_a.py, {name}")
    F.add("DFresAaBiasThresholdM", f"{BIAS_FLAG_M:g}",
          "rule constant BIAS_FLAG_M: an anchor pair counts as biased from this residual, m",
          "Code/datasets/radiocal/scripts/06_data_facts.py, BIAS_FLAG_M")

    # F21 the frozen synthetic draws against their model (session G1 of the
    # data descriptor). Appended after the older facts, whose lines stay as
    # they were.
    st, q = synthetic_residuals(data)
    F.tables["synthetic_quantiles"] = q
    rsrc = "synthetic/*/ranges.parquet, synthetic/nodes.yaml, synthetic/campaign.yaml"
    this = "Code/datasets/radiocal/scripts/06_data_facts.py"

    def m3(v: float) -> str:  # metres to the millimetre, without a negative zero
        return fmt(0.0 if abs(v) < 0.0005 else v, 3)

    F.add("DFsynRangesPerCondition", st["ranges_per_condition"],
          "ranges that one noisy condition pools over its draws", "synthetic/*/ranges.parquet")
    F.add("DFsynResAbsMaxNoiseFree", f"{st['noise_free_max_abs']:g}",
          "largest absolute residual of the noise-free draw against the generating 2-D "
          "distances, m", rsrc)
    F.add("DFsynResMeanGaussian", m3(st["gaussian_mean"]),
          "mean residual of the gaussian draws, m", rsrc)
    F.add("DFsynResSdGaussian", m3(st["gaussian_sd"]),
          "standard deviation of the residuals of the gaussian draws, m", rsrc)
    F.add("DFsynResMedBiased", m3(st["biased_median"]),
          "median residual of the biased draws, m", rsrc)
    F.add("DFsynResSdBiased", m3(st["biased_sd"]),
          "standard deviation of the residuals of the biased draws, m", rsrc)
    F.add("DFsynNlosExcessDevMax", m3(st["nlos_dev_max"]),
          "largest difference, over the nlos draws and all their pairs, between the median "
          "residual of a pair and its stored excess (nlos_pairs, zero for a pair the list "
          "leaves out), m",
          rsrc + ", synthetic/*/session.yaml nlos_pairs")
    F.add("DFsynTailSharePct", f"{100 * st['tail_share']:.1f}",
          "share of the residuals of the nlos_per_range draws above SYN_TAIL_SIGMA sigma_hw, %",
          rsrc)
    F.add("DFsynTailShareModelPct", f"{100 * st['tail_share_model']:.1f}",
          "the share that eq. (5) predicts for the drawn distances (tail_share_model), %",
          "synthetic/campaign.yaml generator.model, " + rsrc)
    for macro, name, value, rule in (
            ("DFsynTolM", "SYN_TOL_M", f"{SYN_TOL_M:g}",
             "tolerance of the gaussian mean and sd and the biased median and sd, m"),
            ("DFsynNlosTolSigma", "SYN_NLOS_TOL_SIGMA", f"{SYN_NLOS_TOL_SIGMA:g}",
             "tolerance of a pair median of the nlos draws against its excess, in sigma_hw"),
            ("DFsynTailSigma", "SYN_TAIL_SIGMA", f"{SYN_TAIL_SIGMA:g}",
             "begin of the tail of the nlos_per_range residuals, in sigma_hw"),
            ("DFsynTailTolPct", "SYN_TAIL_TOL", f"{100 * SYN_TAIL_TOL:g}",
             "tolerance of the tail share against the predicted share, percentage points")):
        F.add(macro, value, f"rule constant {name}: {rule}", f"{this}, {name}")
    return F, sess


# ------------------------------------------------------------------ pairs

def pair_table(data: Path, facts: dict | None = None,
               samples: dict | None = None) -> pd.DataFrame:
    """One row per pair of the nodes of a deployment in its residual sessions
    (AA_SESSIONS): residual, spread and availability, with the field label.

    The nodes are those of nodes.yaml, a node that never ranges included, so
    each deployment gives n(n-1)/2 rows. kind is "tag" for a pair with the
    mobile node and "anchor" otherwise. status is "node never ranges" when a
    node of the pair ranges with no node, "no ranges" when the pair carries no
    range, and "ranges" otherwise.
      residual_median_m   anchor pairs: the rule of WEresAaMed* (median over the
                          sessions of the per-session median of the symmetrized
                          ranges minus the surveyed distance); tag pairs: the
                          median over the sessions of the per-session median
                          residual against the matched, registered pose
                          (mobile_residuals, so only sessions with a wand tie)
      residual_iqr_m      interquartile range of the per-epoch residuals
                          (aa_epoch_residuals, mobile_residuals) of the sessions
      residual_session_*  the smallest and largest per-session median
      ranges_ab, ranges_ba, availability_ab_pct, availability_ba_pct
                          directed ranges and their share of the epochs of the
                          sessions (the QA rule)
      availability_pct    share of the epochs with a range in either direction
    Given facts (macro: value), the rows must reproduce the anchor-pair macros
    WEresAaMed*, DFresAaPairs*, DFresAaBiased*, DFresAaBiasMax* and
    DFresObstructed*, and WEresMaMedSetB by its own rule over the (session,
    anchor) medians. Given a dict as samples, it receives the per-epoch
    residuals that residual_quantiles pools."""
    rows = []
    for c, sessions in AA_SESSIONS.items():
        reg = yaml.safe_load((data / c / "nodes.yaml").read_text(encoding="utf-8"))
        ids = sorted(str(n["id"]) for n in reg["nodes"])
        xyz = {str(n["id"]): np.array(n["prior"]["xyz_local"], dtype=float)
               for n in reg["nodes"] if n["prior"]["type"] == "surveyed"}
        lab = pd.read_csv(data / c / "los_nlos_field_labels.csv", dtype=str)
        label = {tuple(sorted(k)): v for *k, v in zip(lab["node_a"], lab["node_b"], lab["label"])}
        S = {s: radiocal.load_session(data / c / s) for s in sessions}
        epochs = sum(int(g.meta["counts"]["range_epochs"]) for g in S.values())
        directed, either, ranging = {}, {}, set()
        for g in S.values():
            e = g.edges
            ranging |= set(e["from_id"]) | set(e["to_id"])
            for k, v in e.groupby(["from_id", "to_id"]).size().items():
                directed[k] = directed.get(k, 0) + int(v)
            und = pd.DataFrame({"seq": e["seq"], "a": e[["from_id", "to_id"]].min(axis=1),
                                "b": e[["from_id", "to_id"]].max(axis=1)}).drop_duplicates()
            for k, v in und.groupby(["a", "b"]).size().items():
                either[k] = either.get(k, 0) + int(v)
        mobile = next((mobile_of(g) for g in S.values() if mobile_of(g)), None)
        aa_med = pd.concat([aa_residuals(g).assign(session=s) for s, g in S.items()])
        aa_ses = {k: grp.set_index("session")["residual_m"]
                  for k, grp in aa_med.groupby(["node_a", "node_b"])}
        aa_ep = pd.concat([aa_epoch_residuals(g) for g in S.values()])
        aa_vals = {k: grp["residual_m"] for k, grp in aa_ep.groupby(["node_a", "node_b"])}
        anchor_pairs = {k for k in either if k[0] in xyz and k[1] in xyz and mobile not in k}
        if len(aa_ep) != sum(either[k] for k in anchor_pairs):
            raise SystemExit(f"06_data_facts: {c}: per-epoch anchor residuals and epochs disagree")
        mob = [mobile_residuals(g).assign(session=s) for s, g in S.items()]
        mob = pd.concat([m for m in mob if len(m)]) if any(len(m) for m in mob) else None
        if mob is not None:
            tag_ses = {a: grp.groupby("session")["residual_m"].median()
                       for a, grp in mob.groupby("anchor")}
            tag_vals = {a: grp["residual_m"] for a, grp in mob.groupby("anchor")}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                k, tag = (a, b), mobile in (a, b)
                status = ("node never ranges" if a not in ranging or b not in ranging
                          else "no ranges" if k not in either else "ranges")
                per, vals = None, None
                if status == "ranges" and tag and mob is not None:
                    anchor = b if a == mobile else a
                    per, vals = tag_ses.get(anchor), tag_vals.get(anchor)
                elif status == "ranges" and not tag:
                    per, vals = aa_ses.get(k), aa_vals.get(k)
                rows.append({
                    "set": c, "node_a": a, "node_b": b, "kind": "tag" if tag else "anchor",
                    "status": status, "field_label": label.get(k, ""),
                    "surveyed_m": (float(np.linalg.norm(xyz[a] - xyz[b]))
                                   if a in xyz and b in xyz and not tag else np.nan),
                    "residual_median_m": float(per.median()) if per is not None else np.nan,
                    "residual_iqr_m": (float(vals.quantile(0.75) - vals.quantile(0.25))
                                       if vals is not None else np.nan),
                    "residual_session_min_m": float(per.min()) if per is not None else np.nan,
                    "residual_session_max_m": float(per.max()) if per is not None else np.nan,
                    "sessions": " ".join(per.index) if per is not None else "",
                    "epochs": epochs, "ranges_ab": directed.get((a, b), 0),
                    "ranges_ba": directed.get((b, a), 0),
                    "availability_ab_pct": 100.0 * directed.get((a, b), 0) / epochs,
                    "availability_ba_pct": 100.0 * directed.get((b, a), 0) / epochs,
                    "availability_pct": 100.0 * either.get(k, 0) / epochs})
        if samples is not None:
            samples[f"{c}_anchor"] = aa_ep["residual_m"].to_numpy(float)
            if mob is not None:
                samples[f"{c}_tag"] = mob["residual_m"].to_numpy(float)
        if facts is not None and mob is not None:
            got = fmt(float(mob.groupby(["session", "anchor"])["residual_m"].median().median()), 2)
            check_fact(facts, f"WEresMaMed{'SetB' if c == 'set_b' else 'SetA'}", got)
    pairs = pd.DataFrame(rows)
    bad = set(pairs["status"]) - set(PAIR_STATUS)
    if bad:
        raise SystemExit(f"06_data_facts: unknown pair status {bad}")
    if facts is not None:
        for c, tag in (("set_b", "SetB"), ("set_a", "SetA")):
            aa = pairs[(pairs["set"] == c) & (pairs["kind"] == "anchor")]
            res = aa.loc[aa["status"] == "ranges", "residual_median_m"]
            check_fact(facts, f"WEresAaMed{tag}", fmt(float(res.median()), 2))
            if c == "set_a":
                check_fact(facts, "DFresAaPairsSetA", str(len(res)))
                check_fact(facts, "DFresAaPairsPossibleSetA", str(len(aa)))
                check_fact(facts, "DFresAaBiasedPairsSetA", str(int((res >= BIAS_FLAG_M).sum())))
                check_fact(facts, "DFresAaBiasMaxSetA", fmt(float(res.max()), 1))
            else:
                r = aa[(aa["node_a"] == "5BB6") & (aa["node_b"] == "D38F")].iloc[0]
                check_fact(facts, "DFresObstructedSetB", fmt(float(r["residual_median_m"]), 2))
                check_fact(facts, "DFresObstructedSpanSetB",
                           span([r["residual_session_min_m"], r["residual_session_max_m"]], 1))
    return pairs


def check_fact(facts: dict, name: str, value: str) -> None:
    if facts.get(name) != value:
        raise SystemExit(f"06_data_facts: pairs.csv gives {name} = {value}, "
                         f"the macro holds {facts.get(name)}")


def label_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """The field labels against the residuals: per deployment and label, the
    pairs, those with a residual, the median and the largest residual, and the
    pairs at or above BIAS_FLAG_M. The labels judge the line of sight in the
    field and are not corrected by the residuals."""
    rows = []
    for c in AA_SESSIONS:
        p = pairs[(pairs["set"] == c) & (pairs["field_label"] != "")]
        for value in sorted(set(p["field_label"]), key=lambda v: (LABEL_ORDER.get(v, 9), v)):
            q = p[p["field_label"] == value]
            r = q["residual_median_m"].dropna()
            rows.append({"set": c, "field_label": value, "pairs": len(q), "with_residual": len(r),
                         "residual_median_m": float(r.median()) if len(r) else np.nan,
                         "residual_max_m": float(r.max()) if len(r) else np.nan,
                         "biased": int((r >= BIAS_FLAG_M).sum())})
    return pd.DataFrame(rows)


def residual_quantiles(samples: dict) -> pd.DataFrame:
    """The quantiles (numpy 'linear') of the pooled per-epoch residuals at the
    levels k/QUANTILE_LEVELS: the anchor pairs of each deployment's residual
    sessions and the set_b tag at its matched, registered poses."""
    p = np.arange(QUANTILE_LEVELS + 1) / QUANTILE_LEVELS
    return pd.DataFrame({"p": p, **{k: np.quantile(samples[k], p) for k in QUANTILE_COLUMNS}})


def tail_share_model(d: np.ndarray, tau: float, model: dict) -> np.ndarray:
    """The probability that a range at the distance d exceeds d by more than
    tau under eq. (5) of Code/methods/range_errors.py, from the model values
    of synthetic/campaign.yaml: an outlier U(-d, d_max - d) with p_out, else
    multipath where p_eps exceeds mp_base - mp_slope d / d_max and the
    log-normal excess stays below d, else hardware noise N(0, sigma_hw^2).
    The clamp at zero acts on negative errors only, so it leaves the tail."""
    def phi(x):
        return np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in np.ravel(x)])

    def mp_cdf(x):
        return phi((np.log(x) - model["mp_mean"]) / model["mp_sigma"])

    d = np.asarray(d, float)
    p_out = model["p_out"]
    thr = model["mp_base"] - model["mp_slope"] * d / model["d_max_m"]
    if np.any(thr <= p_out):
        raise SystemExit("06_data_facts: the multipath and outlier cases of eq. (5) overlap")
    f_d = mp_cdf(d)
    p_mp = (1.0 - thr) * f_d
    out_tail = np.clip((model["d_max_m"] - d - tau) / model["d_max_m"], 0.0, 1.0)
    mp_tail = (1.0 - thr) * np.clip(f_d - mp_cdf(np.full(d.shape, tau)), 0.0, None)
    hw_tail = 1.0 - phi(np.full(d.shape, tau / model["sigma_hw_m"]))
    return p_out * out_tail + mp_tail + (1.0 - p_out - p_mp) * hw_tail


def synthetic_residuals(data: Path) -> tuple[dict, pd.DataFrame]:
    """The frozen synthetic draws against their model (F21). The residual of a
    range is the range minus its generating 2-D distance, computed from the
    coordinates of synthetic/nodes.yaml by the formula of the generator
    (synthetic._distances), and pooled per condition over its draws. Stops
    when a condition misses its check:
      noise_free      the largest absolute residual is exactly zero;
      gaussian        mean within SYN_TOL_M of zero, sd within SYN_TOL_M of sigma_hw;
      biased          median within SYN_TOL_M of the bias, sd within SYN_TOL_M of sigma_hw;
      nlos            in every draw, the median residual of every pair within
                      SYN_NLOS_TOL_SIGMA sigma_hw of its stored excess (nlos_pairs,
                      zero for a pair that the list leaves out);
      nlos_per_range  the share of residuals above SYN_TAIL_SIGMA sigma_hw within
                      SYN_TAIL_TOL of the share that tail_share_model predicts
                      for the drawn distances.
    Returns the statistics and the quantiles at the levels k/QUANTILE_LEVELS."""
    camp = yaml.safe_load((data / "synthetic" / "campaign.yaml").read_text(encoding="utf-8"))
    gen = camp["generator"]
    model = gen["model"]
    reg = yaml.safe_load((data / "synthetic" / "nodes.yaml").read_text(encoding="utf-8"))
    coords = pd.DataFrame({n["id"]: n["prior"]["xyz_local"] for n in reg["nodes"]},
                          index=["x", "y", "z"]).T.astype(float)
    res = {c: [] for c in gen["conditions"]}
    dist = {c: [] for c in gen["conditions"]}
    nlos_dev = 0.0
    for sp in sorted((data / "synthetic").glob("syn_*/session.yaml")):
        draw = yaml.safe_load(sp.read_text(encoding="utf-8"))["draw"]
        e = pd.read_parquet(sp.parent / "ranges.parquet")
        a = coords.loc[e["from_id"]].to_numpy(float)
        b = coords.loc[e["to_id"]].to_numpy(float)
        d = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)
        r = e["range_m"].to_numpy(float) - d
        res[draw["condition"]].append(r)
        dist[draw["condition"]].append(d)
        if draw["condition"] == "nlos":
            lo = e[["from_id", "to_id"]].min(axis=1).to_numpy()
            hi = e[["from_id", "to_id"]].max(axis=1).to_numpy()
            med = pd.Series(r).groupby([lo, hi]).median()
            # the writer stores the pairs with multipath only, and every other pair has none
            excess = {(p["node_a"], p["node_b"]): p["excess_m"] for p in draw["nlos_pairs"]}
            if not set(excess) <= set(med.index):
                raise SystemExit(f"06_data_facts: {sp.parent.name} stores the excess of a pair "
                                 "without ranges")
            for pair, m in med.items():
                nlos_dev = max(nlos_dev, abs(float(m) - excess.get(pair, 0.0)))
    pool = {c: np.concatenate(v) for c, v in res.items()}
    sigma_hw, bias = model["sigma_hw_m"], gen["bias_m"]
    tau = SYN_TAIL_SIGMA * sigma_hw
    d_tail, n_tail = np.unique(np.concatenate(dist["nlos_per_range"]), return_counts=True)
    st = {"noise_free_max_abs": float(np.abs(pool["noise_free"]).max()),
          "gaussian_mean": float(pool["gaussian"].mean()),
          "gaussian_sd": float(pool["gaussian"].std(ddof=1)),
          "biased_median": float(np.median(pool["biased"])),
          "biased_sd": float(pool["biased"].std(ddof=1)),
          "nlos_dev_max": nlos_dev,
          "tail_share": float((pool["nlos_per_range"] > tau).mean()),
          "tail_share_model": float((tail_share_model(d_tail, tau, model) * n_tail).sum()
                                    / n_tail.sum()),
          "ranges_per_condition": int(len(pool["gaussian"]))}
    failed = [name for name, ok in (
        ("noise_free exact", st["noise_free_max_abs"] == 0.0),
        ("gaussian mean", abs(st["gaussian_mean"]) <= SYN_TOL_M),
        ("gaussian sd", abs(st["gaussian_sd"] - sigma_hw) <= SYN_TOL_M),
        ("biased median", abs(st["biased_median"] - bias) <= SYN_TOL_M),
        ("biased sd", abs(st["biased_sd"] - sigma_hw) <= SYN_TOL_M),
        ("nlos excess", st["nlos_dev_max"] <= SYN_NLOS_TOL_SIGMA * sigma_hw),
        ("nlos_per_range tail", abs(st["tail_share"] - st["tail_share_model"]) <= SYN_TAIL_TOL),
    ) if not ok]
    if failed:
        raise SystemExit(f"06_data_facts: the synthetic draws miss their model: {failed} {st}")
    levels = np.arange(QUANTILE_LEVELS + 1) / QUANTILE_LEVELS
    q = pd.DataFrame({"p": levels, **{c: np.quantile(pool[c], levels) for c in gen["conditions"]}})
    return st, q


def write_csv(df: pd.DataFrame, path: Path, decimals: dict) -> None:
    """A CSV file with LF endings, the float columns at the given decimals and
    empty cells for missing values."""
    out = df.copy()
    for col, nd in decimals.items():
        out[col] = ["" if pd.isna(v) else f"{v:.{nd}f}" for v in out[col]]
    out.to_csv(path, index=False, lineterminator="\n")


# ----------------------------------------------------------------- output

def session_table(data: Path) -> str:
    L = [r"% AUTO-GENERATED by Code/datasets/radiocal/scripts/06_data_facts.py from",
         r"% Data/<set>/<session>/session.yaml - do not edit by hand,",
         r"% regenerate instead.",
         r"\begin{table}[!htb]",
         r"  \caption{Sessions of the Torgau campaigns. Truth names the reference of",
         r"    the mobile node and its grade.}",
         r"  \label{tab:sessions}",
         r"  \centering",
         r"  \footnotesize",
         r"  \setlength{\tabcolsep}{4pt}",
         r"  \begin{tabular}{@{}llrrl@{}}",
         r"    \toprule",
         r"    Session & Motion & Epochs & Ranges & Truth \\"]
    for c in MEASURED:
        L += [r"    \midrule",
              rf"    \multicolumn{{5}}{{@{{}}l}}{{\texttt{{{c.replace('_', chr(92) + '_')}}}}} \\"]
        for sp in sorted((data / c).glob("*/session.yaml")):
            m = yaml.safe_load(sp.read_text(encoding="utf-8"))
            cnt = m["counts"]
            L.append(rf"    \texttt{{{m['session_id']}}} & {MOTION[m['motion']]} & "
                     rf"{cnt['range_epochs']} & {cnt['directed_ranges']} & {truth_cell(m)} \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DATA, metavar="DIR",
                    help="folder with the campaign folders (default: Data/ at the repository root)")
    ap.add_argument("--out", type=Path, required=True, metavar="DIR",
                    help="folder for data_macros.tex, tab_sessions.tex, sessions.csv, "
                         "data_facts.csv, pairs.csv, labels.csv, residual_quantiles.csv, "
                         "synthetic_quantiles.csv")
    args = ap.parse_args()
    F, sess = compute(args.data)
    samples = {}
    pairs = pair_table(args.data, facts={r["macro"]: r["value"] for r in F.rows},
                       samples=samples)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    head = ["% AUTO-GENERATED by Code/datasets/radiocal/scripts/06_data_facts.py from",
            "% the released files under Data/ - do not edit by hand, regenerate instead.",
            "% Rule and source of every macro: data_facts.csv."]
    (out / "data_macros.tex").write_text(
        "\n".join(head + [rf"\newcommand{{\{r['macro']}}}{{{r['value']}}}" for r in F.rows]) + "\n",
        encoding="utf-8", newline="\n")
    (out / "tab_sessions.tex").write_text(session_table(args.data), encoding="utf-8", newline="\n")
    sess.to_csv(out / "sessions.csv", index=False, lineterminator="\n")
    with open(out / "data_facts.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, ["macro", "value", "rule", "sources"], lineterminator="\n")
        w.writeheader()
        w.writerows(F.rows)  # the rule constants are rows of their own (F18 in compute)
    write_csv(pairs, out / "pairs.csv", PAIR_DECIMALS)
    write_csv(label_table(pairs), out / "labels.csv", LABEL_DECIMALS)
    write_csv(residual_quantiles(samples), out / "residual_quantiles.csv", QUANTILE_DECIMALS)
    syn_q = F.tables["synthetic_quantiles"]
    write_csv(syn_q, out / "synthetic_quantiles.csv",
              {"p": 3, **{c: SYN_QUANTILE_DECIMALS for c in syn_q.columns if c != "p"}})
    print(f"06_data_facts: {len(F.rows)} macros, {len(sess)} sessions, {len(pairs)} pairs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
