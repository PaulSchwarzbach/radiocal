"""Convert the NLOS campaign of Torgau (set_a, recorded on 2025-03-13) into Data/set_a.

The campaign of 2025-03-13 (autopositioning_torgau, P. Schwarzbach and A. Jung):
fully-meshed 12-node Qorvo network, no Vicon - all reference positions from the
tachymeter (campaign-local frame). D5B4 actively ranges here (unlike set_b).
The date is the author's record (the descriptor's session P, 2026-11-08), since
the logs carry no timestamps. Earlier versions of this file called the campaign
"2024", which belongs to another campaign in the same hall (HYMN). Sessions:

  s21/s22       all 12 nodes static at their surveyed positions
  s01..s20          quasi-static: node 8B05 (the roving tag - confirmed by
                    multilateration against the assigned markers, 2026-07-06)
                    placed at surveyed marker PXX; all other nodes static.
                    DD98 never moves in set_a and is a regular anchor.

Tag height. The tag stood on a tripod at about anchor height (field record).
Its antenna height above the ground marker is estimated from the ranges: a
robust joint fit of the height and one common range bias over the
tag-anchor median ranges of the sessions the screen passes (below). The fit
gives 1.72 m, the height of the surveyed 8B05 antenna above marker P01 in
the static sessions. groundtruth.parquet carries marker z + that height.
With the bias forced to zero the same ranges put the tag near the ground,
which is where an earlier note "P-marker z fits the ranges with ~0 m
offset" came from.

Truth adjudication, a rule applied to every quasi-static session, with no
estimator output involved:

1. Screen: 8B05 is robustly multilaterated in 2-D from its median ranges to
   the 11 static nodes. A deviation of SCREEN_DEV_M or more from the
   assigned marker flags the session. The fit can end far from the marker even
   where every link agrees with it (s05), so a flag only triggers step 2.
2. Direct range test (ruling D10, 2026-09-30): the median tag-anchor ranges
   against the 3-D distances to the marker at the fitted tag height. A link
   counts as line of sight when its residual lies within LOS_TOL_M of the
   fitted common bias. A flagged session passes when the median residual of
   its line-of-sight links lies within the range of the sessions the screen
   passes. It is then verified, otherwise unreliable.

Every statistic goes to Data/set_a/truth_adjudication.csv, together with a
stationarity figure that is reported only: the span of a 9-epoch rolling
median per line-of-sight link, median over the links. A tag that moves
shifts every link at once, while a burst on one link leaves the median.

Log records carry no timestamps, so seq orders the epochs.

Raw sources are located via --raw-root, a folder that holds
torgau_autopos_ref.txt and the logs (in log/ or at its top level). The
default is raw/set_a at the repository root, where the raw archive of the
Zenodo deposit unpacks.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

from radiocal import convert
from radiocal.graph import TRUTH_COLUMNS
from radiocal.io import coords as coords_io
from radiocal.io import qorvo_log
from radiocal.masks import _PACKAGED_MASKS as PACKAGED_MASKS

REPO = Path(__file__).resolve().parents[4]  # repository root
RAW_DEFAULT = REPO / "raw" / "set_a"  # where the Zenodo raw deposit unpacks
OUT = REPO / "Data" / "set_a"
NODES = ["4197", "4503", "4A23", "4AA4", "5388", "5BB6",
         "8AA5", "8B05", "9911", "D38F", "D5B4", "DD98"]
TAG = "8B05"
CRS = {
    "epsg": None,
    "note": "campaign-local tachymeter frame (metres); not georeferenced",
    "axes": "x, y, z as surveyed (anchor z ~ tripod antenna height, "
            "P-marker z = ground marker; the tag antenna stood on a tripod "
            "above the marker, see campaign.yaml)",
}

SCREEN_DEV_M = 1.0         # screen: 2-D multilateration deviation from the marker
LOS_TOL_M = 0.5            # line of sight: residual within this of the common bias
HEIGHT_FIT_SCALE_M = 0.3   # cauchy scale of the joint height and bias fit
ROLL_EPOCHS = 9            # rolling-median window of the stationarity figure


def truth_row(name: str, ref: pd.DataFrame, node_id: str,
              height: float) -> pd.DataFrame:
    p = ref.set_index("name").loc[name]
    row = {c: np.nan for c in TRUTH_COLUMNS}
    row.update(node_id=node_id, x=float(p["E"]), y=float(p["N"]),
               z=round(float(p["h"]) + height, 3))
    return pd.DataFrame([row], columns=TRUTH_COLUMNS)


def tag_links(log_path: Path) -> pd.DataFrame:
    """Valid tag-anchor ranges of one log (columns seq, other, range_m), the
    same records the session's ranges.parquet keeps."""
    ranges, _ = convert._drop_invalid_ranges(qorvo_log.parse_qorvo_log(log_path))
    t = ranges[(ranges["from_id"] == TAG) | (ranges["to_id"] == TAG)].copy()
    t["other"] = np.where(t["from_id"] == TAG, t["to_id"], t["from_id"])
    return t[["seq", "other", "range_m"]]


def multilaterate_tag(med: pd.Series, surveyed: pd.DataFrame) -> np.ndarray:
    """Robust x/y position of TAG from its median ranges to the static nodes."""
    A = surveyed.loc[med.index, ["E", "N"]].to_numpy(dtype=float)
    d = med.to_numpy()
    x = A.mean(axis=0)
    for _ in range(60):
        diff = x - A
        r = np.linalg.norm(diff, axis=1)
        res = r - d
        w = np.where(np.abs(res) < 1.0, 1.0, 1.0 / np.abs(res))  # Huber-like
        dx = np.linalg.lstsq((diff / r[:, None]) * w[:, None], -res * w, rcond=None)[0]
        x = x + dx
        if np.linalg.norm(dx) < 1e-8:
            break
    return x


def marker_residuals(med: pd.Series, surveyed: pd.DataFrame,
                     marker: pd.Series, height: float) -> pd.Series:
    """Median range minus the 3-D distance from each static node to the
    tag antenna at `height` above the marker."""
    A = surveyed.loc[med.index, ["E", "N", "h"]].to_numpy(dtype=float)
    p = np.array([marker["E"], marker["N"], marker["h"] + height], dtype=float)
    return med - np.linalg.norm(A - p, axis=1)


def fit_tag_height(links: list[tuple[pd.Series, pd.Series]],
                   surveyed: pd.DataFrame) -> tuple[float, float]:
    """Robust joint fit (cauchy) of the tag height above the marker and one
    common range bias over the (median ranges, marker) pairs given."""
    def f(p):
        return np.concatenate([
            marker_residuals(med, surveyed, marker, p[0]).to_numpy() - p[1]
            for med, marker in links])
    sol = least_squares(f, [1.0, 0.0], loss="cauchy", f_scale=HEIGHT_FIT_SCALE_M)
    return float(sol.x[0]), float(sol.x[1])


def roll_span(links: pd.DataFrame, others: list[str]) -> float:
    """Span of the ROLL_EPOCHS rolling median per link, median over the
    given links."""
    spans = []
    for o in others:
        r = links[links["other"] == o].sort_values("seq")["range_m"]
        rm = r.rolling(ROLL_EPOCHS, center=True).median().dropna()
        if len(rm):
            spans.append(float(rm.max() - rm.min()))
    return float(np.median(spans)) if spans else float("nan")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-root", type=Path, default=RAW_DEFAULT, metavar="DIR",
        help="folder with torgau_autopos_ref.txt and the session logs, the "
             "logs either in log/ as in the original checkout or next to the "
             "reference file as in the Zenodo deposit (default: raw/set_a at "
             "the repository root)")
    parser.add_argument("--out", type=Path, default=OUT, metavar="DIR",
                        help="output folder (default: Data/set_a at the repository "
                             "root; destructive, overwrites the released files)")
    args = parser.parse_args()
    raw, OUT = args.raw_root, args.out
    coord_file = raw / "torgau_autopos_ref.txt"
    if not coord_file.exists():
        parser.error(f"{coord_file} not found - pass --raw-root")

    def log_file(name):
        for cand in (raw / "log" / name, raw / name):
            if cand.is_file():
                return cand
        parser.error(f"{name} not found in {raw / 'log'} or {raw} - pass --raw-root")

    OUT.mkdir(parents=True, exist_ok=True)
    ref = coords_io.dedupe_shots(coords_io.parse_totalstation(coord_file))
    refi = ref.set_index("name")
    surveyed_others = refi.loc[[n for n in NODES if n != TAG]]

    sessions = [("s21", log_file("stat_1.log"), None),
                ("s22", log_file("stat_2.log"), None)]
    sessions += [(f"s{i:02d}", log_file(f"dyn_p{i:02d}.log"), f"P{i:02d}")
                 for i in range(1, 21)]

    # truth adjudication: screen, tag height, direct range test
    print("-> truth adjudication (screen, tag height, direct range test)")
    adj = {}
    for sid, log_path, marker in sessions:
        if marker is None:
            continue
        links = tag_links(log_path)
        med = links.groupby("other")["range_m"].median()
        est = multilaterate_tag(med, surveyed_others)
        dev = float(np.linalg.norm(est - refi.loc[marker, ["E", "N"]].to_numpy(dtype=float)))
        adj[sid] = {"marker": marker, "links": links, "med": med,
                    "screen_dev_m": dev, "screen_passed": dev < SCREEN_DEV_M}
    unflagged = [s for s, a in adj.items() if a["screen_passed"]]
    height, bias = fit_tag_height(
        [(adj[s]["med"], refi.loc[adj[s]["marker"]]) for s in unflagged],
        surveyed_others)
    height = round(height, 2)
    print(f"   tag antenna {height:.2f} m above the marker, common range bias "
          f"{bias:+.3f} m (cauchy fit over {len(unflagged)} screened sessions)")
    for sid, a in adj.items():
        res = marker_residuals(a["med"], surveyed_others, refi.loc[a["marker"]], height)
        los = res[(res - bias).abs() <= LOS_TOL_M]
        a.update(n_links=len(res), n_los=len(los),
                 median_los_resid_m=float(los.median()),
                 roll_span_m=roll_span(a["links"], list(los.index)))
    ref_vals = [adj[s]["median_los_resid_m"] for s in unflagged]
    lo, hi = min(ref_vals), max(ref_vals)
    for sid, a in adj.items():
        passed = lo <= a["median_los_resid_m"] <= hi
        a["range_test_passed"] = passed
        a["truth_quality"] = ("verified" if a["screen_passed"] or passed
                              else "unreliable")
        print(f"   {sid}: screen {a['screen_dev_m']:.2f} m, line-of-sight links "
              f"{a['n_los']}/{a['n_links']}, median residual "
              f"{a['median_los_resid_m']:+.3f} m -> {a['truth_quality']}")
    table = pd.DataFrame([{
        "session": sid, "marker": a["marker"],
        "screen_deviation_m": round(a["screen_dev_m"], 3),
        "screen_passed": a["screen_passed"],
        "n_links": a["n_links"], "n_los_links": a["n_los"],
        "median_los_residual_m": round(a["median_los_resid_m"], 3),
        "reference_min_m": round(lo, 3), "reference_max_m": round(hi, 3),
        "range_test_passed": a["range_test_passed"],
        "rolling_median_span_m": round(a["roll_span_m"], 3),
        "tag_height_above_marker_m": height,
        "common_range_bias_m": round(bias, 3),
        "truth_quality": a["truth_quality"]} for sid, a in adj.items()])
    table.to_csv(OUT / "truth_adjudication.csv", index=False, lineterminator="\n")
    n_ver = int((table["truth_quality"] == "verified").sum())

    print("-> nodes.yaml (12 surveyed nodes; 8B05 roves only in quasi-static sessions via overrides)")
    convert.build_registry(coord_file, anchors=NODES, mobiles={}, excluded={},
                           crs=CRS, out_path=OUT / "nodes.yaml")

    print("-> campaign.yaml")
    flagged = sorted(s for s, a in adj.items() if not a["screen_passed"])
    meta = {
        "campaign": "set_a",
        "title": "Torgau UWB auto-positioning campaign, 2025-03-13 (tachymeter-referenced)",
        "date": "2025-03-13",
        "location": "Torgau, Saxony, Germany",
        "hardware": "Qorvo/Decawave MDEK1001 UWB, two-way ranging, fully-meshed 12-node network",
        "groundtruth": "Total Station (tachymeter): all node positions + roving-tag markers P01..P20",
        "nodes": {"anchors": [n for n in NODES if n != TAG], "roving_tag": TAG, "excluded": []},
        "notes": [
            "12 surveyed nodes; D5B4 ranges in this campaign (unlike set_b); DD98 is static here.",
            "s01..s20: 8B05 quasi-static at surveyed marker PXX on a tripod; other nodes static.",
            f"Tag antenna {height:.2f} m above the P-marker (groundtruth z = marker z + "
            f"{height:.2f}): robust joint fit of height and common range bias "
            f"({bias:+.2f} m) over the tag-anchor ranges; the surveyed 8B05 antenna of "
            "s21/s22 stands the same height above marker P01.",
            f"truth_quality: {n_ver}/{len(adj)} verified. Screen = 2-D multilateration "
            f"vs marker (< {SCREEN_DEV_M:g} m); flagged sessions ({', '.join(flagged)}) "
            "were decided by the direct range test at the marker. Statistics in "
            "truth_adjudication.csv.",
            "Log records carry no timestamps; epoch ordering via seq.",
        ],
        "sources": [{"path": p.name, "sha256": convert.sha256(p)} for _, p, _ in sessions] +
                   [{"path": coord_file.name, "sha256": convert.sha256(coord_file)}],
    }
    (OUT / "campaign.yaml").write_text(yaml.safe_dump(meta, sort_keys=False),
                                       encoding="utf-8", newline="\n")

    print("-> masks/")
    (OUT / "masks").mkdir(exist_ok=True)
    for m in PACKAGED_MASKS.glob("*.yaml"):  # LF bytes on every platform
        (OUT / "masks" / m.name).write_bytes(m.read_bytes().replace(b"\r\n", b"\n"))

    for sid, log_path, marker in sessions:
        if marker is None:
            convert.build_session_ranges_only(log_path, OUT / sid, sid, "static", "set_a")
            print(f"   {sid}: static (truth = surveyed priors)")
            continue
        a = adj[sid]
        verification = {
            "screen": {"method": "robust_multilateration_xy_vs_marker",
                       "deviation_m": round(a["screen_dev_m"], 3),
                       "passed": bool(a["screen_passed"])},
            "range_test": {"method": "range_residuals_vs_marker",
                           "tag_height_above_marker_m": height,
                           "median_los_residual_m": round(a["median_los_resid_m"], 3),
                           "n_los_links": int(a["n_los"]),
                           "n_links": int(a["n_links"]),
                           "reference_range_m": [round(lo, 3), round(hi, 3)],
                           "passed": bool(a["range_test_passed"])},
        }
        extra = {
            "node_overrides": [{"id": TAG, "role": "mobile", "prior": {"type": "none"}}],
            "truth_quality": a["truth_quality"],
            "verification": verification,
        }
        notes = [f"roving tag {TAG} on a tripod at surveyed marker {marker}; other nodes static"]
        if not a["screen_passed"]:
            if a["truth_quality"] == "verified":
                notes.append(
                    f"The screen flagged a {a['screen_dev_m']:.1f} m deviation of the "
                    "2-D multilateration. The "
                    "direct range test at the marker passes: median line-of-sight "
                    f"residual {a['median_los_resid_m']:+.3f} m, reference "
                    f"{lo:+.3f} to {hi:+.3f} m. Truth verified.")
            else:
                notes.append(
                    "TRUTH UNRELIABLE: the direct range test at the marker fails "
                    f"(median line-of-sight residual {a['median_los_resid_m']:+.3f} m, "
                    f"reference {lo:+.3f} to {hi:+.3f} m) - use ranges only")
        convert.build_session_ranges_only(
            log_path, OUT / sid, sid, "quasi_static_point", "set_a",
            truth=truth_row(marker, ref, TAG, height), notes=notes, extra_meta=extra)
        print(f"   {sid}: tag @ {marker} -> {a['truth_quality']}")

    print(f"dataset written to {OUT}")


if __name__ == "__main__":
    main()
