"""Step 0: assert the raw-data facts the conversion relies on (read-only).

Codifies the 2026-07-06 inspection findings; run before rebuilding the dataset
to catch silently changed/moved raw files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from campaign_set_b import (ANCHORS, EXCLUDED, MOBILE, VICON_TOPIC, coord_file,
                             parser, sessions)
from radiocal.io import coords, mqtt_db, qorvo_log

# total repr records per backup log (idle + data), from the 2026-07-06 inspection
EXPECT_RECORDS = {"s01": 3834, "s02": 3877, "s03": 2251}


def check(label, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    if not ok:
        raise SystemExit(f"assertion failed: {label} {detail}")


def main():
    raw = parser(__doc__.splitlines()[0]).parse_args().raw_root
    print("== coordinate file ==")
    shots = coords.parse_totalstation(coord_file(raw))
    points = coords.dedupe_shots(shots)
    names = set(points["name"])
    check("all 10 anchors surveyed", set(ANCHORS) <= names)
    check("D5B4 surveyed", "D5B4" in names)
    conflicts = points[points["conflict"]]["name"].tolist()
    check("no anchor survey conflicts", not any(n in ANCHORS + ["D5B4"] for n in conflicts),
          f"control-point conflicts (expected, e.g. GCP-12): {conflicts}")

    for sid, motion, log_path, db_path in sessions(raw):
        print(f"== session {sid} ==")
        check("backup log exists", log_path.is_file(), str(log_path))
        check("mqtt db exists", db_path.is_file(), str(db_path))

        n_records = sum(1 for _ in qorvo_log.iter_log_records(log_path))
        check(f"record count == {EXPECT_RECORDS[sid]}", n_records == EXPECT_RECORDS[sid],
              f"got {n_records}")
        ranges = qorvo_log.parse_qorvo_log(log_path)
        n_epochs = ranges["seq"].nunique()
        check("epochs with ranges plausible", 0.85 * n_records < n_epochs <= n_records,
              f"{n_epochs} epochs from {n_records} records")
        ids = set(ranges["from_id"]) | set(ranges["to_id"])
        check("node ids == 10 anchors + DD98", ids == set(ANCHORS) | {MOBILE}, f"got {sorted(ids)}")
        check("D5B4 never ranges", not (ids & set(EXCLUDED)))
        nonpos = ranges["range_m"] <= 0
        check("non-positive (failed TWR) rate < 0.5%", nonpos.mean() < 0.005,
              f"{int(nonpos.sum())}/{len(ranges)}")
        good = ranges.loc[~nonpos, "range_m"]
        check("positive ranges plausible (<100 m)", bool((good < 100).all()),
              f"max={good.max():.2f} m")

        heads = mqtt_db.read_autopos_heads(db_path)
        matched = heads[heads["seq"].isin(set(ranges["seq"]))]
        check("db auto_pos is a subsample of the log", len(matched) / max(len(heads), 1) > 0.95,
              f"{len(matched)}/{len(heads)} seq matched")
        min_env = (matched["t_broker"] - matched["t_raw"]).min()
        check("clock offset lower envelope in 6..8 s band", 6 < min_env < 8,
              f"{min_env:.3f} s (queue lag beyond this is expected)")

        vic = mqtt_db.read_vicon(db_path, VICON_TOPIC, MOBILE)
        check("vicon poses present", len(vic) > 300, f"{len(vic)} poses")
        check("vicon frames strictly increasing", bool(vic["frame"].is_monotonic_increasing))

        move = ranges[((ranges["from_id"] == MOBILE) | (ranges["to_id"] == MOBILE))
                      & (ranges["range_m"] > 0)]
        std = move.groupby(["from_id", "to_id"])["range_m"].std().median()
        if motion == "dynamic":
            check("DD98 moves (median pair std > 1 m)", std > 1.0, f"{std:.2f} m")
        else:
            check("DD98 static (median pair std < 0.1 m)", std < 0.1, f"{std:.3f} m")

    print("all raw-data assertions passed")


if __name__ == "__main__":
    main()
