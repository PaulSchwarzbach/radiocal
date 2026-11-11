"""Step 1-3: build the canonical Data/set_b dataset from the raw campaign.

Step 4: the Vicon -> local ENU registration of the sessions scored against
Vicon truth, from the surveyed wand points alone (wand_tie).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml

from campaign_set_b import (ANCHORS, CAMPAIGN_META, DB_SHA256, ENU_ORIGIN,
                             EXCLUDED, MOBILE, VICON_EVAL_SESSIONS, VICON_TOPIC,
                             WAND_ORIGIN, WAND_Y_AXIS, coord_file, parser,
                             sessions)
from radiocal import convert, frames
from radiocal.masks import _PACKAGED_MASKS as PACKAGED_MASKS


def wand_tie(nodes_yaml: Path) -> tuple[frames.Transform, dict]:
    """Vicon -> local ENU from the two surveyed wand points: the origin
    (x, y and height) at WAND_ORIGIN, the Vicon +y axis along WAND_ORIGIN ->
    WAND_Y_AXIS in the horizontal plane, the Vicon z axis vertical (the
    wand is levelled at calibration), scale 1. No UWB range enters."""
    reg = yaml.safe_load(nodes_yaml.read_text(encoding="utf-8"))
    cp = {c["name"]: np.asarray(c["xyz_local"], dtype=float)
          for c in reg["control_points"]}
    o, a = cp[WAND_ORIGIN], cp[WAND_Y_AXIS]
    u = (a - o)[:2]
    baseline = float(np.linalg.norm(u))
    u = u / baseline
    yaw = float(np.arctan2(-u[0], u[1]))  # rotation taking +y onto u
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    tf = frames.Transform(R, o.copy(), 1.0, src="vicon", dst="enu")
    provenance = {
        "method": "wand_tie",
        "origin": WAND_ORIGIN,
        "y_axis": WAND_Y_AXIS,
        "yaw_deg": float(np.degrees(yaw)),
        "baseline_m": baseline,
        "height_difference_m": float(a[2] - o[2]),
        "note": ("survey-only registration, independent of the UWB ranges; "
                 "estimate_vicon_transform.py fits the UWB-based cross-check"),
    }
    return tf, provenance


def source_record(path: Path) -> dict:
    """The `sources` entry of a raw file. A database must equal its original
    or its archived copy (campaign_set_b.DB_SHA256), and both hashes are
    recorded, so a rebuild from the raw deposit writes the released YAML."""
    h = convert.sha256(path)
    known = DB_SHA256.get(path.name)
    if known is None:
        return {"path": path.name, "sha256": h}
    if h not in (known["original"], known["archived"]):
        raise SystemExit(f"{path}: SHA-256 {h} is neither the original nor the "
                         f"archived copy of {path.name} (campaign_set_b.DB_SHA256)")
    return {"path": path.name, "sha256": known["original"],
            "sha256_archived": known["archived"]}


def main():
    args = parser(__doc__.splitlines()[0], out=True).parse_args()
    OUT = args.out
    COORD_FILE = coord_file(args.raw_root)
    SESSIONS = sessions(args.raw_root)
    OUT.mkdir(parents=True, exist_ok=True)

    print("-> nodes.yaml (registry)")
    convert.build_registry(
        COORD_FILE,
        anchors=ANCHORS,
        mobiles={MOBILE: {"groundtruth": {"source": "vicon",
                                          "object": VICON_TOPIC.rsplit("/", 1)[-1]}}},
        excluded=EXCLUDED,
        origin=ENU_ORIGIN,
        out_path=OUT / "nodes.yaml",
    )

    print("-> campaign.yaml")
    meta = dict(CAMPAIGN_META)
    meta["sources"] = ([source_record(s[2]) for s in SESSIONS]
                       + [source_record(s[3]) for s in SESSIONS]
                       + [source_record(COORD_FILE)])
    (OUT / "campaign.yaml").write_text(yaml.safe_dump(meta, sort_keys=False),
                                       encoding="utf-8", newline="\n")

    print("-> masks/")
    (OUT / "masks").mkdir(exist_ok=True)
    for m in PACKAGED_MASKS.glob("*.yaml"):  # LF bytes on every platform
        (OUT / "masks" / m.name).write_bytes(m.read_bytes().replace(b"\r\n", b"\n"))

    for sid, motion, log_path, db_path in SESSIONS:
        print(f"-> session {sid} ({motion})")
        out = convert.build_session(log_path, db_path, OUT / sid, sid, motion,
                                    VICON_TOPIC, MOBILE,
                                    db_source=source_record(db_path))
        session = yaml.safe_load((out / "session.yaml").read_text(encoding="utf-8"))
        c, k = session["counts"], session["clock"]
        print(f"   {c['range_epochs']} epochs, {c['directed_ranges']} directed ranges "
              f"({c['dropped_invalid']} invalid dropped), "
              f"{c['truth_poses']} truth poses; clock offset {k['offset_s']:.3f} s "
              f"+/- {k['uncertainty_s']} (n={k['n_matched']}, "
              f"queue lag up to {k['queue_lag_max_s']:.1f} s)")

    tf, provenance = wand_tie(OUT / "nodes.yaml")
    print(f"-> Vicon registration (wand tie): yaw {provenance['yaw_deg']:.3f} deg, "
          f"t [{tf.t[0]:.3f}, {tf.t[1]:.3f}, {tf.t[2]:.3f}] m, "
          f"baseline {provenance['baseline_m']:.3f} m")
    for sid in VICON_EVAL_SESSIONS:
        spath = OUT / sid / "session.yaml"
        session = yaml.safe_load(spath.read_text(encoding="utf-8"))
        session["transform_vicon_to_local"] = tf.to_dict()
        session["transform_provenance"] = provenance
        spath.write_text(yaml.safe_dump(session, sort_keys=False), encoding="utf-8",
                         newline="\n")
        print(f"   {sid}: transform_vicon_to_local written")

    print(f"dataset written to {OUT}")


if __name__ == "__main__":
    main()
