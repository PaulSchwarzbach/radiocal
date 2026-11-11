"""Set B (Torgau, 2025-10-22) campaign configuration, shared by the inspect and build scripts.

The raw files are found under a raw root (--raw-root). Each file is looked
up by name, first at its place in the original campaign folder (coordinate
file under Koordinaten/, backup logs under Data/AutoPos_backup/, databases
under Data/20251022/), then directly in the root, which is the layout of the
raw deposit on Zenodo (raw/set_b/<file>). The default root is raw/set_b at
the repository root, where the deposit unpacks.
"""

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]  # repository root
DEFAULT_RAW = REPO / "raw" / "set_b"
OUT = REPO / "Data" / "set_b"

COORD_NAME = "Tag2_TORGAU-22102025.txt"

# Subfolder of each raw file in the original campaign folder.
LAYOUT = {
    COORD_NAME: "Koordinaten",
    "dyn_torgau_1.log": "Data/AutoPos_backup",
    "dyn_torgau_2.log": "Data/AutoPos_backup",
    "stat_torgau.log": "Data/AutoPos_backup",
    "autopos_run1.db": "Data/20251022",
    "autopos_run2.db": "Data/20251022",
    "autopos_run3_static.db": "Data/20251022",
}

ANCHORS = ["4197", "4503", "4A23", "4AA4", "5388", "5BB6", "8AA5", "8B05", "9911", "D38F"]
MOBILE = "DD98"
EXCLUDED = {"D5B4": "surveyed_never_ranging"}

# campaign-local ENU origin (UTM33N E, N, DHHN2016 h) - round numbers SW/below the site
ENU_ORIGIN = (361600.0, 5715100.0, 80.0)

VICON_TOPIC = "tracking/objects/torgau-final-autopos"

# Vicon registration (wand tie): the total station surveyed the calibration
# wand at the Vicon origin (WAND-0-0) and at a point on the Vicon +y axis
# (WAND-001). One Vicon calibration served every session, so one transform
# serves both walks. It uses no UWB range.
WAND_ORIGIN = "WAND-0-0"
WAND_Y_AXIS = "WAND-001"
VICON_EVAL_SESSIONS = ("s01", "s02")  # sessions scored against Vicon truth

SESSION_FILES = [
    # (session_id, motion, backup log, mqtt db)
    ("s01", "dynamic", "dyn_torgau_1.log", "autopos_run1.db"),
    ("s02", "dynamic", "dyn_torgau_2.log", "autopos_run2.db"),
    ("s03", "static", "stat_torgau.log", "autopos_run3_static.db"),
]

# SHA-256 of each database: the original campaign file, and the copy in the
# raw deposit, stripped of the ZigPos streams by Code/release/stage_zenodo_raw.py
# with SQLite 3.50.4. The builder accepts either file and records both hashes,
# so a rebuild from the raw deposit writes the released campaign.yaml and
# session.yaml.
DB_SHA256 = {
    "autopos_run1.db": {
        "original": "04a596276a3b2b18e64affd3f2716613471a75b0b0e95ae50725e9ac63bbbb43",
        "archived": "4c4e1d513859e944b68fed56a20a639c0de6265377c32aef493f4ce6f3418d51"},
    "autopos_run2.db": {
        "original": "1cc278fbbbd31ca2193f81039c01a7aff050a2a384d5f46ff21f788377003d77",
        "archived": "bea4a57629be6249d19f3d335999e3cb2912cdc691daa3894901a19eccee9782"},
    "autopos_run3_static.db": {
        "original": "5a0aa5ff734f4df5b5777f1ffc80552981ec9d5d64b1cbecce967f7f98b1d759",
        "archived": "7cdcf01a5e6700836decb38d2fa169091d77bd252848336c647f2846655c8f9b"},
}

CAMPAIGN_META = {
    "campaign": "set_b",
    "title": "Torgau UWB anchor-calibration campaign, 2025-10-22",
    "location": "Torgau, Saxony, Germany (industrial hall, the hall of set_a)",
    "date": "2025-10-22",
    "hardware": "Qorvo/Decawave MDEK1001 (DWM1001) UWB, two-way ranging, 11-node mesh",
    "groundtruth": "Vicon motion capture (mobile node), Total Station survey (anchors)",
    "nodes": {"anchors": ANCHORS, "mobile": MOBILE, "excluded": list(EXCLUDED)},
    "notes": [
        "Range stream: full-rate backup logs (~12.5 Hz); MQTT DBs hold a subsampled duplicate.",
        "All timestamps unified onto the Vicon (tracking PC) clock; broker receive times are"
        " contaminated by logger queueing and only used for min-envelope offset estimation.",
        "Distance matrices are directed (row = measuring node); symmetrize in the loader.",
        "Vicon -> local ENU registration: wand tie from the surveyed control points"
        " WAND-0-0 (Vicon origin, incl. height) and WAND-001 (Vicon +y axis), yaw about"
        " the vertical, scale 1, one transform for s01 and s02. Independent of the UWB"
        " ranges; Code/methods/estimate_vicon_transform.py fits a UWB-based transform"
        " as a cross-check.",
    ],
}


def raw_file(root: Path, name: str) -> Path:
    """Path of a raw file under root: original subfolder first, then flat."""
    for cand in (root / LAYOUT[name] / name, root / name):
        if cand.is_file():
            return cand
    raise SystemExit(f"raw file {name} not found under {root} "
                     f"(looked in {LAYOUT[name]}/ and in the root); pass --raw-root")


def coord_file(root: Path) -> Path:
    return raw_file(root, COORD_NAME)


def sessions(root: Path) -> list[tuple[str, str, Path, Path]]:
    return [(sid, motion, raw_file(root, log), raw_file(root, db))
            for sid, motion, log, db in SESSION_FILES]


def parser(description: str, out: bool = False) -> argparse.ArgumentParser:
    """Command line of the set B scripts: --raw-root, and --out for builders."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW, metavar="DIR",
                   help="folder with the set B raw files, in the original "
                        "campaign layout or flat as in the Zenodo deposit "
                        "(default: raw/set_b at the repository root)")
    if out:
        p.add_argument("--out", type=Path, default=OUT, metavar="DIR",
                       help="output folder (default: Data/set_b at the repository "
                            "root; destructive, overwrites the released files)")
    return p
