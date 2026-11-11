"""Raw public data -> canonical radiocal campaigns in EXT_ROOT/data.

    python -B Code/external/convert_external.py --raw-root EXT_ROOT/raw --out EXT_ROOT/data

Reads Code/external/sources.yaml, checks every source file's SHA-256, and
writes ext_bern_a40, ext_selfloc_department and ext_selfloc_reception (session
r01 each) with the conversion rules documented in Code/external/README.md.
--raw-root is the folder with the fetched files and --out the folder that
receives the campaigns. Both default to the ext_root of sources.yaml (raw/
and data/ under it), outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from radiocal.convert import _drop_invalid_ranges  # noqa: E402

import session_writer as sw  # noqa: E402
from readers import selfloc, twr_cloud_bern  # noqa: E402

KAPPA_BASE = (1, 2, 3, 4, 5, 6, 8)
KAPPA_LARGE = (12, 16, 24)
LCC_MIN_SHARE, LCC_MIN_NODES = 0.90, 8


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checked(raw: Path, sources: dict, key: str, rel: str) -> tuple[Path, dict]:
    entry = next(f for f in sources[key]["files"] if f["path"] == rel)
    p = raw / rel
    got = _sha256(p)
    if got != entry["sha256"]:
        raise SystemExit(f"checksum mismatch for {rel}")
    return p, {"path": rel, "sha256": got}


def _build(data_root: Path, campaign: str, nodes: pd.DataFrame, edges: pd.DataFrame,
           meta: dict, src: list[dict], clock_note: str, crs_note: str,
           prior_source: str) -> dict:
    known = set(nodes["id"])
    unknown = ~(edges["from_id"].isin(known) & edges["to_id"].isin(known))
    edges = edges[~unknown]
    edges, dropped = _drop_invalid_ranges(edges)
    pairs = sw.pair_table(edges)
    keep = sw.largest_component(sorted(known), pairs)
    share = len(keep) / len(known)
    status = {i: "outside_largest_component" for i in known - keep}
    edges = edges[edges["from_id"].isin(keep) & edges["to_id"].isin(keep)]
    pairs = sw.pair_table(edges)
    triple = sw.gauge_triple(pairs, keep)
    deg = pd.concat([pairs["a"], pairs["b"]]).value_counts()
    max_deg = int(deg.max())
    kappa = [k for k in KAPPA_BASE if k <= max_deg] + [k for k in KAPPA_LARGE if k < max_deg]

    cdir = data_root / campaign
    sw.write_registry(cdir, nodes, crs_note=crs_note, prior_source=prior_source,
                      prior_sigma_m=0.01, status=status)
    sw.write_masks(cdir, triple)
    meta = dict(meta)
    meta["nodes"] = {"anchors": sorted(keep, key=lambda s: (len(s), s)),
                     "roving_tag": None,
                     "excluded": sorted(known - keep, key=lambda s: (len(s), s))}
    meta["sources"] = src
    meta["benchmark"] = {"gauge_triple": list(triple), "kappa_grid": kappa,
                         "max_degree": max_deg,
                         "largest_component_share": round(share, 4),
                         "provenance": "radiocal Code/external, run on public data"}
    sw.write_campaign(cdir, meta)
    counts = sw.write_session(cdir / "r01", session_id="r01", campaign=campaign,
                              edges=edges, dropped_invalid=dropped,
                              sources=[{**s, "role": "ranges"} for s in src],
                              clock_note=clock_note)
    return {"campaign": campaign, "nodes": len(known), "kept": len(keep),
            "lcc_share": round(share, 3),
            "lcc_ok": share >= LCC_MIN_SHARE and len(keep) >= LCC_MIN_NODES,
            "edges_unknown_id": int(unknown.sum()), "dropped_invalid": dropped,
            "directed_ranges": counts["directed_ranges"],
            "epochs": counts["range_epochs"], "undirected_pairs": len(pairs),
            "max_degree": max_deg, "gauge_triple": " ".join(triple),
            "kappa_grid": " ".join(map(str, kappa))}


def main() -> int:
    cfg = yaml.safe_load((HERE / "sources.yaml").read_text(encoding="utf-8"))
    ext_root = Path(cfg["ext_root"]).expanduser()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-root", type=Path, default=ext_root / "raw", metavar="DIR",
                    help="folder with the fetched files (default: raw/ under the ext_root "
                         "of sources.yaml)")
    ap.add_argument("--out", type=Path, default=ext_root / "data", metavar="DIR",
                    help="folder that receives the campaigns (default: data/ under the "
                         "ext_root of sources.yaml)")
    args = ap.parse_args()
    raw, data_root = args.raw_root, args.out
    S = cfg["sources"]
    out = []

    jp, js = _checked(raw, S, "twr_cloud_bern_v5", "twr_cloud_bern_v5/40-agents.json")
    nodes, edges, stats = twr_cloud_bern.read_scenario(jp)
    out.append(_build(
        data_root, "ext_bern_a40", nodes, edges,
        {"campaign": "ext_bern_a40",
         "title": "TWR-CLOUD-Bern evaluation scan, 40 static nodes (run on public data)",
         "location": "University of Bern, one office floor (Switzerland)",
         "hardware": "Qorvo DWM3001C (DW3000) on Raspberry Pi Zero 2W, 4-message DS-TWR",
         "groundtruth": "Leica BLK360 laser-scan registration, released in cm, converted to m",
         "source": {"citation_key": "xenakis_cooperative_2026",
                    "doi": "10.5281/zenodo.20277908", "version": 5, "licence": "MIT"},
         "notes": ["Ranges: released ToF [ns] times 0.299702547 m/ns (speed of light in air, "
                   "as in the authors' reproduce script).",
                   "Both released iterations pooled: up to 20 samples per directed link, "
                   "random draws without replacement from the full scan table.",
                   "No timestamps; seq is the rank of a sample within its directed link.",
                   f"Scenario file: {stats}"]},
        [js], "no timestamps in the released scenario; seq = rank within directed link",
        "campaign-local frame of the laser-scan registration (metres); not georeferenced",
        "laser_scan"))

    for name in ("department", "reception"):
        dp, ds = _checked(raw, S, "selfloc_39c1271", f"selfloc_39c1271/sw/deployments/{name}.csv")
        lp, ls = _checked(raw, S, "selfloc_39c1271", f"selfloc_39c1271/data/{name}/sstwr.csv")
        nodes = selfloc.read_deployment(dp)
        edges = selfloc.read_sstwr(lp, range_column="dist_drift")
        out.append(_build(
            data_root, f"ext_selfloc_{name}", nodes, edges,
            {"campaign": f"ext_selfloc_{name}",
             "title": f"selfloc {name.upper()} testbed (run on public data)",
             "location": "University of Trento, Italy",
             "hardware": "DW1000 (EVB1000 with Raspberry Pi), single-sided TWR with "
                         "clock-drift compensation",
             "groundtruth": "manually measured anchor positions as released; z = 2 m nominal",
             "source": {"citation_key": "corbalan_self-localization_2023",
                        "repository": "github.com/d3s-trento/selfloc",
                        "commit": S["selfloc_39c1271"]["commit"], "licence": "none stated"},
             "notes": ["Ranges: column dist_drift (drift-compensated, no signal-level bias "
                       "table); dist_drift_bias is a sensitivity only.",
                       "seq is the rank of an exchange within its directed pair by timestamp."]},
            [ls, ds], "Unix timestamps of the initiator log; seq = rank within directed pair",
            "campaign-local frame of the deployment file (metres); z nominal",
            "manual"))

    summary = pd.DataFrame(out)
    data_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(data_root / "conversion_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
