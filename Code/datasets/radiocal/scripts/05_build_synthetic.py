"""Build the synthetic counterpart of the testing ground into Data/synthetic.

SYNTHETIC DATA. Each session holds the directed ranges of one seeded draw of
the range-error generator Code/methods/range_errors.py (P. Schwarzbach,
R. Weber, O. Michler, IEEE Sensors J. 22(6):4836-4843, 2022, with the per-pair
extension of its docstring) on the surveyed anchors of set_b/s03: 10 anchors,
both directions of every pair, 200 synthetic epochs, 2-D (horizontal)
distances. Conditions: noise_free (one draw), gaussian, nlos, nlos_per_range
and biased (+0.27 m), with 20 draws each, draw k drawn with seed k. These are
the 81 draws of the synthetic baseline of the companion tutorial
(Code/methods/run_synthetic.py there, not part of this release).

    python -B Code/datasets/radiocal/scripts/05_build_synthetic.py --out rebuild/synthetic
    python -B Code/datasets/radiocal/scripts/05_build_synthetic.py --check

--data    folder with the campaign folders (default: Data/ at the repository
          root). set_b/s03 gives the geometry.
--out     output folder (default: Data/synthetic at the repository root;
          destructive, overwrites the released files)
--frozen  the reference hashes: frozen.csv of the tutorial's synthetic
          baseline (default: Code/methods/results/synthetic_baseline/frozen.csv
          when it exists). Without it, the ranges_sha256 stored in
          <data>/synthetic/*/session.yaml are the reference.
--check   draw in memory and compare with the reference and with the files
          under <data>/synthetic. Writes nothing.

ranges_sha256 is the first 16 hex digits of a SHA-256 over the little-endian
float64 range_m column in edge order (run_synthetic._sha of the tutorial).
Every draw must match its reference before anything is written. The draws
use numpy Generator streams, which numpy does not guarantee across versions,
so a mismatch under another numpy shows as generator drift.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]  # repository root
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(REPO / "Code" / "methods"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import radiocal  # noqa: E402
import range_errors  # noqa: E402
import synthetic  # noqa: E402
from radiocal.masks import _PACKAGED_MASKS  # noqa: E402

DATA = REPO / "Data"
OUT = DATA / "synthetic"
FROZEN = REPO / "Code" / "methods" / "results" / "synthetic_baseline" / "frozen.csv"
GEOMETRY = "set_b/s03"

# The setup of the tutorial's synthetic baseline (run_synthetic.py: N_EPOCHS,
# DRAWS, BIAS_M, MODEL). The hash check against frozen.csv proves the match.
N_EPOCHS = 200
DRAWS = 20
BIAS_M = 0.27
MODEL = range_errors.PAPER

# The generator that drew the released files: the last commit of the
# tutorial repository that touched both modules, and their SHA-256 with LF
# line endings. A changed module is refused, since its draws need not equal
# the released ones.
GENERATOR_COMMIT = "633fb950fced9ca4097bbd80675c47f2d2b81a85"
GENERATOR_SHA256 = {
    "range_errors.py": "8fe27c80e545c2216af4cc617b3300fc523259519a1350a67843f365e9cc0494",
    "synthetic.py": "bcdc86ca74ac8dd128a7b482a5c7b7449e14b4e4824903801c2fbb5667ec7170",
}
RATE_HZ = synthetic.RATE_HZ  # only sets the synthetic time axis
# Where the stored hashes come from. It is the same text whichever reference a
# run checks against, so that a rebuild writes the released session.yaml.
HASH_REFERENCE = ("equal to frozen.csv of the tutorial's synthetic baseline "
                  "(Code/methods/results/synthetic_baseline/frozen.csv at 633fb95)")


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def ranges_sha256(r) -> str:
    return hashlib.sha256(np.ascontiguousarray(r, dtype="<f8").tobytes()).hexdigest()[:16]


def session_id(condition: str, draw: int) -> str:
    return f"syn_{condition}_{draw:02d}"


def plan() -> list[tuple[str, int]]:
    return [(c, d) for c in range_errors.CONDITIONS
            for d in ([0] if c == "noise_free" else range(DRAWS))]


def reference(data: Path) -> pd.DataFrame:
    """The surveyed anchors of set_b/s03 (columns id, x, y, z), the rows of the
    tutorial's evaluation.surveyed_reference, built from radiocal alone."""
    ref = radiocal.load_session(data / GEOMETRY).priors()
    return ref[ref["role"] == "anchor"][["id", "x", "y", "z"]].reset_index(drop=True)


def check_generator() -> None:
    for name, want in GENERATOR_SHA256.items():
        got = lf_sha256(Path(getattr({"range_errors.py": range_errors,
                                      "synthetic.py": synthetic}[name], "__file__")))
        if got != want:
            raise SystemExit(f"05_build_synthetic: {name} has SHA-256 {got}, the released "
                             f"draws come from {want} (commit {GENERATOR_COMMIT[:7]})")


def reference_hashes(frozen: Path | None, data: Path) -> tuple[dict, str]:
    if frozen is not None and frozen.is_file():
        f = pd.read_csv(frozen, dtype={"ranges_sha256": str})
        per = f.groupby(["condition", "draw"])["ranges_sha256"]
        if (per.nunique() != 1).any():
            raise SystemExit(f"05_build_synthetic: {frozen} holds two hashes for one draw")
        return ({(c, int(d)): h for (c, d), h in per.first().items()},
                "frozen.csv of the tutorial's synthetic baseline")
    out = {}
    for sp in sorted((data / "synthetic").glob("syn_*/session.yaml")):
        d = yaml.safe_load(sp.read_text(encoding="utf-8"))["draw"]
        out[(d["condition"], int(d["index"]))] = d["ranges_sha256"]
    return out, "the released Data/synthetic/*/session.yaml"


def draw_all(ref: pd.DataFrame) -> dict:
    g0 = synthetic.synthetic_static_graph(ref, n_epochs=N_EPOCHS, sigma=0.0)
    out = {}
    for condition, draw in plan():
        g, excess = range_errors.perturbed_graph(
            g0, condition, draw, MODEL,
            bias_m=BIAS_M if condition == "biased" else 0.0)
        out[(condition, draw)] = (g.edges, excess[excess > 0])
    return out


def provenance(data: Path) -> dict:
    geo = data / "set_b" / "nodes.yaml"
    return {"generator_commit": GENERATOR_COMMIT,
            "generator_sha256": dict(GENERATOR_SHA256),
            "radiocal_version": radiocal.__version__,
            "numpy_version": np.__version__,
            "geometry": {"path": "set_b/nodes.yaml", "session": GEOMETRY,
                         "sha256": lf_sha256(geo)}}


def dump(obj, path: Path) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
                    encoding="utf-8", newline="\n")


def write(out: Path, data: Path, ref: pd.DataFrame, draws: dict) -> None:
    prov = provenance(data)
    reg = yaml.safe_load((data / "set_b" / "nodes.yaml").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    nodes = [{"id": r.id, "role": "anchor",
              "prior": {"type": "surveyed",
                        "source": "set_b/nodes.yaml, exact in the synthetic scene",
                        "xyz_local": [float(r.x), float(r.y), float(r.z)],
                        "sigma_m": 0.005}}
             for r in ref.itertuples()]
    dump({"crs": reg["crs"], "nodes": nodes}, out / "nodes.yaml")
    dump({
        "campaign": "synthetic",
        "title": "SYNTHETIC: seeded range draws on the surveyed set_b/s03 anchors",
        "synthetic": True,
        "location": "none (synthetic)",
        "hardware": "none (synthetic ranges of a seeded generator)",
        "groundtruth": "the generating geometry, exact (nodes.yaml)",
        "nodes": {"anchors": list(ref["id"]), "mobile": None, "excluded": []},
        "generator": {
            "module": "Code/methods/range_errors.py",
            "paper": ("P. Schwarzbach, R. Weber, O. Michler, IEEE Sensors J. 22(6):"
                      "4836-4843, 2022, doi:10.1109/JSEN.2021.3121627"),
            "model": {"sigma_hw_m": MODEL.sigma_hw, "mp_mean": MODEL.mp_mean,
                      "mp_sigma": MODEL.mp_sigma, "d_max_m": MODEL.d_max,
                      "p_out": MODEL.p_out, "mp_base": MODEL.mp_base,
                      "mp_slope": MODEL.mp_slope},
            "conditions": list(range_errors.CONDITIONS),
            "draws_per_condition": {c: (1 if c == "noise_free" else DRAWS)
                                    for c in range_errors.CONDITIONS},
            "seeds": f"draw k uses seed k (0 to {DRAWS - 1})",
            "bias_m": BIAS_M,
            "epochs": N_EPOCHS,
            "time_axis_hz": RATE_HZ,
            "distances": "2-D (horizontal) between the surveyed x, y of the anchors",
            "dropouts": False,
            **prov,
            "ranges_sha256_reference": HASH_REFERENCE,
        },
        "sources": [{"path": "set_b/nodes.yaml", "role": "geometry",
                     "sha256": prov["geometry"]["sha256"]}],
        "notes": [
            "SYNTHETIC DATA: no measurement. Every range is d + error of one seeded draw.",
            "Both directions of every anchor pair in every epoch; t = seq / 12.5 Hz is a "
            "synthetic time axis.",
            "nlos draws one multipath state and excess per unordered pair (session.yaml -> "
            "draw.nlos_pairs); nlos_per_range is the published per-range model.",
        ],
    }, out / "campaign.yaml")
    (out / "masks").mkdir(exist_ok=True)
    for m in _PACKAGED_MASKS.glob("*.yaml"):  # LF bytes on every platform
        (out / "masks" / m.name).write_bytes(m.read_bytes().replace(b"\r\n", b"\n"))
    for (condition, draw), (edges, nlos) in draws.items():
        sid = session_id(condition, draw)
        sdir = out / sid
        sdir.mkdir(exist_ok=True)
        edges.to_parquet(sdir / "ranges.parquet", index=False)
        block = {"condition": condition, "index": draw, "seed": draw,
                 "bias_m": BIAS_M if condition == "biased" else 0.0,
                 "ranges_sha256": ranges_sha256(edges["range_m"].to_numpy(float)),
                 "ranges_sha256_reference": HASH_REFERENCE,
                 "generator_commit": prov["generator_commit"],
                 "generator_sha256": prov["generator_sha256"],
                 "radiocal_version": prov["radiocal_version"],
                 "numpy_version": prov["numpy_version"]}
        if condition == "nlos":
            block["nlos_pairs"] = [{"node_a": a, "node_b": b, "excess_m": round(float(v), 6)}
                                   for (a, b), v in nlos.items()]
        dump({
            "session_id": sid, "campaign": "synthetic", "motion": "static",
            "synthetic": True, "draw": block,
            "t_start": float(edges["t"].min()), "t_end": float(edges["t"].max()),
            "sources": [{"path": "set_b/nodes.yaml", "role": "geometry",
                         "sha256": prov["geometry"]["sha256"]}],
            "counts": {"range_epochs": int(edges["seq"].nunique()),
                       "directed_ranges": int(len(edges)),
                       "dropped_invalid": 0, "truth_poses": 0},
            "notes": [f"SYNTHETIC DATA: draw {draw} of condition {condition} (seed {draw})."],
        }, sdir / "session.yaml")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DATA, metavar="DIR")
    ap.add_argument("--out", type=Path, default=OUT, metavar="DIR",
                    help="output folder (default: Data/synthetic at the repository root; "
                         "destructive, overwrites the released files)")
    ap.add_argument("--frozen", type=Path, default=FROZEN if FROZEN.is_file() else None,
                    metavar="CSV")
    ap.add_argument("--check", action="store_true",
                    help="draw in memory and compare, write nothing")
    args = ap.parse_args()

    check_generator()
    want, ref_label = reference_hashes(args.frozen, args.data)
    if len(want) != len(plan()):
        raise SystemExit(f"05_build_synthetic: {len(want)} reference hashes for "
                         f"{len(plan())} draws ({ref_label})")
    ref = reference(args.data)
    draws = draw_all(ref)
    bad = [k for k, (e, _) in draws.items()
           if ranges_sha256(e["range_m"].to_numpy(float)) != want.get(k)]
    print(f"synthetic: {len(draws) - len(bad)} of {len(draws)} ranges_sha256 equal "
          f"{ref_label}")
    for c, d in bad:
        print(f"  DRIFT  {session_id(c, d)}")
    if bad:
        return 1

    if args.check:
        sdir = args.data / "synthetic"
        if sdir.is_dir():
            same = 0
            for (c, d), (edges, _) in draws.items():
                g = radiocal.load_session(sdir / session_id(c, d))
                same += int(np.array_equal(g.edges["range_m"].to_numpy(float),
                                           edges["range_m"].to_numpy(float)))
            print(f"synthetic: {same} of {len(draws)} files under {sdir.name}/ equal the draws")
            return 0 if same == len(draws) else 1
        return 0

    write(args.out, args.data, ref, draws)
    back = sum(ranges_sha256(radiocal.load_session(args.out / session_id(c, d))
                             .edges["range_m"].to_numpy(float)) == want[(c, d)]
               for c, d in draws)
    print(f"synthetic: {len(draws)} sessions written to {args.out}, {back} of "
          f"{len(draws)} hashes equal after reloading")
    return 0 if back == len(draws) else 1


if __name__ == "__main__":
    sys.exit(main())
