"""Draw new synthetic scenes with the released range-error generator.

SYNTHETIC DATA. Every range is a generating distance plus the error of one
seeded draw of Code/methods/range_errors.py (P. Schwarzbach, R. Weber,
O. Michler, IEEE Sensors J. 22(6):4836-4843, 2022, with the per-pair
extension of its docstring). The distances come from the surveyed geometry of
a released session, through the scenes of Code/methods/synthetic.py:

  static   the anchors of the session, both directions of every anchor pair
           in every epoch. The 81 frozen draws under Data/synthetic are this
           scene on set_b/s03 (05_build_synthetic.py writes them).
  marker   the anchors of a quasi-static session of set_a and its tag, which
           stands still at the verified truth point of the session.
  walk     the anchors of a session and its mobile node, which moves on a
           Lissajous or line path within the bounding box of the anchors.
           The geometry session must hold a mobile node (set_b/s01 to s03,
           set_a/s01 to s20).

    python -B Code/datasets/radiocal/scripts/07_draw_synthetic.py --scene walk --condition noise_free --out draws/walk_set_b
    python -B Code/datasets/radiocal/scripts/07_draw_synthetic.py --scene marker --geometry set_a/s07 --seeds 0-4 --out draws/marker_p07
    python -B Code/datasets/radiocal/scripts/07_draw_synthetic.py --reproduce

The condition noise_free gives ranges equal to their generating distances. On a
view whose ranges fix the geometry, such as the full anchor mesh, a method whose
range model matches them (2-D distances unless --ranges-3d) recovers the
geometry up to the gauge, so the error it keeps is its own: its error floor. Under nlos a walk keeps one excess per tag-anchor pair for the
whole walk, drawn at the distance of the first epoch. Without --dropouts a
noise-free draw uses no random numbers, so one draw is written per scene.

--out       the folder of the new campaign (required). It must lie outside
            the repository's Data/ folder and outside --data, and must not
            hold a campaign yet. The script never writes into the released
            files.
--data      folder with the campaign folders (default: Data/ at the
            repository root), read only
--geometry  the session whose surveyed anchors (and mobile node) give the
            geometry, as <set>/<session> (default set_b/s03)
--scene     static (default), marker or walk
--path      the path of a walk: lissajous (default) or line
--condition one or more of the five conditions (default: all five)
--seeds     seeds, as 0-19, 0,3,7 or 5 (default 0). Draw k uses seed k.
--epochs    epochs of a static or marker scene (default 200, as the frozen
            draws)
--duration  seconds of a walk at the generator's epoch rate (default: the
            generator's)
--bias      bias of the biased condition, metres (default 0.27, as the
            frozen draws)
--tag-z     height of the walking tag in the geometry's frame, metres
            (default: the median height of the anchors). Only --ranges-3d
            uses it.
--ranges-3d 3-D slant ranges instead of the default 2-D (horizontal) ones
--dropouts  remove the ranges that eq. (4) of the model drops
--sigma-hw, --mp-mean, --mp-sigma, --d-max, --p-out, --mp-base, --mp-slope
            the model values. Their defaults are the generator's: the 2022
            paper, with p_out from the author's earlier framework.
--reproduce draw the frozen setup in memory through the static scene of this
            script and compare its 81 hashes with frozen.csv of the tutorial
            (--frozen, when present) or with the stored hashes of
            Data/synthetic/*/session.yaml. Writes nothing. It needs the
            generator modules of the release unchanged.

Each campaign gets nodes.yaml (the anchors with their exact positions, the
tag without a prior), the four packaged masks, campaign.yaml with the
options, the command, the model values, the hashes of the generator modules
and of every geometry input, and one session per draw,
syn_<scene>_<condition>_<NN>, with ranges.parquet, session.yaml and, for a
scene with a tag, groundtruth.parquet. The truth lies in the frame of
nodes.yaml (session.yaml -> truth_frame: local). Its frame and rotation
columns are NaN, as the data dictionary defines them for truth without a
motion-capture frame. A changed generator module draws with a warning, and
the campaign records the hashes of the module that drew it.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import inspect
import os
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
from radiocal.graph import TRUTH_COLUMNS  # noqa: E402
from radiocal.masks import _PACKAGED_MASKS  # noqa: E402


def _writer_05():
    """05_build_synthetic.py as a module. Only its constants and pure helpers
    are used here: the frozen setup, the pinned generator hashes and commit,
    the two hash functions, the reference hashes and the YAML dump."""
    spec = importlib.util.spec_from_file_location(
        "build_synthetic_05", HERE.parent / "05_build_synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # main() is guarded, so nothing runs
    return mod


W05 = _writer_05()
DATA = REPO / "Data"
SCENES = ("static", "marker", "walk")
PATHS = ("lissajous", "line")
MOTION = {"static": "static", "marker": "quasi_static_point", "walk": "dynamic"}
DURATION_S = inspect.signature(synthetic.synthetic_trajectory).parameters["duration_s"].default
MODEL_FIELDS = [f.name for f in dataclasses.fields(range_errors.RangeErrorModel)]
# the keys of campaign.yaml -> generator.model, as 05_build_synthetic.py writes them
MODEL_KEYS = {"sigma_hw": "sigma_hw_m", "d_max": "d_max_m"}
MODULES = (("range_errors.py", range_errors), ("synthetic.py", synthetic))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_seeds(text: str) -> list[int]:
    """Seeds as 0-19, 0,3,7 or 5: distinct non-negative integers."""
    seeds: list[int] = []
    try:
        for part in text.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = (int(v) for v in part.split("-", 1))
                if hi < lo:
                    raise argparse.ArgumentTypeError(f"seed range {part!r} runs backwards")
                seeds += range(lo, hi + 1)
            elif part:
                seeds.append(int(part))
    except ValueError:
        raise argparse.ArgumentTypeError(f"seeds {text!r}: give integers as 0-19, 0,3,7 or 5")
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"seeds {text!r}: give distinct non-negative integers")
    return seeds


def seeds_text(seeds: list[int]) -> str:
    """The seeds as runs, for example 0-4,7, so that the recorded command reruns."""
    runs, start = [], seeds[0]
    for prev, cur in zip(seeds, seeds[1:] + [None]):
        if cur != prev + 1:
            runs.append(str(start) if start == prev else f"{start}-{prev}")
            start = cur
    return ",".join(runs)


def inside(path: Path, guard: Path) -> bool:
    """Whether path is guard or lies below it, compared as files, so that case
    and links on the file system do not matter."""
    if not guard.exists():
        return False
    for p in (path, *path.parents):
        if p.exists() and os.path.samefile(p, guard):
            return True
    return False


def generator_provenance() -> dict:
    """The hashes of both generator modules, and the pinned commit only when
    both equal the released ones."""
    got = {name: W05.lf_sha256(Path(mod.__file__)) for name, mod in MODULES}
    pinned = got == W05.GENERATOR_SHA256
    return {"generator_commit": W05.GENERATOR_COMMIT if pinned else "unknown (changed module)",
            "generator_sha256": got, "generator_pinned": pinned}


def geometry(data: Path, session: str) -> tuple:
    """The geometry session: its graph, the surveyed anchors (id, x, y, z) in
    the order of the frozen draws (05_build_synthetic.reference), and its
    mobile node, if it holds one."""
    g = radiocal.load_session(data / session)
    pri = g.priors()
    ref = pri[pri["role"] == "anchor"][["id", "x", "y", "z"]].reset_index(drop=True)
    mobile = list(g.nodes.loc[g.nodes["role"] == "mobile", "id"])
    if len(mobile) > 1:
        raise SystemExit(f"07_draw_synthetic: {session} holds {len(mobile)} mobile nodes")
    return g, ref, (mobile[0] if mobile else None)


def tag_point(g, tag: str | None, session: str) -> tuple[float, float, float]:
    """The verified truth point of the tag of a quasi-static session."""
    if (g.meta.get("motion") != "quasi_static_point" or g.meta.get("truth_quality") != "verified"
            or tag is None):
        raise SystemExit(f"07_draw_synthetic: a marker scene needs a quasi-static session with "
                         f"verified truth (set_a/s01 to s20), and {session} is none")
    tr = g.truth[g.truth["node_id"] == tag]
    if len(tr) != 1:
        raise SystemExit(f"07_draw_synthetic: {session} holds {len(tr)} truth rows of {tag}")
    return tuple(float(v) for v in tr[["x", "y", "z"]].iloc[0])


def scene_graph(scene: str, ref: pd.DataFrame, tag: str | None, *, point=None,
                epochs: int, duration_s: float, path: str, tag_z: float, ranges_3d: bool):
    """The noise-free superset graph of a scene, with exact distances."""
    if scene == "static":
        return synthetic.synthetic_static_graph(ref, n_epochs=epochs, sigma=0.0,
                                                ranges_3d=ranges_3d)
    if scene == "marker":
        return synthetic.synthetic_static_graph(ref, n_epochs=epochs, sigma=0.0,
                                                mobile=(tag, point), ranges_3d=ranges_3d)
    traj = synthetic.synthetic_trajectory(ref, duration_s=duration_s, kind=path)
    return synthetic.synthetic_slam_graph(ref, traj, mobile_id=tag, sigma=0.0, truth_z=tag_z,
                                          ranges_3d=ranges_3d)


def plan(conditions: list[str], seeds: list[int], dropouts: bool) -> list[tuple[str, int]]:
    """(condition, seed) of every draw. A noise-free draw without dropouts uses
    no random numbers, so it is drawn once, with the first seed."""
    return [(c, s) for c in conditions
            for s in (seeds[:1] if c == "noise_free" and not dropouts else seeds)]


def draw(g0, condition: str, seed: int, model, bias_m: float, dropouts: bool):
    return range_errors.perturbed_graph(g0, condition, seed, model,
                                        bias_m=bias_m if condition == "biased" else 0.0,
                                        dropouts=dropouts)


def reproduce(data: Path, frozen: Path | None) -> int:
    """The frozen setup of 05_build_synthetic.py drawn through scene_graph()
    and draw() of this script, compared with the reference hashes."""
    prov = generator_provenance()
    if not prov["generator_pinned"]:
        raise SystemExit("07_draw_synthetic --reproduce: the generator modules differ from the "
                         f"released ones (commit {W05.GENERATOR_COMMIT[:7]})")
    want, label = W05.reference_hashes(frozen, data)
    frozen_plan = W05.plan()
    if len(want) != len(frozen_plan):
        raise SystemExit(f"07_draw_synthetic: {len(want)} reference hashes for "
                         f"{len(frozen_plan)} frozen draws ({label})")
    _, ref, _ = geometry(data, W05.GEOMETRY)
    g0 = scene_graph("static", ref, None, epochs=W05.N_EPOCHS, duration_s=DURATION_S,
                     path=PATHS[0], tag_z=0.0, ranges_3d=False)
    bad = []
    for condition, index in frozen_plan:
        g, _ = draw(g0, condition, index, W05.MODEL, W05.BIAS_M, dropouts=False)
        if W05.ranges_sha256(g.edges["range_m"].to_numpy(float)) != want.get((condition, index)):
            bad.append((condition, index))
    print(f"07_draw_synthetic: {len(frozen_plan) - len(bad)} of {len(frozen_plan)} frozen draws "
          f"reproduced by the static scene ({label})")
    for condition, index in bad:
        print(f"  DRIFT  {W05.session_id(condition, index)}")
    return 1 if bad else 0


def truth_frame(truth: pd.DataFrame) -> pd.DataFrame:
    """The generator's truth in the columns of the data dictionary: the frame
    counter and the rotation do not apply, so they are NaN (float64)."""
    tr = truth.copy()
    for col in ("t_broker", "frame", "rot_x", "rot_y", "rot_z"):
        tr[col] = np.nan
    tr["node_id"] = tr["node_id"].astype(str)
    for col in ("t", "x", "y", "z"):
        tr[col] = tr[col].astype(float)
    return tr[TRUTH_COLUMNS].reset_index(drop=True)


def command(args, conditions: list[str], seeds: list[int]) -> str:
    """The command that redraws the campaign, without --out and --data."""
    parts = ["python -B Code/datasets/radiocal/scripts/07_draw_synthetic.py",
             f"--geometry {args.geometry}", f"--scene {args.scene}"]
    if args.scene == "walk":
        parts += [f"--path {args.path}", f"--duration {args.duration:g}"]
    else:
        parts.append(f"--epochs {args.epochs}")
    parts += ["--condition " + " ".join(conditions), f"--seeds {seeds_text(seeds)}",
              f"--bias {args.bias:g}"]
    if args.scene == "walk" and args.tag_z is not None:
        parts.append(f"--tag-z {args.tag_z:g}")
    if args.ranges_3d:
        parts.append("--ranges-3d")
    if args.dropouts:
        parts.append("--dropouts")
    for name in MODEL_FIELDS:
        value = getattr(args, name)
        if value != getattr(range_errors.PAPER, name):
            parts.append(f"--{name.replace('_', '-')} {value:g}")
    return " ".join(parts + ["--out DIR"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, metavar="DIR")
    ap.add_argument("--data", type=Path, default=DATA, metavar="DIR")
    ap.add_argument("--geometry", default=W05.GEOMETRY, metavar="SET/SESSION")
    ap.add_argument("--scene", choices=SCENES, default=SCENES[0])
    ap.add_argument("--path", choices=PATHS, default=PATHS[0])
    ap.add_argument("--condition", nargs="+", choices=range_errors.CONDITIONS,
                    default=list(range_errors.CONDITIONS))
    ap.add_argument("--seeds", default="0", metavar="SEEDS")
    ap.add_argument("--epochs", type=int, default=W05.N_EPOCHS)
    ap.add_argument("--duration", type=float, default=DURATION_S, metavar="S")
    ap.add_argument("--bias", type=float, default=W05.BIAS_M, metavar="M")
    ap.add_argument("--tag-z", type=float, default=None, metavar="M")
    ap.add_argument("--ranges-3d", action="store_true")
    ap.add_argument("--dropouts", action="store_true")
    for name in MODEL_FIELDS:
        ap.add_argument(f"--{name.replace('_', '-')}", dest=name, type=float,
                        default=getattr(range_errors.PAPER, name), metavar="V")
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--frozen", type=Path,
                    default=W05.FROZEN if W05.FROZEN.is_file() else None, metavar="CSV")
    args = ap.parse_args()

    if args.reproduce:
        return reproduce(args.data, args.frozen)
    if args.out is None:
        ap.error("--out is required: the folder of the new campaign, outside Data/")
    out = args.out.resolve()
    for guard in {DATA.resolve(), args.data.resolve()}:
        if out == guard or out.is_relative_to(guard) or inside(out, guard):
            ap.error(f"--out {args.out} lies inside {guard}, which holds released data")
    if (out / "campaign.yaml").exists():
        ap.error(f"--out {args.out} holds a campaign already; choose an empty folder")
    try:
        seeds = parse_seeds(args.seeds)
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))
    conditions = list(dict.fromkeys(args.condition))
    set_id, _, session = args.geometry.partition("/")
    if not session or not (args.data / args.geometry / "session.yaml").is_file():
        ap.error(f"--geometry {args.geometry}: no released session <set>/<session> there")

    prov = generator_provenance()
    if not prov["generator_pinned"]:
        print("07_draw_synthetic: warning: the generator modules differ from the released ones "
              f"(commit {W05.GENERATOR_COMMIT[:7]}); the campaign records their own hashes",
              file=sys.stderr)
    model = range_errors.RangeErrorModel(**{n: getattr(args, n) for n in MODEL_FIELDS})

    g, ref, tag = geometry(args.data, args.geometry)
    point = None
    if args.scene == "marker":
        point = tag_point(g, tag, args.geometry)
    elif args.scene == "walk" and tag is None:
        ap.error(f"a walk needs a session with a mobile node, and {args.geometry} holds none")
    if args.scene == "static":
        tag = None
    tag_z = (float(np.median(ref["z"])) if args.tag_z is None else args.tag_z)
    g0 = scene_graph(args.scene, ref, tag, point=point, epochs=args.epochs,
                     duration_s=args.duration, path=args.path, tag_z=tag_z,
                     ranges_3d=args.ranges_3d)

    campaign = f"synthetic_{args.scene}_{set_id}"
    reg = yaml.safe_load((args.data / set_id / "nodes.yaml").read_text(encoding="utf-8"))
    sources = [{"path": f"{set_id}/nodes.yaml", "role": "geometry",
                "sha256": W05.lf_sha256(args.data / set_id / "nodes.yaml")},
               {"path": f"{args.geometry}/session.yaml", "role": "node roles",
                "sha256": W05.lf_sha256(args.data / args.geometry / "session.yaml")}]
    if args.scene == "marker":
        sources.append({"path": f"{args.geometry}/groundtruth.parquet", "role": "tag point",
                        "sha256": sha256_file(args.data / args.geometry / "groundtruth.parquet")})
    versions = {"radiocal_version": radiocal.__version__, "numpy_version": np.__version__}
    draws = plan(conditions, seeds, args.dropouts)

    out.mkdir(parents=True, exist_ok=True)
    nodes = []
    for r in g0.nodes.itertuples():
        if r.role == "anchor":
            nodes.append({"id": r.id, "role": "anchor",
                          "prior": {"type": "surveyed",
                                    "source": f"{set_id}/nodes.yaml, exact in the synthetic scene",
                                    "xyz_local": [float(r.x), float(r.y), float(r.z)],
                                    "sigma_m": float(r.sigma_m)}})
        else:
            nodes.append({"id": r.id, "role": "mobile", "prior": {"type": "none"}})
    W05.dump({"crs": reg["crs"], "nodes": nodes}, out / "nodes.yaml")
    distances = ("3-D (slant) between the surveyed x, y, z" if args.ranges_3d
                 else "2-D (horizontal) between the surveyed x, y")
    scene_block = {"scene": args.scene, "geometry": args.geometry, "tag": tag}
    if args.scene == "walk":
        scene_block.update({"path": args.path, "duration_s": args.duration,
                            "tag_z_m": tag_z, "path_area": "within the bounding box of the anchors"})
    elif args.scene == "marker":
        scene_block["tag_point"] = [float(v) for v in point]
    if args.scene != "walk":
        scene_block["epochs"] = args.epochs
    W05.dump({
        "campaign": campaign,
        "title": f"SYNTHETIC: seeded {args.scene} scene on the surveyed geometry of "
                 f"{args.geometry}",
        "synthetic": True,
        "location": "none (synthetic)",
        "hardware": "none (synthetic ranges of a seeded generator)",
        "groundtruth": "the generating geometry, exact (nodes.yaml, and groundtruth.parquet "
                       "for a tag)",
        "nodes": {"anchors": list(ref["id"]), "mobile": tag, "excluded": []},
        "generator": {
            "module": "Code/methods/range_errors.py",
            "scene_module": "Code/methods/synthetic.py",
            "script": "Code/datasets/radiocal/scripts/07_draw_synthetic.py",
            "paper": ("P. Schwarzbach, R. Weber, O. Michler, IEEE Sensors J. 22(6):"
                      "4836-4843, 2022, doi:10.1109/JSEN.2021.3121627"),
            "model": {MODEL_KEYS.get(n, n): float(getattr(model, n)) for n in MODEL_FIELDS},
            **scene_block,
            "conditions": conditions,
            "seeds": seeds,
            "bias_m": args.bias,
            "time_axis_hz": synthetic.RATE_HZ,
            "distances": distances,
            "dropouts": args.dropouts,
            **prov, **versions,
            "command": command(args, conditions, seeds),
        },
        "sources": sources,
        "notes": [
            "SYNTHETIC DATA: no measurement. Every range is d + error of one seeded draw.",
            f"t = seq / {synthetic.RATE_HZ:g} Hz is a synthetic time axis.",
            "The truth lies in the frame of nodes.yaml (session.yaml -> truth_frame: local); "
            "its frame and rotation columns are NaN.",
        ],
    }, out / "campaign.yaml")
    (out / "masks").mkdir(exist_ok=True)
    for m in sorted(_PACKAGED_MASKS.glob("*.yaml")):  # LF bytes on every platform
        (out / "masks" / m.name).write_bytes(m.read_bytes().replace(b"\r\n", b"\n"))

    stored = {}
    for condition, seed in draws:
        gd, excess = draw(g0, condition, seed, model, args.bias, args.dropouts)
        sid = f"syn_{args.scene}_{condition}_{seed:02d}"
        sdir = out / sid
        sdir.mkdir(exist_ok=True)
        edges = gd.edges
        edges.to_parquet(sdir / "ranges.parquet", index=False)
        truth = truth_frame(gd.truth) if tag is not None else None
        if truth is not None:
            truth.to_parquet(sdir / "groundtruth.parquet", index=False)
        stored[sid] = W05.ranges_sha256(edges["range_m"].to_numpy(float))
        block = {"condition": condition, "index": seed, "seed": seed,
                 "bias_m": args.bias if condition == "biased" else 0.0,
                 "dropouts": args.dropouts,
                 "dropouts_removed": int(len(g0.edges) - len(edges)),
                 "ranges_sha256": stored[sid],
                 "generator_commit": prov["generator_commit"],
                 "generator_sha256": prov["generator_sha256"], **versions}
        if condition == "nlos":  # the pairs with multipath, as 05_build_synthetic.py stores them
            block["nlos_pairs"] = [{"node_a": a, "node_b": b, "excess_m": round(float(v), 6)}
                                   for (a, b), v in excess[excess > 0].items()]
        session_yaml = {
            "session_id": sid, "campaign": campaign, "motion": MOTION[args.scene],
            "synthetic": True, "draw": block,
            "t_start": float(edges["t"].min()), "t_end": float(edges["t"].max()),
            "sources": sources,
            "counts": {"range_epochs": int(edges["seq"].nunique()),
                       "directed_ranges": int(len(edges)), "dropped_invalid": 0,
                       "truth_poses": 0 if truth is None else int(len(truth))},
        }
        if truth is not None:
            session_yaml["truth_frame"] = "local"
        if args.scene == "marker":  # exact truth, marked as synthetic.py marks it
            session_yaml["truth_quality"] = "verified"
        session_yaml["notes"] = [f"SYNTHETIC DATA: draw {seed} of condition {condition} "
                                 f"(seed {seed}), {args.scene} scene on {args.geometry}."]
        W05.dump(session_yaml, sdir / "session.yaml")

    back = sum(W05.ranges_sha256(radiocal.load_session(out / sid).edges["range_m"]
                                 .to_numpy(float)) == h for sid, h in stored.items())
    print(f"07_draw_synthetic: {len(stored)} session{'s' * (len(stored) != 1)} of {campaign} "
          f"written to {args.out}, {back} of {len(stored)} hashes equal after reloading")
    return 0 if back == len(stored) else 1


if __name__ == "__main__":
    sys.exit(main())
