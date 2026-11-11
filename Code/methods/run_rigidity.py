"""Rigidity certificates per benchmark view -> results/rigidity.parquet (+ CSV).

One row per (campaign, session, mask) view of the anchor-only families
(fully_meshed and multi_hop, including the parameterized thinning and the
random-ablation masks). The certificate (rigidity.py: components, generic
and redundant rigidity, 3-connectivity, Jackson-Jordan global rigidity) is
evaluated on the pair graph the estimators actually see: the view's
MEASURED anchor pairs, i.e. the finite entries of common.pair_range_matrix
after family_masks.resolve_mask -> apply_mask. On real data the kNN thinning
selects by measured median range, so NLOS bias can reorder neighbours and
the class can differ from the true-distance thinning that
validate_multihop.py certifies on synthetic data.

The certificate is prior-independent, so the prior-sweep masks are skipped
(they share the pair graph of their base mask).

Default cell list: every anchor-only view present in benchmark.parquet,
plus the kNN thinning sweep on EVERY session (the benchmark runs the sweep
on the two static reference sessions only; the certificate is cheap and the
per-site threshold should hold for every session of a site).

    python run_rigidity.py
    python run_rigidity.py --sessions "set_b/s03" --mask multi_hop_sweep
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from radiocal import load_session

import benchmark_io
import evaluation
import family_masks
import rigidity
from common import estimation_order, pair_counts, pair_range_matrix
from run_benchmark import DATA, discover_sessions

STATIC_FAMILIES = ("fully_meshed", "multi_hop")


def default_cells() -> list[tuple[str, str, str]]:
    """Anchor-only views of benchmark.parquet + the kNN sweep everywhere."""
    df = benchmark_io.load()
    cells = set()
    if len(df):
        views = df[["campaign", "session", "mask"]].drop_duplicates()
        for campaign, session, mask in views.itertuples(index=False):
            if (family_masks.family_of(mask) in STATIC_FAMILIES
                    and family_masks.prior_count(mask) is None):
                cells.add((campaign, session, mask))
    sweep = ["fully_meshed", "multi_hop"] + family_masks.expand_masks(
        "multi_hop_sweep")
    for s in discover_sessions():
        campaign, session = s.split("/")
        for mask in sweep:
            cells.add((campaign, session, mask))
    return sorted(cells)


def view_certificate(g, ref: pd.DataFrame) -> dict:
    """Certificate of a masked view's measured anchor pair graph, evaluated
    at the surveyed coordinates."""
    order = estimation_order(g)
    M = pair_range_matrix(g, order)
    iu, ju = np.triu_indices(len(order), k=1)
    obs = np.isfinite(M[iu, ju])
    pairs = list(zip(iu[obs], ju[obs]))
    coords = ref.set_index("id").loc[order, ["x", "y"]].to_numpy(float)
    cert = rigidity.global_rigidity_certificate(coords, pairs)
    available, total = pair_counts(M)
    return {"n_anchors": cert["n"], "n_pairs": available,
            "n_pairs_total": total, "n_components": cert["n_components"],
            "generically_rigid": bool(cert["generically_rigid"]),
            "redundantly_rigid": bool(cert["redundantly_rigid"]),
            "three_connected": bool(cert["three_connected"]),
            "globally_rigid": bool(cert["globally_rigid"]),
            "cert_class": rigidity.certificate_class(cert)}


def mask_axes(mask: str) -> dict:
    """Thinning rule, k and thinning seed encoded in a mask name."""
    if mask == "fully_meshed":
        return {"thinning": "none", "k": -1, "thin_seed": -1}
    r = family_masks._RAND_RE.match(mask)
    if r:
        return {"thinning": "random", "k": int(r.group(1)),
                "thin_seed": int(r.group(2))}
    return {"thinning": "knn", "k": family_masks.thinning_k(mask),
            "thin_seed": -1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", default="all",
                    help="comma-separated campaign/session globs")
    ap.add_argument("--mask", default="",
                    help="optional comma-separated mask filter (expanded "
                         "aliases allowed); default: every default cell")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = default_cells()
    if args.sessions != "all":
        pats = [p for p in args.sessions.split(",") if p]
        cells = [c for c in cells
                 if any(fnmatch.fnmatch(f"{c[0]}/{c[1]}", p) for p in pats)]
    if args.mask:
        keep = set(family_masks.expand_masks(args.mask))
        cells = [c for c in cells if c[2] in keep]
    if args.dry_run:
        for c in cells:
            print("run  " + "  ".join(c))
        print(f"{len(cells)} views")
        return 0

    rows = []
    run_at = datetime.now().isoformat(timespec="seconds")
    for (campaign, session), group in pd.DataFrame(
            cells, columns=["campaign", "session", "mask"]).groupby(
            ["campaign", "session"], sort=True):
        sdir = DATA / campaign / session
        g0 = load_session(sdir)
        ref = evaluation.surveyed_reference(sdir)
        for mask in group["mask"]:
            g = g0.apply_mask(family_masks.resolve_mask(
                mask, mask_dirs=g0.meta.get("_mask_dirs")))
            vc = view_certificate(g, ref)
            rows.append({"campaign": campaign, "session": session,
                         "mask": mask, **mask_axes(mask), **vc,
                         "run_at": run_at})
            print(f"{campaign}/{session}  {mask:<24} "
                  f"{vc['n_pairs']:>2}/{vc['n_pairs_total']} pairs  "
                  f"{vc['cert_class']}")

    benchmark_io.upsert(pd.DataFrame(rows), "rigidity")
    print(f"\nwrote {len(rows)} views to results/rigidity.parquet (+ CSV)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
