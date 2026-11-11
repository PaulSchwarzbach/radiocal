"""Gaussian (Cramer-Rao) bounds per benchmark view -> results/crlb.parquet (+ CSV).

One row per (campaign, session, mask) view of the static anchor-only
families (fully_meshed and multi_hop, including the parameterized thinning,
random-ablation, and prior-sweep masks). The bound is evaluated at the
surveyed geometry with the session's doctrine sigma - the fully_meshed-view
kernels.session_sigma, identical to the benchmark driver's value, asserted
against benchmark.parquet wherever the session has rows. Mobile-bearing
families are out of scope (the anchor-only FIM is not a valid bound once
anchor-mobile factors carry information; paper Sec. IV-E).

The bound consumes the reference information the view's method roster
consumes: the canonical and thinning masks run blind estimators (the
subset some of them expose is ignored there), so their bound is blind;
only the prior-sweep masks enter seeded. n_priors records what the bound
consumed.

status per row: "ok" when the FIM nullity equals the gauge nullity expected
for the consumed prior count (crlb.expected_nullity), "flexible" when it
exceeds it - the view has continuous flexes (or is disconnected), and the
aligned bound conditions on those modes as well.

Default cell list: every static-family view present in benchmark.parquet,
plus the prior sweep on the two static reference sessions.

    python run_crlb.py
    python run_crlb.py --sessions "set_b/s03" --mask "fully_meshed,multi_hop_k3"
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from radiocal import load_session

import benchmark_io
import crlb
import evaluation
import family_masks
import kernels
from common import default_anchor_order
from run_benchmark import DATA, GATE_SESSIONS

STATIC_FAMILIES = ("fully_meshed", "multi_hop")


def default_cells() -> list[tuple[str, str, str]]:
    """Static-family views of benchmark.parquet + the prior sweep."""
    df = benchmark_io.load()
    cells = set()
    if len(df):
        views = df[["campaign", "session", "mask"]].drop_duplicates()
        for campaign, session, mask in views.itertuples(index=False):
            if family_masks.family_of(mask) in STATIC_FAMILIES:
                cells.add((campaign, session, mask))
    for s in GATE_SESSIONS:
        campaign, session = s.split("/")
        for mask in family_masks.expand_masks("prior_sweep"):
            cells.add((campaign, session, mask))
    return sorted(cells)


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

    bench = benchmark_io.load()
    rows = []
    run_at = datetime.now().isoformat(timespec="seconds")
    for (campaign, session), group in pd.DataFrame(
            cells, columns=["campaign", "session", "mask"]).groupby(
            ["campaign", "session"], sort=True):
        sdir = DATA / campaign / session
        g0 = load_session(sdir)
        g_full = g0.apply_mask("fully_meshed")
        sigma_m = float(kernels.session_sigma(g_full,
                                              default_anchor_order(g_full)))
        sel = bench[(bench.campaign == campaign) & (bench.session == session)]
        if len(sel):
            assert np.isclose(sigma_m, float(sel.sigma_m.iloc[0]),
                              rtol=1e-9), \
                f"sigma mismatch vs benchmark table on {campaign}/{session}"
        ref = evaluation.surveyed_reference(sdir)
        for mask in group["mask"]:
            g = g0.apply_mask(family_masks.resolve_mask(
                mask, mask_dirs=g0.meta.get("_mask_dirs")))
            # the bound consumes what the view's roster consumes: the
            # canonical and thinning masks run blind estimators (their
            # exposed subset is ignored), only the prior sweep is seeded
            use_priors = family_masks.prior_count(mask) is not None
            vb = crlb.view_bound(g, ref, sigma_m, use_priors=use_priors)
            expected = crlb.expected_nullity(vb["n_priors"])
            status = "ok" if vb["fim_nullity"] == expected else "flexible"
            rows.append({"campaign": campaign, "session": session,
                         "mask": mask, **vb, "status": status,
                         "run_at": run_at})
            print(f"{campaign}/{session}  {mask:<22} "
                  f"aligned {vb['crlb_rmse_m']:.3f} m  "
                  f"absolute {vb['crlb_abs_rmse_m']:.3f} m  "
                  f"nullity {vb['fim_nullity']}  {status}")

    benchmark_io.upsert(pd.DataFrame(rows), "crlb")
    print(f"\nwrote {len(rows)} views to results/crlb.parquet (+ CSV)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
