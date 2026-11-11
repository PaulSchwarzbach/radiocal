"""One-off maintenance: freeze the factorial matrix as the regression target.

Parses the stats tables (reporting.markdown_table format) of the six
results/<method>/summary.md files and writes results/regression_frozen.csv
with one row per benchmark cell: campaign, session, mask, method, variant,
rmse_m, max_error_m, reflection, source_file. The grid filter's
"mixture(gaussian)" label maps to the variant name "mixture" of the
registry. run_benchmark.py --check-regression consumes the CSV.

The released regression_frozen.csv was written this way before the summary
tables became views of results/benchmark.parquet. A rerun now overwrites the
reference with the current outputs, and the gate would then compare the
outputs with themselves. Do not run it to reproduce results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
METHODS = ("closed_form", "mds", "factor_graph", "ekf",
           "particle_filter", "grid_filter")
VARIANT_ALIASES = {"mixture(gaussian)": "mixture"}


def parse_summary_table(path: Path) -> pd.DataFrame:
    """First markdown table of a summary.md -> DataFrame of strings."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows, header = [], None
    for line in lines:
        if not line.startswith("|"):
            if header is not None:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
        elif not all(set(c) <= {"-"} for c in cells):  # skip the rule row
            rows.append(cells)
    if header is None:
        raise ValueError(f"no markdown table in {path}")
    return pd.DataFrame(rows, columns=header)


def main() -> None:
    frames = []
    for method in METHODS:
        src = RESULTS / method / "summary.md"
        t = parse_summary_table(src)
        frames.append(pd.DataFrame({
            "campaign": t["campaign"],
            "session": t["session"],
            "mask": "fully_meshed",
            "method": method,
            "variant": t["variant"].map(lambda v: VARIANT_ALIASES.get(v, v)),
            "rmse_m": t["rmse_m"].astype(float),
            "max_error_m": t["max_error_m"].astype(float),
            "reflection": t["reflection"].map({"True": True, "False": False}),
            "source_file": f"results/{method}/summary.md",
        }))
    frozen = pd.concat(frames, ignore_index=True)
    out = RESULTS / "regression_frozen.csv"
    frozen.to_csv(out, index=False, float_format="%.3f")
    print(f"wrote {out} ({len(frozen)} rows)")
    print(frozen.to_string(index=False))


if __name__ == "__main__":
    main()
