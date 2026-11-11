"""Load one session and compare measured ranges to the survey.

Loads set_b/s03, aggregates the directed ranging stream into
per-pair medians, and plots them against the Total-Station distances of the
surveyed anchors. Expected output: '45 anchor pairs, median |residual| ~0.3 m'
and 01_ranges_vs_survey.png in the current directory. Runtime: under a minute.
"""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import radiocal

REPO = Path(__file__).resolve().parents[1]

g = radiocal.load_session(REPO / "Data/set_b/s03")

# per-pair median of the undirected ranges
sym = g.symmetrized()
med = sym.groupby(["node_a", "node_b"])["range_m"].median()

# surveyed inter-anchor distances from the node priors
anchors = g.priors().set_index("id")
pairs, surveyed, measured = [], [], []
for a, b in combinations(sorted(anchors.index), 2):
    if (a, b) in med.index:
        d = np.linalg.norm(anchors.loc[a, ["x", "y", "z"]].astype(float)
                           - anchors.loc[b, ["x", "y", "z"]].astype(float))
        pairs.append((a, b)), surveyed.append(d), measured.append(med[a, b])

resid = np.array(measured) - np.array(surveyed)
print(f"{len(pairs)} anchor pairs, median |residual| "
      f"{np.median(np.abs(resid)):.2f} m (UWB TWR bias is visible here)")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(surveyed, measured, s=18)
lim = [0, max(surveyed) * 1.1]
ax.plot(lim, lim, lw=1, color="grey")
ax.set(xlabel="surveyed distance [m]", ylabel="median measured range [m]",
       title="set_b/s03: ranging vs survey")
fig.tight_layout()
out = Path.cwd() / "01_ranges_vs_survey.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
