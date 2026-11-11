"""One session, four topologies: the controlled variable of the result tables.

Loads set_b/s01 (superset graph) and applies each shipped mask.
Every view filters the SAME recording; nothing is re-measured. Expected
output: one table row per view showing how edges, priors, and exposed
ground truth differ. Runtime: a few seconds.
"""
from pathlib import Path

import radiocal

REPO = Path(__file__).resolve().parents[1]
MASKS = ["fully_meshed", "slam_bipartite", "mobile_assisted", "multi_hop"]

g = radiocal.load_session(REPO / "Data/set_b/s01")
print(f"superset: {len(g.nodes)} nodes, {len(g.edges)} directed ranges, "
      f"{len(g.truth)} truth poses\n")

header = f"{'view':<18}{'edges':>9}{'epochs':>8}{'priors':>7}{'truth':>7}"
print(header)
print("-" * len(header))
for name in MASKS:
    v = g.apply_mask(name)
    print(f"{name:<18}{len(v.edges):>9}{v.edges['seq'].nunique():>8}"
          f"{len(v.priors()):>7}{len(v.truth):>7}")

print("\npriors = nodes keeping their surveyed position; truth = ground-truth"
      "\nposes exposed to the method (evaluation always uses the full survey).")
