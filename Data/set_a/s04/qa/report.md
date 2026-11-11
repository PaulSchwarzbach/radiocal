# QA report - s04 (quasi_static_point)

- epochs: 170, directed ranges: 16092 (19 invalid dropped at conversion)
- no timestamps in this campaign's logs; epoch ordering via seq
- ground-truth quality: **verified** (screen: multilateration deviation 0.436 m, range test: median line-of-sight residual +0.217 m, reference +0.158 to +0.251 m)

## Per-pair availability
- directed pairs observed: 122 of 132 possible
- weakest 5 directed pairs (availability % of epochs): 4503->D5B4 0%, 5388->4AA4 0%, 4AA4->8AA5 0%, 4AA4->5388 0%, 5388->9911 0%

## Mobile node (8B05)
- mobile<->anchor range std: median 0.04 m, max 3.56 m
- quasi-static truth: 8B05 at (3.44, -2.06, 1.27) (surveyed marker, z at the tag antenna)

## Geometry sanity (median measured vs surveyed anchor-anchor distance)
- 50 anchor pairs; residual mean +1.290 m, std 3.880 m, max |.| 23.719 m
- worst 5 pairs: 9911-D5B4 +23.72 m, 5BB6-8AA5 +14.14 m, 4503-5388 +5.52 m, 4197-D5B4 +4.27 m, 4AA4-D5B4 +1.64 m

## Figures
![pairs](pair_availability.png)
![geometry](geometry_residuals.png)
