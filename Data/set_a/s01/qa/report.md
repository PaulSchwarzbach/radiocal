# QA report - s01 (quasi_static_point)

- epochs: 225, directed ranges: 21713 (13 invalid dropped at conversion)
- no timestamps in this campaign's logs; epoch ordering via seq
- ground-truth quality: **verified** (screen: multilateration deviation 0.306 m, range test: median line-of-sight residual +0.251 m, reference +0.158 to +0.251 m)

## Per-pair availability
- directed pairs observed: 124 of 132 possible
- weakest 5 directed pairs (availability % of epochs): 4503->D5B4 0%, 5388->9911 0%, 4AA4->8AA5 0%, 9911->8AA5 0%, 9911->5388 0%

## Mobile node (8B05)
- mobile<->anchor range std: median 0.03 m, max 2.91 m
- quasi-static truth: 8B05 at (7.54, 4.16, 1.31) (surveyed marker, z at the tag antenna)

## Geometry sanity (median measured vs surveyed anchor-anchor distance)
- 52 anchor pairs; residual mean +1.566 m, std 4.341 m, max |.| 23.771 m
- worst 5 pairs: 9911-D5B4 +23.77 m, 4AA4-8AA5 +16.40 m, 5BB6-8AA5 +14.14 m, 4503-5388 +5.46 m, 4197-D5B4 +3.80 m

## Figures
![pairs](pair_availability.png)
![geometry](geometry_residuals.png)
