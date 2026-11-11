# QA report - s06 (quasi_static_point)

- epochs: 227, directed ranges: 21586 (19 invalid dropped at conversion)
- no timestamps in this campaign's logs; epoch ordering via seq
- ground-truth quality: **verified** (screen: multilateration deviation 0.309 m, range test: median line-of-sight residual +0.176 m, reference +0.158 to +0.251 m)

## Per-pair availability
- directed pairs observed: 124 of 132 possible
- weakest 5 directed pairs (availability % of epochs): 4503->D5B4 0%, 5388->9911 0%, 4AA4->8AA5 0%, 9911->8AA5 0%, 9911->5388 0%

## Mobile node (8B05)
- mobile<->anchor range std: median 0.02 m, max 6.17 m
- quasi-static truth: 8B05 at (-3.48, 1.40, 1.28) (surveyed marker, z at the tag antenna)

## Geometry sanity (median measured vs surveyed anchor-anchor distance)
- 51 anchor pairs; residual mean +1.358 m, std 3.967 m, max |.| 24.231 m
- worst 5 pairs: 9911-D5B4 +24.23 m, 5BB6-8AA5 +14.14 m, 4197-D5B4 +7.33 m, 4503-5388 +5.37 m, 4AA4-D5B4 +1.74 m

## Figures
![pairs](pair_availability.png)
![geometry](geometry_residuals.png)
