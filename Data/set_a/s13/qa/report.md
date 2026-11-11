# QA report - s13 (quasi_static_point)

- epochs: 206, directed ranges: 19894 (11 invalid dropped at conversion)
- no timestamps in this campaign's logs; epoch ordering via seq
- ground-truth quality: **verified** (screen: multilateration deviation 0.16 m, range test: median line-of-sight residual +0.250 m, reference +0.158 to +0.251 m)

## Per-pair availability
- directed pairs observed: 124 of 132 possible
- weakest 5 directed pairs (availability % of epochs): 4503->D5B4 0%, 5388->9911 0%, 4AA4->8AA5 0%, 9911->8AA5 0%, 9911->5388 0%

## Mobile node (8B05)
- mobile<->anchor range std: median 0.02 m, max 3.06 m
- quasi-static truth: 8B05 at (-12.82, 7.76, 1.28) (surveyed marker, z at the tag antenna)

## Geometry sanity (median measured vs surveyed anchor-anchor distance)
- 51 anchor pairs; residual mean +1.047 m, std 2.441 m, max |.| 14.138 m
- worst 5 pairs: 5BB6-8AA5 +14.14 m, 9911-D5B4 +9.07 m, 4503-5388 +5.44 m, 4197-D5B4 +5.24 m, 4AA4-D5B4 +1.89 m

## Figures
![pairs](pair_availability.png)
![geometry](geometry_residuals.png)
