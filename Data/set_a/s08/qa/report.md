# QA report - s08 (quasi_static_point)

- epochs: 189, directed ranges: 17860 (16 invalid dropped at conversion)
- no timestamps in this campaign's logs; epoch ordering via seq
- ground-truth quality: **verified** (screen: multilateration deviation 0.195 m, range test: median line-of-sight residual +0.158 m, reference +0.158 to +0.251 m)

## Per-pair availability
- directed pairs observed: 122 of 132 possible
- weakest 5 directed pairs (availability % of epochs): 4503->D5B4 0%, 5388->4AA4 0%, 4AA4->8AA5 0%, 4AA4->5388 0%, 5388->9911 0%

## Mobile node (8B05)
- mobile<->anchor range std: median 0.03 m, max 3.56 m
- quasi-static truth: 8B05 at (-3.40, 11.99, 1.28) (surveyed marker, z at the tag antenna)

## Geometry sanity (median measured vs surveyed anchor-anchor distance)
- 50 anchor pairs; residual mean +1.325 m, std 3.901 m, max |.| 23.617 m
- worst 5 pairs: 9911-D5B4 +23.62 m, 5BB6-8AA5 +14.14 m, 4197-D5B4 +5.95 m, 4503-5388 +5.54 m, 4AA4-D5B4 +1.44 m

## Figures
![pairs](pair_availability.png)
![geometry](geometry_residuals.png)
