# QA report - s09 (quasi_static_point)

- epochs: 241, directed ranges: 23558 (25 invalid dropped at conversion)
- no timestamps in this campaign's logs; epoch ordering via seq
- ground-truth quality: **verified** (screen: multilateration deviation 0.241 m, range test: median line-of-sight residual +0.208 m, reference +0.158 to +0.251 m)

## Per-pair availability
- directed pairs observed: 123 of 132 possible
- weakest 5 directed pairs (availability % of epochs): 4503->D5B4 0%, 4AA4->8AA5 0%, 4AA4->5388 0%, 5388->9911 0%, 9911->5388 0%

## Mobile node (8B05)
- mobile<->anchor range std: median 0.05 m, max 5.31 m
- quasi-static truth: 8B05 at (-3.39, 17.53, 1.26) (surveyed marker, z at the tag antenna)

## Geometry sanity (median measured vs surveyed anchor-anchor distance)
- 51 anchor pairs; residual mean +1.304 m, std 3.876 m, max |.| 23.933 m
- worst 5 pairs: 9911-D5B4 +23.93 m, 5BB6-8AA5 +14.14 m, 4197-D5B4 +5.26 m, 4503-5388 +5.19 m, 4AA4-D5B4 +1.60 m

## Figures
![pairs](pair_availability.png)
![geometry](geometry_residuals.png)
