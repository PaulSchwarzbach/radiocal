# QA report - s14 (quasi_static_point)

- epochs: 174, directed ranges: 16322 (18 invalid dropped at conversion)
- no timestamps in this campaign's logs; epoch ordering via seq
- ground-truth quality: **verified** (screen: multilateration deviation 0.24 m, range test: median line-of-sight residual +0.221 m, reference +0.158 to +0.251 m)

## Per-pair availability
- directed pairs observed: 122 of 132 possible
- weakest 5 directed pairs (availability % of epochs): 4503->D5B4 0%, 5388->4AA4 0%, 4AA4->8AA5 0%, 4AA4->5388 0%, 5388->9911 0%

## Mobile node (8B05)
- mobile<->anchor range std: median 0.02 m, max 5.54 m
- quasi-static truth: 8B05 at (-16.17, 2.77, 1.29) (surveyed marker, z at the tag antenna)

## Geometry sanity (median measured vs surveyed anchor-anchor distance)
- 50 anchor pairs; residual mean +1.311 m, std 3.915 m, max |.| 23.979 m
- worst 5 pairs: 9911-D5B4 +23.98 m, 5BB6-8AA5 +14.13 m, 4503-5388 +5.44 m, 4197-D5B4 +4.73 m, 4AA4-D5B4 +1.64 m

## Figures
![pairs](pair_availability.png)
![geometry](geometry_residuals.png)
