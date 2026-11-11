# frame_choice - the choice of the defining pair and of the seed triple

The benchmark fixes the frame of every method by one convention: the anchor
order starts with the triple 5BB6, 4A23, 8B05 (`common.GAUGE_TRIPLE`), so
5BB6 sits at the origin and 4A23 on the positive x-axis, and the prior sweep
seeds nested prefixes of the same triple. `run_frame_choice.py` varies both
choices on the full mesh of set_b/s03 and set_a/s21 (see its docstring for
the definitions). Every RMSE below is in meters. "Aligned" is the benchmark
metric, "absolute" the error in the survey frame without alignment.

<!-- BEGIN GENERATED (run_frame_choice.py) -->
## Part A, the defining pair

Aligned RMSE over the pairs that place every anchor; rank 1 is the best, given only where the release pair places every anchor too ("partial": the release construction leaves an anchor unplaced and scores fewer anchors). "Other minimum" counts factor-graph pairs whose robust cost differs from the release pair's by more than 1e-06 relative or whose RMSE moves by 1e-4 m or more, and "max |d| to release" is taken over the pairs at the release minimum.

| session | engine | pairs | ok | all placed | min | median | max | release | release rank | wrong side > 0 | max |d| to release | other minimum |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| set_a/s21 | closed_form:disambiguated | 66 | 65 | 25 | 0.870 | 9.189 | 19.266 | 18.551 | partial | 44 |  |  |
| set_a/s21 | closed_form:faithful | 66 | 65 | 25 | 1.625 | 10.532 | 15.663 | 5.277 | partial | 50 |  |  |
| set_a/s21 | ekf:cauchy | 132 | 132 | 132 | 0.647 | 0.651 | 1.512 | 0.649 | 40 of 132 | 10 |  |  |
| set_a/s21 | factor_graph:cauchy | 66 | 66 | 66 | 0.999 | 0.999 | 0.999 | 0.999 | 64 of 66 | 4 | 0.000 | 0 |
| set_a/s21 | factor_graph:gaussian | 66 | 66 | 66 | 5.935 | 5.935 | 5.935 | 5.935 | 54 of 66 | 39 | 0.000 | 0 |
| set_a/s21 | factor_graph:huber | 66 | 66 | 66 | 2.331 | 2.331 | 6.536 | 2.331 | 8 of 66 | 10 | 0.000 | 1 |
| set_b/s03 | closed_form:disambiguated | 45 | 45 | 30 | 0.286 | 1.715 | 6.433 | 1.694 | 15 of 30 | 19 |  |  |
| set_b/s03 | closed_form:faithful | 45 | 45 | 30 | 0.318 | 3.377 | 5.917 | 1.694 | 7 of 30 | 32 |  |  |
| set_b/s03 | ekf:cauchy | 90 | 90 | 90 | 0.224 | 0.224 | 0.225 | 0.224 | 76 of 90 | 0 |  |  |
| set_b/s03 | factor_graph:cauchy | 45 | 45 | 45 | 0.223 | 0.223 | 0.223 | 0.223 | 40 of 45 | 0 | 0.000 | 0 |
| set_b/s03 | factor_graph:gaussian | 45 | 45 | 45 | 0.965 | 0.965 | 0.965 | 0.965 | 3 of 45 | 14 | 0.000 | 0 |
| set_b/s03 | factor_graph:huber | 45 | 45 | 45 | 0.241 | 0.241 | 0.241 | 0.241 | 41 of 45 | 1 | 0.000 | 0 |
| set_b/s03 | grid_filter:cauchy | 90 | 90 | 90 | 0.215 | 0.251 | 4.814 | 0.251 | 45 of 90 | 3 |  |  |
| set_b/s03 | grid_filter:mixture | 90 | 90 | 90 | 0.191 | 0.243 | 7.145 | 0.238 | 37 of 90 | 8 |  |  |

### set_a/s21 closed_form:disambiguated, best and worst five pairs

| origin | axis | rmse_m | n_placed | unplaced | wrong_side | pair_length_m | pair_residual_m | pair_outlier | pair_hull_edge | n_minority_side |
|---|---|---|---|---|---|---|---|---|---|---|
| 4AA4 | D38F | 0.398 | 10 | 8AA5:radicand;D5B4:radicand | 0.000 | 16.859 | 0.273 | False | False | 4 |
| 4AA4 | 9911 | 0.449 | 9 | 5388:radicand;8AA5:radicand;D5B4:radicand | 0.000 | 7.857 | 0.209 | False | False | 1 |
| 5BB6 | D38F | 0.647 | 11 | 8AA5:radicand | 0.000 | 21.615 | 0.348 | False | False | 5 |
| 8B05 | D38F | 0.721 | 11 | D5B4:radicand | 0.000 | 20.700 | 0.192 | False | False | 4 |
| 4A23 | D38F | 0.870 | 12 |  | 0.000 | 17.308 | 0.172 | False | False | 4 |
| 4A23 | 9911 | 17.584 | 12 |  | 2.000 | 28.765 | 0.457 | False | False | 3 |
| 8B05 | DD98 | 17.981 | 12 |  | 2.000 | 21.493 | 0.214 | False | False | 2 |
| 5BB6 | 4A23 | 18.551 | 11 | 4197:radicand | 2.000 | 23.900 | 0.175 | False | False | 0 |
| 5BB6 | 8B05 | 18.635 | 12 |  | 2.000 | 35.827 | 0.189 | False | False | 3 |
| 4A23 | 8B05 | 19.266 | 12 |  | 2.000 | 13.635 | 0.160 | False | False | 1 |

### set_a/s21 closed_form:faithful, best and worst five pairs

| origin | axis | rmse_m | n_placed | unplaced | wrong_side | pair_length_m | pair_residual_m | pair_outlier | pair_hull_edge | n_minority_side |
|---|---|---|---|---|---|---|---|---|---|---|
| 5BB6 | DD98 | 0.892 | 10 | 4197:radicand;8AA5:radicand | 0.000 | 15.021 | 0.323 | False | True | 0 |
| 4AA4 | 9911 | 1.112 | 9 | 5388:radicand;8AA5:radicand;D5B4:radicand | 1.000 | 7.857 | 0.209 | False | False | 1 |
| 9911 | D38F | 1.498 | 9 | 5388:radicand;8AA5:radicand;D5B4:radicand | 1.000 | 15.933 | 0.287 | False | False | 4 |
| 4A23 | DD98 | 1.625 | 12 |  | 0.000 | 8.880 | 0.123 | False | True | 0 |
| 4197 | DD98 | 1.661 | 11 | 5BB6:radicand | 0.000 | 20.538 | 0.073 | False | False | 0 |
| 4A23 | 8AA5 | 14.244 | 12 |  | 5.000 | 25.881 | 0.608 | False | False | 5 |
| 8B05 | 4197 | 14.471 | 12 |  | 1.000 | 7.770 | 0.082 | False | False | 1 |
| 5388 | 8AA5 | 14.549 | 12 |  | 0.000 | 8.716 | 0.176 | False | True | 0 |
| 5BB6 | 9911 | 15.663 | 12 |  | 0.000 | 15.280 | 0.500 | False | True | 0 |
| 8AA5 | D5B4 | 17.275 | 11 | 4503:missing range | 1.000 | 13.589 | 0.308 | False | False | 1 |

### set_b/s03 closed_form:disambiguated, best and worst five pairs

| origin | axis | rmse_m | n_placed | unplaced | wrong_side | pair_length_m | pair_residual_m | pair_outlier | pair_hull_edge | n_minority_side |
|---|---|---|---|---|---|---|---|---|---|---|
| 5BB6 | 8B05 | 0.204 | 9 | D38F:radicand | 0.000 | 15.306 | 0.405 | False | False | 1 |
| 5388 | D38F | 0.258 | 9 | 5BB6:radicand | 0.000 | 10.768 | 0.155 | False | False | 3 |
| 4503 | 8AA5 | 0.286 | 10 |  | 0.000 | 5.826 | 0.174 | False | False | 3 |
| 4A23 | 4197 | 0.304 | 10 |  | 0.000 | 11.298 | 0.526 | False | False | 1 |
| 4503 | 5388 | 0.310 | 10 |  | 0.000 | 3.158 | 0.015 | False | False | 2 |
| 4AA4 | 9911 | 6.022 | 9 | 5388:radicand | 2.000 | 16.907 | 0.992 | False | False | 2 |
| 4AA4 | 5388 | 6.365 | 9 | 9911:radicand | 1.000 | 2.582 | 0.376 | False | False | 2 |
| 4503 | 9911 | 6.433 | 10 |  | 1.000 | 11.928 | 0.272 | False | False | 1 |
| 4A23 | 4AA4 | 6.445 | 9 | 4503:radicand | 3.000 | 14.273 | 0.670 | False | True | 0 |
| 4503 | 4AA4 | 6.860 | 9 | 4A23:radicand | 1.000 | 5.531 | 0.168 | False | False | 1 |

### set_b/s03 closed_form:faithful, best and worst five pairs

| origin | axis | rmse_m | n_placed | unplaced | wrong_side | pair_length_m | pair_residual_m | pair_outlier | pair_hull_edge | n_minority_side |
|---|---|---|---|---|---|---|---|---|---|---|
| 5BB6 | 8B05 | 0.204 | 9 | D38F:radicand | 0.000 | 15.306 | 0.405 | False | False | 1 |
| 8B05 | 4197 | 0.318 | 10 |  | 0.000 | 8.680 | 0.199 | False | False | 0 |
| 4A23 | 9911 | 0.343 | 10 |  | 0.000 | 7.159 | 0.405 | False | True | 0 |
| 4AA4 | D38F | 0.415 | 9 | 5BB6:radicand | 0.000 | 12.079 | 0.333 | False | False | 1 |
| 4A23 | 4197 | 0.425 | 10 |  | 1.000 | 11.298 | 0.526 | False | False | 1 |
| 4197 | 4AA4 | 5.060 | 10 |  | 4.000 | 18.911 | 0.468 | False | False | 4 |
| 5BB6 | 4197 | 5.380 | 10 |  | 3.000 | 22.125 | 0.292 | False | False | 3 |
| 4503 | D38F | 5.469 | 10 |  | 4.000 | 11.554 | 0.155 | False | False | 4 |
| 8B05 | 4503 | 5.602 | 10 |  | 3.000 | 11.058 | 0.157 | False | False | 3 |
| 4A23 | D38F | 5.917 | 10 |  | 3.000 | 16.060 | 0.300 | False | False | 3 |

## Part B, the seed triple

Over the triples that are not near-collinear (smallest height at least 0.8 m). The quartile columns compare the best and the worst quarter of the triples by the measure: the median smallest height (m), the share of triples whose three anchors are hull vertices, and the share holding an outlier pair.

| session | measure | triples | near-collinear | min | median | max | release | release rank | best q: height | worst q: height | best q: all hull | worst q: all hull | best q: outlier | worst q: outlier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| set_a/s21 | B1 alignment after, absolute | 220 | 8 | 1.019 | 1.228 | 2.001 | 1.112 | 54 of 212 | 14.422 | 6.841 | 0.340 | 0.491 | 0.396 | 0.245 |
| set_a/s21 | B2 seeds as priors, absolute | 220 | 8 | 0.571 | 1.115 | 34.288 | 0.985 | 69 of 212 | 13.510 | 6.677 | 0.415 | 0.453 | 0.585 | 0.226 |
| set_a/s21 | B2 seeds as priors, aligned | 220 | 8 | 0.556 | 0.892 | 2.759 | 0.876 | 93 of 212 | 12.221 | 6.904 | 0.453 | 0.321 | 0.642 | 0.113 |
| set_b/s03 | B1 alignment after, absolute | 120 | 14 | 0.225 | 0.247 | 0.364 | 0.246 | 50 of 106 | 3.996 | 3.347 | 0.192 | 0.154 | 0.038 | 0.077 |
| set_b/s03 | B2 seeds as priors, absolute | 120 | 14 | 0.185 | 0.261 | 0.521 | 0.239 | 27 of 106 | 3.345 | 3.569 | 0.038 | 0.308 | 0.038 | 0.154 |
| set_b/s03 | B2 seeds as priors, aligned | 120 | 14 | 0.183 | 0.217 | 0.296 | 0.206 | 18 of 106 | 4.669 | 3.571 | 0.308 | 0.192 | 0.077 | 0.154 |

### set_a/s21, near-collinear triples (reported apart)

| a | b | c | min_height_m | b1_abs_rmse_m | b2_abs_rmse_m | b2_rmse_m | b2_abs_reflection |
|---|---|---|---|---|---|---|---|
| 4503 | 9911 | D38F | 0.448 | 21.923 | 21.985 | 0.843 | False |
| 4A23 | 4197 | DD98 | 0.072 | 31.007 | 31.006 | 0.986 | False |
| 5388 | 8AA5 | 9911 | 0.429 | 1.445 | 1.079 | 0.794 | False |
| 5BB6 | 4197 | DD98 | 0.224 | 31.342 | 31.311 | 0.894 | False |
| 5BB6 | 4A23 | 4197 | 0.200 | 31.360 | 31.337 | 0.921 | False |
| 5BB6 | 4A23 | DD98 | 0.099 | 31.043 | 31.041 | 0.972 | False |
| 5BB6 | 4AA4 | 9911 | 0.661 | 1.475 | 1.396 | 0.985 | False |
| 8B05 | 4197 | 4503 | 0.540 | 1.422 | 1.340 | 0.980 | False |

### set_b/s03, near-collinear triples (reported apart)

| a | b | c | min_height_m | b1_abs_rmse_m | b2_abs_rmse_m | b2_rmse_m | b2_abs_reflection |
|---|---|---|---|---|---|---|---|
| 4197 | 4AA4 | 5388 | 0.712 | 0.233 | 0.222 | 0.207 | False |
| 4503 | 4AA4 | 5388 | 0.763 | 0.265 | 0.355 | 0.206 | False |
| 4A23 | 4197 | 9911 | 0.502 | 0.333 | 0.332 | 0.212 | False |
| 4A23 | 4503 | 4AA4 | 0.324 | 0.263 | 0.272 | 0.205 | False |
| 4A23 | 4503 | 5388 | 0.344 | 0.310 | 0.318 | 0.219 | False |
| 4AA4 | 5388 | 9911 | 0.189 | 0.226 | 0.205 | 0.199 | False |
| 4AA4 | 8AA5 | D38F | 0.034 | 0.296 | 0.324 | 0.221 | False |
| 5BB6 | 4503 | 4AA4 | 0.232 | 0.333 | 13.017 | 0.301 | False |
| 5BB6 | 4503 | 5388 | 0.633 | 0.261 | 0.405 | 0.230 | False |
| 5BB6 | 4A23 | 4503 | 0.243 | 0.261 | 0.279 | 0.211 | False |
| 5BB6 | 4A23 | 4AA4 | 0.131 | 0.302 | 13.821 | 0.251 | False |
| 5BB6 | 4A23 | 5388 | 0.797 | 0.267 | 0.273 | 0.208 | False |
| 5BB6 | 4AA4 | 5388 | 0.621 | 0.305 | 0.477 | 0.214 | False |
| 8B05 | 4197 | D38F | 0.201 | 0.281 | 0.295 | 0.224 | False |

## The line of the release pair

Signed offsets of the other anchors from the line of the release pair (positive on the side of 8B05), and what the release closed form (faithful) does with them.

| session | pair | within 0.8 m | far side max | unplaced |
|---|---|---|---|---|
| set_a/s21 | 5BB6-4A23 | 2 | 0.099 | 4197:radicand (+0.30 m) |
| set_b/s03 | 5BB6-4A23 | 3 | 0.131 |  |

### set_a/s21

| id | offset_m | on_hull | cf_placed | cf_reason | cf_error_m |
|---|---|---|---|---|---|
| DD98 | -0.099 | True | True |  | 1.277 |
| 4A23 | 0.000 | True | True |  | 0.665 |
| 5BB6 | 0.000 | True | True |  | 2.217 |
| 4197 | 0.297 | True | False | radicand |  |
| 4AA4 | 7.467 | False | True |  | 2.143 |
| 8B05 | 8.040 | False | True |  | 1.903 |
| D38F | 15.109 | False | True |  | 1.941 |
| 9911 | 15.272 | True | True |  | 2.721 |
| 4503 | 15.912 | True | True |  | 2.148 |
| 8AA5 | 23.169 | True | True |  | 16.199 |
| 5388 | 27.144 | True | True |  | 2.514 |
| D5B4 | 27.220 | True | True |  | 2.578 |

### set_b/s03

| id | offset_m | on_hull | cf_placed | cf_reason | cf_error_m |
|---|---|---|---|---|---|
| 4AA4 | -0.131 | True | True |  | 0.632 |
| 5BB6 | 0.000 | True | True |  | 1.582 |
| 4A23 | 0.000 | True | True |  | 0.234 |
| 4503 | 0.243 | False | True |  | 0.119 |
| 5388 | 0.797 | False | True |  | 0.595 |
| 8AA5 | 3.996 | False | True |  | 0.914 |
| 9911 | 7.086 | True | True |  | 0.627 |
| 8B05 | 11.266 | False | True |  | 0.963 |
| 4197 | 11.269 | True | True |  | 1.230 |
| D38F | 11.542 | True | True |  | 4.658 |

## Reproduction of the release cells

- set_a/s21 closed_form:disambiguated stored 18.550814294 got 18.550814294 (d=0.0e+00)
- set_a/s21 closed_form:faithful stored 5.277286608 got 5.277286608 (d=0.0e+00)
- set_a/s21 ekf:cauchy stored 0.649241672 got 0.649241672 (d=0.0e+00)
- set_a/s21 factor_graph:cauchy stored 0.998592230 got 0.998592230 (d=0.0e+00)
- set_a/s21 factor_graph:gaussian stored 5.934699429 got 5.934699429 (d=0.0e+00)
- set_a/s21 factor_graph:huber stored 2.331229560 got 2.331229560 (d=0.0e+00)
- set_b/s03 closed_form:disambiguated stored 1.693961867 got 1.693961867 (d=0.0e+00)
- set_b/s03 closed_form:faithful stored 1.693961867 got 1.693961867 (d=0.0e+00)
- set_b/s03 ekf:cauchy stored 0.223613700 got 0.223613700 (d=0.0e+00)
- set_b/s03 factor_graph:cauchy stored 0.223357951 got 0.223357951 (d=0.0e+00)
- set_b/s03 factor_graph:gaussian stored 0.964595778 got 0.964595778 (d=0.0e+00)
- set_b/s03 factor_graph:huber stored 0.241227404 got 0.241227404 (d=0.0e+00)
- set_b/s03 grid_filter:cauchy stored 0.251190060 got 0.251190060 (d=0.0e+00)
- set_b/s03 grid_filter:mixture stored 0.237643583 got 0.237643583 (d=0.0e+00)
- set_a/s21 fg_anchored:cauchy p3 aligned d=0.0e+00, absolute d=0.0e+00
- set_b/s03 fg_anchored:cauchy p3 aligned d=0.0e+00, absolute d=0.0e+00
<!-- END GENERATED (run_frame_choice.py) -->
