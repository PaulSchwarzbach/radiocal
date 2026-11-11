# prior_sweep - reference-information axis (C1 arm)

The fully_meshed and canonical multi_hop (k = 3) views with the priors
field overridden to nested known-anchor subsets |K| = 0..3 of the gauge
triple 5BB6 / 4A23 / 8B05 (family_masks.PRIOR_SUBSETS, masks
fully_meshed_p0..3 / multi_hop_p0..3). Roster: the prior-anchored factor
graph (fg_anchored) only - full 2n parameterization with Gaussian unary
prior factors at the nodes-table survey sigma; blind methods on a p-mask
would replicate their base-mask cells. Same kernels, constants, and shared
per-session sigma as the static suite.

Two metrics per cell: the aligned RMSE of the whole benchmark, and from
|K| = 2 on the absolute (non-aligned) RMSE in the reference frame, with
the undecidable |K| = 2 mirror branch released
(evaluation.evaluate_absolute). The CRLB rows are the Gaussian reference
floors at the session sigma from results/crlb.parquet (run_crlb.py):
aligned = trace of the FIM pseudo-inverse on the gauge quotient, absolute
= plain inverse once the nullity is zero. p0 doubles as a standing
blind-consistency check against the factor_graph cells of the base masks.
Validation: validate_crlb.py.

<!-- BEGIN GENERATED (summarize.py) -->
### set_b/s03 fully_meshed - fg_anchored RMSE m by |K| (aligned p0-p3; absolute from |K| = 2; CRLB rows = Gaussian floors at the session sigma)

| variant       | p0    | p1    | p2    | p3    | p2_abs | p3_abs |
| ------------- | ----- | ----- | ----- | ----- | ------ | ------ |
| gaussian      | 0.965 | 0.965 | 0.906 | 0.661 | 1.789  | 1.022  |
| huber         | 0.241 | 0.241 | 0.317 | 0.213 | 0.583  | 0.253  |
| cauchy        | 0.223 | 0.223 | 0.280 | 0.206 | 0.497  | 0.239  |
| CRLB aligned  | 0.083 | 0.083 | 0.080 | 0.074 | -      | -      |
| CRLB absolute | -     | -     | -     | -     | 0.110  | 0.091  |

### set_b/s03 multi_hop - fg_anchored RMSE m by |K| (aligned p0-p3; absolute from |K| = 2; CRLB rows = Gaussian floors at the session sigma)

| variant       | p0    | p1    | p2    | p3    | p2_abs | p3_abs |
| ------------- | ----- | ----- | ----- | ----- | ------ | ------ |
| gaussian      | 0.425 | 0.425 | 0.916 | 0.680 | 1.334  | 0.835  |
| huber         | 0.425 | 0.425 | 0.913 | 0.690 | 1.325  | 0.843  |
| cauchy        | 0.426 | 0.426 | 0.905 | 0.708 | 1.292  | 0.870  |
| CRLB aligned  | 0.984 | 0.984 | 0.907 | 0.249 | -      | -      |
| CRLB absolute | -     | -     | -     | -     | 1.379  | 0.320  |

### set_a/s21 fully_meshed - fg_anchored RMSE m by |K| (aligned p0-p3; absolute from |K| = 2; CRLB rows = Gaussian floors at the session sigma)

| variant       | p0    | p1    | p2    | p3    | p2_abs | p3_abs |
| ------------- | ----- | ----- | ----- | ----- | ------ | ------ |
| gaussian      | 5.935 | 5.935 | 5.818 | 7.120 | 7.207  | 7.753  |
| huber         | 2.331 | 2.331 | 2.189 | 1.846 | 3.059  | 1.989  |
| cauchy        | 0.999 | 0.999 | 0.997 | 0.876 | 1.441  | 0.985  |
| CRLB aligned  | 0.912 | 0.912 | 0.890 | 0.843 | -      | -      |
| CRLB absolute | -     | -     | -     | -     | 1.372  | 1.059  |

### set_a/s21 multi_hop - fg_anchored RMSE m by |K| (aligned p0-p3; absolute from |K| = 2; CRLB rows = Gaussian floors at the session sigma)

| variant       | p0            | p1            | p2            | p3            | p2_abs   | p3_abs   |
| ------------- | ------------- | ------------- | ------------- | ------------- | -------- | -------- |
| gaussian      | 11.144        | 8.256         | 11.047        | 10.877        | 13.190   | 30.916   |
| huber         | 11.887        | 9.466         | 11.526        | 12.542        | 13.424   | 26.243   |
| cauchy        | 11.996        | 9.628         | 8.266         | 11.124        | 9.248    | 30.608   |
| CRLB aligned  | 10.059 (flex) | 11.087 (flex) | 10.946 (flex) | 10.958 (flex) | -        | -        |
| CRLB absolute | -             | -             | -             | -             | - (flex) | - (flex) |

Generated from results/benchmark.parquet (prior-sweep masks, fg_anchored roster, shared per-session fully_meshed sigma) and results/crlb.parquet
(bounds consume the mask's subset; (flex) marks views whose FIM nullity exceeds the gauge
expectation, where the absolute bound is undefined). On rigid views p0 is verified equal
to the blind factor_graph within 0.0005 m; on flexible views the reduced and full
parameterizations may select different flex configurations and the check is not asserted.
The absolute metric is defined exactly from |K| = 2 on.
<!-- END GENERATED (summarize.py) -->
