# Factor-graph NLS on fully_meshed - summary

Session-median symmetrized ranges; every observed pair is a range
factor; 2-D anchor coordinates, gauge fixed by parameterization
(anchor 0 at the origin, anchor 1 on the x-axis); LM with analytic
Jacobian, initialized from classical MDS. Evaluation: rigid Umeyama
alignment (reflection allowed) against the surveyed anchors of the
unmasked session, 2-D (x, y).

<!-- BEGIN GENERATED (summarize.py) -->
| campaign | session | variant  | n_anchors | pairs_available | rmse_m | max_error_m | reflection | n_iter | converged | cost0   | cost    | sigma_m |
| -------- | ------- | -------- | --------- | --------------- | ------ | ----------- | ---------- | ------ | --------- | ------- | ------- | ------- |
| set_b    | s03     | gaussian | 10        | 45/45           | 0.965  | 2.441       | False      | 8      | True      | 808.675 | 434.926 | 0.129   |
| set_b    | s03     | huber    | 10        | 45/45           | 0.241  | 0.397       | False      | 17     | True      | 85.172  | 67.692  | 0.129   |
| set_b    | s03     | cauchy   | 10        | 45/45           | 0.223  | 0.314       | False      | 14     | True      | 44.991  | 27.933  | 0.129   |
| set_a    | s21     | gaussian | 12        | 65/66           | 5.935  | 14.777      | False      | 18     | True      | 342.379 | 143.251 | 1.621   |
| set_a    | s21     | huber    | 12        | 65/66           | 2.331  | 5.007       | False      | 87     | True      | 109.339 | 83.209  | 1.621   |
| set_a    | s21     | cauchy   | 12        | 65/66           | 0.999  | 1.770       | False      | 13     | True      | 80.754  | 56.690  | 1.621   |
<!-- END GENERATED (summarize.py) -->

Notes:

- Parameters: accepted-step convergence at relative cost change
  < 1e-12 or max parameter change < 1e-10; lambda 1e-3 x/10 per
  acceptance/rejection, Marquardt diagonal scaling; max_iter 10000
  is a safety cap only. All runs above converged on the rule
  (huber/2024 needs 87 iterations).
- Kernels and noise scale come from the shared benchmark module
  kernels.py (factorial design: same kernel constants and the same
  session sigma for every method): sigma = max(median per-pair
  epoch std of symmetrized ranges, 1.4826 * MAD of the MDS-init
  residuals) = 0.129 m (2025) / 1.621 m (2024); huber k = 1.345,
  cauchy c = 2.385 on u = r / sigma. The variant formerly labeled
  l2 is now gaussian (identical objective and results).
- Exact-recovery check: noise-free distance matrices built from the
  surveyed coordinates of both sites are recovered by init + solver
  (all variants) with < 1e-13 m aligned RMSE.
- The gaussian variant reproduces the SMACOF geometry (0.965 /
  5.935 m vs 0.96 / 5.93 m): same objective, different optimizer.
  Dropping the NLOS pair 5BB6-D38F moves it on set_b from
  0.965 to 0.228 m, matching the SMACOF precedent - the objective,
  not the solver, is what NLOS breaks; the robust kernels repair it.
- Scale sweep (scale_sweep.csv, cauchy, sigma multiplier 0.25-4):
  on set_a the result is strongly scale-dependent (see the
  cross-method overview) - the a-priori sigma is kept for the
  benchmark row; selecting the multiplier by test RMSE would be
  circular.
- Down-weighted pairs (see pair_weights.csv). set_b: the one
  persistent NLOS pair 5BB6-D38F (residual -5.2/-5.4 m) gets weight
  0.033 (huber) / 0.003 (cauchy); next-lowest is 4AA4-9911 at
  0.46 / 0.41, all other pairs stay >= 0.57 / 0.66. set_a:
  8AA5-9911 (resid -21.1/-27.6 m), 9911-D5B4, 8B05-D5B4, 4AA4-8AA5,
  5388-9911 and 5BB6-8AA5 are the six lowest in both kernels
  (huber weights 0.10-0.27, cauchy 0.02-0.11; next pair 4503-5388
  at 0.52 / 0.51).
- Init sensitivity (set_a, all kernels): starting LM from the
  gauged surveyed coordinates instead of the MDS init converges to
  the same cost and the same RMSE - no local-minimum dependence on
  the initialization was observed, including for the non-convex
  cauchy kernel.
- Remaining cauchy floor on set_a (~1.0 m at 1x sigma):
  removing the six down-weighted pairs entirely and re-solving
  still leaves 0.79 m (gaussian) / 0.68 m (cauchy), and 11 of 65
  pairs have > 1 m median-range bias against the surveyed distances
  (19 > 0.5 m) - the residual error is a broad moderate-NLOS floor,
  not a single culprit pair.
