# Sequential EKF on fully_meshed - summary (factorial round 2)

Per-epoch symmetrized ranges processed sequentially (state updated
between pairs of an epoch); evaluation: rigid Umeyama alignment
(reflection allowed) against the surveyed anchors of the unmasked
session, 2-D (x, y).

<!-- BEGIN GENERATED (summarize.py) -->
| campaign | session | variant  | init  | n_anchors | sigma_m | rmse_m | max_error_m | reflection | gated_frac |
| -------- | ------- | -------- | ----- | --------- | ------- | ------ | ----------- | ---------- | ---------- |
| set_b    | s03     | gaussian | mds20 | 10        | 0.129   | 1.070  | 2.734       | False      | -          |
| set_b    | s03     | huber    | mds20 | 10        | 0.129   | 0.240  | 0.387       | False      | -          |
| set_b    | s03     | cauchy   | mds20 | 10        | 0.129   | 0.224  | 0.315       | False      | -          |
| set_b    | s03     | gated    | mds20 | 10        | 0.129   | 0.220  | 0.312       | False      | 0.031      |
| set_a    | s21     | gaussian | mds20 | 12        | 1.621   | 3.286  | 7.516       | False      | -          |
| set_a    | s21     | huber    | mds20 | 12        | 1.621   | 0.900  | 1.643       | False      | -          |
| set_a    | s21     | cauchy   | mds20 | 12        | 1.621   | 0.649  | 1.219       | False      | -          |
| set_a    | s21     | gated    | mds20 | 12        | 1.621   | 0.767  | 1.629       | False      | 0.038      |
<!-- END GENERATED (summarize.py) -->

Parameters: state = 2-D anchor coordinates minus gauge (anchor 5BB6
fixed at the origin, 4A23 y = 0; dim 2n-3); init = classical MDS,
shortest-path completed, on the median matrix of the first 20
epochs, re-gauged (mds20; a closed-form init remains available in
ekf.py); initial sigma 2 m per state coordinate; process noise zero
(static anchors). Noise model shared across all C4 methods:
sigma_m = kernels.session_sigma (scatter / init-residual scale),
kernel constants a-priori (huber k = 1.345, cauchy c = 2.385).
Robust variants use IRLS-style R inflation per scalar update:
u = innovation / sigma, R_eff = sigma^2 / kernels.weight(u, kernel).
gated = gaussian kernel + NIS > 9 innovation gate (3-sigma, 1 DOF).

Notes (all statements verified by targeted experiments):

- Exact-recovery check: on synthetic streams built from the
  surveyed coordinates (200 epochs, all pairs, exact distances and
  1 mm noise) the full pipeline (init + filter) recovers both
  sites to < 0.1 mm RMSE, all four variants.
- Changes vs round 1: the shared session sigma replaces the
  per-method scatter estimate (0.050 -> 0.129 m on 2025, 1.039 ->
  1.621 m on 2024) and the kernel axis was added. The gaussian
  results match the round-1 plain filter (1.07 / 3.28-3.29 m):
  a uniform rescaling of R changes gains, not the fixed point.
- Kernel axis: gaussian adopts the NLOS biases (2025: 1.07 m,
  dominated by 5BB6-D38F at +5.8 m - the round-1 drop experiment
  gives 0.23 m without that pair; 2024: 3.29 m). huber (0.240 /
  0.900 m) improves but keeps the linear tail influence of large
  outliers; cauchy redescends and gives 0.224 / 0.649 m.
- Gate behavior change on 2025 (sigma 0.050 -> 0.129 m): the
  3-sigma gate is now ~0.39 m instead of ~0.15 m, wider than the
  +0.2..0.5 m systematic LOS biases, so the round-1 pathology of
  wholesale LOS rejection is gone: 3.1% of updates gated (round 1:
  17.4%), and only the true NLOS pair 5BB6-D38F is > 99% gated
  (100%, from epoch 1); next is 4AA4-9911 (+0.99 m) at 49%. The
  gated filter now re-estimates the geometry (0.220 m) instead of
  freezing the 0.264 m init as in round 1 (0.238 m).
- 2024 gate: the six worst NLOS pairs (+14.1 to +30.3 m median
  bias) are 100% gated from their first observation, 3.8% of all
  updates. gated (0.767 m) still trails cauchy (0.649 m): pairs
  with moderate bias that pass the gate enter at full gaussian
  weight, while cauchy downweights them continuously.
- Cauchy scale sweep (*_scale_sweep.csv): on 2025 the result is
  flat (0.218-0.251 m over sigma multiples 0.25-4) - the one
  dominant outlier is rejected at every scale. On 2024 it is
  monotone (0.278 m at 0.25x to 2.475 m at 4x): the shared sigma
  of the NLOS-rich session is itself inflated by NLOS scatter, so
  the a-priori scale under-rejects there; reported as scale
  sensitivity, not retuned (constants stay a-priori).
- Init dependence persists in this configuration: from the
  closed-form init (reflected, 17.8 m wrong on 2024) gaussian ends
  at 13.0 m and cauchy at 18.3 m - the robust kernel downweights
  exactly the measurements that contradict the wrong geometry, so
  robustness amplifies init lock-in. On 2025 both are init-
  independent (1.071 / 0.224 m from the closed-form init). All
  table rows use the mds20 init (0.264 m on 2025, 3.097 m on
  2024, no reflection).
- The filter remains optimistic: final per-coordinate posterior
  sigma is 0.002-0.004 m (2025) and 0.015-0.034 m (2024) while
  actual per-anchor errors reach 2.73 m / 7.52 m (gaussian) and
  0.32-0.39 m / 1.2-1.6 m (robust variants); R inflation does not
  repair the unmodeled systematic bias in the covariance.
- Convergence (*_convergence.csv): on 2025 the robust and gated
  traces are flat at their final level from epoch 50; gaussian
  drifts 0.78 m (epoch 400) -> 1.07 m as biased epochs accumulate.
  On 2024 the robust traces improve slowly (cauchy 0.80 m at epoch
  50 -> 0.65 m at the end).
