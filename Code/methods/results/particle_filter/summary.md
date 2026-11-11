# Particle filter baseline on fully_meshed - summary (round 2)

Per-anchor conditional bootstrap particle filter over the epoch
stream (grouped by seq); evaluation: rigid Umeyama alignment
(reflection allowed) against the surveyed anchors of the unmasked
session, 2-D (x, y).

<!-- BEGIN GENERATED (summarize.py) -->
| campaign | session | variant  | init  | n_anchors | rmse_m | max_error_m | reflection | sigma_m | n_particles | sweeps |
| -------- | ------- | -------- | ----- | --------- | ------ | ----------- | ---------- | ------- | ----------- | ------ |
| set_b    | s03     | gaussian | mds20 | 10        | 0.762  | 1.970       | False      | 0.129   | 2000        | 2      |
| set_b    | s03     | huber    | mds20 | 10        | 0.241  | 0.383       | False      | 0.129   | 2000        | 2      |
| set_b    | s03     | cauchy   | mds20 | 10        | 0.226  | 0.321       | False      | 0.129   | 2000        | 2      |
| set_a    | s21     | gaussian | mds20 | 12        | 2.942  | 6.346       | False      | 1.621   | 2000        | 2      |
| set_a    | s21     | huber    | mds20 | 12        | 1.100  | 2.146       | False      | 1.621   | 2000        | 2      |
| set_a    | s21     | cauchy   | mds20 | 12        | 0.964  | 2.078       | False      | 1.621   | 2000        | 2      |
<!-- END GENERATED (summarize.py) -->

Notes:

- Round 2 (factorial benchmark): the likelihood is the shared
  kernel module kernels.py - per-range log-weight
  -rho((r - d) / sigma, kernel), kernels gaussian / huber
  (k = 1.345) / cauchy (c = 2.385), one session scale sigma =
  kernels.session_sigma: 0.129 m (set_b), 1.621 m
  (set_a). Round 1 used a per-method scale (median per-pair
  epoch std floored at 0.05 m; cauchy gamma = that sigma, 1.04 m
  at 2024). The 2024 cauchy change 0.28 -> 0.96 m is this scale
  unification (effective half-width 2.385 * 1.621 = 3.87 m admits
  more NLOS influence), not an estimator change.
- Parameters unchanged from round 1: N = 2000 particles per
  anchor, init = mds20 (classical MDS on the median range matrix
  of the first 20 epochs, shortest-path completed, gauge-fixed;
  the same init recipe as the EKF and factor-graph methods) + 2 m
  Gaussian scatter, annulus/disk fallback for anchors without a
  finite init position, systematic resampling at ESS < N/2 with
  0.02 m jitter, 2 sweeps (sweep 1: condition on earlier anchors
  only; sweep 2: re-scatter 2 m around the sweep-1 means,
  condition on all anchors), seed 0. mds20 init RMSE after
  alignment: 0.26 m (2025), 3.10 m (2024, worst anchor 8AA5 at
  5.2 m).
- Exact-recovery check (synthetic ranges from the surveyed
  coordinates, 1 mm noise, 200 epochs; the session sigma resolves
  to the noise floor): 0.06-0.22 cm RMSE on both site geometries,
  all three kernels, sweeps 1 and 2.
- Seed spread (seeds 0/1/2, committed config): <= 0.006 m on all
  set_b cells; set_a gaussian 2.940-2.969, huber
  1.049-1.100, cauchy 0.902-1.274. The 2024 cauchy spread is
  dominated by 8AA5 (2.08 m at seed 0 vs 3.33 m at seed 1): under
  the wide shared scale the redescending posterior at the most
  NLOS-loaded anchor is broad, so the selected mode varies with
  the sampling stream.
- Sweeps 1 vs 2 (seed 0): gaussian 0.30 -> 0.76 m (2025) and
  3.94 -> 2.94 m (2024); huber 0.23 -> 0.24 m and 3.96 -> 1.10 m;
  cauchy 0.23 -> 0.23 m and 3.95 -> 0.96 m. Two sweeps stay the
  default (the 2024 robust cells improve by ~3 m; sweep 1 is
  limited by the conditioning order, not the kernel). The
  gaussian/2025 regression is the NLOS pair 5BB6-D38F (+5.8 m
  bias): in sweep 2 all anchors condition on the displaced D38F
  mean and the bias propagates. Dropping that pair gives 0.23 m
  for both sweep counts (re-verified under the round-2
  likelihood; same pair that moved SMACOF 0.96 -> 0.23 m). It
  also collapses ESS at D38F: 1111 of 4160 updates below 1% of N
  under the 0.129 m Gaussian; huber and cauchy have <= 2 such
  events per anchor.
- Init-mode capture, re-verified under the round-2 likelihood:
  with init='closed_form' (8AA5 seeded 16.2 m off by the +14.1 m
  NLOS pair 8AA5-5BB6) the 2024 cauchy cell degrades to 1.51 m
  RMSE with 8AA5 at 4.28 m, vs 0.96 m / 2.08 m under mds20 - the
  effect persists, attenuated by the wider scale. Round-1
  experiments at the tight per-method scale (gamma = 1.04 m)
  showed the full mechanism: closed-form init 3.88 m RMSE / 8AA5
  at 12.5 m; clean init + full NLOS stream 0.28 m; biased init +
  cleaned stream 14.9 m - a redescending kernel that rejects NLOS
  votes equally defends a wrong init mode, while gaussian is
  init-insensitive but dragged by the stream bias.
- Scale-sweep diagnostic (cauchy, sigma_used = m * sigma, seed 0;
  RMSE per multiplier, see the scale_sweep CSVs):
  set_b: 0.23 / 0.22 / 0.23 / 0.23 / 0.25 m at x0.25 / x0.5 / x1 / x2 / x4.
  set_a: 0.27 / 0.38 / 0.96 / 1.81 / 2.86 m at x0.25 / x0.5 / x1 / x2 / x4.
- posterior_std_m is the spread of the per-anchor conditional
  filter given point estimates of the other anchors; it does not
  include the uncertainty of the conditioning set and understates
  the joint uncertainty.

Replication over the set_a quasi-static sessions (benchmark.parquet, seed 0):

- The cauchy cell is the method's weak spot in the sparse quasi-static session
  mesh, and the worst anchor is 8AA5 in every elevated session (s06
  6.83 m, s03 3.97 m, s14 3.61 m, s08 3.40 m anchor error; session
  RMSE 2.57 / 1.71 / 1.46 / 1.42 m). Root cause, verified on s06:
  with 8B05 roving, 8AA5 keeps only 8 of its 11 anchor pairs and 5 of
  the 8 are NLOS-biased (8AA5-5BB6 +14.1 m, four more at +0.6..1.1 m),
  so the clean ranges are a 3-of-8 minority. The redescending
  likelihood then supports a displaced mode for the conditional
  per-anchor filter; seeds 0-4 all land 8AA5 between 3.7 and 6.8 m
  off, so this is a systematic multi-modality, not seed noise. Batch
  cauchy (0.88 m on s06) and the cauchy EKF (0.73 m) place 8AA5
  correctly from the same data - the global view outweighs the
  biased minority, the conditional sequential update does not. This
  is the sparse-mesh, always-on version of the 8AA5 seed-spread
  caveat above.
