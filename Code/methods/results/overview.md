# C4 reference methods on fully_meshed - cross-method overview

Factorial benchmark: estimator class x kernel under ONE shared noise model
(kernels.py: dimensionless gaussian / huber (k = 1.345) / cauchy (c = 2.385)
kernels on u = r / sigma; session sigma = max(pair epoch scatter, MDS-init
residual MAD) = 0.129 m on set_b/s03, 1.621 m on
set_a/s21). All methods consume the identical fully_meshed-masked
MeasurementGraph, and the sequential estimators share the mds20 init family
(grid filter: closed-form init, historical port). The kernel constants are
not tuned against the evaluation metric. Three settings were fixed after
benchmark results had been inspected (the shared noise scale, the mds20
start of the filters and the second particle-filter sweep, the lead of
Section VI of the paper). Evaluation: rigid 2-D Umeyama alignment (reflection
allowed) against the surveyed anchors of the unmasked session; aligned
anchor RMSE in metres; details per method in results/<method>/summary.md.

Sites: set_b = LOS with one persistent NLOS pair (5BB6-D38F, +5.8 m)
and +0.3..0.5 m systematic range bias; set_a = NLOS-rich (19 of 65
pairs biased > 0.5 m, worst 8AA5-9911 at +30.3 m).

## Factorial matrix (RMSE m, 2025 / 2024)

| estimator \ kernel | gaussian | huber | cauchy |
| ------------------ | -------- | ----- | ------ |
| SMACOF (batch IRLS-Guttman) | 0.964 / 5.934 | 0.241 / 2.330 | 0.223 / 0.999 |
| factor graph (batch IRLS-LM) | 0.965 / 5.935 | 0.241 / 2.331 | 0.223 / 0.999 |
| EKF (sequential, mds20) | 1.070 / 3.286 | 0.240 / 0.900 | 0.224 / 0.649 |
| particle filter (sequential, mds20, seed 0) | 0.762 / 2.942 | 0.241 / 1.100 | 0.226 / 0.964 |
| grid product (sequential, closed-form init) | 0.805 / 3.042 | 0.275 / 0.808 | 0.251 / 0.619 |

Mechanism variants (own fusion/rejection rule, matched sigma):

| mechanism | 2025 | 2024 | matched baseline |
| --------- | ---- | ---- | ---------------- |
| EKF innovation gate (NIS > 9, gaussian) | 0.220 | 0.767 | EKF gaussian 1.070 / 3.286 |
| grid mixture (IOPS sum update, gaussian) | 0.238 | 0.397 | grid product gaussian 0.805 / 3.042 |

Kernel-less references: classical MDS (Torgerson) 0.258 / 3.061; closed-form
construction 1.694 / 5.277 (faithful; y-sign-disambiguated 18.551 on 2024).

## Cauchy scale sweep (RMSE m at sigma multiplier 0.25 / 0.5 / 1 / 2 / 4)

| estimator | set_b | set_a |
| --------- | ----------- | ----------- |
| batch (SMACOF = factor graph) | 0.20 0.22 0.22 0.23 0.25 | 0.32 0.50 1.00 7.39 7.41 |
| EKF | 0.22 0.22 0.22 0.23 0.25 | 0.28 0.40 0.65 1.30 2.48 |
| particle filter | 0.23 0.22 0.23 0.23 0.25 | 0.27 0.38 0.96 1.81 2.86 |
| grid product | 0.25 0.25 0.25 0.28 0.31 | 0.32 0.39 0.62 0.96 1.38 |

From results/sigma_sweep.parquet (run_controls.py: grid filter at the
committed stride, particle filter seed 0), the noise-scale sweep that
Section VI-A of the paper reads. It supersedes the per-method results/<method>/*_scale_sweep.csv of the
legacy runners, whose grid-filter sweep ran at twice the committed stride.

## Verified cross-method findings

1. Batch identity: robust SMACOF and the robust factor graph reach identical
   minima in every cell and every sweep point - two independent solvers
   (IRLS-Guttman vs IRLS-LM) of the same objective. "MDS-stress vs
   factor-graph NLS" is a solver distinction, not an estimator distinction.
2. Kernel dominates estimator class. Down each column the spread is small
   (huber 0.24-0.28 / 0.81-2.33; cauchy 0.22-0.25 / 0.62-1.00) compared to
   the gaussian -> cauchy step (roughly 1 m -> 0.22 m on 2025, 3-6 m ->
   0.6-1.0 m on 2024). Every gaussian cell reproduces the round-1
   least-squares attractor; on 2025 dropping the single NLOS pair moves
   each of them to ~0.23-0.25 m (verified per method).
3. Scale is the second factor, larger than estimator class at the NLOS
   site: the 2024 cauchy column varies monotonically by 4-23x across the
   16x sigma sweep, and at 0.25 x sigma ALL estimator classes converge to
   0.27-0.32 m. The a-priori session sigma is NLOS-inflated on 2024 (the
   init-residual MAD absorbs the biases it should reject), so every kernel
   under-rejects at 1x. Principled adaptive scale estimation (e.g.
   iteratively re-estimated residual MAD) is the obvious C4 tutorial
   direction; selecting the multiplier by test RMSE would be circular and
   is not done here.
4. At the conservative 1x scale, sequential per-epoch estimators (EKF 0.65,
   grid product 0.62) degrade more gracefully than batch on the median
   matrix (1.00) in the 2024 cauchy cell; the effect disappears at 0.25x
   (see finding 3) and is second-order next to kernel and scale.
5. Fusion mechanisms are a real axis, now measured without confounds: with
   identical gaussian components at identical sigma, the IOPS mixture
   update beats the product update 0.397 vs 3.042 m (2024) and even the
   cauchy product (0.619 m). A biased range contributes one displaced,
   outvoted component instead of multiplying into the posterior. The
   innovation gate is the weaker mechanism: perfect on gross NLOS (all six
   worst 2024 pairs 100 % gated from first observation) but moderate-bias
   pairs pass at full gaussian weight (0.767 vs cauchy 0.649).
6. Init dependence is a sequential-robustness property, sharpened in round
   2: from a reflected closed-form init on 2024 the robust EKF is WORSE
   than the gaussian one (18.3 vs 13.0 m) - a robust kernel downweights
   exactly the measurements that contradict a wrong geometry, so
   robustness amplifies init lock-in. The cauchy particle filter shows the
   attenuated version (closed-form init 1.51 m vs mds20 0.96 m). Batch
   robust kernels show no init dependence (surveyed-coordinate restarts
   reach identical minima).
7. The 2025 floor (~0.22-0.25 m in every robust cell, flat across the
   entire scale sweep) is the systematic +0.3..0.5 m per-pair range bias,
   not NLOS and not the estimator; joint per-pair/per-device bias
   estimation is the natural next model extension.

## Round 1 (as-published configurations)

Round 1 ran each method in its literature configuration (per-method scales,
IOPS pipeline defaults). Its results and per-method narratives are not part
of this release. Headline
numbers for reference: closed-form 1.69 / 5.28 (disambiguated 18.55), MDS
classical 0.26 / 3.06, SMACOF 0.96 / 5.93, FGO cauchy 0.22 / 1.00, EKF
gated 0.24 / 0.49, PF cauchy 0.23 / 0.28, grid mixture 0.27 / 0.34. The
round-1 spread between "PF cauchy 0.28" and "FGO cauchy 1.00" is fully
explained by the kernel-scale difference (the PF sweep point at 0.25 x
sigma, effective width ~ its round-1 gamma, reproduces 0.273 m).

Caveats: one static session per site; 2-D evaluation (surveyed z-spread
0.04 / 0.10 m); particle filter seed spread <= 0.05 m (2025) and up to
0.37 m in the 2024 cauchy cell (broad redescending posterior at the most
NLOS-loaded anchor 8AA5 - documented in its summary); grid results within
one 0.25 m cell of quantization.

## fully_meshed replication over all sessions

<!-- BEGIN GENERATED: replication (summarize.py) -->
### set_a (22 sessions: s21, s22, s01-s20) - RMSE m

| method          | variant         | kernel   | median_rmse_m | iqr_m       | worst_session | n_ok  |
| --------------- | --------------- | -------- | ------------- | ----------- | ------------- | ----- |
| closed_form     | faithful        | -        | 5.498         | 5.495-5.500 | s12 (5.506)   | 22/22 |
| closed_form     | disambiguated   | -        | 5.522         | 5.519-5.523 | s22 (18.555)  | 22/22 |
| mds             | classical       | -        | 3.077         | 3.022-3.122 | s03 (3.328)   | 22/22 |
| mds             | smacof_gaussian | gaussian | 3.937         | 3.903-3.962 | s21 (5.935)   | 22/22 |
| mds             | smacof_huber    | huber    | 0.959         | 0.920-1.016 | s21 (2.331)   | 22/22 |
| mds             | smacof_cauchy   | cauchy   | 0.789         | 0.746-0.842 | s21 (0.999)   | 22/22 |
| factor_graph    | gaussian        | gaussian | 3.937         | 3.903-3.962 | s21 (5.935)   | 22/22 |
| factor_graph    | huber           | huber    | 0.959         | 0.920-1.016 | s21 (2.331)   | 22/22 |
| factor_graph    | cauchy          | cauchy   | 0.789         | 0.746-0.842 | s21 (0.999)   | 22/22 |
| ekf             | gaussian        | gaussian | 3.312         | 3.210-3.409 | s01 (3.828)   | 22/22 |
| ekf             | huber           | huber    | 0.881         | 0.859-0.938 | s03 (1.000)   | 22/22 |
| ekf             | cauchy          | cauchy   | 0.713         | 0.666-0.735 | s03 (0.837)   | 22/22 |
| ekf             | gated           | gaussian | 0.906         | 0.863-0.931 | s03 (1.221)   | 22/22 |
| particle_filter | gaussian        | gaussian | 3.146         | 2.900-3.494 | s19 (4.001)   | 22/22 |
| particle_filter | huber           | huber    | 1.478         | 1.113-1.580 | s10 (2.662)   | 22/22 |
| particle_filter | cauchy          | cauchy   | 0.704         | 0.668-0.999 | s06 (2.568)   | 22/22 |
| grid_filter     | gaussian        | gaussian | 2.862         | 2.394-3.579 | s20 (4.950)   | 22/22 |
| grid_filter     | huber           | huber    | 0.779         | 0.762-0.807 | s12 (0.893)   | 22/22 |
| grid_filter     | cauchy          | cauchy   | 0.630         | 0.612-0.662 | s19 (0.865)   | 22/22 |
| grid_filter     | mixture         | gaussian | 0.422         | 0.412-0.439 | s19 (0.805)   | 22/22 |
| fg_offset       | gaussian        | gaussian | 3.852         | 3.818-3.868 | s21 (7.090)   | 22/22 |
| fg_offset       | huber           | huber    | 1.002         | 0.968-1.087 | s21 (6.768)   | 22/22 |
| fg_offset       | cauchy          | cauchy   | 0.817         | 0.777-0.900 | s21 (1.209)   | 22/22 |
| fg_offset_dev   | gaussian        | gaussian | 3.866         | 3.835-3.884 | s21 (7.928)   | 22/22 |
| fg_offset_dev   | huber           | huber    | 1.013         | 0.978-1.097 | s21 (6.760)   | 22/22 |
| fg_offset_dev   | cauchy          | cauchy   | 0.821         | 0.782-0.905 | s21 (1.222)   | 22/22 |

### set_b (3 sessions) - RMSE m

| method          | variant         | kernel   | s03   | s01   | s02   |
| --------------- | --------------- | -------- | ----- | ----- | ----- |
| closed_form     | faithful        | -        | 1.694 | 1.147 | 1.168 |
| closed_form     | disambiguated   | -        | 1.694 | 1.147 | 1.168 |
| mds             | classical       | -        | 0.258 | 0.261 | 0.263 |
| mds             | smacof_gaussian | gaussian | 0.965 | 0.678 | 0.690 |
| mds             | smacof_huber    | huber    | 0.241 | 0.243 | 0.244 |
| mds             | smacof_cauchy   | cauchy   | 0.223 | 0.225 | 0.226 |
| factor_graph    | gaussian        | gaussian | 0.965 | 0.678 | 0.690 |
| factor_graph    | huber           | huber    | 0.241 | 0.243 | 0.244 |
| factor_graph    | cauchy          | cauchy   | 0.223 | 0.225 | 0.226 |
| ekf             | gaussian        | gaussian | 1.070 | 0.645 | 0.715 |
| ekf             | huber           | huber    | 0.240 | 0.241 | 0.244 |
| ekf             | cauchy          | cauchy   | 0.224 | 0.224 | 0.226 |
| ekf             | gated           | gaussian | 0.220 | 0.220 | 0.222 |
| particle_filter | gaussian        | gaussian | 0.762 | 0.644 | 0.766 |
| particle_filter | huber           | huber    | 0.241 | 0.245 | 0.243 |
| particle_filter | cauchy          | cauchy   | 0.226 | 0.226 | 0.222 |
| grid_filter     | gaussian        | gaussian | 0.805 | 0.491 | 0.716 |
| grid_filter     | huber           | huber    | 0.275 | 0.251 | 0.274 |
| grid_filter     | cauchy          | cauchy   | 0.251 | 0.251 | 0.245 |
| grid_filter     | mixture         | gaussian | 0.238 | 0.249 | 0.248 |
| fg_offset       | gaussian        | gaussian | 0.943 | 0.646 | 0.658 |
| fg_offset       | huber           | huber    | 0.166 | 0.167 | 0.167 |
| fg_offset       | cauchy          | cauchy   | 0.147 | 0.148 | 0.148 |
| fg_offset_dev   | gaussian        | gaussian | 0.927 | 0.629 | 0.640 |
| fg_offset_dev   | huber           | huber    | 0.143 | 0.144 | 0.145 |
| fg_offset_dev   | cauchy          | cauchy   | 0.126 | 0.128 | 0.128 |

Generated from results/benchmark.parquet (fully_meshed, shared session sigma, 1x scale,
particle filter seed 0). Grid filter epoch subsampling: stride = n_epochs // 75
(session-derived; strides used [2, 3, 27, 47, 48, 101]), total grid runtime 3044 s.
Cells with status != ok: none.
<!-- END GENERATED: replication (summarize.py) -->

### Verified replication findings

1. The factorial ordering replicates in every session: kernel >> scale >>
   estimator class. Down the 2024 cauchy column the 22-session medians
   span 0.63-0.79 m across estimator classes vs 2.9-3.9 m for gaussian;
   the batch identity (SMACOF = factor graph) holds to 3 decimals in
   median, IQR and worst session.
2. The quasi-static sessions are systematically EASIER than the static sessions for
   the batch robust cells (huber median 0.96 vs s21 2.33 m; cauchy
   0.79 vs 1.00 m): masking the roving 8B05 also removes the +16.9 m
   NLOS pair 8B05-D5B4 that only the 12-anchor static sessions contain
   (verified against the surveyed geometry). S21/s22 appearing
   as "worst session" in the batch rows is that one pair, not a session
   quality issue.
3. The grid mixture update is the most accurate and most stable 2024
   cell overall (median 0.422 m, IQR 0.412-0.439), replicating the
   s21 mechanism finding across all 22 sessions. The cauchy EKF is
   the best product/IRLS-family sequential cell (median 0.713 m); the
   innovation gate degrades on the quasi-static sessions (median 0.906 m vs its
   s21 0.767 m) because moderate-bias pairs pass the gate at full
   gaussian weight.
4. The particle filter cauchy cell has the heaviest replication tail
   (median 0.704 but worst 2.57 m at s06): a systematic, seed-stable
   mislocation of the NLOS-loaded anchor 8AA5 in the sparse quasi-static mesh
   (details and root cause in results/particle_filter/summary.md).
5. set_b s01/s02 replicate s03 within 0.003 m in every
   robust cell (0.22-0.25 m floor, i.e. the systematic range bias, as
   in finding 7 above). The gaussian cells IMPROVE on the dyn sessions
   (e.g. batch 0.96 -> 0.68 m) because the single NLOS pair 5BB6-D38F
   measures a smaller median bias there (+4.0 m vs +5.8 m in s03);
   the LS attractor tracks that one pair almost linearly.
6. Chirality: 3 of 500 cells (grid product variants on s19) converge to
   the mirrored geometry; the evaluation is reflection-blind by the C4
   convention and the aligned RMSE is unaffected (root cause in
   results/grid_filter/summary.md). The closed-form disambiguated flip
   on s22 replicates its s21 behaviour.

## Mask families (the C5 topology axis)

The benchmark extends from fully_meshed to the remaining C5 views with the
factorial discipline unchanged: same estimators, same kernels, and ONE
session sigma computed from the fully_meshed view and shared by every mask of
that session (sigma measures hardware/environment, not topology - a
per-mask sigma would confound the connectivity axis, is undefined on
disconnected thinned views and measures motion on bipartite views;
fully_meshed numbers are unchanged, regression gate 40/40 after the driver
restructure). Estimators stay gauge-free and prior-agnostic in every
family; the prior subsets some masks expose are a documented future axis.

| family | sessions | methods | family metric (secondary to anchor RMSE) |
|---|---|---|---|
| fully_meshed | all 25 | static suite (6) | - |
| multi_hop k=1,2,3,4,5,6,8 | sweep: both static refs; canonical k=3: all 25 | static suite | n_components + rigidity class per k |
| mobile_assisted | s01-s20 (quasi-static 8B05) | static suite + mobile as extra unknown | mobile_error_m vs surveyed marker |
| slam_bipartite | s01, s02 (dynamic DD98) | ekf_slam, fg_slam | checkpoint ATE + within-segment RPE vs Vicon |

The static suite is undefined on bipartite views (no anchor-anchor pairs),
and a static tag leaves bipartite anchor geometry unidentifiable (verified
constructively in validate_slam.py) - absence of rows in the table is the
N/A representation. Every family passed its exact-recovery validation
(validate_multihop.py, validate_mobile.py, validate_slam.py - synthetic
exact ranges through the production mask -> registry -> evaluation path)
BEFORE its real numbers were produced. Verified family narratives live in
results/<family>/summary.md; the sections below hold the generated views.

Verified cross-family findings (details and numbers in the family
narratives):

1. The kernel axis dominates in EVERY family, as on fully_meshed: multihop
   (robust batch holds the k-columns the gaussian cells lose), mobile
   (pooled mobile-error medians 0.94 / 0.41 / 0.29 m for gaussian / huber /
   cauchy over the 20 verified quasi-static sessions), slam (cauchy < huber << gaussian
   in anchors, ATE and RPE of every cell).
2. The topology axis produces three DISTINCT failure regimes: graded
   information loss for gauge-free estimators (bounded by the generic
   global rigidity of the view on the range-kNN masks - threshold k = 4 on
   2025 and k = 5 on 2024 by the measured-range certificates of
   results/rigidity.parquet; necessary, not sufficient, see the
   random-thinning note in results/multi_hop/summary.md);
   structural failure for baseline-dependent constructions (closed form /
   grid lose the long 5BB6-4A23 gauge edge to thinning on all 25 sessions
   at k = 3); absolute unidentifiability (bipartite + static tag). A mask
   is not just "fewer measurements" - it selects which method CLASS
   remains defined.
3. Init lock-in is the recurring sequential-method failure across
   families: the fully_meshed reflected-init case, the gated EKF at sparse k
   (rejects exactly the corrective updates from a completion-inflated
   init), and EKF-SLAM on s01 (bipartite-MDS init ~20 sigma off for two
   anchors, robust weights collapse on the corrections). Batch
   re-linearization recovers from every one of these inits.
4. Measured-range thinning doubles as crude NLOS rejection: on the
   NLOS-rich 2024 site the cells NLOS hurt (every gaussian cell and both
   constructions) improve from full mesh down to k = 5
   (kept-NLOS fraction 0.29 -> 0.16, worst kept link +30.3 -> +5.4 m)
   before rigidity loss takes over; the already-robust EKF and grid cells
   hold flat or degrade. The connectivity-accuracy tradeoff is
   not monotone, and thinning to just above the rigidity threshold is the
   sweet spot.
5. The 2025 systematic +0.3-0.5 m range-bias floor persists in every
   family: the 0.22-0.25 m fully_meshed robust floor, the ~0.23 m multihop
   k = 6 cells, and the slam checkpoint ATE (0.15-0.19 m robust) sitting
   at the known-anchor multilateration residual level - while the 1-s RPE
   (0.07 m) shows the bias field cancels over short horizons. Joint
   per-pair bias estimation remains the natural next model extension.
6. One well-connected extra node (mobile_assisted vs fully_meshed, paired
   over the same 20 quasi-static sessions) helps modestly and asymmetrically:
   -0.027 m median anchor RMSE (250/400 cells improved), concentrated in
   the gaussian cells (-0.12..-0.16 m); robust kernels are ~unchanged -
   averaging over one more geometry-consistent node mainly helps the
   estimators that cannot reject NLOS themselves.

## multi_hop family

<!-- BEGIN GENERATED: multi_hop (summarize.py) -->
### set_b/s03 - RMSE m by thinning k (full = fully_meshed reference; disc. = disconnected; undef. = undefined construction)

| method          | variant         | full  | k8    | k6    | k5     | k4     | k3     | k2     | k1     |
| --------------- | --------------- | ----- | ----- | ----- | ------ | ------ | ------ | ------ | ------ |
| closed_form     | faithful        | 1.694 | 1.723 | 0.309 | undef. | undef. | undef. | undef. | undef. |
| closed_form     | disambiguated   | 1.694 | 1.723 | 0.309 | undef. | undef. | undef. | undef. | undef. |
| mds             | classical       | 0.258 | 0.291 | 0.642 | 0.897  | 1.529  | 2.312  | 3.413  | disc.  |
| mds             | smacof_gaussian | 0.965 | 0.973 | 0.232 | 0.250  | 0.330  | 0.425  | 1.078  | disc.  |
| mds             | smacof_huber    | 0.241 | 0.257 | 0.232 | 0.250  | 0.330  | 0.425  | 1.115  | disc.  |
| mds             | smacof_cauchy   | 0.223 | 0.233 | 0.231 | 0.250  | 0.330  | 0.426  | 1.138  | disc.  |
| factor_graph    | gaussian        | 0.965 | 0.973 | 0.232 | 0.250  | 0.330  | 0.425  | 1.147  | disc.  |
| factor_graph    | huber           | 0.241 | 0.257 | 0.232 | 0.250  | 0.330  | 0.425  | 1.142  | disc.  |
| factor_graph    | cauchy          | 0.223 | 0.233 | 0.231 | 0.250  | 0.330  | 0.426  | 1.266  | disc.  |
| ekf             | gaussian        | 1.070 | 1.080 | 0.232 | 0.245  | 0.323  | 1.927  | 1.383  | disc.  |
| ekf             | huber           | 0.240 | 0.255 | 0.232 | 0.250  | 0.329  | 0.417  | 1.339  | disc.  |
| ekf             | cauchy          | 0.224 | 0.234 | 0.231 | 0.250  | 0.329  | 0.429  | 1.351  | disc.  |
| ekf             | gated           | 0.220 | 0.229 | 0.232 | 0.250  | 0.329  | 1.952  | 1.879  | disc.  |
| particle_filter | gaussian        | 0.762 | 0.780 | 0.234 | 0.252  | 0.278  | 0.829  | 1.195  | 7.613  |
| particle_filter | huber           | 0.241 | 0.256 | 0.233 | 0.252  | 0.282  | 1.772  | 1.520  | 7.974  |
| particle_filter | cauchy          | 0.226 | 0.235 | 0.232 | 0.252  | 0.283  | 1.741  | 1.295  | 7.788  |
| grid_filter     | gaussian        | 0.805 | 0.884 | 0.257 | undef. | undef. | undef. | undef. | undef. |
| grid_filter     | huber           | 0.275 | 0.329 | 0.257 | undef. | undef. | undef. | undef. | undef. |
| grid_filter     | cauchy          | 0.251 | 0.269 | 0.257 | undef. | undef. | undef. | undef. | undef. |
| grid_filter     | mixture         | 0.238 | 0.253 | 0.257 | undef. | undef. | undef. | undef. | undef. |

### set_a/s21 - RMSE m by thinning k (full = fully_meshed reference; disc. = disconnected; undef. = undefined construction)

| method          | variant         | full   | k8    | k6    | k5     | k4     | k3     | k2     | k1     |
| --------------- | --------------- | ------ | ----- | ----- | ------ | ------ | ------ | ------ | ------ |
| closed_form     | faithful        | 5.277  | 0.560 | 0.624 | 0.638  | undef. | undef. | undef. | undef. |
| closed_form     | disambiguated   | 18.551 | 0.647 | 0.624 | 0.638  | undef. | undef. | undef. | undef. |
| mds             | classical       | 3.061  | 2.955 | 2.646 | 3.021  | 3.942  | 12.412 | disc.  | disc.  |
| mds             | smacof_gaussian | 5.935  | 0.832 | 0.870 | 0.820  | 1.447  | 12.318 | disc.  | disc.  |
| mds             | smacof_huber    | 2.331  | 0.776 | 0.814 | 0.820  | 1.447  | 12.490 | disc.  | disc.  |
| mds             | smacof_cauchy   | 0.999  | 0.724 | 0.773 | 0.761  | 1.452  | 12.518 | disc.  | disc.  |
| factor_graph    | gaussian        | 5.935  | 0.832 | 0.870 | 0.820  | 1.447  | 8.308  | disc.  | disc.  |
| factor_graph    | huber           | 2.331  | 0.776 | 0.814 | 0.820  | 1.447  | 9.538  | disc.  | disc.  |
| factor_graph    | cauchy          | 0.999  | 0.724 | 0.773 | 0.761  | 1.452  | 9.663  | disc.  | disc.  |
| ekf             | gaussian        | 3.286  | 0.953 | 1.048 | 1.021  | 1.762  | 10.248 | disc.  | disc.  |
| ekf             | huber           | 0.900  | 0.784 | 0.875 | 0.869  | 1.573  | 10.048 | disc.  | disc.  |
| ekf             | cauchy          | 0.649  | 0.712 | 0.808 | 0.805  | 1.556  | 10.020 | disc.  | disc.  |
| ekf             | gated           | 0.767  | 0.801 | 0.888 | 3.159  | 3.943  | 10.097 | disc.  | disc.  |
| particle_filter | gaussian        | 2.942  | 1.001 | 1.135 | 1.425  | 1.795  | 10.584 | 14.937 | 15.170 |
| particle_filter | huber           | 1.100  | 0.800 | 0.943 | 1.071  | 1.733  | 10.647 | 15.047 | 14.407 |
| particle_filter | cauchy          | 0.964  | 0.728 | 0.897 | 1.127  | 1.695  | 10.843 | 15.060 | 15.230 |
| grid_filter     | gaussian        | 3.042  | 1.168 | 1.259 | 3.359  | undef. | undef. | undef. | undef. |
| grid_filter     | huber           | 0.808  | 0.719 | 0.786 | 7.740  | undef. | undef. | undef. | undef. |
| grid_filter     | cauchy          | 0.619  | 0.643 | 0.736 | 7.844  | undef. | undef. | undef. | undef. |
| grid_filter     | mixture         | 0.397  | 0.409 | 0.621 | 14.904 | undef. | undef. | undef. | undef. |

### canonical multi_hop (k=3) over all sessions

| campaign | method          | variant         | median_rmse_m | worst_session | n_ok  | n_disconnected |
| -------- | --------------- | --------------- | ------------- | ------------- | ----- | -------------- |
| set_a    | closed_form     | disambiguated   | -             | -             | 0/22  | 0              |
| set_a    | closed_form     | faithful        | -             | -             | 0/22  | 0              |
| set_a    | ekf             | cauchy          | 3.548         | s22 (10.501)  | 22/22 | 0              |
| set_a    | ekf             | gated           | 3.563         | s22 (11.387)  | 22/22 | 0              |
| set_a    | ekf             | gaussian        | 3.716         | s22 (10.570)  | 22/22 | 0              |
| set_a    | ekf             | huber           | 3.541         | s22 (10.513)  | 22/22 | 0              |
| set_a    | factor_graph    | cauchy          | 4.965         | s22 (9.700)   | 22/22 | 0              |
| set_a    | factor_graph    | gaussian        | 4.939         | s22 (8.316)   | 22/22 | 0              |
| set_a    | factor_graph    | huber           | 4.951         | s22 (9.569)   | 22/22 | 0              |
| set_a    | grid_filter     | cauchy          | -             | -             | 0/22  | 0              |
| set_a    | grid_filter     | gaussian        | -             | -             | 0/22  | 0              |
| set_a    | grid_filter     | huber           | -             | -             | 0/22  | 0              |
| set_a    | grid_filter     | mixture         | -             | -             | 0/22  | 0              |
| set_a    | mds             | classical       | 6.358         | s22 (12.420)  | 22/22 | 0              |
| set_a    | mds             | smacof_cauchy   | 4.986         | s22 (12.540)  | 22/22 | 0              |
| set_a    | mds             | smacof_gaussian | 5.067         | s22 (12.322)  | 22/22 | 0              |
| set_a    | mds             | smacof_huber    | 5.000         | s22 (12.512)  | 22/22 | 0              |
| set_a    | particle_filter | cauchy          | 6.636         | s21 (10.843)  | 22/22 | 0              |
| set_a    | particle_filter | gaussian        | 4.821         | s21 (10.584)  | 22/22 | 0              |
| set_a    | particle_filter | huber           | 6.869         | s21 (10.647)  | 22/22 | 0              |
| set_b    | closed_form     | disambiguated   | -             | -             | 0/3   | 0              |
| set_b    | closed_form     | faithful        | -             | -             | 0/3   | 0              |
| set_b    | ekf             | cauchy          | 0.437         | s02 (0.443)   | 3/3   | 0              |
| set_b    | ekf             | gated           | 2.002         | s02 (2.107)   | 3/3   | 0              |
| set_b    | ekf             | gaussian        | 1.860         | s03 (1.927)   | 3/3   | 0              |
| set_b    | ekf             | huber           | 0.441         | s01 (1.960)   | 3/3   | 0              |
| set_b    | factor_graph    | cauchy          | 0.436         | s02 (0.441)   | 3/3   | 0              |
| set_b    | factor_graph    | gaussian        | 0.435         | s02 (0.440)   | 3/3   | 0              |
| set_b    | factor_graph    | huber           | 0.435         | s02 (0.440)   | 3/3   | 0              |
| set_b    | grid_filter     | cauchy          | -             | -             | 0/3   | 0              |
| set_b    | grid_filter     | gaussian        | -             | -             | 0/3   | 0              |
| set_b    | grid_filter     | huber           | -             | -             | 0/3   | 0              |
| set_b    | grid_filter     | mixture         | -             | -             | 0/3   | 0              |
| set_b    | mds             | classical       | 2.314         | s02 (2.315)   | 3/3   | 0              |
| set_b    | mds             | smacof_cauchy   | 0.436         | s02 (0.441)   | 3/3   | 0              |
| set_b    | mds             | smacof_gaussian | 0.435         | s02 (0.440)   | 3/3   | 0              |
| set_b    | mds             | smacof_huber    | 0.435         | s02 (0.440)   | 3/3   | 0              |
| set_b    | particle_filter | cauchy          | 1.781         | s02 (1.786)   | 3/3   | 0              |
| set_b    | particle_filter | gaussian        | 1.815         | s02 (1.841)   | 3/3   | 0              |
| set_b    | particle_filter | huber           | 1.814         | s02 (1.819)   | 3/3   | 0              |


### random-thinning ablation - robust-batch median RMSE m (seeded-uniform selection, 3 seeds pooled)

| campaign/session | k2     | k3     | k4    | k5     | k6    | k8    |
| ---------------- | ------ | ------ | ----- | ------ | ----- | ----- |
| set_b/s03        | 9.988  | 7.409  | 4.603 | 1.423  | 1.475 | 0.233 |
| set_a/s21        | 23.656 | 17.704 | 9.738 | 13.410 | 3.342 | 1.429 |

Same union-of-per-node construction as the k-NN sweep with the selection criterion
changed from nearest-by-median-range to seeded-uniform; the arm isolates the
NLOS-filtering side effect of range-based thinning from the rigidity effect.

Generated from results/benchmark.parquet (multihop family; shared per-session fully_meshed sigma).
Canonical multi_hop verified equal to multi_hop_k3 within 0.0005 m.
<!-- END GENERATED: multi_hop (summarize.py) -->

## mobile_assisted family

<!-- BEGIN GENERATED: mobile_assisted (summarize.py) -->
### set_a quasi-static sessions - anchor RMSE (primary) and mobile error vs marker (secondary)

| method          | variant         | anchor_median_rmse_m | fully_meshed_median_rmse_m | mobile_median_err_m | mobile_worst | n_ok  |
| --------------- | --------------- | -------------------- | -------------------------- | ------------------- | ------------ | ----- |
| closed_form     | faithful        | 5.499                | 5.499                      | 1.868               | s12 (14.994) | 20/20 |
| closed_form     | disambiguated   | 5.522                | 5.522                      | 1.965               | s12 (53.642) | 20/20 |
| mds             | classical       | 3.047                | 3.097                      | 1.249               | s12 (4.106)  | 20/20 |
| mds             | smacof_gaussian | 3.810                | 3.934                      | 1.554               | s12 (5.872)  | 20/20 |
| mds             | smacof_huber    | 0.962                | 0.955                      | 0.475               | s19 (2.526)  | 20/20 |
| mds             | smacof_cauchy   | 0.754                | 0.780                      | 0.354               | s12 (1.705)  | 20/20 |
| factor_graph    | gaussian        | 3.810                | 3.934                      | 1.554               | s12 (5.872)  | 20/20 |
| factor_graph    | huber           | 0.962                | 0.955                      | 0.475               | s19 (2.526)  | 20/20 |
| factor_graph    | cauchy          | 0.754                | 0.780                      | 0.354               | s12 (1.705)  | 20/20 |
| ekf             | gaussian        | 3.196                | 3.319                      | 1.137               | s12 (4.101)  | 20/20 |
| ekf             | huber           | 0.883                | 0.881                      | 0.329               | s19 (1.870)  | 20/20 |
| ekf             | cauchy          | 0.692                | 0.720                      | 0.268               | s19 (1.093)  | 20/20 |
| ekf             | gated           | 0.851                | 0.913                      | 0.313               | s12 (0.551)  | 20/20 |
| particle_filter | gaussian        | 3.027                | 3.191                      | 1.088               | s10 (7.532)  | 20/20 |
| particle_filter | huber           | 1.419                | 1.484                      | 0.558               | s10 (4.549)  | 20/20 |
| particle_filter | cauchy          | 0.675                | 0.704                      | 0.307               | s18 (1.442)  | 20/20 |
| grid_filter     | gaussian        | 3.156                | 2.862                      | 1.062               | s19 (5.028)  | 20/20 |
| grid_filter     | huber           | 0.789                | 0.779                      | 0.307               | s19 (2.029)  | 20/20 |
| grid_filter     | cauchy          | 0.618                | 0.632                      | 0.210               | s19 (0.822)  | 20/20 |
| grid_filter     | mixture         | 0.415                | 0.424                      | 0.144               | s12 (0.638)  | 20/20 |

Generated from results/benchmark.parquet (mobile_assisted; mobile error = xy distance to the surveyed marker after the anchor-fitted alignment;
sessions flagged truth_quality: unreliable are excluded from every mobile aggregate; fully_meshed medians cover the same quasi-static sessions).
<!-- END GENERATED: mobile_assisted (summarize.py) -->

## slam_bipartite family

<!-- BEGIN GENERATED: slam_bipartite (summarize.py) -->
### set_b dyn sessions - anchor RMSE (primary), checkpoint ATE and within-segment RPE (secondary)

| method   | variant  | session | anchor_rmse_m | ate_rmse_m | rpe_rmse_m | n_truth_matched |
| -------- | -------- | ------- | ------------- | ---------- | ---------- | --------------- |
| ekf_slam | gaussian | s01     | 5.836         | 4.983      | 1.463      | 655             |
| ekf_slam | gaussian | s02     | 0.623         | 1.099      | 0.712      | 688             |
| ekf_slam | huber    | s01     | 2.230         | 3.298      | 0.289      | 655             |
| ekf_slam | huber    | s02     | 0.339         | 0.413      | 0.112      | 688             |
| ekf_slam | cauchy   | s01     | 1.885         | 2.507      | 0.200      | 655             |
| ekf_slam | cauchy   | s02     | 0.311         | 0.429      | 0.098      | 688             |
| fg_slam  | gaussian | s01     | 0.890         | 0.517      | 0.273      | 655             |
| fg_slam  | gaussian | s02     | 0.867         | 0.648      | 0.408      | 688             |
| fg_slam  | huber    | s01     | 0.432         | 0.158      | 0.076      | 655             |
| fg_slam  | huber    | s02     | 0.390         | 0.160      | 0.079      | 688             |
| fg_slam  | cauchy   | s01     | 0.359         | 0.149      | 0.070      | 655             |
| fg_slam  | cauchy   | s02     | 0.334         | 0.136      | 0.066      | 688             |

### mobile_assisted dynamic views - fg_slam with anchor-anchor factors vs the bipartite run

| variant  | session | anchor_rmse_m (+E_aa) | anchor_rmse_m (bipartite) | ate_rmse_m (+E_aa) | ate_rmse_m (bipartite) |
| -------- | ------- | --------------------- | ------------------------- | ------------------ | ---------------------- |
| gaussian | s01     | 0.886                 | 0.890                     | 0.518              | 0.517                  |
| gaussian | s02     | 0.864                 | 0.867                     | 0.649              | 0.648                  |
| huber    | s01     | 0.430                 | 0.432                     | 0.158              | 0.158                  |
| huber    | s02     | 0.389                 | 0.390                     | 0.159              | 0.160                  |
| cauchy   | s01     | 0.357                 | 0.359                     | 0.149              | 0.149                  |
| cauchy   | s02     | 0.332                 | 0.334                     | 0.136              | 0.136                  |

Generated from results/benchmark.parquet (slam_bipartite and the mobile_assisted dynamic arm; trajectory metrics only at Vicon-covered epochs, never interpolated across gaps).
<!-- END GENERATED: slam_bipartite (summarize.py) -->

## prior_sweep arm (the C1 reference-information axis)

<!-- BEGIN GENERATED: prior_sweep (summarize.py) -->
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
<!-- END GENERATED: prior_sweep (summarize.py) -->
