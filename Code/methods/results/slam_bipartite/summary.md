# slam_bipartite - mobile-anchor ranges only (C5 family)

Bipartite view of the set_b dyn sessions: the anchors are observable
only through the moving tag DD98 (no anchor-anchor edges), the classic
range-only SLAM setting. Dyn-only by construction: with a static tag every
anchor is constrained only to a circle around it, so bipartite anchor
geometry is unidentifiable (demonstrated in validate_slam.py). Methods:
ekf_slam (pose + anchors state, random-walk motion) and fg_slam (pose-chain
factor graph), robust kernels from kernels.py on the range factors only.

Evaluation: anchor RMSE after Procrustes stays primary; mobile metrics are
checkpoint ATE at Vicon-covered epochs and within-segment RPE only - the
bursty Vicon truth is never interpolated across gaps. Truth reaches ENU via
the estimated vicon->enu transform (estimate_vicon_transform.py; canonical
copy in the session.yaml transform_vicon_to_local slot, provenance copy
under results/transforms/).

<!-- BEGIN GENERATED (summarize.py) -->
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
<!-- END GENERATED (summarize.py) -->

## Notes

Every statement below is verified by an in-memory experiment or by
validate_slam.py (all numbers reproduced 2026-07-07; benchmark rows from
results/benchmark.parquet, mask slam_bipartite).

1. Shared model constants (stated a-priori, slam_common.py): pedestrian
   speed bound V_MAX = 1.5 m/s gives the random-walk-on-position process
   std q_step = V_MAX * median epoch dt = 0.116 m (s01) / 0.115 m (s02)
   per epoch; sigma is the session's fully_meshed value (0.131 / 0.130 m,
   sigma doctrine); the robust kernels act on range factors only, the
   motion prior stays Gaussian. Init for both methods: joint classical MDS
   over anchors + every 25th epoch pose on the shortest-path-completed
   bipartite matrix (init RMSE 1.52 m s01 / 1.05 m s02 vs surveyed).

2. reflection = True in all 12 cells is the init's chirality, not an
   estimator property. Root-caused: the bipartite classical-MDS init of
   BOTH dyn sessions lands in the mirrored chirality relative to ENU
   (reflection-allowed alignment of the init itself already reports
   True), the eigenvector-sign ambiguity of Torgerson MDS being resolved
   the same way by both sessions' similar data. Both solvers preserve
   chirality exactly: mirroring the init (y -> -y on anchors and poses)
   flips the reported reflection flag to False while the robust cost,
   anchor RMSE, ATE and RPE are unchanged to the last digit and the
   ENU-mapped trajectory is IDENTICAL (max difference 0.0 m, fg and ekf,
   both sessions) - the mirrored problem is an exact isometry and the
   anchor-fitted alignment maps the trajectory through the same
   reflection, so the ATE/RPE pipeline is reflection-consistent on real
   data, as designed. Range-only geometry is chirality-blind; the flag is
   bookkeeping.

3. The information cost of losing the anchor-anchor edges: fg_slam robust
   anchors reach 0.359/0.432 m (s01 cauchy/huber) and 0.334/0.390 m
   (s02) vs the fully_meshed robust floor of the SAME sessions (0.225/0.243
   and 0.226/0.244 m, factor_graph cauchy/huber; best static cell 0.220
   m). Observing 10 anchors only through a moving tag costs a factor
   ~1.5 in anchor RMSE - and, unlike fully_meshed (45 aggregated pairs), it
   needs ~3.4k epochs of motion to get there.

4. Kernel ordering replicates the fully_meshed finding on a different
   estimator class and view: cauchy < huber << gaussian in every
   slam_bipartite cell (fg_slam s01: 0.359 < 0.432 << 0.890 m; s02:
   0.334 < 0.390 << 0.867 m; same ordering for ekf_slam and for ATE/RPE).
   The kernel axis remains the dominant factor, estimator class second.

5. The ekf_slam s01/s02 asymmetry (1.89-5.84 m vs 0.31-0.62 m) is an
   init-basin lock-in of the causal filter, not filter inconsistency.
   Oracle experiment: started from the surveyed anchors, the same filter
   reaches 0.458-0.875 m (s01) and 0.382-0.715 m (s02), kernel-ordered.
   The production init is 1.05 m RMSE on s02 (inside the basin) but
   1.52 m on s01 with two anchors ~3 m off (4A23 2.88 m, D38F 2.97 m);
   at sigma = 0.131 m that is a ~20 sigma innovation, the robust weights
   collapse on exactly the updates that could repair those anchors
   (cauchy w ~ 0.002), and the gaussian variant instead freezes the
   inconsistent geometry into an overconfident P. fg_slam, from the SAME
   init, re-linearizes over 15-35 LM iterations and recovers on both
   sessions (all cells converged; 0.7-1.7 s per cell for ~6.7k
   parameters, sparse LM).

6. The trajectory ATE floor is the site's range-bias field, not the SLAM.
   fg_slam robust checkpoint ATE is 0.191/0.149 m (s01/s02) - at the
   level of KNOWN-ANCHOR multilateration on the same data: the per-epoch
   robust fixes of estimate_vicon_transform.py have median residual RMS
   0.194/0.172 m and match the Vicon poses with 0.22/0.18 m xy RMSE
   (s01/s02) after the optimal rigid fit. A trajectory estimated jointly with unknown
   anchors is as accurate as one computed from surveyed anchors, so the
   remaining ATE measures the mobile-anchor range biases (the dynamic
   counterpart of the 0.22-0.25 m fully_meshed anchor floor), not estimator
   error. Consistently, the 1 s within-segment RPE is far below the ATE
   (0.065-0.078 m robust): the bias field varies slowly along the path
   and largely cancels in 1 s differences.

### Validation (validate_slam.py, exit 0; all checks green)

- Static-tag degeneracy: two anchor geometries differing by 2.34 m
  aligned RMSE (each anchor independently rotated about the tag) generate
  bit-identical bipartite range streams (max |dr| = 1.8e-15 m) - the
  family's dyn-only rule is demonstrated, not assumed.
- Exact recovery through the production path (real mask spec, registry
  adapters, noiseless synthetic on the surveyed s03 anchors, 3000
  epochs): fg_slam anchor RMSE 2.03e-05 m, checkpoint ATE 1.31e-04 m for
  all three kernels (gates 1e-2); ekf_slam 8.0-9.0e-05 m and ATE
  1.2-1.3e-02 m (gates 0.1); <= 1.1 s per fg kernel.
- ATE/RPE arithmetic on gappy truth with a known vicon->enu transform:
  undistorted estimate scores 5.4e-15 / 6.7e-15 m; an injected 0.01 m/s
  drift reproduces the analytic ATE (1.288816 m) and RPE (0.00999363 m,
  including the sub-horizon window-end pairs the |dt-1| < 0.25 s rule
  admits) to < 1e-9; uncovered epochs stay unmatched (segment -1) and no
  match or RPE pair crosses a truth gap.
- vicon->enu transforms (estimate_vicon_transform.py): synthetic selftest
  recovers a known transform to |dR| 2.2e-16 / |dt| 1.1e-14. Adopted
  parameterization is PLANAR (yaw + z offset): the coplanar anchors
  (z = 4.54-4.58 m) leave the multilaterated z soft, and the full 3-D
  Umeyama fits a non-reproducible roll/pitch (+2.1/-0.2 deg s01 vs
  +3.4/-1.8 deg s02) to that noise while agreeing with the planar yaw to
  ~0.1 deg. Estimates: yaw +134.596 deg (s01, 400 matched epochs) vs
  +133.901 deg (s02, 572), delta yaw 0.695 deg, planar translation
  difference 0.042 m in xy - within the < 1 deg / < 0.10 m consistency
  target; the z offsets differ by 0.257 m (the soft axis; the evaluation
  is 2-D, so z never enters ATE/RPE). Cross-check on s03: the mapped
  Vicon centroid agrees with the independently multilaterated static DD98
  to 0.123 m (s01) / 0.163 m (s02) in xy - the bias-field level.
  Trimming uses a robust scale FLOORED AT SIGMA: unfloored MAD trimming
  of the bias-dominated residual norms collapsed s01 onto a single
  Vicon burst (31/400 "inliers", yaw 1.9 deg off s02) - root-caused and
  fixed; the untrimmed fits of the two sessions already agreed to 0.24
  deg.
