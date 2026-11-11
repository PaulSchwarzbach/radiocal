# multi_hop - connectivity-thinned anchor mesh (C5 family)

The full anchor mesh thinned to each anchor's k nearest neighbours by median
measured range (union semantics: a pair survives if either endpoint lists
it), emulating limited radio range / multi-hop deployments. Same static
estimator suite and shared per-session sigma as fully_meshed; the new factorial
axis is connectivity k (sweep via parameterized masks multi_hop_k1..8,
canonical YAML k = 3, fully_meshed = k = infinity reference).

Under thinning, shortest-path completion (classical MDS / all MDS inits)
systematically overestimates the missing distances, and at low k the graph
loses (global) rigidity or disconnects - both structural outcomes recorded,
not errors. Validation and rigidity certificates: validate_multihop.py,
rigidity.py.

<!-- BEGIN GENERATED (summarize.py) -->
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
<!-- END GENERATED (summarize.py) -->

## Notes

Validation (validate_multihop.py, exit 0; exact synthetic ranges through the
production mask -> registry -> evaluation path): on exact ranges the kNN
thinning selects by TRUE distance, and the global-rigidity threshold of
those synthetic views is k = 4 on both geometries. The REAL views thin by
measured median range, and their certificates differ on set_a
(results/rigidity.parquet, run_rigidity.py, all 25 sessions): set_b is
rigid-but-not-globally-rigid at k = 3 and globally rigid from k = 4;
set_a is FLEXIBLE at k = 3, rigid-but-not-globally-rigid at k = 4 (not
3-connected) and globally rigid from k = 5; k = 1 disconnects both. On
exact data the classical-MDS error tracks the shortest-path completion
inflation almost linearly (2025 k = 2..8: RMSE 3.04..0.016 m vs max
inflation 7.99..0.09 m) - completion IS its error mechanism. The factor
graph is machine-exact (3e-15..1e-14 m) everywhere globally rigid, and
SMACOF run to its stopping rule reaches <= 3e-11 m there (revised
2026-09-30: until then the solver stopped at a fixed 500-iteration budget,
which left it unconverged on every thinned real view and produced the
0.19-0.21 m "stall" formerly reported for 2025 k3); the per-anchor
conditional PF keeps an error on sparse views (0.12-0.23 m at 2025 k4) that
is not separated from slow convergence.

Verified observations on the real data:

1. Baseline-dependent constructions fail STRUCTURALLY, not gradually. kNN
   thinning by measured range removes long edges first, and the 5BB6-4A23
   gauge baseline is long by design: it survives only k >= 6 on 2025 and
   k >= 5 on 2024 (verified on the real s21 views), and is absent from
   the canonical k = 3 view of all 25/25 sessions - exactly the 150
   closed_form + grid_filter error cells ("only 1 common ids"). Where the
   baseline survives but gauge visibility is sparse, the grid locks onto
   wrong modes with every anchor placed (2024 k5: 3.4-14.9 m). The
   gauge-free suite is untouched by this mechanism.
2. Classical MDS degrades by completion inflation, monotonically: 2025
   0.258 (full) -> 0.291 / 0.642 / 0.897 / 1.529 / 2.312 / 3.413 m for
   k = 8..2 - the real k-curve mirrors the synthetic inflation stats. On
   2024 it collapses to 12.4 m at k3 where flexibility compounds the
   completion error.
3. Robust batch methods track the rigidity class on the range-kNN views:
   on 2025 they hold 0.23-0.33 m over the entire globally rigid range
   (k >= 4), sit at 0.43 m on the rigid-not-globally-rigid k3 view (correct
   basin, uncertifiable; converged SMACOF and the factor graph agree to
   2e-6 m), jump to 1.1-1.3 m on the flexible k2 view and disconnect at k1.
   On 2024: 0.72-0.82 m while globally rigid (k >= 5), 1.45 m on the
   rigid-not-globally-rigid k4 view, 8.3-12.5 m on the flexible k3 view,
   where the factor graph and SMACOF reach the same objective value at
   different points of the flex. The canonical k = 3 mask is therefore a
   deliberate stress case on both sites; the threshold is k = 4 on 2025
   and k = 5 on 2024. The random-thinning arm shows the class is necessary,
   not sufficient (note 8).
4. Measured-range thinning is a crude NLOS filter - the connectivity-
   accuracy tradeoff is NOT monotone on an NLOS-rich site. NLOS-inflated
   links look long and are dropped preferentially: on set_a/s21 the
   kept-NLOS fraction falls 0.29 (full, worst +30.3 m) -> 0.23 (k8) ->
   0.16 (k5) -> 0.10 (k3, worst kept +1.0 m), and the cells NLOS hurt
   IMPROVE from full mesh down to k5 (gaussian batch 5.93 -> 0.83 m at k8;
   cauchy batch 1.00 -> 0.72-0.76 m) before rigidity loss destroys the
   estimate below k5. The closed_form drop 5.28 -> 0.56 m at k8 is NOT
   part of this effect: it is scored on the 9 of 12 anchors the
   construction still places, and the full-mesh estimate scored on the
   same 9 anchors gives the identical 0.56 m (the unplaced 8AA5 alone
   carries 16.2 m). The already-robust
   EKF and grid cells hold flat or degrade (grid wrong-mode lock at k5,
   see point above). On 2025 the single NLOS
   pair 5BB6-D38F is dropped from k6 on (verified), moving every gaussian
   cell from ~0.97-1.08 to ~0.23 m - the k6 column beats the full mesh.
   Sweet spot: thin to just above the global-rigidity threshold.
5. Sequential estimators add their own failure modes at sparse k. At 2025
   k3 the gaussian EKF (1.93 m) and the gated EKF (1.95 m) stay near the
   completion-inflated classical-MDS init (2.31 m, verified) while huber/
   cauchy EKF recover to 0.42-0.43 m - the validation's convergence
   mechanism (robust R inflation delays premature covariance collapse from
   a biased init; the hard gate instead rejects exactly the corrective
   updates). The PF ends at 1.7-1.8 m (2025 k3 canonical), the real-data
   counterpart of its error on noise-free thinned views, which is not
   separated from slow convergence.
6. The seven k2 reflection flags on 2025 are chirality bookkeeping of a
   FLEXIBLE view (mirror-equivalent realizations exist by the certificate;
   the same sessions' fully_meshed rows do not flip); the evaluation is
   reflection-blind, so the recorded 1.1-1.3 m is the flex ambiguity, not
   the mirror.
7. Family hypothesis verdict: CONFIRMED on the range-kNN views, with
   sharpened mechanisms - the closed form collapses structurally (gauge
   baseline), classical MDS collapses metrically (shortest-path
   completion), and the robust batch methods hold as long as the thinned
   graph stays generically globally rigid. Global rigidity is necessary,
   not sufficient (note 8).
8. Random thinning separates identifiability from solvability
   (results/oracle_start.parquet, run_controls.py; cauchy factor graph).
   31 of the 36 random views are globally rigid (all 30 at k >= 3 and one
   k2 view; the union of random picks keeps more pairs than kNN at the
   same k), yet the benchmark cells fail at metre level on most of them. Started from the surveyed geometry instead of the MDS
   completion, the same problem gives 0.21-0.44 m on all 16 globally rigid
   2025 views, at a lower robust cost on each of the 11 views where the two
   starts differ: there the initialization decides, not the view. On 2024
   the two starts differ on 9 of 15 globally rigid views, the surveyed
   start still ends at 1.1-9.5 m and reaches the lower cost on only 4 of
   them: the random views keep the gross NLOS pairs that range-kNN thinning
   drops, and the robust objective itself no longer favours the true
   geometry. On every rigid kNN view both starts reach the same minimum.
   On the flexible 2024 k3 view the surveyed start gives 0.43 m against
   9.66 m from the MDS start at the same cost - the error there measures
   the start, not the data.
