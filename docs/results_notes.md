# Notes on the released results

This file traces a few anomalies in the result tables to their causes
(Sections 1 to 4) and gives the numbers behind four statements that the
companion paper makes without them (Section 5). Every number is read from
the released tables under `Code/methods/results/` or from `Data/`, and
recomputed where stated. Sessions are addressed as `<set>/<session>`, and
"the factor graph" means `factor_graph` with the cauchy kernel. The notes
under `Code/methods/results/`, and the scripts of `Code/methods` that write
parts of them, name the sets by year: "2024" is set A, the NLOS deployment,
which was recorded on 2025-03-13, and "2025" is set B, the LOS deployment.

## 1. `set_a/s21`: three gross anchor pairs

On the fully meshed view the factor graph scores 1.00 m on `set_a/s21`,
against a median of 0.78 m over the 20 quasi-static sessions of set A and
0.81 m on the second static session `s22`. Three anchor pairs of `s21` range
far above their surveyed distance (median of the symmetrized ranges minus
the surveyed distance):

| pair | `s21` | `s22` | remark |
|---|---|---|---|
| 8AA5–9911 | +30.3 m | not observed | 5 epochs, in no other set A session |
| 5388–9911 | +16.6 m | not observed | 279 epochs, in no other set A session |
| 8B05–D5B4 | +16.9 m | +17.5 m | |

The pair 8B05–D5B4 is equally biased on `s22`, which is not elevated, so the
two pairs seen only on `s21` set it apart. Removing all three pairs from the
session-median range matrix and re-solving at the same noise scale
(1.62 m) gives 0.76 m, from the original start and from a start recomputed on
the reduced matrix alike. This re-solve is not part of the result tables.

## 2. `set_a/s06`: the particle filter and anchor 8AA5

The particle filter places anchor 8AA5 of `set_a/s06` far off its surveyed
position in all five seeds, by 3.7 to 6.8 m with the cauchy kernel on the
fully meshed view, and 8AA5 is the worst anchor of every seed. The same
holds for the other kernels (huber 4.8 to 5.6 m, gaussian 8.2 to 8.4 m) and
on the mobile-assisted view. The table cells average the five seeds
(`benchmark_estimates`, column `error_m`).

## 3. Reflection flags

The column `reflection` records whether the best rigid alignment of an
estimate to the surveyed anchors needs a reflection. Ranges cannot tell a
geometry from its mirror image, and the evaluation aligns with reflections
allowed (`Code/methods/evaluation.py`), so a flag never changes `rmse_m`. It
records which of the two chiralities an estimate ended in.

**Fully meshed views.** Five cells carry the flag: the gaussian, huber and
cauchy grid filters on `set_a/s19`, which converge to the mirrored geometry,
and the disambiguated closed form on `set_a/s21` and `s22`, whose sign test
is fooled by the non-line-of-sight pattern of the static set A sessions
(18.55 m on `s21` against 5.28 m for the faithful variant). The gauge-free
batch and filter methods never flip on a fully meshed view.

**Thinning sweep.** Flags per view of the range-thinning sweep, with the
rigidity class from `results/rigidity` and the chirality of the
classical-MDS start of the batch methods (the factor graph and the
multidimensional scaling, both started on the session-median range matrix).
The EKF and the particle filter start from a classical embedding of their
first 20 epochs instead.

| view | rigidity class | start mirrored | flagged / ok cells | flagged methods |
|---|---|---|---|---|
| `set_b/s03` κ=1 | disconnected | (no start) | 9 / 15 | particle filter |
| `set_b/s03` κ=2 | flexible | yes | 26 / 26 | MDS, factor graph, EKF, particle filter |
| `set_b/s03` κ=3 | rigid, not globally | no | 0 / 26 | |
| `set_b/s03` κ=4, 5 | globally rigid | yes | 26 / 26 each | MDS, factor graph, EKF, particle filter |
| `set_b/s03` κ=6, 8 | globally rigid | yes | 26 / 32 each | MDS, factor graph, EKF, particle filter |
| `set_a/s21` κ=1, 2 | disconnected | (no start) | 4 / 15, 10 / 15 | particle filter |
| `set_a/s21` κ=3 | flexible | no | 0 / 26 | |
| `set_a/s21` κ=4 | rigid, not globally | no | 0 / 26 | |
| `set_a/s21` κ=5 | globally rigid | no | 20 / 32 | EKF, particle filter, grid filter (mixture) |
| `set_a/s21` κ=6 | globally rigid | yes | 7 / 32 | MDS, factor graph |
| `set_a/s21` κ=8 | globally rigid | no | 0 / 32 | |

On `set_b/s03` the flags of the batch methods follow the chirality of their
start, and the EKF and the particle filter end in the same chirality. On
`set_a/s21` the batch methods follow their start as well, with flags at κ=6
only, while the EKF and the particle filter end mirrored at κ=5 and
unmirrored at κ=6. Apart from the mixture grid filter at κ=5, which starts
from its own initialization, the closed form and the grid filter never carry
the flag in the sweep.

**Mobile-assisted views of `set_a/s15` and `s16`.** On both sessions 32 of
the 38 ok cells carry the flag (the multidimensional scaling, the factor
graph and its offset variants, the EKF and the particle filter), against 0
of 38 on the fully meshed views of the same sessions. Appending the mobile
node 8B05 to the completed distance matrix mirrors the classical-MDS start
of the batch methods, which is unmirrored on the fully meshed views, and the
filters end in the same chirality. The closed form and the grid filter keep
their own constructions and carry no flag.

**Mobile-only views.** All 12 cells of the set B walks (EKF-SLAM and
factor-graph SLAM, three kernels each) carry the flag. Both estimators start
from the shared bipartite embedding, and mirroring it changes neither the
cost nor the aligned error, because a reflection is an exact isometry of the
bipartite range problem.

## 4. The particle filter on disconnected views

The rigidity table certifies three views of the thinning sweep as
disconnected: `set_b/s03` at κ=1 and `set_a/s21` at κ=1 and κ=2. On these
views the multidimensional scaling, the factor graph and the EKF record the
status `disconnected`, and the closed form and the grid filter `undefined`.
The particle filter returns status `ok` on all 15 of its rows per view (three
kernels, five seeds), with errors of 6.5 to 8.0 m on `set_b/s03` and 14.1 to
16.9 m on `set_a/s21`. The released `benchmark` table keeps that status. The
paper counts these runs as structural outcomes and draws them as such in its
κ-sweep figure, so a reader who aggregates the table should exclude them in
the same way.

## 5. The numbers behind four statements of the paper

Section VI of the companion paper states the following results without
their numbers. Each is recomputed from the released files, with the
particle filter as the mean over its five seeds, as the paper reports it.

- **The filters below the batch solver** (Section VI-A). Under the cauchy
  kernel the EKF ends below the factor graph on the fully meshed view of all
  22 sessions of set A, and the grid filter on 21 of them. The exception is
  `set_a/s19`, where the grid filter ends 0.09 m above. On the
  mobile-assisted view of the 20 quasi-static sessions both filters end
  below it in every session (`benchmark`, column `rmse_m`). The filters
  read the epoch stream and the static batch engines one median per pair,
  so the advantage may come from this data interface rather than from the
  engine, and no control in the release separates the two.
- **Anchors with clean pairs under the gaussian kernel** (Section VI-A).
  For every anchor of a fully meshed view of set A, the absolute
  session-median residual of each of its pairs (the median over the epochs
  of the symmetrized range minus the surveyed distance, as in Section 1 of
  this file) is
  averaged over its pairs. This average is at most 0.40 m for 65 of the 244
  anchor estimates, the same 65 under both kernels. Their median aligned
  error (`benchmark_estimates`, column `error_m`) is 2.06 m under the
  gaussian factor graph and 0.19 m under the cauchy one, so under a
  quadratic loss the outlier pairs also move the anchors whose own pairs are
  clean. The residuals come from the sessions under `Data/`, not from a
  result table.
- **Offsets on the NLOS deployment** (Section VI-C). On the fully meshed
  view of set A, a common range offset lowers the median RMSE of the factor
  graph only at a reduced noise scale. At a quarter of the session noise
  scale it goes from 0.326 m to 0.227 m, and per-device offsets under a
  0.10 m prior reach 0.230 m. At half of it the two give 0.357 m and
  0.361 m against 0.397 m. At the session noise scale both offset models end
  above the estimate without offsets, at 0.817 m and 0.821 m against
  0.789 m (`offset_control`).
- **One extra node** (Section VI-D). Over the 20 quasi-static sessions of
  set A, the mobile-assisted view adds the tag and the surveyed priors of
  two anchors to the full mesh. Paired per cell, it lowers the anchor RMSE
  in 251 of 360 cells and raises it in 109, with a median change of
  −0.046 m. The gain per cell is largest for the five gaussian variants and
  the kernel-less classical MDS, whose paired changes have medians of −0.08
  to −0.13 m, against −0.07 to +0.01 m for the variants with a robust
  kernel, gate or mixture. The
  closed form is left out, since its sequential construction places every
  anchor before the mobile and leaves its 40 cells unchanged. With them the
  count is 251 of 400 and the median −0.027 m.
  `Code/methods/results/overview.md` and `mobile_assisted/summary.md` give
  250 of 400, since they count seed 0 of the particle filter.

## 6. Table VI and the pair weight of Section V-B

Table VI of the companion paper gives the anchor RMSE without and with the
action of each lesson. Its rows select the released tables as follows, with
the median where a row covers several sessions or views:

1. **Full meshes, NLOS deployment.** `benchmark`, fully meshed view, the 22
   sessions of set A, `factor_graph` with the variant `gaussian` against
   `cauchy`, column `rmse_m`.
2. **Globally rigid random views of `s03`.** `oracle_start`, the random
   thinning of `set_b/s03`, restricted to the views whose class in
   `rigidity` is `globally_rigid`, column `rmse_mds_start_m` against
   `rmse_oracle_start_m`.
3. **Full meshes, LOS deployment.** `benchmark`, fully meshed view, the three
   sessions of set B, `factor_graph` against `fg_offset_dev`, both `cauchy`,
   column `rmse_m`.
4. **Walk `s01`.** `benchmark`, mobile-only view (`slam_bipartite`) of
   `set_b/s01`, `ekf_slam` against `fg_slam`, both `cauchy`, column `rmse_m`.
5. **The κ = 3 view of `s03`.** `benchmark`, the prior sweep of `set_b/s03`,
   `fg_anchored` with the variant `cauchy` on `multi_hop_p0` (no seed)
   against `multi_hop_p3` (three seeds), column `rmse_m`.

Section V-B states the weight that the cauchy kernel gives the pair 5BB6 to
D38F at the solution on the full mesh of `set_b/s03`, 0.003. It stands in
`Code/methods/results/factor_graph/set_b_s03_pair_weights.csv` (0.0032),
which the per-method runner `run_factor_graph.py` writes and the documented
command list does not regenerate.
