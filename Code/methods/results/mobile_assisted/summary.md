# mobile_assisted - anchor mesh plus a quasi-static mobile (C5 family)

The full anchor mesh plus the mobile-anchor edges of the roving tag. Entry
via the set_a quasi-static sessions: 8B05 parked at a surveyed marker is one
extra unknown node, so the static estimator suite runs unchanged with the
mobile appended to the estimation order (common.estimation_order). Anchor
RMSE after Procrustes stays the primary metric; the mobile's xy error
against its marker (after the anchor-fitted alignment) is secondary.
Sessions flagged truth_quality: unreliable run fully but are excluded from
every mobile aggregate. Since the truth adjudication of 2026-09-30 (Note 8)
all 20 quasi-static sessions are verified. Validation: validate_mobile.py
(exact recovery incl. the mobile; 3-D slant projection systematic
quantified separately).

<!-- BEGIN GENERATED (summarize.py) -->
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
<!-- END GENERATED (summarize.py) -->

## Notes

Every statement below was verified against results/benchmark.parquet
(mask == mobile_assisted, 400/400 cells ok) or reproduced with targeted
in-memory experiments on the unmasked sessions; the anomaly worklist is
`summarize.py --flag-anomalies` (54 family rows, all root-caused).

1. Headline: the extra well-connected node helps the NLOS-rich 2024 mesh,
   but only marginally. Paired per-cell anchor RMSE (mobile_assisted minus
   fully_meshed, same 20 quasi-static sessions) has median -0.027 m and improves 250 of
   400 cells. The gain concentrates in the gaussian kernels (median deltas
   -0.12 to -0.16 m for smacof/factor_graph/ekf/particle gaussian); robust
   kernels move within -0.07 to +0.01 m, and grid_filter:gaussian is the
   one degradation (+0.29 m on the median). The closed form is bitwise
   unaffected (max |delta| = 0.0 over all 40 cells): its sequential
   construction places every anchor before the mobile, confirming the
   family view only appends one unknown.

2. Kernel effect on the mobile, analysed over the 18 sessions verified
   before the adjudication of Note 8 admitted s05 and s12 (with all 20 the
   pooled mobile medians stay at 0.94 / 0.41 / 0.29 m, the pooled gaussian
   anchor RMSE becomes 3.13 m, and the table above gives the per-variant
   values): per-variant
   medians gaussian 1.06-1.55 m, huber 0.31-0.56 m, cauchy 0.21-0.35 m
   (pooled 0.94 / 0.41 / 0.29 m); anchor RMSE pools to
   3.09 / 0.91 / 0.71 m. The mobile error tracks anchor quality, and the
   kernel, not the estimator class, stays the dominant axis - the
   fully_meshed finding extends to the mobile metric. The mechanism variants
   are the family's best: grid_filter:mixture 0.138 m median, and
   ekf:gated 0.313 m with worst verified session s08 (0.523 m) - the only
   variant whose worst case is not one of the NLOS sessions below.

3. s15/s16 reflections (28 of the 54 flags) are family-induced chirality
   flips of the shared classical-MDS initialization, not errors. Per
   session exactly the 14 MDS-initialized cells flip (mds x4,
   factor_graph x3, ekf x4, particle_filter x3); closed_form (own +y
   gauge) and grid_filter never do, and the same sessions' fully_meshed rows
   show 0/20 reflections. In-memory rerun: classical MDS on s15/s16
   aligns without reflection on the fully_meshed view and with reflection as
   soon as the 8B05 row/column enters the completed distance matrix
   (control s03: no flip on either view) - an eigen-embedding chirality
   flip that every downstream solver inherits as its basin. Evaluation
   allows reflection because range-only geometry is chirality-blind;
   anchor RMSE is unaffected (paired session medians +0.009 m on s15,
   -0.037 m on s16).

4. s19 (and, milder, s18) carry gross NLOS on the mobile links
   themselves. Median 8B05-anchor range minus true 3-D marker-anchor
   distance: s19 has 3 of 11 links grossly inflated (D5B4 +17.4 m,
   8AA5 +9.3 m, 5388 +6.5 m; all others <= 0.32 m), s18 has D5B4 +11.1 m
   and 5388 +1.8 m. With 27 % of its mobile links corrupted, s19 is the
   worst verified session of nearly every variant (factor_graph:cauchy
   1.605 m vs its 0.354 m median) - robust kernels bound the damage but
   cannot remove a 3-of-11 contamination. The s19 grid_filter reflections
   (gaussian/huber/cauchy) occur in the same session's fully_meshed rows
   too, so they are a session property of the grid placement, not
   family-induced; the mixture variant does not flip.

5. s10: the closed_form:disambiguated 51.5 m mobile error is a
   wrong-branch adoption driven by a single NLOS range. The mobile is
   intersected from its ranges to 5BB6 (34.02 m, bias +0.60) and 4A23
   (26.16 m, +0.20), giving x = 21.86 m, |y| = 26.06 m; the +-y decision
   is then resolved against the measured 8B05-4503 range of 32.44 m,
   which is NLOS-inflated by +15.11 m (true distance 17.33 m). The
   mirrored branch (44.28 m from 4503) matches the corrupted range better
   than the true one (17.06 m), so the mobile lands reflected across the
   5BB6-4A23 baseline, 2|y| = 52 m off (51.5 m after alignment). The
   faithful variant keeps +y by convention and scores 2.64 m. The same
   link (4503, one of only 9 mobile links in s10) plus the campaign-worst
   anchor-anchor outliers (9911-D5B4 +23.7 m, 5BB6-8AA5 +14.1 m) explain
   the s10 gaussian cluster (mobile 3.3-7.5 m) and particle_filter:huber
   (RMSE 4.50 m, the family's only flagged anchor RMSE).

6. The mobile links are cleaner than the anchor mesh. Per-session median
   mobile-link residual is 0.04-0.26 m (ranging bias plus the projection
   systematic of note 9); 8 of 20 sessions have every link within 1 m.
   Single gross links (> 3 m) occur in 8 verified sessions (s01 D5B4
   +19.9 m, s09 9911 +8.3 m, s10 4503 +15.1 m, s15 8AA5 +8.8 m, s17 8AA5
   +7.6 m, s18 D5B4 +11.1 m, s20 D5B4 +14.9 m, s19 three links); one
   gross link in ~11 is absorbed by the robust kernels (s01
   factor_graph:cauchy mobile 0.56 m), three are not (s19, note 4). The
   anchors D5B4 and 8AA5 recur as the biased endpoints.

7. Cross-reference with the curators' verification: over the 15 verified
   sessions without gross mobile NLOS, factor_graph:cauchy mobile errors
   (median 0.29 m, range 0.12-0.56 m) sit on the same few-decimeter scale
   as the session.yaml multilateration deviations (median 0.25 m, range
   0.09-0.47 m), median absolute difference 0.19 m. The two diverge
   exactly where they should: the curators multilaterate against the
   surveyed anchors, the benchmark estimates the anchors jointly, so
   mobile-link NLOS hits the benchmark harder (s18 0.88 vs 0.35 m, s19
   1.61 vs 0.26 m).

8. Truth adjudication. A rule in 04_build_set_a.py decides the truth of
   every quasi-static session, with no estimator output involved. The
   multilateration screen flags s05 and s12, and a direct range test at
   the marker (tag antenna 1.72 m above it) decides both: the median
   residual of the line-of-sight links must lie within the range of the
   18 unflagged sessions (+0.16 to +0.25 m). s05 gives +0.24 m and s12
   +0.21 m, so both are verified and enter every mobile aggregate. The
   large per-link scatter of s12 comes from four non-line-of-sight links
   (4503, 5BB6, 4AA4, D38F, +1 to +18 m). Its six line-of-sight links fit
   P12 at +0.09 to +0.42 m and stay as stationary as in the other
   sessions, and no field record reports a relocation. Its wide mobile
   errors are a non-line-of-sight effect on the estimators, not a truth
   defect. Statistics: Data/set_a/truth_adjudication.csv.

9. Validation (validate_mobile.py; noise-free synthetic superset from
   the surveyed 2024 anchors with the mobile at the real P01 marker, run
   through the production mask -> registry -> evaluation path; numbers
   of the 2026-08 run, the 3-D block is updated below): the
   batch solvers (factor graph, SMACOF, and on this full mesh also
   classical MDS and closed_form:disambiguated) recover anchors and
   mobile to < 1e-13 m, the EKF to < 1e-14 m, the particle filter to
   <= 0.04 m and the grid filter to < 0.22 m (quantization floor). With
   3-D slant ranges instead, the 2-D projection systematic depends on the
   height of the tag against the anchors. With the tag antenna at its
   adjudicated height (z +1.31 m against an anchor mean of +1.29 m,
   2026-09-30) the offset is -0.02 m and the batch solvers stay exact to
   < 1 mm on anchors and mobile. The 0.065 m mobile bias reported before
   came from a tag assumed at marker height (1.70 m below the anchors).
   Flag propagation (s01, s05 and s12 verified, Note 8), the estimation
   order on the real s03 view
   (11 anchors + 8B05) and the (NaN, None) degradation of a missing
   mobile estimate are asserted by the same script.
