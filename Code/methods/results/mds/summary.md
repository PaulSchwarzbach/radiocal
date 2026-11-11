# MDS baselines on fully_meshed - summary

Session-median symmetrized ranges; evaluation: rigid Umeyama
alignment (reflection allowed) against the surveyed anchors of the
unmasked session, 2-D (x, y).

<!-- BEGIN GENERATED (summarize.py) -->
| campaign | session | variant         | n_anchors | pairs_available | pairs_completed | rmse_m | max_error_m | reflection | stress1 | neg_eig_frac | sigma_m | n_iter |
| -------- | ------- | --------------- | --------- | --------------- | --------------- | ------ | ----------- | ---------- | ------- | ------------ | ------- | ------ |
| set_b    | s03     | classical       | 10        | 45/45           | 0               | 0.258  | 0.496       | False      | 0.062   | 0.012        | nan     | 0      |
| set_b    | s03     | smacof_gaussian | 10        | 45/45           | 0               | 0.965  | 2.441       | False      | 0.045   | 0.012        | nan     | 126    |
| set_b    | s03     | smacof_huber    | 10        | 45/45           | 0               | 0.241  | 0.397       | False      | 0.063   | 0.012        | 0.129   | 172    |
| set_b    | s03     | smacof_cauchy   | 10        | 45/45           | 0               | 0.223  | 0.314       | False      | 0.065   | 0.012        | 0.129   | 176    |
| set_a    | s21     | classical       | 12        | 65/66           | 1               | 3.061  | 5.246       | False      | 0.194   | 0.044        | nan     | 0      |
| set_a    | s21     | smacof_gaussian | 12        | 65/66           | 1               | 5.935  | 14.777      | False      | 0.126   | 0.044        | nan     | 105    |
| set_a    | s21     | smacof_huber    | 12        | 65/66           | 1               | 2.331  | 5.007       | False      | 0.169   | 0.044        | 1.621   | 267    |
| set_a    | s21     | smacof_cauchy   | 12        | 65/66           | 1               | 0.999  | 1.770       | False      | 0.208   | 0.044        | 1.621   | 103    |
<!-- END GENERATED (summarize.py) -->

Notes:

- `classical` = Torgerson double centering on the shortest-path-
  completed matrix (MDS-MAP), kernel-less; `smacof_gaussian` =
  stress majorization over observed pairs only, initialized from
  the classical solution (formerly labeled `smacof`, identical
  results); `smacof_huber` / `smacof_cauchy` = IRLS-robust weighted
  SMACOF with the shared benchmark kernels and session sigma
  (kernels.py; huber k = 1.345, cauchy c = 2.385, sigma 0.129 m on
  2025 / 1.621 m on 2024).
- Scale sweep (scale_sweep.csv, smacof_cauchy, sigma multiplier
  0.25-4): see the cross-method overview; the a-priori sigma is
  kept for the benchmark rows.
- `stress1` is Kruskal stress-1 over observed pairs; `neg_eig_frac`
  is the negative eigenvalue mass of the centered squared-distance
  matrix - both indicate non-Euclidean noise (NLOS, ranging bias).
- All anchors are always placed (connected range graph), and every
  observed pair enters the estimate, unlike the closed-form
  construction.
- The two variants split under NLOS: classical strain spreads a
  biased range across the Gram matrix, while SMACOF fits distances
  directly and adopts the bias - lower stress1 can coincide with
  worse geometry (verified on set_b: dropping the one NLOS
  pair 5BB6-D38F moves SMACOF 0.96 -> 0.23 m, classical stays).
