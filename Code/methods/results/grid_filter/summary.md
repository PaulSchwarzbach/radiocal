# Grid filter (IOPS auto-positioning port) on fully_meshed - summary

Round 2 (controlled factorial): one shared noise model per session
(kernels.session_sigma: 0.129 m in set_b, 1.621 m in
set_a) and the kernel axis gaussian / huber / cauchy on the
textbook log-domain product update; `mixture(gaussian)` keeps the
original IOPS mechanism (sum of Gaussian likelihood components over
the references) at the same sigma. Round 1 confounded kernel, scale
and fusion rule; its private scatter sigma (0.05 / 1.039 m) and the
reference-variance inflation are gone (grid_filter.py, deviations 1
and 6). Per-epoch symmetrized ranges (grouped by seq); evaluation:
rigid Umeyama alignment (reflection allowed) against the surveyed
anchors, 2-D.

<!-- BEGIN GENERATED (summarize.py) -->
| campaign | session | variant           | n_anchors | rmse_m | max_error_m | reflection | grid_res_m | likelihood_sigma_m |
| -------- | ------- | ----------------- | --------- | ------ | ----------- | ---------- | ---------- | ------------------ |
| set_b    | s03     | gaussian          | 10        | 0.805  | 2.067       | False      | 0.250      | 0.129              |
| set_b    | s03     | huber             | 10        | 0.275  | 0.454       | False      | 0.250      | 0.129              |
| set_b    | s03     | cauchy            | 10        | 0.251  | 0.342       | False      | 0.250      | 0.129              |
| set_b    | s03     | mixture(gaussian) | 10        | 0.238  | 0.416       | False      | 0.250      | 0.129              |
| set_a    | s21     | gaussian          | 12        | 3.042  | 6.784       | False      | 0.250      | 1.621              |
| set_a    | s21     | huber             | 12        | 0.808  | 1.496       | False      | 0.250      | 1.621              |
| set_a    | s21     | cauchy            | 12        | 0.619  | 1.075       | False      | 0.250      | 1.621              |
| set_a    | s21     | mixture(gaussian) | 12        | 0.397  | 0.748       | False      | 0.250      | 1.621              |
<!-- END GENERATED (summarize.py) -->

Notes:

- Fixed pipeline pieces (historical, outside the kernel axis):
  closed-form init bump (std 0.5 m), 0.25 m grid (half-width = max
  median range around anchor 0 + 5 m: 27.4 m -> 48400 cells in
  2025, 45.4 m -> 132496 cells in 2024), probability floor 6e-3 x
  uniform, estimate = probability-weighted mean within 4 m of the
  MAP cell. Committed subsampling: stride 27 (78 update epochs,
  2025) and 48 (76 epochs, 2024); the 4-variant table runs in
  ~15 s (2025) / ~130 s (2024). Stride is stable: cauchy 2024
  gives 0.609/0.619/0.634 m at stride 36/48/72, mixture 0.385/
  0.397/0.421 m; cauchy 2025 gives 0.251 m at stride 20/27/41.
- Exact-recovery validation (synthetic exact distances from the
  surveyed coordinates, 200 epochs, shared sigma): 2025 RMSE
  0.099 m / max 0.153 m for all four variants; 2024 RMSE 0.075-
  0.098 m / max 0.176 m (kernels) and 0.248 m (mixture), all
  anchors placed, no reflection - within grid quantization (cell
  half-diagonal 0.177 m; the mixture posterior at sigma 1.621 m is
  broad, so its 4-m window mean smooths up to one cell).
- Kernel axis (product update, shared sigma): robustness is
  monotone in the kernel at both sites - see table. The gaussian
  cell fails exactly like round 1's product variant: in 2025,
  dropping the single persistent NLOS pair 5BB6-D38F (+5.82 m
  median bias) moves it from 0.805 to 0.247 m (committed stride) -
  that pair is the root cause; in 2024 the bias is distributed
  (p95 pair bias +16.6 m; in round 1 dropping the worst pair moved
  the product update only from 2.70 to 2.43 m).
- Sum vs product, now at identical kernel and scale: the original
  mixture update beats the gaussian product 0.238 vs 0.805 m
  (2025) and 0.397 vs 3.042 m (2024) - the mechanism, not the
  scale, is what made the historical pipeline NLOS-tolerant. It
  also matches or beats the redescending cauchy product (0.251 m
  in 2025, 0.619 m in 2024): a biased range contributes one
  displaced component that the other references outvote, while
  any product update must down-weight it explicitly.
- Mixture under the shared sigma (vs round 1): 2025 0.266 ->
  0.238 m; 2024 0.342 -> 0.397 m. The 2024 change decomposes
  cleanly at the committed stride: rerunning with the round-1
  sigma 1.039 m gives 0.358 m (+0.039 m from the wider scale),
  and the round-1 sigma at round-1 stride 18 gave 0.342 m
  (+0.016 m from coarser subsampling). No mechanism surprise.
- Cauchy and mixture(gaussian) agree closely on 2025 (both fully
  reject the NLOS pair); at stride 20 they produced cell-identical
  estimates (max difference 2.3 mm), at the committed stride 27
  they differ by up to 0.35 m per anchor at near-equal RMSE -
  grid-cell-level coincidence, not an equivalence.
- Scale sweep of the legacy runner (cauchy, 2x committed stride, 39/38
  epochs, superseded by results/sigma_sweep.parquet at the committed
  stride, see results/overview.md): RMSE
  0.245 / 0.245 / 0.251 / 0.281 / 0.306 m (2025) and 0.319 /
  0.395 / 0.598 / 0.900 / 1.257 m (2024) at sigma_mult 0.25 / 0.5
  / 1 / 2 / 4. At the NLOS site the cauchy filter improves
  monotonically as the scale shrinks below the shared sigma - a
  tighter scale sharpens the redescend and rejects the distributed
  biased pairs harder; in 2025 the sweep is flat below 1x, where
  grid quantization dominates.

Replication over all fully_meshed sessions (benchmark.parquet):

- s19 is the one session where the product variants (gaussian /
  huber / cauchy) converge to the mirrored geometry
  (reflection=True; aligned RMSE unremarkable at 0.75-2.21 m), while
  the mixture update lands in the surveyed chirality. Root cause:
  the first 2-D grid anchor (4197) misses its closed-form triple
  init on s19 and starts from the uniform grid, so the +y convention
  of meas_init never anchors the chirality; both mirror modes stay
  alive until the fusion rule collapses one, and the product and
  mixture rules happen to collapse different ones. The shared C4
  evaluation is chirality-blind (alignment with reflection allowed),
  so the metric is unaffected; the flag is reported for completeness.
