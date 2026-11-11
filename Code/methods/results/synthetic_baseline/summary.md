# synthetic_baseline - controlled-truth check of the lessons (paper, Sec. VI)

The range-error generator of `range_errors.py`, a seeded port of the
simulation framework of Schwarzbach, Weber and Michler (IEEE Sensors J.
22(6):4836-4843, 2022, doi 10.1109/JSEN.2021.3121627) with the values of
that paper: hardware noise sigma_hw = 0.33 m, log-normal multipath (0.8 and
1.07 on the log scale) where p_eps > 0.8 - 0.3 d / d_max and the excess is
shorter than d, d_max = 100 m, and uniform outliers U(-d, d_max - d) with
p_out = 0.05, the setting of this baseline. Dropouts are off, since the
topology views set the edges.

Geometry: the surveyed anchors of set_b/s03 (10 anchors, 45 pairs), 200
synthetic epochs, 2-D distances, 20 draws per condition with noise and one
without (seeds 0-19, one generator stream per error component). Conditions:
noise_free, gaussian,
nlos (state and excess per anchor pair, both directions, plus per-range
noise and outliers), nlos_per_range (the published per-range model) and
biased (+0.27 m on every range, plus noise). Views: fully_meshed and
multi_hop_k3 through the production masks. Kernel scales: the session sigma
of the benchmark doctrine and the generator's sigma_hw (the noise-free
condition only at sigma_hw, since its session sigma is zero up to rounding).
Methods: the factor graph (gaussian, cauchy) on both views, also from an
oracle start at the true geometry, and fg_offset (one common range offset)
on the full mesh.

Why the nlos condition holds the multipath per pair: through the per-pair
medians of the batch engines, the per-range model of the 2022 paper becomes
a positive bias of a few decimeters on every pair, growing with the distance,
which acts like a range bias (rows nlos_per_range below), whereas the
measured sessions show single pairs biased by meters (Sec. VI-A of the
paper).

Regenerate with `python run_synthetic.py --force`. `python run_synthetic.py
--check` (also run by `run_benchmark.py --check-regression`) compares a rerun
with `frozen.csv`.

<!-- BEGIN GENERATED (run_synthetic.py) -->
### Aligned anchor RMSE m, median (min-max) over the draws

| condition | view | scale | method | variant | draws | rmse_m | oracle_start_m | b_hat_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noise_free | fully_meshed | hardware | factor_graph | gaussian | 1 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | - |
| noise_free | fully_meshed | hardware | factor_graph | cauchy | 1 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | - |
| noise_free | fully_meshed | hardware | fg_offset | gaussian | 1 | 0.000 (0.000-0.000) | - | 0.000 (0.000-0.000) |
| noise_free | fully_meshed | hardware | fg_offset | cauchy | 1 | 0.000 (0.000-0.000) | - | 0.000 (0.000-0.000) |
| noise_free | multi_hop_k3 | hardware | factor_graph | gaussian | 1 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | - |
| noise_free | multi_hop_k3 | hardware | factor_graph | cauchy | 1 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | - |
| gaussian | fully_meshed | session | factor_graph | gaussian | 20 | 0.014 (0.010-0.023) | 0.014 (0.010-0.023) | - |
| gaussian | fully_meshed | session | factor_graph | cauchy | 20 | 0.014 (0.010-0.023) | 0.014 (0.010-0.023) | - |
| gaussian | fully_meshed | session | fg_offset | gaussian | 20 | 0.015 (0.010-0.024) | - | -0.004 (-0.019-0.011) |
| gaussian | fully_meshed | session | fg_offset | cauchy | 20 | 0.015 (0.010-0.024) | - | -0.003 (-0.019-0.011) |
| gaussian | fully_meshed | hardware | factor_graph | gaussian | 20 | 0.014 (0.010-0.023) | 0.014 (0.010-0.023) | - |
| gaussian | fully_meshed | hardware | factor_graph | cauchy | 20 | 0.014 (0.010-0.023) | 0.014 (0.010-0.023) | - |
| gaussian | fully_meshed | hardware | fg_offset | gaussian | 20 | 0.015 (0.010-0.024) | - | -0.004 (-0.019-0.011) |
| gaussian | fully_meshed | hardware | fg_offset | cauchy | 20 | 0.015 (0.010-0.024) | - | -0.003 (-0.019-0.011) |
| gaussian | multi_hop_k3 | session | factor_graph | gaussian | 20 | 0.102 (0.026-0.435) | 0.102 (0.026-0.435) | - |
| gaussian | multi_hop_k3 | session | factor_graph | cauchy | 20 | 0.102 (0.026-0.435) | 0.102 (0.026-0.435) | - |
| gaussian | multi_hop_k3 | hardware | factor_graph | gaussian | 20 | 0.102 (0.026-0.435) | 0.102 (0.026-0.435) | - |
| gaussian | multi_hop_k3 | hardware | factor_graph | cauchy | 20 | 0.102 (0.026-0.435) | 0.102 (0.026-0.435) | - |
| nlos | fully_meshed | session | factor_graph | gaussian | 20 | 1.089 (0.323-3.028) | 1.050 (0.323-1.774) | - |
| nlos | fully_meshed | session | factor_graph | cauchy | 20 | 1.072 (0.322-3.011) | 1.037 (0.322-1.701) | - |
| nlos | fully_meshed | session | fg_offset | gaussian | 20 | 1.131 (0.306-2.751) | - | 0.105 (-0.703-1.156) |
| nlos | fully_meshed | session | fg_offset | cauchy | 20 | 1.103 (0.305-2.739) | - | 0.146 (-0.655-1.148) |
| nlos | fully_meshed | hardware | factor_graph | gaussian | 20 | 1.089 (0.323-3.028) | 1.050 (0.323-1.774) | - |
| nlos | fully_meshed | hardware | factor_graph | cauchy | 20 | 0.177 (0.090-2.495) | 0.167 (0.090-0.570) | - |
| nlos | fully_meshed | hardware | fg_offset | gaussian | 20 | 1.131 (0.306-2.751) | - | 0.105 (-0.703-1.156) |
| nlos | fully_meshed | hardware | fg_offset | cauchy | 20 | 0.184 (0.076-2.469) | - | 0.142 (-0.061-0.755) |
| nlos | multi_hop_k3 | session | factor_graph | gaussian | 20 | 1.002 (0.188-4.875) | 0.896 (0.188-4.838) | - |
| nlos | multi_hop_k3 | session | factor_graph | cauchy | 20 | 1.002 (0.188-4.875) | 0.896 (0.188-4.875) | - |
| nlos | multi_hop_k3 | hardware | factor_graph | gaussian | 20 | 1.002 (0.188-4.875) | 0.896 (0.188-4.838) | - |
| nlos | multi_hop_k3 | hardware | factor_graph | cauchy | 20 | 1.001 (0.188-4.993) | 0.897 (0.188-4.985) | - |
| nlos_per_range | fully_meshed | session | factor_graph | gaussian | 20 | 0.166 (0.156-0.181) | 0.166 (0.156-0.181) | - |
| nlos_per_range | fully_meshed | session | factor_graph | cauchy | 20 | 0.166 (0.156-0.181) | 0.166 (0.156-0.181) | - |
| nlos_per_range | fully_meshed | session | fg_offset | gaussian | 20 | 0.089 (0.070-0.129) | - | 0.130 (0.068-0.157) |
| nlos_per_range | fully_meshed | session | fg_offset | cauchy | 20 | 0.089 (0.070-0.129) | - | 0.130 (0.068-0.157) |
| nlos_per_range | fully_meshed | hardware | factor_graph | gaussian | 20 | 0.166 (0.156-0.181) | 0.166 (0.156-0.181) | - |
| nlos_per_range | fully_meshed | hardware | factor_graph | cauchy | 20 | 0.166 (0.156-0.181) | 0.166 (0.156-0.181) | - |
| nlos_per_range | fully_meshed | hardware | fg_offset | gaussian | 20 | 0.089 (0.070-0.129) | - | 0.130 (0.068-0.157) |
| nlos_per_range | fully_meshed | hardware | fg_offset | cauchy | 20 | 0.089 (0.070-0.129) | - | 0.130 (0.068-0.157) |
| nlos_per_range | multi_hop_k3 | session | factor_graph | gaussian | 20 | 0.424 (0.241-0.876) | 0.424 (0.241-0.876) | - |
| nlos_per_range | multi_hop_k3 | session | factor_graph | cauchy | 20 | 0.424 (0.241-0.876) | 0.424 (0.241-0.876) | - |
| nlos_per_range | multi_hop_k3 | hardware | factor_graph | gaussian | 20 | 0.424 (0.241-0.876) | 0.424 (0.241-0.876) | - |
| nlos_per_range | multi_hop_k3 | hardware | factor_graph | cauchy | 20 | 0.424 (0.241-0.876) | 0.424 (0.241-0.876) | - |
| biased | fully_meshed | session | factor_graph | gaussian | 20 | 0.176 (0.172-0.181) | 0.176 (0.172-0.181) | - |
| biased | fully_meshed | session | factor_graph | cauchy | 20 | 0.175 (0.172-0.181) | 0.175 (0.172-0.181) | - |
| biased | fully_meshed | session | fg_offset | gaussian | 20 | 0.015 (0.010-0.024) | - | 0.266 (0.251-0.281) |
| biased | fully_meshed | session | fg_offset | cauchy | 20 | 0.015 (0.010-0.024) | - | 0.267 (0.251-0.281) |
| biased | fully_meshed | hardware | factor_graph | gaussian | 20 | 0.176 (0.172-0.181) | 0.176 (0.172-0.181) | - |
| biased | fully_meshed | hardware | factor_graph | cauchy | 20 | 0.175 (0.172-0.181) | 0.175 (0.172-0.181) | - |
| biased | fully_meshed | hardware | fg_offset | gaussian | 20 | 0.015 (0.010-0.024) | - | 0.266 (0.251-0.281) |
| biased | fully_meshed | hardware | fg_offset | cauchy | 20 | 0.015 (0.010-0.024) | - | 0.267 (0.251-0.281) |
| biased | multi_hop_k3 | session | factor_graph | gaussian | 20 | 0.731 (0.619-0.857) | 0.731 (0.619-0.857) | - |
| biased | multi_hop_k3 | session | factor_graph | cauchy | 20 | 0.731 (0.619-0.857) | 0.731 (0.619-0.857) | - |
| biased | multi_hop_k3 | hardware | factor_graph | gaussian | 20 | 0.731 (0.619-0.857) | 0.731 (0.619-0.857) | - |
| biased | multi_hop_k3 | hardware | factor_graph | cauchy | 20 | 0.731 (0.619-0.857) | 0.731 (0.619-0.857) | - |

### Classical start ended more than 0.01 m above the oracle start at a higher cost (factor graph)

| condition | view | scale | variant | draws | classical_start_worse |
| --- | --- | --- | --- | --- | --- |
| noise_free | fully_meshed | hardware | gaussian | 1 | 0 |
| noise_free | fully_meshed | hardware | cauchy | 1 | 0 |
| noise_free | multi_hop_k3 | hardware | gaussian | 1 | 0 |
| noise_free | multi_hop_k3 | hardware | cauchy | 1 | 0 |
| gaussian | fully_meshed | session | gaussian | 20 | 0 |
| gaussian | fully_meshed | session | cauchy | 20 | 0 |
| gaussian | fully_meshed | hardware | gaussian | 20 | 0 |
| gaussian | fully_meshed | hardware | cauchy | 20 | 0 |
| gaussian | multi_hop_k3 | session | gaussian | 20 | 0 |
| gaussian | multi_hop_k3 | session | cauchy | 20 | 0 |
| gaussian | multi_hop_k3 | hardware | gaussian | 20 | 0 |
| gaussian | multi_hop_k3 | hardware | cauchy | 20 | 0 |
| nlos | fully_meshed | session | gaussian | 20 | 0 |
| nlos | fully_meshed | session | cauchy | 20 | 0 |
| nlos | fully_meshed | hardware | gaussian | 20 | 0 |
| nlos | fully_meshed | hardware | cauchy | 20 | 2 |
| nlos | multi_hop_k3 | session | gaussian | 20 | 0 |
| nlos | multi_hop_k3 | session | cauchy | 20 | 1 |
| nlos | multi_hop_k3 | hardware | gaussian | 20 | 0 |
| nlos | multi_hop_k3 | hardware | cauchy | 20 | 1 |
| nlos_per_range | fully_meshed | session | gaussian | 20 | 0 |
| nlos_per_range | fully_meshed | session | cauchy | 20 | 0 |
| nlos_per_range | fully_meshed | hardware | gaussian | 20 | 0 |
| nlos_per_range | fully_meshed | hardware | cauchy | 20 | 0 |
| nlos_per_range | multi_hop_k3 | session | gaussian | 20 | 0 |
| nlos_per_range | multi_hop_k3 | session | cauchy | 20 | 0 |
| nlos_per_range | multi_hop_k3 | hardware | gaussian | 20 | 0 |
| nlos_per_range | multi_hop_k3 | hardware | cauchy | 20 | 0 |
| biased | fully_meshed | session | gaussian | 20 | 0 |
| biased | fully_meshed | session | cauchy | 20 | 0 |
| biased | fully_meshed | hardware | gaussian | 20 | 0 |
| biased | fully_meshed | hardware | cauchy | 20 | 0 |
| biased | multi_hop_k3 | session | gaussian | 20 | 0 |
| biased | multi_hop_k3 | session | cauchy | 20 | 0 |
| biased | multi_hop_k3 | hardware | gaussian | 20 | 0 |
| biased | multi_hop_k3 | hardware | cauchy | 20 | 0 |

### Per draw: sigma terms, NLOS pairs, per-pair median residuals of the full mesh, kappa = 3 view classes

| condition | sigma_scatter_m | sigma_mds_m | sigma_robust_m | nlos_pairs | pair_resid_median_m | pair_resid_max_m | pair_resid_corr_d | nlos_pairs_kept_k3 | k3_view_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noise_free | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.0 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | - | 0.0 | rigid_not_global 1 |
| gaussian | 0.233 (0.229-0.236) | 0.017 (0.013-0.029) | 0.232 (0.227-0.239) | 0.0 | -0.000 (-0.006-0.005) | 0.045 (0.029-0.062) | 0.029 (-0.173-0.257) | 0.0 | rigid_not_global 20 |
| nlos | 7.291 (6.916-7.930) | 0.407 (0.165-1.100) | 0.264 (0.256-0.269) | 8.0 | 0.031 (0.023-0.045) | 7.530 (2.920-11.650) | 0.063 (-0.160-0.251) | 2.0 | rigid_not_global 14, globally_rigid 3, flexible 3 |
| nlos_per_range | 7.474 (6.723-8.022) | 0.055 (0.038-0.081) | 0.571 (0.541-0.624) | 0.0 | 0.239 (0.219-0.271) | 0.414 (0.339-0.534) | 0.689 (0.614-0.828) | 0.0 | rigid_not_global 20 |
| biased | 0.233 (0.229-0.236) | 0.066 (0.057-0.080) | 0.232 (0.227-0.239) | 0.0 | 0.270 (0.264-0.275) | 0.315 (0.299-0.332) | 0.029 (-0.173-0.257) | 0.0 | rigid_not_global 20 |
<!-- END GENERATED (run_synthetic.py) -->
