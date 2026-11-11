# Code/methods: reference implementations

Reference implementations of the engines of the companion tutorial (Section V),
run on the topology views of the measured sessions (Section IV-C).

**Rule:** methods consume only `radiocal.MeasurementGraph` (via
`radiocal.load_session(session_dir, mask=...)`). No raw-file access, no
modification of `radiocal` or `Data/`. The Vicon->ENU transform in the
`transform_vicon_to_local` slot of the session.yaml files of the set_b walks comes from
the dataset build (wand tie, `02_build_set_b.py`).
`estimate_vicon_transform.py` fits a UWB-based transform as a cross-check
and writes it under `results/transforms/` only.

## Layout

| file | role |
|---|---|
| `common.py` | shared graph -> range-matrix utilities (anchor ordering, session-aggregated pair matrix) |
| `closed_form.py` | closed-form sequential trilateration (port of the IOPS auto-positioning baseline; faithful and y-sign-disambiguated variants) |
| `kernels.py` | shared noise model of the factorial benchmark: gaussian/huber/cauchy kernels (fixed constants) + the session sigma |
| `mds.py` | MDS self-calibration: classical (MDS-MAP, kernel-less) and SMACOF gaussian/huber/cauchy (IRLS-robust Guttman) |
| `factor_graph.py` | batch NLS over all pair ranges, LM with analytic Jacobian; gaussian/huber/cauchy kernels (IRLS); also the prior-anchored solver (`fg_anchored`) and the range-offset solver (`fg_offset`: one common offset, `fg_offset_dev`: plus per-device offsets under a 0.10 m prior) |
| `ekf.py` | sequential EKF over epochs, gauge-reduced state; gaussian/huber/cauchy (M-estimation R inflation) + innovation gate, mds20 init |
| `particle_filter.py` | per-anchor conditional bootstrap PF, 2 sweeps; gaussian/huber/cauchy likelihoods, mds20 init, seeded |
| `grid_filter.py` | port of the IOPS histogram + probability-grid filters; kernelized product (gaussian/huber/cauchy) + original mixture update |
| `evaluation.py` | shared evaluation: surveyed reference from the unmasked session + `frames.align` (rigid, reflection allowed), 2-D |
| `reporting.py` | shared per-anchor result tables and Markdown formatting |
| `registry.py` | method registry: uniform `run(graph, variant, sigma_m, sigma_scale, seed)` adapters over the entry points + the variant -> (kernel, mechanism) map |
| `run_benchmark.py` | THE driver: session x mask x method x variant, keyed upserts into the long tables; `--check-regression` gate; `--mask` takes a comma list incl. parameterized `multi_hop_k<INT>`, `multi_hop_random_k<INT>_s<INT>`, `fully_meshed_p<K>`, `multi_hop_p<K>` and the aliases `multi_hop_sweep`, `multi_hop_random_sweep`, `prior_sweep` |
| `family_masks.py` | mask families: parameterized multihop spec dicts, family -> method roster, family -> session applicability (session.yaml `motion`) |
| `synthetic.py` | exact-range superset MeasurementGraphs from surveyed geometry (static / quasi-static mobile / trajectory with bursty truth) - the data source of every `validate_*.py` |
| `range_errors.py` | seeded port of the UWB range-error generator of Schwarzbach et al. (IEEE Sensors J. 2022): hardware noise, log-normal multipath, uniform outliers and dropouts, per range as published or with the multipath state and excess held per anchor pair; perturbs a `synthetic.py` superset graph under one condition |
| `run_synthetic.py` | synthetic baseline: the generator on the surveyed set_b/s03 anchors in five conditions with 20 draws each (one for noise_free, 81 in all), factor graph (gaussian, cauchy) on the full mesh and kappa = 3 at the session sigma and the generator's sigma, plus an oracle start, and fg_offset on the full mesh -> `results/synthetic_baseline/`; `--check` reruns it in memory against `frozen.csv` |
| `run_frame_choice.py` | the choice of the frame on the full mesh of set_b/s03 and set_a/s21: every defining pair (origin, x-axis) of closed_form, factor_graph, ekf and, on set_b/s03, grid_filter, and every seed triple, mapped onto the survey after the blind factor graph or used as fg_anchored priors -> `results/frame_choice/`; reads the stored benchmark cells, so it runs after `run_benchmark.py`; `--check` reruns its gated subset in memory against `frozen.csv` |
| `benchmark_io.py` | long-table storage (parquet + CSV mirror, idempotent upsert on campaign/session/mask/method/variant/seed; seed -1 = deterministic method) |
| `regression.py`, `freeze_expected.py` | regression gate against `results/regression_frozen.csv` (40 fully meshed cells of the two reference sessions, frozen at three decimals, tolerance 5e-4 m); the gate then reruns the synthetic baseline against `results/synthetic_baseline/frozen.csv` and the gated subset of the frame-choice sweep against `results/frame_choice/frozen.csv`. `freeze_expected.py` is the one-off maintenance step that wrote this reference. It overwrites the file from the current per-method summaries, which now render the result tables, so a rerun would make the gate check the outputs against themselves. Do not run it to reproduce results |
| `summarize.py` | regenerates the marker-delimited stats tables (per-method views, fully_meshed replication, per-family sections incl. the k3==canonical consistency assertion); `--flag-anomalies` worklist (family-aware; disconnected = structural outcome) |
| `rigidity.py`, `validate_multihop.py` | multihop family: 2-D (global) rigidity certificates + exact-recovery validation over the k sweep |
| `run_rigidity.py` | certificate class of every anchor-only view on its MEASURED pair graph (kNN sweep on all sessions, random-thinning views) -> `results/rigidity.parquet` |
| `run_controls.py` | controls: factor graph started from the surveyed geometry (`results/oracle_start.parquet`) and the cauchy noise-scale sweep at benchmark settings (`results/sigma_sweep.parquet`) on the static reference sessions; range offsets at three prior widths and noise scales with the offset-aware bounds (`results/offset_control.parquet`, `results/offset_crlb.parquet`) and the two terms of the session noise scale (`results/sigma_terms.parquet`) on all sessions |
| `crlb.py`, `run_crlb.py`, `validate_crlb.py` | Gaussian bounds on the gauge quotient per view (`results/crlb.parquet`), incl. the offset-aware bound (`crlb.offset_view_bound`) |
| `validate_offset.py` | range-offset solver: zero-offset identity with the blind solver, exact recovery of common and per-device offsets, bound ordering, Monte-Carlo closure |
| `evaluation_mobile.py`, `validate_mobile.py` | mobile_assisted family: marker evaluation of the quasi-static mobile (honors `truth_quality`) + exact-recovery validation |
| `slam_common.py`, `fg_slam.py`, `ekf_slam.py` | slam_bipartite family: shared constants/init (V_MAX_MS=1.5 random-walk motion prior, joint bipartite MDS init), pose-chain robust factor graph (sparse LM), causal EKF-SLAM |
| `estimate_vicon_transform.py` | UWB cross-check of the Vicon->ENU wand tie: robust per-epoch multilateration + trimmed planar Umeyama, compared with the session.yaml transform |
| `gt_error_budget.py` | ground-truth error budget (registration, wand baseline, clock, coverage, set A tag height, nominal instrument figures) -> `results/transforms/error_budget.yaml` |
| `evaluation_slam.py`, `validate_slam.py` | bursty-Vicon trajectory evaluation (checkpoint ATE + within-segment RPE, never interpolated) + the family validation (incl. the static-tag degeneracy demonstration) |
| `run_<method>.py` | legacy per-method runners (provenance of the frozen gate-session artifacts, scale sweeps and convergence traces); superseded by `run_benchmark.py` for benchmark cells |
| `results/benchmark.parquet` (+ `.csv`) | long-format benchmark table, one row per session x mask x method x variant x seed; family metrics (`mobile_error_m`, `ate_rmse_m`, `rpe_rmse_m`, `n_truth_matched`, `n_components`) are NaN where a family does not define them |
| `results/benchmark_estimates.parquet` (+ `.csv`) | per-node estimates of every benchmark cell (the mobile's row rides next to the anchors in the mobile_assisted family) |
| `results/benchmark_trajectories.parquet` (+ `.csv`) | per-epoch mobile trajectories of the SLAM cells, `ekf_slam` and `fg_slam` on slam_bipartite and `fg_slam` also on the mobile-assisted view of the set_b walks (solver frame + ENU + matched truth/segment) |
| `results/<method>/` | committed outputs: `<campaign>_<session>_estimates.csv` (per-anchor, static reference) + `summary.md` (stats table generated between markers, verified notes hand-written) |
| `results/<family>/summary.md` | per-family views (generated between markers) + verified family narratives (multi_hop, mobile_assisted, slam_bipartite) |
| `results/transforms/` | UWB-based cross-check transforms with their comparison to the wand tie, and the ground-truth error budget |
| `results/overview.md` | frozen factorial matrix + verified findings (hand-written) + generated fully_meshed replication and per-family sections |
| `results/synthetic_baseline/` | the synthetic baseline of `run_synthetic.py`: `cells.parquet` (+ `.csv`), one row per condition x draw x view x kernel scale x method x variant; `frozen.csv`, the gate's reference, written once by `--freeze`; `summary.md` (generated between markers) |
| `results/frame_choice/` | the frame-choice sweep of `run_frame_choice.py`: `pairs.parquet` (+ `.csv`), one row per defining pair x engine variant; `triples.parquet` (+ `.csv`), one row per seed triple; `line.parquet` (+ `.csv`), the offsets of the anchors from the line of the release pair; `frozen.csv`, the gate's reference, written once by `--freeze`; `summary.md` (generated between markers) |

## Run

The commands assume a POSIX shell such as bash (Git Bash or WSL on Windows)
and run from `Code/methods`, in the order of the repository README:

```
python run_benchmark.py --check-regression
python run_benchmark.py --mask fully_meshed,multi_hop,mobile_assisted,slam_bipartite --force
python run_benchmark.py --sessions "set_a/s21,set_b/s03" --mask multi_hop_sweep --force
python run_benchmark.py --sessions "set_a/s21,set_b/s03" --mask multi_hop_random_sweep --methods closed_form,mds,factor_graph --force
python run_benchmark.py --sessions "set_a/s21,set_b/s03" --mask prior_sweep --force
for s in 1 2 3 4; do
  python run_benchmark.py --methods particle_filter --seed $s --mask fully_meshed,multi_hop,mobile_assisted --force
  python run_benchmark.py --methods particle_filter --seed $s --sessions "set_a/s21,set_b/s03" --mask multi_hop_sweep --force
done
python run_rigidity.py
python run_crlb.py
python run_controls.py
python run_synthetic.py --force
python run_frame_choice.py --force
python estimate_vicon_transform.py --session set_b/s01,set_b/s02 --static-crosscheck
python gt_error_budget.py
python summarize.py
```

The gate (first line) must pass before new numbers are produced: 40/40 frozen
benchmark cells, then the frozen cells of the synthetic baseline
(`python run_synthetic.py --check` runs the latter alone) and the gated
rows of the frame-choice sweep (`python run_frame_choice.py --check`). The
order matters where a script reads the output of another: `run_controls.py`
needs the rigidity and bound tables, `run_frame_choice.py` the stored
benchmark cells, and `gt_error_budget.py` the transforms of
`estimate_vicon_transform.py`. `run_synthetic.py` and `run_frame_choice.py`
stop without an option while their tables exist, hence `--force`. The
validations and the anomaly worklist
write nothing:

```
# validations on synthetic exact ranges, in memory
python validate_multihop.py
python validate_mobile.py
python validate_slam.py
python validate_offset.py
python validate_crlb.py
# worklist of anomalous cells
python summarize.py --flag-anomalies
```

Requires the `radiocal` package with its extras, installed from the
repository root with `pip install -e "Code/datasets/radiocal[qa,methods]"`
(the `methods` extra brings scipy). `requirements.txt` at the repository root
pins the environment of the released tables. Existing cells are skipped unless
`--force` is given, and the repository README describes what each command
regenerates.

## Method roster (family x methods)

| family | sessions | methods |
|---|---|---|
| fully_meshed | all 25 | static suite: closed_form, mds, factor_graph, ekf, particle_filter, grid_filter; plus fg_offset, fg_offset_dev |
| multi_hop (k sweep) | all 25 (sweep: both static references) | static suite (disconnection at low k recorded as `status=disconnected`) |
| mobile_assisted | the 20 quasi-static set A sessions (8B05 at a marker) and the set B walks s01, s02 | static suite + the mobile as one extra unknown (`common.estimation_order`), plus fg_offset, fg_offset_dev, on set A; fg_slam with the anchor-anchor factors on the set B walks |
| slam_bipartite | s01, s02 (dynamic DD98) | ekf_slam, fg_slam (static suite undefined: no anchor-anchor pairs; a static tag is unidentifiable - see validate_slam.py) |

- [x] closed-form / LS baseline (`closed_form.py`) — kernel-less
- [x] MDS: classical + SMACOF x {gaussian, huber, cauchy} (`mds.py`)
- [x] factor-graph NLS x {gaussian, huber, cauchy} (`factor_graph.py`)
- [x] factor graph with range offsets x {gaussian, huber, cauchy}: one common offset (`fg_offset`) and per-device offsets under a prior (`fg_offset_dev`)
- [x] prior-anchored factor graph x {gaussian, huber, cauchy} (`fg_anchored`) - prior-sweep masks only
- [x] EKF x {gaussian, huber, cauchy} + gate (`ekf.py`)
- [x] particle filter x {gaussian, huber, cauchy} (`particle_filter.py`)
- [x] grid filter: product x {gaussian, huber, cauchy} + mixture (`grid_filter.py`)
- [x] EKF-SLAM + pose-chain factor graph x {gaussian, huber, cauchy} (`ekf_slam.py`, `fg_slam.py`) — slam_bipartite

## Conventions

- Primary estimate: session-aggregated (median) symmetrized range per pair,
  one solve per session; per-epoch statistics are diagnostics.
- Anchor ordering / gauge: origin `5BB6`, x-axis `4A23`, +y `8B05`
  (the paper convention of `radiocal.frames.fix_gauge`); the aligned RMSE is
  gauge-independent.
- Evaluation is 2-D against surveyed x, y; masks like `fully_meshed` strip the
  priors from the method's view, so the reference is always taken from the
  unmasked session (`evaluation.surveyed_reference`).
- Init family: batch and sequential estimators initialize from classical MDS
  (factor graph: the session-median matrix of the view over all epochs;
  EKF/PF: first 20 epochs, "mds20") so the benchmark compares estimation
  behavior, not init failures. The grid
  filter keeps the closed-form init of the historical IOPS pipeline.
- Frame of the per-anchor filters: the particle filter and the grid filter
  keep one belief per anchor, a particle cloud or a grid on a bounded box,
  which cannot carry the continuous family of joint solutions that a free
  frame allows. Their start therefore fixes the frame before inference,
  with anchor 0 at the origin and anchor 1 on the positive x-axis, from the
  classical-MDS start of the first 20 epochs (particle filter) or the
  closed-form construction around anchor 0 (grid filter).
- Factorial noise model: every method uses the shared kernels and session
  sigma from `kernels.py` (gaussian / huber k=1.345 / cauchy c=2.385 on
  u = r/sigma; sigma = max(pair epoch scatter, MDS-init residual MAD)), so
  the benchmark separates estimator class from kernel and scale. Mechanism
  variants (EKF gate, grid mixture) are compared against their
  matched-kernel baseline. The legacy per-method runners also wrote cauchy
  scale-sweep CSVs (sigma multiplier 0.25 to 4). `results/sigma_sweep.parquet`
  of `run_controls.py` supersedes them, and the legacy grid-filter sweep ran
  at twice the committed stride.
- Sigma doctrine (mask families): the driver computes sigma ONCE per
  session from its fully_meshed view and passes it to every mask cell of the
  session - sigma measures hardware/environment, not topology, so the mask
  axis stays a pure connectivity/information axis (identical on fully_meshed
  itself; the masked view's own sigma is recorded as
  `extras_json["sigma_masked_m"]`).
- Priors: the mask subsets (`multi_hop`, `mobile_assisted`) expose
  anchor priors. The static suite ignores them and evaluation stays
  Procrustes-aligned anchor RMSE, to preserve comparability with
  fully_meshed. The prior-information axis is covered by `fg_anchored` on
  the prior-sweep masks (`fully_meshed_p0..3`, `multi_hop_p0..3`), which
  turns the exposed priors into unary factors and adds the absolute
  (unaligned) error. Each prior is a Gaussian unary factor whose standard
  deviation is the `sigma_m` of its anchor in `nodes.yaml` (0.01 m for the
  surveyed anchors), so the surveyed coordinates fix the frame instead of a
  convention, as seed anchors do in Yudanto et al. (IPIN 2023) and Corbalán
  et al. (IEEE Access, 2023).
- Mobile evaluation: mobile_assisted scores the transform-applied mobile
  estimate against its surveyed marker (`mobile_error_m`; sessions with
  `truth_quality: unreliable` run fully but are excluded from every mobile
  aggregate). slam_bipartite scores checkpoint ATE at Vicon-covered epochs
  and within-segment RPE (1 s horizon) only - the bursty Vicon truth is
  never interpolated across gaps. Anchor RMSE stays the primary metric in
  every family.
- Method parameters (grid resolution, particle counts, seeds, iteration
  caps) are data-derived or documented constants, and the kernel constants
  are not tuned against the evaluation metric. Three settings were fixed
  after benchmark results had been inspected, as the lead of Section VI of
  the paper states: the shared session noise scale, which replaced per-method scales
  between two benchmark rounds, the MDS start of the EKF and the particle
  filter, and the second sweep of the particle filter.
- Benchmark table conventions: the `fully_meshed` replication covers all 25
  sessions. On set A (s21, s22, s01 to s20) the roving 8B05 is masked out of
  the quasi-static sessions and the 11 surveyed anchors remain. On set B
  (s03, s01, s02) the anchor mesh persists in the walks. Particle-filter
  cells run with five seeds (seed 0 with the other commands, seeds 1 to 4 in
  a loop), and the paper reports their mean. The per-method summary views
  and the noise-scale sweep use seed 0. The grid filter keeps its
  per-session epoch subsampling (stride = n_epochs // 75).
