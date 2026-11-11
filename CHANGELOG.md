# Changelog

All notable changes to this repository are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versioning follows [Semantic Versioning](https://semver.org/): MAJOR =
canonical-schema breaking change, MINOR = new sessions, methods, or masks,
PATCH = docs and metadata.

## [1.0.0] — 2026-11-11

Initial public release.

### Added

- **Reference implementations** (MIT): closed form (sequential
  trilateration), MDS and SMACOF, factor graph, EKF, particle filter, grid filter, EKF-SLAM and
  FG-SLAM, with shared robust kernels (Gaussian, Huber, Cauchy) and a
  shared session noise scale, and a prior-anchored factor-graph variant
  (`fg_anchored`) that takes the anchor priors a view exposes as Gaussian
  unary factors.
- **Driver and result tables**:
  - `run_benchmark.py` (session × mask × method × variant × seed,
    idempotent upserts, `--check-regression` gate) with the long-form result
    tables `benchmark`, `benchmark_estimates` and `benchmark_trajectories`
    as Parquet + CSV and per-family summaries
  - the reference-information sweep masks `fully_meshed_p0..3` and
    `multi_hop_p0..3` (alias `prior_sweep`), with absolute (non-aligned)
    error columns from two known anchors on
  - Gaussian (Cramér–Rao) bounds per view (`crlb.py`, `run_crlb.py`,
    `results/crlb.parquet`, validated by `validate_crlb.py`)
- **Offset variants**: `fg_offset` (one common range offset) and
  `fg_offset_dev` (per-device offsets under a 0.10 m Gaussian prior) on the
  fully meshed and quasi-static mobile-assisted views, with the control
  tables `offset_control` and `offset_crlb` and the check script
  `validate_offset.py`.
- **Thinning sweeps**: range thinning `multi_hop_k<K>` (alias
  `multi_hop_sweep`) and random thinning `multi_hop_random_k<K>_s<S>`
  (alias `multi_hop_random_sweep`) on the reference sessions, and the
  rigidity certificate of every anchor-only view (`run_rigidity.py`,
  `results/rigidity`).
- **Controls**: oracle start, noise-scale sweep and sigma terms
  (`run_controls.py`, `results/oracle_start`, `sigma_sweep`,
  `sigma_terms`), and the ground-truth error budget (`gt_error_budget.py`,
  `results/transforms/error_budget.yaml`).
- **Solver settings**: SMACOF runs to a convergence rule (relative objective
  change below 1e-13, stagnation guard 1e-12 m), the factor graph to at most
  10,000 iterations, and the particle filter runs seeds 0 to 4 on every cell.
- **Synthetic baseline**: `run_synthetic.py` (five conditions on the
  surveyed anchors of `set_b/s03`, the full mesh and the kappa = 3 view, the
  factor graph at two kernel scales, `--force`, `--freeze`, `--check`) with
  `results/synthetic_baseline/` (cells, frozen reference, summary).
  `examples/04_what_if_ranges.py` varies the settings of the range
  simulator.
- **Frame choice**: `run_frame_choice.py` (every defining anchor pair and
  every seed triple on the full mesh of `set_b/s03` and `set_a/s21`,
  `--force`, `--freeze`, `--check`) with `results/frame_choice/` (pairs,
  triples, line, frozen reference, summary).
- **Regression gate**: `run_benchmark.py --check-regression` compares 40
  fully meshed cells of the two reference sessions with their frozen
  values, reruns the synthetic baseline against its frozen reference, and
  checks the 196 gated rows of the frame-choice sweep against theirs.
- **Public range recordings**: the runner of `Code/external/`
  (`run_external.py`, `qa_external.py`, `ext_context.py`) runs the
  reference implementations on public recordings that the readers of
  UWB-ASC convert, outside the repository and without redistributing either
  recording.
- **Copy of UWB-ASC 1.0.0** (<https://doi.org/10.5281/zenodo.24055896>):
  the measured sessions of `set_a` and `set_b` under `Data/` (CC BY 4.0,
  without the synthetic draws), the `radiocal` package (MIT: loaders,
  topology masks, frames, QA, and the builders and further scripts), the range
  simulator (`range_errors.py`, a seeded port of the range-error generator
  of Schwarzbach et al., IEEE Sensors J., 2022, and `synthetic.py`), and the
  readers and the converter of public range recordings with their fetch
  list. UWB-ASC documents the data, their raw sources and the rebuild.
- **Pinned environment**: `requirements.txt`, the library versions in which
  the result tables were last written.
- **Documentation and metadata**: `CITATION.cff`, `.zenodo.json`,
  `docs/FAIR.md`, `docs/results_notes.md` (root causes of anomalies in the
  result tables, and the numbers behind four statements of the companion
  paper), `docs/method_paper_map.csv`, the literature classification record
  of the companion paper (`docs/corpus/`), examples, smoke test.
