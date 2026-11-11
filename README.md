# radiocal: reference implementations and result tables for infrastructure self-calibration

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24134342.svg)](https://doi.org/10.5281/zenodo.24134342)
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE-CODE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](Code/datasets/radiocal/pyproject.toml)

Where are my anchors? Radio positioning systems need the coordinates of
their fixed infrastructure, and surveying them by hand is the part everyone
would like to skip. This repository is the release of the tutorial paper
*A Tutorial on Infrastructure Self-Calibration for Radio-Based Indoor
Positioning* (under review, IEEE Communications Surveys & Tutorials). It
holds the code and the result tables behind the paper's examples and the
scripts that recompute those tables within the tolerances stated below. The
scripts that compute the range residuals, the tables and the numbers of the
paper's text and draw its figures stay with the manuscript (see
*Reproducibility: code-to-paper mapping*). The repository contains

- the **reference implementations** of the paper's engines (a closed form,
  multidimensional scaling, a factor graph, also with anchor priors and with
  range-offset states, an extended Kalman filter (EKF), a particle filter, a
  grid filter, and EKF and factor-graph estimators of simultaneous
  localization and mapping (SLAM)), with shared robust kernels and one noise
  scale per session,
- the **driver and the scripts that write the result tables** (the cells of
  the `benchmark` tables, rigidity certificates, bounds, controls, the
  synthetic baseline of Section VI and the frame-choice sweep), with a
  regression gate and the pinned environment that last wrote them
  (`requirements.txt`), and the runner that evaluates converted public range
  recordings (`Code/external/`),
- the **record of the literature search** of Section II-B of the paper, from
  which its Fig. 2 draws works for further study, and
- a **copy of UWB-ASC 1.0.0**, the UWB Anchor Self-Calibration Dataset: its
  measured sessions under `Data/` and the parts of its code that this
  release uses, which are the `radiocal` package that loads a session as a
  measurement graph and applies the topology views, the range simulator,
  and the readers and converter of public range recordings with their fetch
  list.

This release copies the measured sessions without UWB-ASC's 81 synthetic
draws. Script 05 of the package redraws them, and script 06 runs only in
UWB-ASC's repository. The repository shares its name with UWB-ASC's
`radiocal` package, which it copies under `Code/datasets/radiocal`. UWB-ASC
holds the data, their documentation, the raw recordings and the code that
this release copies, and its record is the canonical copy (see *The
measured sessions*).

The design idea in one sentence: every topology (fully meshed, multi-hop,
mobile-assisted, mobile-only) is a masked *view* of the same recorded
superset graph, never a separate recording, so differences between methods
and topologies are not confounded by different data.

## Companion paper

> P. Schwarzbach, "A Tutorial on Infrastructure Self-Calibration for
> Radio-Based Indoor Positioning," under review, *IEEE Communications
> Surveys & Tutorials*, 2026.

Section II-D of the paper introduces the running example and the
deployments of UWB-ASC, Section IV-C the topology views, Section V the
engines with their reference implementations (Table V), and Section VI the
examples, with the synthetic check in its lead. Its Data and Code
Availability statement summarizes what the release holds and reproduces,
and its Fig. 13 shows how the release is organized. This README carries
what the paper leaves to the release: the repository layout, the commands,
the storage format of the result tables, and the runtimes.

## Repository layout

```
radiocal/
├── Data/                        copy of the measured sessions of UWB-ASC 1.0.0 (CC BY 4.0)
│   ├── README.md                data dictionary (columns, units, frames)
│   ├── set_a/                   the NLOS deployment: 12-node mesh, 22 sessions
│   │   ├── README.md, campaign.yaml, nodes.yaml, truth_adjudication.csv,
│   │   │   los_nlos_field_labels.csv, masks/
│   │   └── sNN/                 ranges.parquet, session.yaml, qa/, and
│   │                            groundtruth.parquet for s01 to s20 only
│   └── set_b/                   the LOS deployment: 10 anchors + mobile, 3 sessions
│       ├── README.md, campaign.yaml, nodes.yaml, los_nlos_field_labels.csv, masks/
│       └── sNN/                 ranges.parquet, groundtruth.parquet, session.yaml, qa/
├── Code/
│   ├── datasets/radiocal/       UWB-ASC's installable package: loaders, masks, frames,
│   │                            QA, and the builders and further scripts (scripts/)
│   ├── external/                UWB-ASC's readers and converter for public range
│   │                            recordings, and the runner of this release
│   └── methods/                 estimators, driver, controls, bounds, and UWB-ASC's
│       │                        range simulator (range_errors.py, synthetic.py)
│       └── results/             released result tables (Parquet + CSV)
├── docs/
│   ├── FAIR.md                  FAIR principles and how the release meets them
│   ├── results_notes.md         root causes behind anomalies in the result tables,
│   │                            and the numbers behind four statements of the paper
│   ├── method_paper_map.csv     literature sources of the reference implementations
│   │                            and of kernels, evaluation, masks, noise scale
│   └── corpus/                  record of the literature search (Section II-B)
├── examples/                    four short scripts, the entry point
├── requirements.txt             pinned environment of the result tables
└── tests/                       smoke test
```

`groundtruth.parquet` is optional per session: the static set A sessions
`s21` and `s22` have no mobile node and therefore no file.

## Code map (where to look first)

| goal | look at |
|---|---|
| load a session as a measurement graph | `radiocal.load_session` (`Code/datasets/radiocal/`) |
| understand the file formats | `Data/README.md`, then the READMEs of `set_a` and `set_b` |
| apply a topology view | `Data/<campaign>/masks/*.yaml`, `graph.apply_mask(...)` |
| see a minimal end-to-end run | `examples/03_reproduce_benchmark_cell.py` |
| run the reference implementations | `Code/methods/run_benchmark.py` (commands below) |
| inspect the released results | `Code/methods/results/overview.md` and `benchmark.csv` |
| trace an anomaly in the tables | `docs/results_notes.md` |
| rebuild the sessions from the raw recordings | UWB-ASC's repository and record (see *Raw recordings*) |
| run them on public recordings | `Code/external/README.md` (see *Public range recordings*) |
| draw synthetic ranges with known truth | `Code/methods/range_errors.py`, `examples/04_what_if_ranges.py` |

## Requirements

- Python 3.10 or newer and no GPU, tested on Windows 11 with Python 3.10.20
  and 3.14.6 (versions and times under *Runtimes*)
- a POSIX shell such as bash for the commands of this README (Git Bash or
  WSL on Windows)
- about 1 GB of free memory
- about 50 MB of disk for the repository
- MATLAB users can read the data directly with `parquetread` (R2019a or
  newer) and any YAML parser

## Installation

```
git clone https://github.com/PaulSchwarzbach/radiocal.git
cd radiocal
pip install -e "Code/datasets/radiocal[qa,methods]"
```

The editable install resolves current versions within the bounds of
`pyproject.toml`, which keep pandas below 3 (see *Runtimes*), on every
supported Python. The `qa` extra adds matplotlib (plots, quality reports),
and `methods` adds scipy (estimator implementations). Plain
`pip install -e Code/datasets/radiocal` suffices for loading data.

To reproduce the result tables as *What agrees across environments*
describes, install the pinned library versions in which they were last
written before the editable install. They need Python 3.12 or newer:

```
pip install -r requirements.txt
pip install -e "Code/datasets/radiocal[qa,methods]"
```

## Quick start

```python
import radiocal

g = radiocal.load_session("Data/set_b/s01")   # superset graph
view = g.apply_mask("slam_bipartite")          # mobile-only view

view.edges           # DataFrame: t, t_raw, seq, from_id, to_id, range_m
view.nodes           # DataFrame: id, role, prior_type, x, y, z, sigma_m
view.symmetrized()   # undirected ranges
```

Then work through `examples/`: loading and plotting, the four topology
views, reproducing one benchmark cell against the released results, and a
what-if run of the range generator.

## Quick check

From the repository root:

```
python tests/test_smoke.py
```

The smoke test loads two sessions, checks the invariants of the four masks,
re-derives five frozen cells of `Code/methods/results/regression_frozen.csv`
within 5.1e-4 m (among them the cauchy factor graph, which uses the stored
noise scale) and reads the benchmark table. It runs within seconds
(*Runtimes*) and prints

```
ok  sessions load (191423 + 351263 ranges)
ok  mask invariants hold on Data/set_b/s01
ok  mask invariants hold on Data/set_a/s07
ok  set_b/closed_form/disambiguated: rmse 1.694 m == frozen
ok  set_a/closed_form/disambiguated: rmse 18.551 m == frozen
ok  set_b/mds/classical: rmse 0.258 m == frozen
ok  set_a/mds/classical: rmse 3.061 m == frozen
ok  set_b/factor_graph/cauchy: rmse 0.223 m == frozen
ok  benchmark table readable (3348 rows)

smoke test passed
```

No warnings are expected. Only the first plot in a fresh environment
(example 01) may print matplotlib's notice that it builds its font cache.
`python examples/03_reproduce_benchmark_cell.py` recomputes one cell from
scratch and ends with "OK: the shipped results are reproducible on this
machine." The regression gate under *Views and reference implementations*
is the thorough check.

## The measured sessions

`Data/` holds a copy of the measured sessions of UWB-ASC 1.0.0, two
deployments with two-way ranging in one industrial hall in Torgau, Germany.
The companion paper calls `set_b` the LOS deployment and `set_a` the NLOS
deployment.

| set | name in the paper | sessions | `groundtruth.parquet` |
|---|---|---|---|
| `set_b` | LOS deployment | `s01`, `s02` walks, `s03` static | every session (`s03`: unregistered poses of the standing tag) |
| `set_a` | NLOS deployment | `s01` to `s20` quasi-static, `s21`, `s22` static | `s01` to `s20` only |

The data descriptor of UWB-ASC (P. Schwarzbach, preprint
[arXiv:2611.11579](https://arxiv.org/abs/2611.11579)) documents the
deployments and their ground truth, and UWB-ASC's Zenodo record
(<https://doi.org/10.5281/zenodo.24055896>) archives the sessions and the
raw recordings. `Data/README.md` defines the columns, units, fields and
frames that the code of this release reads, and `Data/set_a/README.md` and
`Data/set_b/README.md` state the limits of each deployment. The copied
`Data/README.md` also documents UWB-ASC's synthetic draws, which this
release leaves out.

## Views and reference implementations

Four masks per set define the topology dimension:

- `fully_meshed`: anchor-to-anchor ranges only, no priors (blind self-calibration)
- `multi_hop`: the anchor mesh thinned by range to the kappa nearest
  neighbors, kappa = 3 in the canonical view
- `mobile_assisted`: the full mesh plus mobile edges and a minimal subset of
  anchors with priors, three on set B and two on set A, where 8B05, the third
  anchor of the subset, is the mobile
- `slam_bipartite`: the mobile-only topology, mobile-to-anchor ranges only,
  solved by the SLAM estimators (the key is kept for compatibility with the
  result tables)

The registry in `Code/methods/registry.py` holds the eight reference
implementations (`closed_form`, `mds`, `factor_graph`, `ekf`,
`particle_filter`, `grid_filter`, `ekf_slam`, `fg_slam`) and three further
factor-graph keys for the analyses of Section VI: `fg_anchored` (anchor
priors as unary factors, the prior sweep) and `fg_offset` and
`fg_offset_dev` (one common range offset, or per-device offsets under a weak
prior). Every implementation runs through one adapter,
`run(graph, variant, *, sigma_m, sigma_scale=1.0, seed=None)`, which the
paper calls `run` (Section II-D). Among the result tables, only the
noise-scale sweep changes `sigma_scale`. Both SLAM estimators start from a
joint classical-MDS embedding of the anchors and every 25th epoch pose
(`slam_common.POSE_STRIDE`), with the poses in between interpolated, and
every epoch enters the solve itself.

### Reproducing the released results

Run the commands from `Code/methods`, in a POSIX shell such as bash (Git
Bash or WSL on Windows). The driver skips every cell whose key already exists
in the result tables, so on a fresh clone all cells count as done, and
`--force` recomputes them. The benchmark lines below carry it, and they
rewrite the released rows in place, so keep a copy of `results/` to compare
against. `run_rigidity.py`, `run_crlb.py` and
`run_controls.py` always recompute their views. Keep `results/` in place: the
gate reads its reference `results/regression_frozen.csv` there, and later
steps read the tables of earlier ones. The list below regenerates every row
of the released result tables and the YAML files under `results/transforms/`. A
dry run of each benchmark command (`--dry-run`) confirms the coverage against
`benchmark.parquet`.

```
cd Code/methods
python run_benchmark.py --check-regression
python run_benchmark.py --mask fully_meshed,multi_hop,mobile_assisted,slam_bipartite --force
python run_benchmark.py --sessions "set_a/s21,set_b/s03" --mask multi_hop_sweep --force
python run_benchmark.py --sessions "set_a/s21,set_b/s03" --mask multi_hop_random_sweep --methods closed_form,mds,factor_graph --force
python run_benchmark.py --sessions "set_a/s21,set_b/s03" --mask prior_sweep --force
for s in 1 2 3 4; do
  python run_benchmark.py --methods particle_filter --seed $s --mask fully_meshed,multi_hop,mobile_assisted --force
  python run_benchmark.py --methods particle_filter --seed $s --sessions "set_a/s21,set_b/s03" --mask multi_hop_sweep --force
done
# rigidity certificate of every anchor-only view
python run_rigidity.py
# Gaussian bounds (Cramer-Rao) per view
python run_crlb.py
# oracle start, noise-scale sweep, offset control, sigma terms
python run_controls.py
# synthetic baseline of Section VI (rewrites results/synthetic_baseline/cells)
python run_synthetic.py --force
# frame-choice sweep of Sections IV-A and VI-E (rewrites results/frame_choice/)
python run_frame_choice.py --force
# Vicon registration cross-check, then the ground-truth error budget
python estimate_vicon_transform.py --session set_b/s01,set_b/s02 --static-crosscheck
python gt_error_budget.py
# summary tables in the Markdown files under results/
python summarize.py
```

What the commands cover:

- the default mask list runs every session on which a view is defined: the
  fully meshed and canonical multi-hop views on all 25 sessions, the
  mobile-assisted view on the quasi-static set A sessions and on the set B
  walks (`fg_slam` with the anchor-to-anchor factors), and the mobile-only
  view on the set B walks;
- `multi_hop_sweep` thins by range to kappa = 1, 2, 3, 4, 5, 6 and 8, and
  `multi_hop_random_sweep` thins at random (three seeds per kappa), both on
  the static reference sessions `set_a/s21` and `set_b/s03`;
- `prior_sweep` runs `fg_anchored` with 0 to 3 known anchors on the fully
  meshed and the kappa = 3 views of the reference sessions;
- the particle filter is the only stochastic estimator. Seed 0 runs with the
  other commands, and the loop adds seeds 1 to 4 on every particle-filter
  cell. The paper reports the mean over the five seeds. The other random
  steps carry fixed seeds in the code: the random thinning seeds 0, 1 and 2
  (`family_masks.RANDOM_SEEDS`), the perturbation of the rigidity test seed 0
  (`rigidity.rigidity_rank`), the synthetic draws seeds 0 to 19
  (`run_synthetic.py`), and each validation script its own.

`run_controls.py` reads the rigidity and bound tables, `run_frame_choice.py`
the stored benchmark cells, and `gt_error_budget.py` the transforms of
`estimate_vicon_transform.py`, so each runs after the scripts that write its
inputs.

Four kinds of released files are not regenerated by the list: the CSV files
under `results/<method>/` (per-anchor estimates, scale sweeps, convergence
traces and pair weights of the two reference sessions),
`results/regression_frozen.csv`, the gate's frozen reference, and
`results/synthetic_baseline/frozen.csv` and `results/frame_choice/frozen.csv`,
the frozen references of the synthetic baseline and of the frame-choice
sweep, which `run_synthetic.py --freeze` and `run_frame_choice.py --freeze`
write once and never overwrite. Both scripts stop without an option while
their tables exist, so the list calls them with `--force`. The per-method
scripts `run_<method>.py` that wrote the former predate the driver. They take
no options, run one method on the two reference sessions with its
noise-scale sweep and convergence traces, and overwrite the files under
`results/<method>/`, including the released summary. Their scale sweeps are
superseded by `results/sigma_sweep.parquet`.

### Regression gate

`run_benchmark.py --check-regression` recomputes the fully meshed cells of
the reference sessions `set_b/s03` and `set_a/s21` and compares 40 of them
with the frozen values in `results/regression_frozen.csv`. A cell passes when
its status is ok and its `rmse_m` lies within 5e-4 m of the frozen value.
The frozen values carry three decimals, so this tolerance includes their
rounding. Differences in `max_error_m` or the reflection flag only produce a
warning.
The gate then reruns the synthetic baseline of `run_synthetic.py` in memory
against `results/synthetic_baseline/frozen.csv`, with the same tolerance,
and reports a changed hash of the drawn ranges as generator drift. Last, it
reruns the gated rows of the frame-choice sweep of `run_frame_choice.py` in
memory against `results/frame_choice/frozen.csv`, with the same tolerance:
196 rows of `set_b/s03`, the defining pairs of the faithful closed form and
the cauchy factor graph and the seed triples of `fg_anchored` that are not
near-collinear. In the synthetic part, cells on flexible views are reported
but not gated, and the synthetic and frame-choice parts write nothing. The gate runs only under
this flag. A run that passes all three parts writes all 52 benchmark cells
it computed back into the result tables, the
40 gated cells and the 12 cells of `fg_offset` and `fg_offset_dev`, which
have no frozen value. At the
precision of the CSV mirror only `run_at` and `runtime_s` change, and the
Parquet floats can differ in their last digits (*What agrees across
environments*). The smoke test of *Quick check* is the faster check.

### What agrees across environments

In the pinned environment of `requirements.txt`, a rerun reproduces the
shipped values. The regenerated benchmark cells agree within 1e-10 m. On
views that are not rigid their metrics agree within 1e-9 m and their
per-anchor estimates within 1e-8 m. The rows of `oracle_start`,
`sigma_sweep`, `offset_control`, `offset_crlb` and `sigma_terms` agree bit
for bit. The synthetic baseline also reproduces bit for bit there: two runs
of `run_synthetic.py --force` wrote identical files. Outside the pinned
environment the Parquet files differ in their writer metadata, which names
the pyarrow version, even where every value is equal, so compare values
rather than file hashes. One shipped cell is the exception. The
cauchy factor graph on the kappa = 2 view of `set_b/s03` stores 1.266 m from
a run before the last change of the solver code, and the released code gives
1.236 m, the value that `oracle_start` stores for the start from classical
MDS. The view is flexible, so the solver ends somewhere on a continuum of
geometries of equal cost, and this end point also moves with the library
versions (1.253 m with numpy 2.5). Apart from this cell, the other tested
environments, Python 3.10 with numpy 2.2 and pandas 2.3 and Python 3.14 with
numpy 2.5 and pandas 2.3 or 3.0, agree with the shipped tables far inside the
acceptance bound of the gate (5e-4 m). The metric columns differ by less
than 1e-6 m and the solver-frame coordinates of the SLAM trajectories by less
than 1e-4 m. pandas 3 changes no value but runs many times slower
(*Runtimes*).

Some fields change on every run: `run_at` and `runtime_s` in every table,
the runtime totals that `summarize.py` writes from them, and the `date` in
the provenance of the transform YAML files. Rows written before
the last code changes leave `n_priors`, `n_components` and
`mobile_truth_quality` empty in places, name a failed construction
`ValueError` where the released code writes `UndefinedConstructionError`, and
carry fewer keys in `extras_json`. A rerun fills these fields and leaves the
metric columns unchanged, apart from the kappa = 2 cell above.

### Storage format

Results land in `Code/methods/results/` as long-format Parquet tables with a
CSV mirror, keyed by campaign, session, mask, method, variant and seed.
`benchmark` holds one row per cell, `benchmark_estimates` one row per anchor
and cell, and `benchmark_trajectories` the mobile poses of the SLAM cells.
Deterministic methods are stored under the seed -1, and the particle filter
records its actual seed. Every run upserts the rows it computes, so partial
and repeated runs are safe. The tables `rigidity`, `crlb`, `oracle_start`,
`sigma_sweep`, `offset_control`, `offset_crlb` and `sigma_terms` hold the
rigidity tests, bounds and controls of Sections IV and VI.
`synthetic_baseline/cells` holds the synthetic baseline of Section VI,
one row per condition, draw, view, kernel scale, method and variant, with a
hash of the drawn ranges. `frame_choice/` holds the frame-choice sweep of
Sections IV-A and VI-E: `pairs` one row per defining pair and engine
variant, `triples` one row per seed triple, and `line` the offsets of the
anchors from the line of the release pair. Cells whose construction
leaves too few anchors record the status `undefined`, and cells on a view
that falls apart record `disconnected`. The particle filter is the exception:
it returns status ok on the three views that the rigidity table certifies as
disconnected (`docs/results_notes.md`).

### Runtimes

Wall times of the documented commands in the pinned environment of
`requirements.txt` (Python 3.14.6, Windows 11), on a laptop with an Intel
Core Ultra 7 355 (8 threads) and 31.5 GB of RAM. No command needed more than
0.8 GB of memory.

| command | wall time |
|---|---|
| `pip install -r requirements.txt`, then the editable install | about 2 min |
| `python tests/test_smoke.py` | 4 s |
| `python examples/01_load_and_plot.py`, `02_...`, `03_...`, `04_...` | 6 s, 2 s, 2 s, 2 s |
| `python run_benchmark.py --check-regression` | about 4.3 min |
| `python run_rigidity.py` | 52 s |
| `python run_crlb.py` | 41 s |
| `python run_controls.py` | 7.5 min |
| `python run_synthetic.py --force` | about 45 s |
| `python run_synthetic.py --check`, `python run_frame_choice.py --check` | 40 s, 29 s |
| `python run_frame_choice.py --force` | about 19 min |
| `python validate_multihop.py`, `validate_mobile.py`, `validate_slam.py`, `validate_offset.py`, `validate_crlb.py` | 3.5 min, 75 s, 9 s, 3 s, 3 s |
| `python estimate_vicon_transform.py --session set_b/s01,set_b/s02 --static-crosscheck` | 10 s |
| `python gt_error_budget.py`, `python summarize.py` | 2 s each |

The benchmark commands recompute cells whose solver time the result tables
store in `runtime_s`. Summed over the released rows, the four default views
take about 2 h, the particle-filter seeds 1 to 4 about 43 min, the kappa
sweep about 10 min, the random sweep about 2 min and the prior sweep a few
seconds, 2.9 h of solver time in all, plus the loading of the sessions. The
tables of the frame-choice sweep store about 16 min of solver time for
`run_frame_choice.py --force`.
Python 3.10 with numpy 2.2 and Python 3.14 with numpy 2.5 ran about as fast
as the pinned environment. With pandas 3 the same commands ran 20 to 30 times
slower, with identical values, which is why `pyproject.toml` keeps pandas
below 3. Median runtime per benchmark cell over the released rows with
status ok:

| method | median [s] | method | median [s] |
|---|---|---|---|
| `fg_offset`, `fg_offset_dev` | 0.02 | `ekf` | 0.25 |
| `closed_form` | 0.05 | `ekf_slam` | 0.45 |
| `fg_anchored` | 0.08 | `particle_filter` (per seed) | 0.68 |
| `factor_graph` | 0.11 | `fg_slam` | 1.4 |
| `mds` | 0.17 | `grid_filter` | 31 |

A full rerun with `--force` therefore takes hours, and the released result
tables make that unnecessary for most uses.

## Public range recordings

`Code/external/` holds UWB-ASC's readers and converter, which turn public UWB
range recordings into the format of `Data/`, and the runner of this release,
which runs the reference implementations on them unchanged. Section II-D of
the paper introduces the evaluation scan of TWR-CLOUD-Bern (Zenodo, MIT
license), and Section VI-B reports the run on it. The selfloc testbeds
(GitHub, no license stated) are converted for analysis only. Neither
recording is part of this repository.
`Code/external/README.md` lists the files to fetch with their checksums, the
run order, and how the result files map to the numbers of the paper. The
converted campaigns and the results go to a folder outside the repository,
and the released result tables stay untouched.

## Reproducibility: code-to-paper mapping

Every figure, table and algorithm box of the paper with the script that
produces it. The scripts marked *paper-side* live in the paper's source
repository as part of its build and are not part of this release. They read
the released files named in the last column, and items without a released
generator or input say so. The numbers in the text of the paper come from
the same paper-side generators. `derive_residuals.py` computes the range
residuals from the sessions under `Data/` into a cache that the later
scripts read.

| paper item | label | producing script | released input |
|---|---|---|---|
| Figs. 1, 3, 12 and 13 | `fig:graphical_abstract`, `fig:topologies_fg`, `fig:decision_tree`, `fig:release_structure` | drawn (TikZ, paper-side) | none |
| Fig. 2 | `fig:two_axis_space` | drawn by hand (TikZ, paper-side). 23 of its 31 works come from the literature record `docs/corpus/classification.csv` | none |
| Fig. 4 | `fig:range_model` | `derive_residuals.py`, `fig_range_model.py` with `fig_grid_belief.py` (paper-side), with the grid update of `Code/methods/grid_filter.py` | `Data/set_b` |
| Fig. 5 | `fig:identifiability` | drawn (TikZ, paper-side) for (a) to (c), `fig_identifiability.py` (paper-side) for their geometry and for (d) to (f) | none |
| Fig. 6 | `fig:views_graph` | `derive_residuals.py`, `fig_views_graph.py` (paper-side) | `Data/set_b`, `benchmark`, `benchmark_trajectories` |
| Fig. 7 | `fig:anchor_scatter` | `fig_anchor_cloud.py`, then `fig_flex_view.py` (paper-side) | `Data/set_a/s21`, `benchmark`, `benchmark_estimates`, `oracle_start`, `crlb` |
| Fig. 8 | `fig:error_propagation` | `derive_residuals.py`, `fig_error_propagation.py` (paper-side) | `Data/`, `benchmark`, `benchmark_estimates` |
| Fig. 9 | `fig:multi_hop_ksweep` | `derive_residuals.py`, `fig_ksweep_causes.py` (paper-side) | `Data/`, `benchmark`, `crlb`, `rigidity`, `oracle_start` |
| Fig. 10 | `fig:pair_stretch` | `derive_residuals.py`, `fig_pair_stretch.py` (paper-side) | `Data/`, `benchmark`, `benchmark_estimates` |
| Fig. 11 | `fig:slam_trajectory` | `fig_slam_init.py` (paper-side), replaying `Code/methods/ekf_slam.py` | `Data/set_b/s01`, `Data/set_b/s02`, `benchmark`, `benchmark_estimates`, `benchmark_trajectories` |
| Tables I to V, Algorithms 1 to 5 | `tab:tech_primitives`, `tab:notation`, `tab:kernels`, `tab:nuisance_states`, `tab:engines`, `alg:mds`, `alg:lm`, `alg:ekf`, `alg:pf`, `alg:grid` | written in the text | none |
| Table VI and the numbers in the text | `tab:lessons_effect` | `derive_residuals.py`, `fig_ksweep_causes.py`, then `gen_benchmark_tables.py` (paper-side) | the result tables of `Code/methods/results/` with `synthetic_baseline/cells`, `frame_choice/` and `transforms/error_budget.yaml`, `Data/set_a/truth_adjudication.csv`, `Data/set_b/s03`, `Data/set_b/nodes.yaml` |
| the numbers of the public scan in Section VI-B | none | `Code/external/convert_external.py`, `qa_external.py` and `run_external.py`, then `gen_external_tables.py` (paper-side) | none (the public TWR-CLOUD-Bern files, fetched as `Code/external/README.md` describes) |

The scripts in `Code/methods` recompute the released result tables as listed
under *Reproducing the released results*, apart from the four kinds of files
named there. `docs/results_notes.md` (section 6) gives the selection behind
each row of Table VI and the file behind the pair weight of Section V-B.

## Raw recordings

The raw recordings, their SHA-256 manifest and the scripts that rebuild the
measured sessions from them belong to UWB-ASC. Its Zenodo record archives
the recordings, and its README gives the commands (section *What the
scripts rebuild*), which run in UWB-ASC's repository
(<https://github.com/PaulSchwarzbach/UWB-ASC-dataset>). The scripts under
`Code/datasets/radiocal/scripts/` are the code of UWB-ASC 1.0.0. Their
docstrings and `Code/datasets/radiocal/README.md` describe UWB-ASC's
repository, so "this release" there means UWB-ASC. Script 06 reads
`Data/synthetic`, which this copy leaves out, and runs only there. The
reference implementations, the driver and the examples of this release read
the canonical sessions under `Data/` and need no raw file.

## FAIR

UWB-ASC maps its data, their documentation and the rebuild to the FAIR
principles in its own `docs/FAIR.md`. [`docs/FAIR.md`](docs/FAIR.md) of
this release does the same for the code, the result tables and the
literature record.

## Versioning

Semantic versioning, see [`CHANGELOG.md`](CHANGELOG.md). Each GitHub release
is archived on Zenodo under a version DOI, and the concept DOI
`10.5281/zenodo.24134342` resolves to the latest version.

## Citation

If you use this release, cite the companion paper:

```bibtex
@article{schwarzbach_selfcal_tutorial_2026,
  author  = {Schwarzbach, Paul},
  title   = {A Tutorial on Infrastructure Self-Calibration for Radio-Based
             Indoor Positioning},
  journal = {IEEE Communications Surveys \& Tutorials},
  year    = {2026},
  note    = {Under review}
}
```

and the archived release:

```bibtex
@misc{schwarzbach_radiocal_2026,
  author    = {Schwarzbach, Paul},
  title     = {radiocal: reference implementations and result tables for
               infrastructure self-calibration},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.24134343},
  note      = {Code and data}
}
```

For the measured sessions, cite UWB-ASC by its record and its data
descriptor:

```bibtex
@misc{schwarzbach_uwbasc_2026,
  author    = {Schwarzbach, Paul},
  title     = {{UWB} Anchor Self-Calibration Dataset ({UWB-ASC})},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.24055896}
}

@misc{schwarzbach_uwbasc_descriptor_2026,
  author       = {Schwarzbach, Paul},
  title        = {Descriptor: {UWB} Anchor Self-Calibration Dataset ({UWB-ASC})},
  year         = {2026},
  howpublished = {submitted to IEEE Data Descriptions},
  note         = {arXiv:2611.11579}
}
```

The readers of `Code/external` convert recordings of other groups. If you
use them, cite those recordings and their descriptions as well. For
TWR-CLOUD-Bern:

```bibtex
@misc{xenakis_twr-cloud-bern_2026,
  author    = {Xenakis, Dimitrios and Di Maio, Antonio and Braun, Torsten},
  title     = {Two-Way-Ranging-based Cooperative Localization with an Open
               {UWB} Dataset from {Bern-University} ({TWR-CLOUD-Bern})},
  year      = {2026},
  publisher = {Zenodo},
  version   = {5},
  doi       = {10.5281/zenodo.20277908}
}

@article{xenakis_cooperative_2026,
  author  = {Xenakis, Dimitris and Di Maio, Antonio and Braun, Torsten},
  title   = {Descriptor: Two-Way-Ranging-Based Cooperative Localization With
             an Open {UWB} Dataset From {Bern-University} ({TWR-CLOUD-Bern})},
  journal = {IEEE Data Descriptions},
  volume  = {3},
  pages   = {441--452},
  year    = {2026},
  doi     = {10.1109/IEEEDATA.2026.3701019}
}
```

For selfloc:

```bibtex
@article{corbalan_self-localization_2023,
  author  = {Corbal{\'a}n, Pablo and Picco, Gian Pietro and Coors, Martin and
             Jain, Vivek},
  title   = {Self-Localization of Ultra-Wideband Anchors: From Theory to
             Practice},
  journal = {IEEE Access},
  volume  = {11},
  pages   = {29711--29725},
  year    = {2023},
  doi     = {10.1109/ACCESS.2023.3261567}
}

@misc{corbalan_selfloc_2023,
  author       = {{d3s-trento}},
  title        = {selfloc: Anchor Self-Localization},
  year         = {2023},
  howpublished = {GitHub repository, commit 39c1271},
  url          = {https://github.com/d3s-trento/selfloc}
}
```

## Licenses

Licensed by folder, as in UWB-ASC:

- **Code** (everything under `Code/`, `examples/` and `tests/`, their
  READMEs included, apart from `Code/methods/results/`): MIT, see
  [`LICENSE-CODE`](LICENSE-CODE). The copied code of UWB-ASC keeps
  UWB-ASC's license notice, which the copied `radiocal` package carries
  (`Code/datasets/radiocal/LICENSE`), and [`NOTICE`](NOTICE) lists its
  files.
- **Data and documentation** (everything under `Data/`,
  `Code/methods/results/` and `docs/`, the literature record included, and
  the files at the repository root apart from the licenses and `NOTICE`):
  Creative Commons Attribution 4.0 International, see
  [`LICENSE-DATA`](LICENSE-DATA).

## Questions and issues

Open a [GitHub issue](https://github.com/PaulSchwarzbach/radiocal/issues)
for questions, bug reports, or requests. For anything else, contact the
author (see `CITATION.cff`).

## Acknowledgments

The acknowledgments for the measurement campaigns stand in UWB-ASC's README.

## Declaration on Generative AI

The author used Anthropic's Claude models through the Claude Code
command-line interface. The model versions were not pinned. The models
helped develop the reference implementations with their driver, the
evaluation behind the result tables, the run on the public scan, the summary
scripts, the examples and the tests. The author set the content of the
documentation, and the models formulated and edited its text. They also
triaged and coded the literature record under `docs/corpus/`, and the author
approved the works that Fig. 2 of the paper draws from it. UWB-ASC's data
descriptor and README declare the use for its code, its documentation and
its field labels, of which this release holds a copy. That code includes the
conversion of the public scan and the port of the range-error generator. The
author reviewed and revised the code and the documentation and takes full
responsibility for the content of this repository.
