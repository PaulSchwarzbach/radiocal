# Code/external: readers and benchmark runs for public range recordings

These scripts convert public UWB range recordings into the campaign format of
`Data/README.md` and run the released benchmark code on them, unchanged.
Section II-D of the companion tutorial introduces the evaluation scan of
TWR-CLOUD-Bern, Section VI-B reports the run on it, and the two selfloc
testbeds are converted for analysis only. The scripts write nothing into `Data/` or
`Code/methods/results/`. Converted campaigns and results go to a folder of
your choice outside the repository, called `ext_root` below.

The readers, `sources.yaml`, `session_writer.py` and `convert_external.py`
need only the `radiocal` package, and they also ship with the UWB Anchor
Self-Calibration Dataset (UWB-ASC). `ext_context.py`, `qa_external.py` and
`run_external.py` run the benchmark of `Code/methods/`, which only the
tutorial's release holds. The parts of this file that need them are marked
"(benchmark)".

## Sources and licenses

Nothing of either source is redistributed here. Fetch the files yourself into
`<ext_root>/raw/`, keeping the paths given below.

| source | files the converter reads | license | folder under `<ext_root>/raw/` |
|---|---|---|---|
| TWR-CLOUD-Bern, Zenodo record [10.5281/zenodo.20277908](https://doi.org/10.5281/zenodo.20277908), version 5 | `40-agents.json` | MIT (Zenodo record) | `twr_cloud_bern_v5/` |
| selfloc, [github.com/d3s-trento/selfloc](https://github.com/d3s-trento/selfloc), commit `39c127198a5247dda60e7e711969a286053bac35` | `data/department/sstwr.csv`, `data/reception/sstwr.csv`, `sw/deployments/department.csv`, `sw/deployments/reception.csv` | none stated | `selfloc_39c1271/`, with the repository paths |

For selfloc, cloning the repository into `<ext_root>/raw/selfloc_39c1271` and
checking out the commit gives the expected layout:

```
git clone https://github.com/d3s-trento/selfloc <ext_root>/raw/selfloc_39c1271
git -C <ext_root>/raw/selfloc_39c1271 checkout 39c127198a5247dda60e7e711969a286053bac35
```

`sources.yaml` lists the SHA-256 of every file the converter reads, and
`convert_external.py` stops on any mismatch. It also lists the files of the
Bern record that were read only as documentation (the authors' scripts and
the registration report) and those never fetched. The converter parses JSON
and CSV files and executes nothing from the downloads. selfloc states no
license, so the tutorial reports no selfloc result, and the converted selfloc
campaigns and their per-node estimates stay in your `ext_root`.

## Setup

1. Install the release as described in the top-level README. The converter
   needs only the `radiocal` package. The benchmark scripts also need the
   `methods` extra (`pip install -e "Code/datasets/radiocal[qa,methods]"`).
2. Choose an empty folder outside the repository as `<ext_root>`. The
   converter takes it on the command line (`--raw-root <ext_root>/raw --out
   <ext_root>/data`), so `sources.yaml` can stay as shipped, and the smoke
   test of UWB-ASC, which checks its SHA-256, still passes. `qa_external.py`
   and `run_external.py` read `ext_root` from `sources.yaml`, so set it there
   for them. Both refuse a data folder inside the repository and any path
   that contains the word `nextcloud`.
3. Fetch the files of the table above into `<ext_root>/raw/`.

## Run order for TWR-CLOUD-Bern

From the repository root:

```
python -B Code/external/convert_external.py --raw-root <ext_root>/raw --out <ext_root>/data
python -B Code/external/qa_external.py
python -B Code/external/run_external.py fm ext_bern_a40
python -B Code/external/run_external.py sens ext_bern_a40
```

(benchmark) The last three lines read the converted campaigns from
`<ext_root>/data/` and write to `<ext_root>/results/`, both set by
`sources.yaml`.

`-B` keeps bytecode caches out of the repository. The run of the tutorial also
used `-P` (Python 3.11 or newer), which keeps the current folder off the
module path.

- `convert_external.py` checks the hashes and writes three campaigns to the
  folder of `--out` (default `<ext_root>/data/`), `ext_bern_a40`,
  `ext_selfloc_department`, and `ext_selfloc_reception`, each with one session
  `r01`, together with `conversion_summary.csv`. It always converts all
  three, so both sources must be present. `--raw-root` (default
  `<ext_root>/raw/`) names the folder with the fetched files.
- (benchmark) `qa_external.py` checks every converted campaign: units against the
  truth, repeated measurements per pair, the rigidity certificate of the
  fully meshed view, truth for every node, planarity, the noise-scale rule,
  recovery from noise-free ranges, and one cauchy factor-graph cell from both
  starts. It writes `results/ext_sessions.csv` and `results/day1_checks.md`.
  For comparison, it repeats the planarity check on `Data/set_b/s03` and
  `Data/set_a/s21` in memory.
- (benchmark) `run_external.py fm` runs the fully meshed roster of the benchmark through
  the released driver, with particle-filter seeds 0 to 4, and the factor
  graph started from the surveyed geometry with the gaussian, huber, and
  cauchy kernels.
- (benchmark) `run_external.py sens` reruns the factor graph, SMACOF, the EKF, and the
  particle filter with the gaussian and cauchy kernels at a quarter of the
  session noise scale, from the shared start, and the factor graph from the
  surveyed geometry.

The driver runs with `write=False`, and `ext_context.py` restores every module
setting it changes for an external campaign, so the released result tables
stay as they are.

### Conversion rules

- TWR-CLOUD-Bern releases times of flight in ns. They are converted with
  0.299702547 m/ns, the speed of light in air that the dataset authors use.
- Both released iterations of the Bern scan are pooled, and equal values are
  kept, because the times of flight are quantized.
- The Bern samples carry no timestamps, so an epoch is the rank of a sample
  within its directed pair. selfloc epochs are ranked by timestamp within the
  directed pair, and its ranges come from the column `dist_drift`.
- Ranges outside the valid interval of the released conversion are dropped,
  and nodes outside the largest connected component of the measured pairs are
  marked as excluded.
- The gauge triple of a campaign, which also forms the prior subset of its
  multi-hop mask, is the node with the most measured pairs, its neighbor
  with the most shared neighbors, and the common neighbor that spans the
  largest triangle.

## Results and the numbers of the tutorial (benchmark)

All files land in `<ext_root>/results/`.

| file | written by | content | in the tutorial |
|---|---|---|---|
| `ext_bern_a40_fm.csv` | `run_external.py fm` | one row per method, variant, and seed, with the columns of `Code/methods/results/benchmark.csv` | Section VI-B: the span of the engines from the classical start (`rmse_m`, particle filter averaged over seeds 0 to 4) |
| `ext_bern_a40_fm_oracle_start.csv` | `run_external.py fm` | factor graph from the surveyed geometry per kernel | Section VI-B: the span from the survey and the spread of the three kernels (`rmse_surveyed_start_m`) |
| `ext_bern_a40_fm_sigma0.25.csv` | `run_external.py sens` | the quarter-scale cells, `start` shared or surveyed, particle filter already averaged over seeds 0 to 4 | not used |
| `ext_sessions.csv`, `day1_checks.md` | `qa_external.py` | the checks per campaign | Section II-D: the rigidity class (`fm_certificate`). The share of measured node pairs, the noise-free recovery from the classical start, the other checks and the selfloc statistics are not used |
| `ext_bern_a40_fm_h1.csv` | `run_external.py fm` | median step from the gaussian to the cauchy kernel over the four engines, against the spread of their cauchy cells, with a verdict that reads `initialization-confounded` when the two starts of the cauchy factor graph end more than 1 mm apart | not used, flagged as confounded on Bern |
| `ext_bern_a40_fm_estimates.csv` | `run_external.py fm` | per-anchor estimates of every cell | not used |

The tutorial typesets the numbers of its Section VI-B from these files
with a paper-side script, `gen_external_tables.py`, which checks them
against the values of the run of 7 October 2026.

## Files

| file | role |
|---|---|
| `sources.yaml` | `ext_root`, the fetched files with SHA-256 (and the Zenodo MD5), licenses, files not fetched |
| `readers/twr_cloud_bern.py`, `readers/selfloc.py` | parsers of the public files into the radiocal edge schema |
| `session_writer.py` | writes a campaign in the released layout: node registry, masks, session, pair table, largest component, gauge triple |
| `convert_external.py` | raw files to `<ext_root>/data/ext_*` (or the folder of `--out`) |
| `ext_context.py` | (benchmark) points the unchanged driver at an external campaign and restores it afterwards |
| `qa_external.py` | (benchmark) checks of the converted campaigns |
| `run_external.py` | (benchmark) fully meshed roster, factor graph from the surveyed geometry, kernel summary, quarter-scale run |

## Adding a recording

A new source needs a reader that returns the nodes (`id`, `x`, `y`, `z` in
meters) and the directed ranges in the columns of `radiocal.graph.EDGE_COLUMNS`,
an entry with its SHA-256 in `sources.yaml`, and one more `_build` call in
`convert_external.py`. External campaign ids start with `ext_`.
