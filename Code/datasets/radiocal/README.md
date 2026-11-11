# radiocal

Measurement-graph dataset tooling for radio infrastructure calibration. It
converts the raw UWB recordings of both Torgau campaigns (set A, 2025-03-13, and
set B, 2025-10-22) into a canonical Parquet and YAML dataset and loads them as
`MeasurementGraph` objects, the single interface all calibration methods
consume.

## Install

From the repository root. The commands assume a POSIX shell such as bash
(Git Bash or WSL on Windows).

```
# numpy, pandas, pyarrow, pyyaml: enough for loading data
pip install -e Code/datasets/radiocal
# adds matplotlib (QA reports, examples) and scipy (estimators, set A rebuild)
pip install -e "Code/datasets/radiocal[qa,methods]"
```

A non-editable install (`pip install Code/datasets/radiocal`, or a wheel)
works as well, since the four masks ship as package data.
`requirements.txt` at the repository root pins the environment that wrote the
released files.

## Concept

A **MeasurementGraph** bundles `nodes` (id, role, positional prior), `edges`
(directed, timestamped range measurements), `truth` (ground-truth poses of
mobile nodes) and `meta`. Topologies of the calibration literature
(fully-meshed, mobile-only, mobile-assisted, multi-hop) are declarative
**edge masks** applied to one superset graph, not separate datasets:

The snippet runs from the repository root:

```python
import radiocal

g = radiocal.load_session("Data/set_b/s01")   # superset
view = g.apply_mask("slam_bipartite")          # or mask=... in load_session

view.edges          # DataFrame: t, t_raw, seq, from_id, to_id, range_m
view.nodes          # DataFrame: id, role, prior_type, x, y, z, sigma_m
view.truth          # DataFrame: Vicon poses, for evaluation only
view.symmetrized()  # undirected ranges (reciprocal pairs aggregated)
view.priors()       # nodes with positional unary priors
t0 = view.edges["t"].min()
first_minute = view.time_window(t0, t0 + 60.0)
for t, epoch in view.epochs(): ...
```

Shipped masks: `fully_meshed`, `slam_bipartite`, `mobile_assisted`,
`multi_hop` (see `src/radiocal/mask_specs/*.yaml`, package data; campaigns
carry their own copies).
The mask key `slam_bipartite` is the mobile-only topology; the identifier is
kept for compatibility with the released benchmark tables.

## Coordinate frames and gauge freedom

Self-calibration estimates live in a frame defined only up to a similarity
transform (translation, rotation, reflection, possibly scale). `radiocal.frames`
names the frames and implements the conversions (full doctrine in the module
docstring):

| frame | definition |
|---|---|
| `enu` | survey frame of `nodes.yaml`, the publication frame: on set B a local grid frame of UTM zone 33N (x grid east, y grid north), on set A the campaign-local tachymeter frame |
| `vicon` | motion-capture local frame of the set B truth, metric. For the walks `s01` and `s02` the wand tie in `session.yaml -> transform_vicon_to_local` registers it to `enu` |
| `radio_free` | whatever frame a gauge-free estimator converges in |
| `radio_conv` | paper convention: declared origin anchor at (0,0), second anchor on +x, third with +y |

Illustrative calls, in which `est_pts`, `ref_pts`, `est_nodes`, `ref_nodes`
and `nodes` stand for your own point arrays and node tables:

```python
from radiocal import frames

tf = frames.umeyama(est_pts, ref_pts, with_scale=False)     # rigid fit (Umeyama 1991)
tf, resid = frames.align(est_nodes, ref_nodes, dim=2)       # id-matched; resid.attrs["rmse_m"]
tf_g = frames.fix_gauge(nodes, "5BB6", "4A23", "8B05")      # radio_free -> radio_conv (2-D)
tf.to_dict()                                                # YAML-serializable (session.yaml slots)
```

Conventions: evaluation alignment is rigid (no scale) with reflection
allowed, because range-only geometry is chirality-blind. `fix_gauge` resolves
the reflection by the positive-y declaration instead. In the set B walks,
`session.yaml -> transform_vicon_to_local` stores the vicon->enu wand tie, a
rigid transform from two surveyed wand points without any UWB range, in the
`Transform.to_dict()` format. The other sessions carry `null`.

Datum. Ranges fix a layout only up to a rigid motion and a reflection, so an
estimate needs a datum, and its choice is the user's.

- The priors of a view give an absolute datum in `enu`. `mobile_assisted` and
  `multi_hop` keep the surveyed priors of 5BB6, 4A23 and 8B05, and
  `fully_meshed` and `slam_bipartite` keep none. In set A `s01` to `s20`,
  8B05 is the roving tag, so only 5BB6 and 4A23 keep their priors.
- Without priors, `fix_gauge` sets a relative datum through a declared triple.
  The default triple is the same three anchors, in this order: origin 5BB6,
  x-axis 4A23, positive y 8B05. In set A `s01` to `s20` only 5BB6 and 4A23
  remain of it as anchors, so, as with the two priors, the reflection about the
  line through them stays open unless you declare another third anchor.
- Both defaults are conventions of the testing ground. Every fixed node is
  surveyed, so a mask of your own (`priors: {subset: [ids]}`, or `surveyed`
  for every anchor) or another triple sets any other datum. The choice matters
  for methods that build the layout outward from the first anchors, since the
  errors of their ranges pass on to every anchor placed later. In set B the
  field labels mark two sides of the default triangle NLOS, 4A23-5BB6 and
  5BB6-8B05 (`los_nlos_field_labels.csv`, and their residuals in the
  `pairs.csv` that `06_data_facts.py` writes).

Comparing an estimate with the survey:

- `align` over all common nodes removes the frame of an estimate, so estimates
  of every datum compare. This is the convention of the papers, in the
  horizontal plane (`dim=2`). It also removes the error that the datum puts
  into the frame.
- A study of the datum therefore compares an estimate that uses priors in
  `enu` directly, without the alignment. With two priors the reflection about
  the line through them stays undetermined (set A `s01` to `s20`: the line
  through 5BB6 and 4A23), so score the better of the estimate and its mirror
  image about that line.
- An estimate in the frame of a gauge triple compares with the survey brought
  into the same frame, `frames.fix_gauge(ref, a, b, c).apply_df(ref)`, without
  further alignment.
- `align(est, ref, ids=[...])` fits the frame on a chosen subset of nodes and
  returns the residuals of those nodes only, so apply the returned transform to
  the whole estimate to score the others.

## Rebuilding the dataset from raw files

Unpack the raw archives of the Zenodo deposit at the repository root
(`raw/set_a/`, `raw/set_b/`) and run from there:

```
# assert the raw-data invariants of set B (reads only)
python -B Code/datasets/radiocal/scripts/01_inspect_raw.py --raw-root raw/set_b
# rebuild both campaigns into rebuild/
python -B Code/datasets/radiocal/scripts/02_build_set_b.py --raw-root raw/set_b --out rebuild/set_b
python -B Code/datasets/radiocal/scripts/04_build_set_a.py --raw-root raw/set_a --out rebuild/set_a
# QA reports of the rebuild, written into the rebuild
python -B Code/datasets/radiocal/scripts/03_qa_all.py --data rebuild --out rebuild
# the synthetic counterpart (imports Code/methods/range_errors.py and synthetic.py)
python -B Code/datasets/radiocal/scripts/05_build_synthetic.py --out rebuild/synthetic
# the data facts of the papers, from the released files
python -B Code/datasets/radiocal/scripts/06_data_facts.py --out facts
# new synthetic scenes on a released geometry, outside Data/ (the draw tool)
python -B Code/datasets/radiocal/scripts/07_draw_synthetic.py --scene walk --condition noise_free --out draws/walk_set_b
# the frozen draws redrawn through the static scene of the draw tool (writes nothing)
python -B Code/datasets/radiocal/scripts/07_draw_synthetic.py --reproduce
```

Scripts 01, 02 and 04 take `--raw-root` for raw files in another folder.
Without `--out`, scripts 02, 03, 04 and 05 write into `Data/`, which is
destructive: they overwrite the released files. Script 06 requires `--out`,
and so does 07 unless it runs `--reproduce`, which writes nothing. 07 refuses
any folder inside `Data/`, and 01 writes nothing. The docstring of 07 lists
its scenes (`static`, `marker`, `walk`) and options. The builders write their text files with LF line
endings and record both hashes of a set B database (`sha256` of the original,
`sha256_archived` of the stripped copy in the deposit), so a rebuild from the
raw deposit equals the released files byte for byte in the pinned environment
of `requirements.txt`. The set A rebuild needs scipy (the `methods` extra).
Campaign-specific configuration (paths, node roles, local origin, the
database hashes) lives in `scripts/campaign_set_b.py` and
`scripts/04_build_set_a.py`, and the library itself is campaign-agnostic.

## Module map

| module | role |
|---|---|
| `radiocal.graph` | `MeasurementGraph` and its views |
| `radiocal.load` | `load_session` / `load_campaign` (incl. per-session node overrides) |
| `radiocal.masks` | mask spec loading + application |
| `radiocal.frames` | frame taxonomy, `Transform`, Umeyama alignment, gauge fixing |
| `radiocal.convert` | raw -> canonical builders, min-envelope clock offset |
| `radiocal.io.qorvo_log` | backup-log parser (repr records, NxN matrices) |
| `radiocal.io.mqtt_db` | mqtt-logger SQLite readers (Vicon, Qorvo headers) |
| `radiocal.io.coords` | Total Station file parser, ENU conversion |
| `radiocal.qa` | per-session data-quality reports |
