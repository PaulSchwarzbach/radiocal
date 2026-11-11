# Data — canonical campaign datasets

Two UWB two-way-ranging campaigns and a synthetic counterpart in one
canonical Parquet + YAML layout, built for topology-controlled
infrastructure-calibration benchmarking. This file is the shared data
dictionary; campaign-specific properties (site, sessions, known limitations)
live in the per-campaign READMEs.

| campaign | nodes | sessions | ground truth | frame |
|---|---|---|---|---|
| [`set_b/`](set_b/README.md) | 10 fixed anchors + mobile `DD98` (LOS-dominant) | `s01`, `s02`, `s03` | Total Station (anchors), Vicon (mobile) | local UTM 33N grid frame, georeferenced (EPSG:25833) |
| [`set_a/`](set_a/README.md) | 12-node static mesh, roving tag `8B05` (NLOS-rich) | `s21`, `s22`, `s01`…`s20` | Total Station only | campaign-local (not georeferenced) |
| `synthetic/` | SYNTHETIC: the 10 surveyed anchors of `set_b/s03` | 81 seeded draws `syn_<condition>_<NN>`; the draw tool writes further campaigns outside `Data/` | the generating geometry, exact | the set_b frame |

Load with the `radiocal` package (`Code/datasets/radiocal`) or read the
files directly (MATLAB: `parquetread`, R2019a+; any Parquet reader works).

## Identifier convention

A session identifier says what the session **is** inside the benchmark and
nothing else. Where and when the data was recorded is provenance: it lives in
`<set>/campaign.yaml` (`location`, `hardware`, `groundtruth`, and `date`), not
in the identifier. set_a was recorded on 2025-03-13 and set_b on 2025-10-22. The
day of set_a is the author's record, since its logs carry no timestamps.

Sessions are addressed as `<set>/<session>`, e.g. `set_a/s01`. A bare session
id is **not** a key: `s01` exists in both sets. Motion regime and truth grade
are properties, read from `session.yaml` (`motion`, `truth_quality`) rather
than parsed out of the name.

Within `set_a`, session `sNN` for NN = 01..20 corresponds to surveyed control
point `PNN` in `nodes.yaml`; the static references sort last.

Synthetic sessions are `synthetic/syn_<condition>_<NN>`, where NN is the draw
index and the seed (for example `synthetic/syn_nlos_07`). Every name of the
synthetic campaign marks it as synthetic. The draw tool
`Code/datasets/radiocal/scripts/07_draw_synthetic.py` writes further synthetic
campaigns in this layout to a folder outside `Data/`, named
`synthetic_<scene>_<set>` with the sessions `syn_<scene>_<condition>_<NN>`,
where the scene is `static`, `marker` or `walk` and NN is the seed.

Renamed 2026-08-25 (previously site- and year-based). Old to new:

| old | new | | old | new |
|---|---|---|---|---|
| `torgau_2024/p01`…`p20` | `set_a/s01`…`s20` | | `torgau_2025/dyn1` | `set_b/s01` |
| `torgau_2024/static1` | `set_a/s21` | | `torgau_2025/dyn2` | `set_b/s02` |
| `torgau_2024/static2` | `set_a/s22` | | `torgau_2025/static1` | `set_b/s03` |

The year in the old set_a names is wrong: set_a was recorded on 2025-03-13, and
2024 is the year of another campaign in the same hall.

Raw source filenames inside the deposit (`dyn_p01.log`, `stat_torgau.log`,
`autopos_run1.db`) are acquisition artifacts and keep their original names;
they anchor the SHA-256 chain in `campaign.yaml` `sources[]`.

## File inventory

| file | format | scope | produced by |
|---|---|---|---|
| `<campaign>/campaign.yaml` | YAML | one per campaign | curation (`radiocal` scripts 02/04; `synthetic`: 05) |
| `<campaign>/nodes.yaml` | YAML | one per campaign | Total Station survey + curation |
| `<campaign>/masks/*.yaml` | YAML | four per campaign | benchmark design (declarative views) |
| `<campaign>/los_nlos_field_labels.csv` | CSV | set_a, set_b | field LOS/NLOS judgement, transcribed twice |
| `set_a/truth_adjudication.csv` | CSV | set_a | truth adjudication of the quasi-static sessions (`scripts/04_build_set_a.py`) |
| `<campaign>/<session>/ranges.parquet` | Parquet | one per session | raw-log conversion; `synthetic`: seeded draws (`scripts/05_build_synthetic.py`) |
| `<campaign>/<session>/groundtruth.parquet` | Parquet | sessions with truth | Vicon logs (set_b) / survey markers (set_a) |
| `<campaign>/<session>/session.yaml` | YAML | one per session | conversion (provenance, clock, counts) |
| `<campaign>/<session>/qa/` | Markdown, CSV, PNG | one per measured session | `radiocal` QA (`scripts/03_qa_all.py`) |

## Column dictionary: `ranges.parquet`

Directed range measurements; one row is one measurement taken by node
`from_id` of node `to_id`. Symmetrize explicitly when an undirected graph is
needed (`MeasurementGraph.symmetrized()`).

| column | type | units | description |
|---|---|---|---|
| `t` | double | Unix seconds | unified timestamp on the Vicon/tracking-PC clock (set_b). **set_a: NaN** (logs carry no timestamps). Synthetic sessions: `seq` / 12.5 Hz, a synthetic time axis |
| `t_raw` | double | seconds | original Qorvo-pipeline timestamp before the clock shift. **set_a: NaN**. Synthetic sessions: equal to `t` |
| `seq` | int64 | — | Qorvo frame counter; one value = one ranging epoch. Sole ordering key in set_a |
| `from_id` | string | — | measuring node id (hex, e.g. `4197`) |
| `to_id` | string | — | measured node id |
| `range_m` | double | metres | measured two-way-ranging distance. Failed exchanges (negative or ≥ 500 m) were dropped at conversion; NLOS-biased ranges remain by design, and so do the few ranges far shorter than the surveyed distance |

The package keeps pandas below 3 (`pyproject.toml`). A rebuild outside that
bound, with pandas 3 and pyarrow 26, stores `from_id` and `to_id` as the
Arrow type `large_string` instead of `string`, with identical values. Parquet
readers return both as strings, but the byte checksums of rebuilt files
differ from those of the shipped files.

## Column dictionary: `groundtruth.parquet`

Both campaigns have the same columns. A column that does not apply holds NaN,
so `frame` is double in set_a, where no Vicon frame exists, and int64 in set_b.

| column | type | units | description |
|---|---|---|---|
| `t` | double | Unix seconds | Vicon clock (set_b). set_a: NaN (one static row per quasi-static session). Synthetic scenes of the draw tool: the synthetic time axis of their ranges, 0.0 for the one row of a marker scene |
| `t_broker` | double | Unix seconds | MQTT broker receive time; queueing-delayed, keep for diagnostics only |
| `frame` | int64 (set_b), double (set_a) | — | Vicon frame counter of the pose (set_b); NaN in set_a and in the synthetic scenes of the draw tool. The coordinate frame itself is named in `session.yaml -> vicon.frame` (`vicon_local`), and for a synthetic tag in `session.yaml -> truth_frame` (`local`, the frame of `nodes.yaml`) |
| `node_id` | string | — | tracked node (`DD98` in set_b, `8B05` in set_a) |
| `x`, `y`, `z` | double | metres | position in the source frame (set_b: local Vicon frame, **not** ENU; see the transform below. set_a: campaign-local survey frame, directly comparable to `nodes.yaml`) |
| `rot_x`, `rot_y`, `rot_z` | double | radians | orientation as the Vicon stream gave it (set_b only; NaN in set_a and in synthetic scenes). Which rotation convention the three angles follow was not recorded |

set_b truth is bursty (~24 Hz effective, ~20 % coverage of each dynamic
run). This is a logging loss, not a tracking loss; evaluate at covered epochs
and do not interpolate across gaps (details in the campaign README).

## Field dictionary: `nodes.yaml`

| field | description |
|---|---|
| `crs.epsg` | EPSG code of the survey frame. `25833` (ETRS89/UTM 33N) in set_b; `null` in set_a (campaign-local tachymeter frame) |
| `crs.height_system` | vertical datum (`DE_DHHN2016` in set_b) |
| `crs.local_origin_utm` | UTM offset subtracted from all coordinates; add it back to georeference |
| `crs.axes` | axis convention. In set_b x=East, y=North, z=Up, metres, where East and North are the grid directions of UTM zone 33N (see *Coordinate systems*) |
| `nodes[].id` | node id (hex string) |
| `nodes[].role` | `anchor` or `mobile` |
| `nodes[].status` | optional; `surveyed_never_ranging` marks nodes excluded from graphs (set_b: `D5B4`) |
| `nodes[].prior.type` | `surveyed` (Total Station) or `none` (mobile nodes) |
| `nodes[].prior.xyz_local` | surveyed position, metres, local frame |
| `nodes[].prior.sigma_m` | survey uncertainty, metres: 0.01 for every surveyed node of set_a and set_b. The synthetic campaigns give each position 0.005, set by the synthetic writer and the draw tool, although their positions are exact |
| `nodes[].groundtruth` | mobile nodes: truth source binding (e.g. Vicon object name) |
| `control_points` | independent survey shots (ground control points, Vicon wand). Repeated shots of a point are averaged when all lie within 0.10 m of each other (`n_shots`, and `spread_m`, the largest distance between them). Otherwise the first shot is kept and `conflict: true` marks the point for manual resolution (set_b: `GCP-12`, two shots 22 m apart, which no anchor and no transform uses). The same rule averages the shots of the nodes (`radiocal.io.coords.dedupe_shots`) |

## Field dictionary: `session.yaml`

| field | description |
|---|---|
| `session_id`, `campaign`, `motion` | identifiers; `motion` is `dynamic`, `static`, or `quasi_static_point` (a tag that stands still at one surveyed marker) |
| `t_start`, `t_end` | session window, Unix seconds (set_b only) |
| `sources[]` | raw files this session was built from: `path`, `role`, SHA-256. A set_b database also carries `sha256_archived`, the hash of its copy in the raw deposit (stripped of the vendor streams); the builder accepts either file and records both |
| `clock` | clock-unification record (set_b): reference clock, min-envelope offset and uncertainty in seconds, match counts, observed MQTT queue lag |
| `counts` | conversion tallies: range epochs, directed ranges, dropped invalid ranges, truth poses |
| `vicon` | truth-stream binding (set_b): MQTT topic, object, node id, effective rate |
| `transform_vicon_to_local` | one rigid Vicon → ENU transform with unit scale from the surveyed wand points (wand tie), the same for set_b `s01` and `s02` (`radiocal.frames.Transform.to_dict()` format); `null` elsewhere, including the static `s03`, whose motion-capture poses are not registered. `transform_provenance` records the method and the control points. Its note, like one note of `set_b/campaign.yaml`, names `Code/methods/estimate_vicon_transform.py`, the UWB-based cross-check of the companion tutorial's release, which the UWB-ASC release does not contain |
| `node_overrides` | per-session role/prior changes (set_a quasi-static sessions: `8B05` becomes `role: mobile, prior: none`); applied by the loader |
| `verification`, `truth_quality` | set_a quasi-static sessions: multilateration screen plus a direct range test at the marker (`set_a/truth_adjudication.csv`); `verified` or `unreliable` (all 20 verified) |
| `synthetic`, `draw` | synthetic sessions only: `synthetic: true` and the draw: `condition`, `index`, `seed`, `bias_m`, `ranges_sha256` (the first 16 hex digits of a SHA-256 over the little-endian float64 `range_m` column in edge order) with its reference, the generator's commit and SHA-256, the `radiocal` and numpy versions, and for `nlos` the drawn excess of every pair with multipath (`nlos_pairs`, a pair not listed has none). The draw tool adds `dropouts` and `dropouts_removed` |
| `truth_frame` | synthetic sessions of the draw tool with a tag: `local`, the frame of the campaign's `nodes.yaml`, in which `groundtruth.parquet` gives the exact tag positions. A marker scene also carries `truth_quality: verified`, as the generator marks its exact truth |

## Field dictionary: `campaign.yaml`

| field | description |
|---|---|
| `campaign`, `title`, `location`, `hardware`, `groundtruth` | campaign metadata in plain text |
| `date` | recording day (for set_a the author's record, since its logs carry no timestamps) |
| `nodes` | node roster by role: `anchors`, `mobile`/`roving_tag`, `excluded` |
| `notes` | curation facts that apply campaign-wide |
| `sources[]` | complete raw-file manifest: `path`, SHA-256 (set_b databases: also `sha256_archived`). Every canonical file traces to entries in this list via its `session.yaml -> sources` |
| `synthetic`, `generator` | synthetic campaigns only: the generator module and paper, the model values, conditions, draws per condition, seeds, bias, epochs, the 2-D distances, and the provenance shared by the sessions. A campaign of the draw tool also records the scene, the geometry session, the tag with its point or path and the height of a walking tag (`tag_z_m`), the command, and whether both generator modules equal the released ones (`generator_pinned`), and its `sources` list every geometry input with its SHA-256 |

## Column dictionary: `los_nlos_field_labels.csv`

One row per pair of static nodes: the line-of-sight condition judged by eye in
the field. A field judgement, not a measured truth. set_a: from the analysis
code of the study of set_a. set_b: from a hand-written matrix of the set_b campaign,
which is not part of the deposit. Each file was transcribed twice
independently.

| column | description |
|---|---|
| `node_a`, `node_b` | node ids, `node_a` before `node_b` in string order, as in `MeasurementGraph.symmetrized()` |
| `label` | `LOS`, `NLOS`, or in set_a also `OLOS` (obstructed line of sight). Two values joined by `/` where the record holds both |
| `label_two_class` | set_a only: the label with OLOS counted as LOS, as the study's later analysis did |
| `note` | how the record holds a cell that is not a single clear value |

## Mask grammar: `masks/*.yaml`

A mask turns the superset measurement graph of a session into one benchmark
topology view; the four shipped masks are the benchmark's controlled
variable. Masks filter, they never add data.

| field | description |
|---|---|
| `name`, `description` | view identity and evaluation convention |
| `edges.keep[]` | directed role-pair patterns to keep (`from_role`, `to_role`) |
| `edges.thin` | optional sparsifier; `anchor_anchor_knn: k` keeps each anchor's k nearest neighbours by median measured range |
| `priors` | `none` (all positions unknown) or `subset: [ids]` (listed anchors keep their surveyed prior) |
| `groundtruth` | truth exposure: `none`, or `motion_only`, which keeps the mobile trajectory in the view for evaluation only. No estimator of the benchmark reads it |

Shipped views: `fully_meshed` (anchor↔anchor only, no priors), `slam_bipartite`
(mobile↔anchor only, no priors, motion truth kept for evaluation),
`mobile_assisted` (full mesh + mobile edges, minimal prior subset `5BB6`,
`4A23`, `8B05`), `multi_hop` (anchor mesh thinned to 3 nearest neighbours,
same prior subset). A listed prior survives only on a node that is an anchor
in the session. In the set_a sessions `s01` to `s20` 8B05 is the mobile, so
these two views keep the priors of `5BB6` and `4A23` only.

## Coordinate systems

| name | axes and units | definition |
|---|---|---|
| `enu` (set_b) | x=East, y=North, z=Up, metres (grid directions) | local grid frame of UTM zone 33N: ETRS89/UTM 33N (EPSG:25833) minus `crs.local_origin_utm`, heights DHHN2016. Grid north deviates from true north by the grid convergence, about 1.6 degrees at the site, and the point scale of 0.99983 shortens 25 m by 4 mm. The publication frame |
| campaign-local (set_a) | x, y, z metres, right-handed | tachymeter frame of the set_a survey; no geodetic datum (`crs.epsg: null`); internally consistent for all nodes and markers. Its origin and axis directions in the hall were not recorded, and no surveyed point ties it to the set_b frame |
| `vicon` (set_b) | x, y, z metres, radians | motion-capture local frame; metric but not aligned to ENU. `transform_vicon_to_local` of the dynamic sessions stores the wand-tie transform |
| `radio_free` | metres | whatever frame a gauge-free estimator converges in; ranges fix it only up to a rigid motion and a reflection |
| `radio_conv` | metres | reporting convention: declared origin anchor at (0,0), second anchor on +x, third with positive y |

Frame handling, Umeyama alignment, and gauge fixing are implemented in
`radiocal.frames` (module docstring has the full doctrine).

## Adding a campaign

A new campaign is a folder `Data/<set>/` that `radiocal.load_session` and the
benchmark driver read without code changes. Required:

| file | required content |
|---|---|
| `campaign.yaml` | `campaign`, `title`, `location`, `hardware`, `groundtruth`, `nodes` (roster by role), `sources[]` with `path` and `sha256` of every raw file; `date` and `notes` where known |
| `nodes.yaml` | `crs` (with `epsg: null` for a local frame) and `nodes[]` with `id`, `role` (`anchor` or `mobile`) and `prior` (`type: surveyed` with `xyz_local` and `sigma_m`, or `type: none`); nodes that never range carry a `status` and are left out of graphs |
| `masks/*.yaml` | the four views; copy them from `Code/datasets/radiocal/src/radiocal/mask_specs/` and adapt the prior subset of `mobile_assisted` and `multi_hop` to anchors of the new campaign |
| `<session>/ranges.parquet` | the columns of `ranges.parquet` above; `t` and `t_raw` may be NaN when the source carries no timestamps, and `seq` then orders the epochs |
| `<session>/session.yaml` | `session_id`, `campaign`, `motion` (`static`, `quasi_static_point` or `dynamic`), `sources[]`, `counts` (`range_epochs`, `directed_ranges`) |

Optional: `<session>/groundtruth.parquet` with the truth columns above (the
loader returns an empty truth table without it), `truth_quality` and
`node_overrides` in `session.yaml` for sessions in which a node moves to a
surveyed marker, and the transform fields for motion-capture truth. The
conversion scripts of `Code/datasets/radiocal/scripts/` are worked examples
of such a build, and `scripts/03_qa_all.py --data DIR --out DIR --campaigns
SET ...` writes the quality reports. Without `--out` the builders, the
synthetic writer and the QA write into `Data/`, which is destructive: they
overwrite the released files. The draw tool has no such default. It requires
`--out` and refuses any folder inside `Data/`.

## Provenance and quality

Every canonical file but the field labels traces to its raw sources through
SHA-256 hashes (`campaign.yaml` and `session.yaml -> sources`). The field
labels transcribe a field judgement whose sources the deposit does not hold
(see the column dictionary of `los_nlos_field_labels.csv`). The builders rebuild
every canonical file byte for byte from the raw deposit with the pinned
environment. The UWB-ASC release also lists the SHA-256 of each of its files
in `SHA256SUMS` at its root. Per-session QA reports under
`<session>/qa/` document pair availability, geometry residuals against the
survey, and coverage; read them before drawing conclusions from a session.
The datasets are released under CC-BY 4.0 (see `LICENSE-DATA` at the
repository root).
