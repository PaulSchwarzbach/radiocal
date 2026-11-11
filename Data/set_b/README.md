# UWB-ASC: the LOS deployment (`set_b`, Torgau, 2025-10-22)

Superset measurement graph for topology-controlled infrastructure-calibration
benchmarking. One UWB two-way-ranging mesh (Qorvo/Decawave MDEK1001, 11 nodes)
recorded in an industrial hall in Torgau, Germany, on 2025-10-22, the hall of
`set_a`, in a mostly line-of-sight layout: **10 fixed anchors** (surveyed by
a Leica TS16 total station) and **one mobile tag `DD98`** (Vicon motion-capture
ground truth). Three sessions: two dynamic walks (`s01`, `s02`) and one
static reference (`s03`).

The design intent: all benchmark topologies (fully-meshed, mobile-only,
mobile-assisted, multi-hop) are *views* of this single graph, realized by the
edge masks in `masks/` — never by separate recordings. Load with the
`radiocal` Python package (`Code/datasets/radiocal`) or read the Parquet/YAML
files directly (MATLAB: `parquetread`, R2019a+).

## Layout

| file | content |
|---|---|
| `campaign.yaml` | campaign metadata, source-file SHA-256 manifest (a database carries `sha256` of the original and `sha256_archived` of its copy in the raw deposit) |
| `nodes.yaml` | node registry: roles, surveyed positions (local ENU), prior types, control points |
| `masks/*.yaml` | four canonical topology views (declarative edge/prior filters) |
| `los_nlos_field_labels.csv` | LOS/NLOS label per pair of static nodes, judged by eye in the field (a field judgement, not truth) |
| `<session>/ranges.parquet` | directed range measurements |
| `<session>/groundtruth.parquet` | Vicon 6-DoF poses of the mobile tag |
| `<session>/session.yaml` | session metadata: clock model, counts, provenance |
| `<session>/qa/report.md` | data-quality report (availability, geometry residuals, coverage) |

## Schemas

`ranges.parquet` (directed; row = one measurement of node `from_id` to `to_id`):

| column | type | meaning |
|---|---|---|
| `t` | double | unified timestamp (Vicon/tracking-PC clock), Unix seconds |
| `t_raw` | double | original Qorvo-pipeline timestamp |
| `seq` | int64 | Qorvo frame counter; one value = one ranging epoch |
| `from_id`, `to_id` | string | node ids (hex, e.g. `4197`) |
| `range_m` | double | measured distance in metres |

`groundtruth.parquet`: `t` (Vicon clock), `t_broker`, `frame`, `node_id`,
`x`, `y`, `z` (metres, local Vicon frame), `rot_x/y/z` (radians). `t` is the
time at which the UDP-to-MQTT bridge on the tracking computer received the
pose, not the capture time of the frame: the bridge script of the campaign,
which is not part of the deposit, stamps each pose with `time.time()` at
receipt. `frame` is the Vicon frame counter.

Coordinates in `nodes.yaml` lie in a local grid frame of UTM zone 33N:
ETRS89/UTM 33N (EPSG:25833) minus the origin recorded in
`nodes.yaml -> crs.local_origin_utm`, with heights in DHHN2016. Its x and y
axes follow grid east and grid north, which deviate from true east and north
by the grid convergence (`Data/README.md`, *Coordinate systems*).
The Vicon frame is **not** aligned to ENU. The registration is a wand tie
from the survey alone: the total station shot the Vicon calibration wand at
the Vicon origin (control point `WAND-0-0`, which also gives the height) and
at a point on the Vicon +y axis (`WAND-001`). The resulting rigid transform
(yaw about the vertical, translation, scale 1, model `y = scale * R @ x + t`)
is stored in `session.yaml -> transform_vicon_to_local` of `s01` and `s02`.
One Vicon calibration served both walks, so both carry the same transform.
No UWB range enters it. A UWB-based fit of the same transform
(`Code/methods/estimate_vicon_transform.py` of the companion tutorial's
release, not part of UWB-ASC) serves the tutorial as a cross-check of the
wand tie.

## Known properties and limitations (see per-session QA reports)

The numbers in this section come from
`Code/datasets/radiocal/scripts/06_data_facts.py` (its `data_facts.csv`
names the rule of each) unless another source is named.

- Ranging is **directed and asymmetric**; symmetrize explicitly if needed.
  ~12.5 Hz epoch rate, all 110 directed pairs observed.
- Invalid garbage ranges (failed TWR exchanges: negative or >= 500 m), 0.066 %
  of all ranges, were dropped at conversion (`counts.dropped_invalid`).
  NLOS-corrupted ranges remain by design — e.g. the obstructed pair
  `5BB6-D38F` (+4.0 to +5.8 m bias over the three sessions). Median
  anchor-anchor residual vs survey: +0.27 m (TWR bias).
- Vicon ground truth is **bursty**: ~120 Hz bursts covering ~20% of each
  dynamic run. Under the evaluation's 0.5 s gap rule s01 has 28 bursts
  (median 1.44 s, max 7.03 s) and s02 25 bursts (median 1.69 s, max 8.18 s).
  The static s03 keeps a similar share (17% of its tag epochs) although the
  tag never moves.
  This is a LOGGING loss, not a tracking loss: the frame counter advances at
  120 Hz across every gap, i.e. tracking ran continuously and the
  poses were dropped by the congested MQTT logging chain. Consequences:
  evaluate at covered epochs (~700 UWB epochs per run have ground truth,
  statistically ample) and use within-segment relative metrics (<= ~9 s
  horizons); do not interpolate across gaps. Only the logged poses exist: no
  native captures of the Vicon Tracker PC were archived (only the .vsk subject
  files were kept by the author and are not part of this release), so the gaps
  cannot be recovered.
- The QA reports flag a few implausible Vicon speeds (12 in s01, 13 in s02,
  faster than 5 m/s between poses less than 0.5 s apart). Most of them (10
  and 9) are steps of a few millimetres between consecutive frames whose `t`
  stamps lie much closer than the 8.3 ms of 120 Hz, so they come from jitter
  of `t`. The others are isolated pose jumps of 8 to 12 cm. Within a burst,
  `frame` divided by 120 Hz gives a steadier time base than `t`.
- Clock: range timestamps were shifted onto the Vicon clock by a per-session
  min-envelope offset (~6.9 s, uncertainty ~0.1 s; `session.yaml -> clock`).
  MQTT broker receive times are unusable mid-run: range messages arrived up to
  ~55 s after their stamp, a lag that includes the clock offset.
- The offset between the tracked Vicon object (`torgau-final-autopos`) and the
  UWB antenna of `DD98` was not recorded, so the poses give the origin of the
  object, not the antenna. The markers of the object sat on the tag, and where
  the antenna sits inside its case is not known.
- Node `D5B4` was surveyed but never ranged (kept in `nodes.yaml` with
  `status: surveyed_never_ranging`, excluded from graphs).
- `los_nlos_field_labels.csv` gives the LOS/NLOS judgement of the 55 pairs
  of the 11 static nodes (D5B4 included, the mobile DD98 not), transcribed
  twice from a hand-written matrix of the campaign. Three pairs carry both
  values, and two cells hold a value written after a struck character
  (column `note`). The labels are a field judgement, not truth.

## Quick start (Python)

```python
import radiocal
g = radiocal.load_session("Data/set_b/s01")            # superset graph
g_slam = radiocal.load_session("Data/set_b/s01", mask="slam_bipartite")
sym = g.symmetrized()                                          # undirected ranges
for t, epoch in g.epochs(): ...                                # per-epoch iteration
```

MATLAB: `r = parquetread("Data/set_b/s01/ranges.parquet");`

## Citation / provenance

Recorded on 2025-10-22 (all three sessions, between 13:22 and 13:46 UTC)
during a two-day campaign of TUD Dresden University of Technology
(2025-10-21/22). A person carried the tag DD98 on the walks. Not recorded:
the UWB physical layer settings (channel, pulse repetition frequency,
preamble, data rate), the firmware version of the modules, the configuration
of the Vicon system (cameras, software, calibration residual), and how many
persons took part and with which expertise. Curated 2026-07 from the raw backup logs and MQTT records listed
with SHA-256 hashes in `campaign.yaml`. Release/DOI: 10.5281/zenodo.24055895
(concept DOI of the UWB-ASC deposit, the canonical copy of these files).
