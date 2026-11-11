# UWB-ASC: the NLOS deployment (`set_a`, Torgau, 2025-03-13)

Companion campaign to `set_b`, recorded on 2025-03-13 in an industrial hall in
Torgau, Germany (`set_b`: the same hall on 2025-10-22, in a mostly line-of-sight
layout), with the same
hardware (Qorvo/Decawave MDEK1001, two-way ranging), for the study of
collaborative self-calibration by P. Schwarzbach and A. Jung (IEEE J. Indoor
Seamless Position. Navig. 4:161-170, 2026). Fully-meshed **12-node** network,
no motion capture: every reference position comes from a survey with a Leica
TS16 total station in a campaign-local frame. Distinctive properties vs set_b:

- **D5B4 actively ranges** (12-node mesh instead of 11), **DD98 is static**
  here (regular anchor).
- The site is **NLOS-rich**: several anchor pairs carry +5…+30 m bias and
  some directed pairs never range at all — a natural stress case for robust
  and multi-hop calibration methods.
- **22 sessions**: `s21`/`s22` (all nodes at surveyed positions) and
  `s01`…`s20` (quasi-static: the roving tag **8B05** placed at surveyed
  marker PXX to vary the geometry; all other nodes static). The quasi-static sessions
  were never used in the original study.

## Ground-truth verification (built into the conversion)

The tag stood on a tripod above its marker. `groundtruth.parquet` holds the
marker x/y and the antenna height: marker z plus 1.72 m. The conversion
estimates that height from the ranges, by a robust joint fit of the height
and one common range bias (+0.22 m) over the tag-anchor median ranges. The
surveyed 8B05 antenna of `s21`/`s22` stands 1.719 m above marker P01, the
same height. With the bias forced to zero the ranges put the tag near the
ground, which an earlier note read as "~0 m antenna offset".

The session→marker mapping is adjudicated by a rule, with no estimator
output involved (`04_build_set_a.py`, statistics in
`truth_adjudication.csv`, per session in `session.yaml -> verification`):

1. Screen: 8B05 is multilaterated in 2-D from its median ranges to the 11
   static nodes. 18 of 20 sessions agree with their marker within 0.5 m.
   `s05` and `s12` deviate by 13.8 and 9.9 m. The fit can end far from the
   marker even where every link agrees with it, as in `s05`, so a flag only
   triggers step 2.
2. Direct range test for the flagged sessions: the median residual of the
   line-of-sight links against the marker at the fitted tag height must lie
   within the range of the 18 unflagged sessions (+0.16 to +0.25 m). `s05`
   (+0.24 m, 11 of 11 links line of sight) and `s12` (+0.21 m, 6 of 10) pass.
   A link counts as line of sight within 0.5 m of the fitted bias, a tolerance
   set while `s05` and `s12` were examined, so the test is no independent
   confirmation of these two sessions.

All 20 quasi-static sessions carry `truth_quality: verified`. There is no
field record of a relocation during `s12`, and its per-link rolling medians
stay as stationary as those of the other sessions. Earlier versions of these
files flagged `s12` as "moved mid-session" on the strength of the screen alone.

## Layout and schema

Same structure as `set_b` (see its README): `campaign.yaml`,
`nodes.yaml`, `masks/`, and per session `ranges.parquet` + `session.yaml` +
`qa/report.md`. Differences:

- Log records carry **no timestamps**: `t` and `t_raw` are NaN; use `seq`
  (the ranging-epoch counter) for ordering. Sessions are order-only.
- quasi-static sessions have a one-row `groundtruth.parquet` (8B05 at the surveyed
  marker) and a `node_overrides` block in `session.yaml` that switches 8B05
  to `role: mobile, prior: none`; the loader applies it automatically. In the
  static sessions all 12 nodes are surveyed anchors and there is no separate
  ground-truth file.
- The `mobile_assisted` and `multi_hop` masks list 8B05 as a prior anchor
  next to 5BB6 and 4A23. In `s01` to `s20` 8B05 is the mobile, so these two
  views keep only two priors there, and three in `s21` and `s22`.
- `nodes.yaml -> crs.epsg` is null: the frame is campaign-local (tachymeter),
  not georeferenced.
- `los_nlos_field_labels.csv` gives the line-of-sight label of every pair of
  the 12 nodes, judged in the field during the study and kept in its analysis
  code: LOS, OLOS (obstructed line of sight) or NLOS. The column
  `label_two_class` counts OLOS as LOS, as the study's later analysis did. The
  labels describe the static positions of `s21` and `s22`, and in `s01` to
  `s20` only the pairs without 8B05. They are a field judgement, not truth.

The numbers of this file come from
`Code/datasets/radiocal/scripts/06_data_facts.py` (its `data_facts.csv`
names the rule of each).

## Quick start

```python
import radiocal
g = radiocal.load_session("Data/set_a/s07")      # 8B05 mobile, 11 anchors
g.meta["truth_quality"]                                 # 'verified'
view = g.apply_mask("slam_bipartite")                   # tag<->anchor edges only
gs = radiocal.load_session("Data/set_a/s21")  # 12 static anchors
```

## Recording details and provenance

Recorded on 2025-03-13, the author's record, since the logs carry no
timestamps. Earlier versions of these files gave 2024, the year of another
campaign in the same hall. Not recorded: the time of day, the UWB physical
layer settings (channel, pulse repetition frequency, preamble, data rate) and
the firmware version of the modules. Curated 2026-07 from the raw backup logs
and the survey file listed with SHA-256 hashes in `campaign.yaml`.
Release/DOI: 10.5281/zenodo.24055895 (concept DOI of the UWB-ASC deposit, the
canonical copy of these files).
