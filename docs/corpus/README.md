# Literature classification record

This folder holds the coded record of the 121 works read in full in the literature search of
the companion paper, and the screening decisions of that search. The paper draws most of the
works of its Fig. 2 (Section II-B) from this record.

| File | Content |
|---|---|
| `classification.csv` | One row per classified work: codes and their provenance |
| `screening_search.csv` | First screening pass over the 5,241 records of the database search |
| `screening_snowball.csv` | First screening pass over the 1,847 snowball candidates |
| `screening_second_pass.csv` | Second pass over the 968 records marked maybe in the first pass |

## `classification.csv`

`classification.csv` is the primary record of the classification. It was exported from the
final record of the re-audit. The export script and the inputs of the earlier agreement
checks (the earlier model record, the three readings per work and the author's blind codes)
stay with the manuscript sources and are not part of this release.

| Column | Meaning |
|---|---|
| `bib_key` | Key of the work in the bibliography behind the paper. The paper cites only some of the works |
| `year`, `title`, `doi` | From that bibliography (`doi` is empty for 4 works) |
| `class` | `core` (infrastructure positions among the estimated states), `adjacent` (other infrastructure states only), `context` (no infrastructure state estimated) |
| `strand` | `SC` self-calibration, `RS` radio SLAM, `NL` network localization, `CO` cooperative localization, `AC` acoustic self-calibration, `SO` signals of opportunity, `--` none (context works) |
| `topology` | `FM` fully meshed, `MH` multi-hop, `MA` mobile-assisted, `MO` mobile-only, `MX` several edge sets of equal standing, `OT` other |
| `primitive` | `TWR` two-way ranging, `TOA` time of arrival, `TDoA` time difference of arrival, `RSS` received signal strength, `AoA` angle, `CSI` channel state information, `CP` carrier phase, `DO` Doppler, `OTH` other, `none` |
| `estimator` | `LS` iterative least squares, `MDS` multidimensional scaling, `SDP` convex relaxation, `EKF` Kalman filtering, `FG` factor graph, `PF` particle filter, `CF` closed form, `LRN` learned model, `OTH` other, `none` |
| `states` | `position`, `clock`, `bias`, `orientation`, `none` |
| `reference` | `B` blind, `S` semi-blind, `N` non-blind, `M` mobile-referenced, `U` undecidable from the text, `NA` not applicable |
| `evaluation` | `S` simulation, `E` experiment with real hardware, `none` |
| `technology` | `UWB`, `WiFi`, `BLE`, `Cell` cellular, `LEO` low-Earth-orbit satellites, `Acu` acoustic, `Gen` no radio named, `Oth` other radio |
| `strand_vocabulary` | For context works: the strand vocabularies that title, abstract or keywords use. Not a code |
| `source_<field>` | Provenance of the code: `unanimous` (all three readings), `majority` (two of three), `author check` (one of the 50 randomly drawn codes the author confirmed against the paper) |

Fields with several codes separate them by `; ` and put the main code first.

## Screening tables

The screening tables are the outputs of the language-model screening as recorded on the
search day (first pass) and in the second pass. They are released unchanged.

| Column | Meaning |
|---|---|
| `idx` | Position of the record in the deduplicated database export (search) or in the screened snowball pool (snowball). In the second pass, the `idx` of the record in the first-pass table of its `origin` |
| `decision` | `keep`, `maybe` or `drop` (second pass: `keep` or `drop`) |
| `score` | Relevance score between 0 and 1 |
| `clusters` | Strand tags (`SC`, `RS`, `NL`, `CO`, `AC`, `SO`), separated by `;` |
| `data_flag` | `data` for dataset and benchmark papers, empty otherwise |
| `source` | Database of the surviving copy (search records) |
| `n_seeds`, `dir` | Number of linked seeds, and whether the record is cited by a seed (`ref`), cites one (`cit`) or both (`cit+ref`) (snowball records) |
| `year`, `doi`, `title` | Bibliographic data of the record |
| `reason` | One-line reason given by the model |
| `origin`, `prior_*` | Second pass: origin of the record and its first-pass score, tags, flag and reason |

No prompt of the screening passes was archived, and the screening decisions were not
audited.
