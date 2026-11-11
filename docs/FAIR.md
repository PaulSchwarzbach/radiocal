# FAIR compliance

How this release implements the FAIR principles (Wilkinson et al., 2016,
[doi:10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18)) for its
code, its result tables and its literature record. The measured sessions
under `Data/` are a copy of UWB-ASC 1.0.0, whose own `docs/FAIR.md` maps the
data, their documentation, the raw recordings and the rebuild to the same
principles (<https://github.com/PaulSchwarzbach/UWB-ASC-dataset>).

## Findable

| mechanism | where |
|---|---|
| Persistent DOI (concept + version) | Zenodo deposit with the concept DOI `10.5281/zenodo.24134342` and the version DOI `10.5281/zenodo.24134343` of 1.0.0, badge in `README.md` |
| Machine-readable citation metadata | `CITATION.cff` (CFF 1.2.0, ORCID), `.zenodo.json` |
| Keywords and abstract | `.zenodo.json`, `CITATION.cff`, GitHub topics |
| Cross-links to the companion papers and to UWB-ASC | `CITATION.cff` (`preferred-citation`, `references`), Zenodo `related_identifiers` |

## Accessible

| mechanism | where |
|---|---|
| Open access without registration | public GitHub repository + Zenodo record (HTTPS, no auth) |
| Long-term retention | Zenodo (CERN-hosted, ≥ 10-year policy, survives the GitHub repo); an OPARA (TUD institutional repository) secondary deposit is planned |
| Data and raw recordings | UWB-ASC's Zenodo record (<https://doi.org/10.5281/zenodo.24055896>), whose measured sessions `Data/` copies |

## Interoperable

| mechanism | where |
|---|---|
| Open, columnar and text-based formats | Parquet with CSV mirrors (result tables), YAML (sessions, transforms), CSV (literature record) |
| Documented tables and columns | `README.md` (*Storage format*: tables, keys and conventions), the column names in `Code/methods/benchmark_io.py`, `docs/corpus/README.md`, and the copied data dictionary `Data/README.md` |
| Language-independent access | any Parquet or YAML reader, MATLAB `parquetread` as `README.md` describes |

## Reusable

| mechanism | where |
|---|---|
| Explicit licenses | `LICENSE-CODE` (MIT) for the code, with `NOTICE` for the copied code of UWB-ASC, and `LICENSE-DATA` (CC BY 4.0) for the data, the result tables, the literature record and the documentation, as the *Licenses* section of `README.md` maps them |
| Pinned environment | `requirements.txt`, the library versions in which the shipped result tables were last written |
| Regression-gated results | `run_benchmark.py --check-regression` recomputes gated cells in memory and compares them with frozen values: benchmark cells, synthetic baseline and frame-choice sweep |
| Recomputable result tables | the commands of `README.md` (*Reproducing the released results*), within the tolerances of *What agrees across environments* |
| Provenance of the data | UWB-ASC: SHA-256 manifests from every canonical file to its raw source, and the scripts that rebuild the sessions |
| Versioned releases | SemVer tags, `CHANGELOG.md`, one Zenodo record per version |
