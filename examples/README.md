# Examples

Four short, self-contained scripts. With the pinned environment each runs
within seconds on a laptop-class CPU, the third included, although it
recomputes the session noise scale from scratch (measured times in the
repository README). Install first from the repository root as the repository
README describes (`pip install -e "Code/datasets/radiocal[qa,methods]"`, after
`pip install -r requirements.txt` for the pinned environment), then run each
script with plain `python`. Example 01 writes its figure into the current
directory.

| script | teaches | paper section |
|---|---|---|
| `01_load_and_plot.py` | loading a session, undirected ranges, survey priors, the UWB ranging bias | II-D (the running example), VI-C (the range bias) |
| `02_topology_views.py` | the four topology masks as views of one superset graph | IV-C (views of one measurement graph), II-B (topologies) |
| `03_reproduce_benchmark_cell.py` | the benchmark cell recipe (shared noise scale, registry, aligned evaluation) and that the shipped results reproduce | V (engines), VI (protocol and results) |
| `04_what_if_ranges.py` | synthetic ranges with known truth from the range-error generator: how noise, NLOS, outliers and a range bias move the anchor error, with your own settings | VI (the synthetic check in its lead) |

Concepts and paper sections: the superset measurement graph and its views
are defined in Section IV-C and realize the topologies of Section II-B. The
robust kernels and the noise scale are introduced in Section III-C, and the
session noise scale of the examples in the lead of Section VI. Section V describes the engines and their reference implementations,
and the commands that reproduce the released results are in the repository
README.
