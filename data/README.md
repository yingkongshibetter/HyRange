# Inputs

| Input | Used for | In this repository? | How to obtain |
|---|---|---|---|
| Ten benchmark datasets of MARIOH (Crime, Hosts, Directors, Foursquare, DBLP, MAG-TopCS, Enron, P.School, H.School, Eu), training and target files of both settings | Benchmark and co-authorship experiments | no | `python data/get_marioh.py` |
| MARIOH code (baseline, refutation, learned completion of the dense benchmarks) | Baseline reconstruction, refutation, and dense inputs | no | same script (the code and the data are one repository) |
| Three OpenAlex samples OA-DB, OA-CS, OA-Bio | Co-authorship experiments | yes, `data/openalex/` (1.8 MB) | shipped; see `data/openalex/README.md` |
| A Zenodo deposit of a VOSviewer co-authorship map | Co-authorship case study (the deposited map; `experiments/zenodo_exports.py`) | no | `python data/get_zenodo.py` |

## MARIOH datasets and code

Source: the public release of the MARIOH authors, <https://github.com/KyuhanLee/MARIOH>, commit
`134affb3e6d8ea81f905d0021b195f7ed75c1c3b` (the commit used for the experiments; K. Lee, G. Lee, K. Shin, "MARIOH:
Multiplicity-Aware Hypergraph Reconstruction", ICDE 2025). `data/get_marioh.py` clones the repository at that commit
into `data/MARIOH/` (or downloads the GitHub archive of the commit if git is missing) and checks the 40 data files
(`data/<dataset>/{train,test}{,_dup}.txt`) against `data/marioh_sha256.txt`. Set `HYRANGE_MARIOH` to use a clone
elsewhere.

License. The repository carries the MIT license of its authors, but its data files are the benchmark collections
that MARIOH took from earlier public sources, whose own terms are not restated in the repository. We therefore do
not redistribute either the data or the code and fetch them from the authors' release instead.

Protocol. The split into training (source) and target hypergraphs is the one shipped by the authors
(`train*.txt` = source, `test*.txt` = target). Preserved setting: target `test_dup.txt` (repeated lines kept).
Reduced setting: target `test.txt`. HyRange reads only the pair counts and degrees of the target lines
(`hyrange/bench.py`); MARIOH also reads its training file. MARIOH runs with the authors' functions in the order of
their `main.py`, the settings of their `params.py` for each dataset and setting (`best_thres`, `ratio_prob`,
`best_params`), seeds 0, 1, 2, on one CPU thread (`experiments/marioh_runner.py`, environment in
`requirements-marioh.txt`).

## OpenAlex samples (shipped)

OpenAlex is CC0, so the three samples are part of this repository: `data/openalex/<name>/incidence.npz` (works x
authors), `node_mapping.csv` (OpenAlex author ids), `hyperedge_mapping.csv` (OpenAlex work ids). Filters and sizes are
in `data/openalex/README.md`.

## Zenodo deposit (downloaded)

| Name | DOI | Database, unit | License | Files |
|---|---|---|---|---|
| `zenodo22729673_authors` | [10.5281/zenodo.22729673](https://doi.org/10.5281/zenodo.22729673) | Web of Science Core Collection, authors | CC-BY-4.0 | `vosviewer_authors_map.txt`, `vosviewer_authors_network.txt` |

`data/get_zenodo.py` downloads the two files through the public Zenodo API into `data/zenodo/` and checks their
sha256. They are third-party files: the scripts parse them as text only.
