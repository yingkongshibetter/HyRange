# OpenAlex samples

Three samples of OpenAlex works (<https://openalex.org>, data under CC0), one hyperedge per work and its authors as
members. The sample names are OA-DB, OA-CS and OA-Bio. `hyrange/views.py` loads the samples.

| Folder | Name in the experiments | Filter of the works | Hyperedges | Authors | Volume | Sizes |
|---|---|---|---|---|---|---|
| `openalex_db` | OA-DB | primary_location.source.id in {SIGMOD, VLDB, ICDE, PODS} | 2,823 | 3,027 | 11,819 | 3 to 24 |
| `openalex_cs_size3plus` | OA-CS | concepts.id C41008148 (Computer science), publication years 2018 to 2024 | 8,219 | 24,286 | 44,525 | 3 to 25 |
| `openalex_bio` | OA-Bio | concepts.id C95444343 (Biology) | 1,799 | 3,764 | 9,279 | 3 to 25 |

All three keep works with three or more retained authors. The samples were built from the OpenAlex works API in
May 2026; OA-DB and OA-Bio keep authors with at least two works in the crawl and a known country and institution
type, and OA-CS keeps the first 30,000 works returned by the API for its filter, authors with at least two works, and
then the works with three or more authors.

Files per folder:

* `incidence.npz`: scipy sparse CSR matrix, rows = works (hyperedges), columns = authors (nodes);
* `hyperedge_mapping.csv`: row index -> OpenAlex work id (`external_id`), raw and kept number of authors;
* `node_mapping.csv`: column index -> OpenAlex author id.
