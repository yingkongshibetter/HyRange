# TwinSearch on the inputs of paragraph Enumeration

TwinSearch (T. LaRock, R. Lambiotte, "Exploring the non-uniqueness of node co-occurrence matrices of hypergraphs",
arXiv:2506.01479) lists the hypergraphs without repeated hyperedges that share a co-occurrence matrix. It is
third-party code under the GNU GPL v3 and is **not** part of this repository: the build below downloads it.

| Item | Version used |
|---|---|
| TwinSearch | <https://github.com/tlarock/twin_search>, commit `cc9e1ea3c3a6517784455e2e6c2f00a82c5ff8a0` (sources unmodified) |
| oneTBB | tag `v2021.12.0` (the version of the repository's `conanfile.txt`) |
| Boost headers | 1.86.0 (directory `include/` of each library of github.com/boostorg at tag `boost-1.86.0`, list in `boost_list.txt`) |
| compiler | zig 0.13.0 from the PyPI wheel `ziglang==0.13.0` (clang 18.1.6, libc++), in a venv |

Files of this folder (MIT, like the rest of the repository):

* `ts_run.cpp`: a thin driver around the authors' classes: it reads a hypergraph, builds the authors'
  `ProjectedGraph` (with the degrees on the diagonal when `use_diag = 1`) and `TwinSearch` objects, calls
  `TwinSearch::parallel_search` and prints the list and its timings;
* `shim/sys/resource.h`: a stand-in for the POSIX header, used only by `rss_mark()` of `twin_search.cpp`, which returns
  at once unless the environment variable `TWIN_RSS_TRACE` is set;
* `build_tbb.sh`, `build_ts.sh`, `fetch_boost_headers.sh`: build without CMake and Conan (the five library sources the
  search needs are compiled directly, oneTBB from its sources as static objects; flags of the repository's
  `CMakeLists.txt`, Release: `-std=c++23 -O3`);
* `run_ts.py`: runs `build/ts_run.exe` on the random inputs of
  `experiments/enumeration.py` (degrees on the diagonal, sizes 2..n, isomorphism filter off, one TBB thread, limit
  600 s), re-verifies every listed hypergraph in Python, and compares the hyperedges common to the list with the
  certified hyperedges of HyRange stored in the same records.

Build (Git Bash on Windows, from this folder; on Linux the repository's own CMake build with gcc 13 also works, with
the driver added):

```
python -I -m venv venv
venv/Scripts/python.exe -I -m pip install --no-cache-dir ziglang==0.13.0
git clone https://github.com/tlarock/twin_search.git twin_search_src && git -C twin_search_src checkout cc9e1ea3c3a6517784455e2e6c2f00a82c5ff8a0
git clone --depth 1 -b v2021.12.0 https://github.com/uxlfoundation/oneTBB.git deps/oneTBB
mkdir -p deps/boost_libs build/tbb build/ts tmp && (cd deps/boost_libs && sh ../../fetch_boost_headers.sh)
sh build_tbb.sh
sh build_ts.sh          # -> build/ts_run.exe  (or set HYRANGE_TWINSEARCH_EXE to a binary built elsewhere)
```

Run (from the repository root, after `experiments/enumeration.py` has written `results/enumeration/parts/`):

```
python -I -B -X utf8 experiments/twinsearch/run_ts.py random 23:59
python -I -B -X utf8 experiments/twinsearch/run_ts.py table        # -> results/twinsearch/results.json
```

The summary records completion within the time limit and compares the hyperedges common to the enumerated solutions with the HyRange certificates.
