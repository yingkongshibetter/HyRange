#!/bin/sh
# Builds the authors' TwinSearch library sources (twin_search_src, unmodified) and the driver ts_run.cpp into
# build/ts_run.exe with the zig C++ compiler of the local venv.  Run from this folder:  sh build_ts.sh
# Flags follow the repository's CMakeLists.txt (Release): C++23, -O3, NDEBUG not defined.
R="$(pwd -W)"
export ZIG_GLOBAL_CACHE_DIR="$R/zigcache/global" ZIG_LOCAL_CACHE_DIR="$R/zigcache/local"
export TMP="$R/tmp" TEMP="$R/tmp"
ZIG="$R/venv/Scripts/python.exe -I -m ziglang"
: > "$R/build/inc.rsp"
for d in "$R"/deps/boost_libs/*/include; do
  [ -d "$d/boost" ] && echo "-isystem $d" >> "$R/build/inc.rsp"
done
INC="-I$R/twin_search_src/include/twin_search -I$R/deps/oneTBB/include -I$R/shim @$R/build/inc.rsp"
cc1() {
  f="$1"; o="$R/build/ts/$(basename "$f" .cpp).o"
  [ -f "$o" ] && return 0
  $ZIG c++ -std=c++23 -O3 -D_WIN32_WINNT=0x0A00 -w $INC -c "$f" -o "$o" || echo "FAILED $f"
}
S="$R/twin_search_src/src/twin_search"
for f in "$S/utils.cpp" "$S/hypergraph.cpp" "$S/projected_graph.cpp" "$S/factor_graph.cpp" "$S/twin_search.cpp" "$R/ts_run.cpp"; do
  cc1 "$f" &
  if [ $(jobs -r | wc -l) -ge 2 ]; then wait -n 2>/dev/null || wait; fi
done
wait
$ZIG c++ -O3 -o "$R/build/ts_run.exe" "$R"/build/ts/*.o "$R"/build/tbb/*.o -static && echo LINKED
ls -la "$R/build/ts_run.exe"
