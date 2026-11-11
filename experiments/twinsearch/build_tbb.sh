#!/bin/sh
# Builds oneTBB v2021.12.0 (deps/oneTBB) as static objects with the zig C++ compiler of the local venv.
# Run from this folder:  sh build_tbb.sh
R="$(pwd)"
export ZIG_GLOBAL_CACHE_DIR="$R/zigcache/global" ZIG_LOCAL_CACHE_DIR="$R/zigcache/local"
export TMP="$R/tmp" TEMP="$R/tmp"
ZIG="$R/venv/Scripts/python.exe -I -m ziglang"
cc1() {
  f="$1"; o="$R/build/tbb/$(basename "$f" .cpp).o"
  [ -f "$o" ] && return 0
  $ZIG c++ -std=c++17 -O2 -D__TBB_BUILD=1 -D__TBB_DYNAMIC_LOAD_ENABLED=0 -D__TBB_SOURCE_DIRECTLY_INCLUDED=1 \
      -D_WIN32_WINNT=0x0A00 -mrtm -mwaitpkg -w -I"$R/deps/oneTBB/include" -I"$R/deps/oneTBB/src" -c "$f" -o "$o" \
      || echo "FAILED $f"
}
for f in "$R"/deps/oneTBB/src/tbb/*.cpp; do
  cc1 "$f" &
  if [ $(jobs -r | wc -l) -ge 2 ]; then wait -n 2>/dev/null || wait; fi
done
wait
ls "$R/build/tbb" | wc -l
