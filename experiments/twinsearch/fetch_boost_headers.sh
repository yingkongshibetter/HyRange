#!/bin/sh
# Fetches the headers (directory include/ only) of every Boost 1.86.0 library from github.com/boostorg, tag boost-1.86.0.
# Run from deps/boost_libs:  sh ../../fetch_boost_headers.sh   (../../boost_list.txt: <repository> <directory>)
one() {
  n="$1"; d="$2"
  [ -d "$d/include" ] && return 0
  rm -rf "$d"
  git clone -q --depth 1 --no-checkout -c advice.detachedHead=false -b boost-1.86.0 "https://github.com/boostorg/$n.git" "$d" 2>/dev/null \
    && git -C "$d" checkout -q boost-1.86.0 -- include 2>/dev/null
  [ -d "$d/include" ] || echo "NOINCLUDE $n $d"
}
while read n d; do
  one "$n" "$d" &
  while [ $(jobs -r | wc -l) -ge 10 ]; do sleep 0.2; done
done < ../../boost_list.txt
wait
