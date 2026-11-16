#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/build"
CXX=${CXX:-g++}
read -r -a FLAGS <<< "${CXXFLAGS:--O3 -std=c++17 -march=native -Wall -Wextra}"
"$CXX" "${FLAGS[@]}" -pthread -I "$ROOT/src/index" "$ROOT/src/index/roleweaver_build.cpp" -o "$ROOT/build/roleweaver_build"
"$CXX" "${FLAGS[@]}" -pthread -I "$ROOT/src/index" "$ROOT/src/index/roleweaver_search.cpp" -o "$ROOT/build/roleweaver_search"
echo "Built build/roleweaver_build and build/roleweaver_search"
