#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cmake -S "$root_dir/cuda" -B "$root_dir/build_cuda" -DCMAKE_BUILD_TYPE=Release
cmake --build "$root_dir/build_cuda" --config Release -j
