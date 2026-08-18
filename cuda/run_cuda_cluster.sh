#!/usr/bin/env bash
set -Eeuo pipefail

# Whitespace-safe launcher: every path and forwarded argument remains a single
# shell word even when the project or scratch directory contains spaces.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
python_bin="${PYTHON:-python3}"

# Do not let BLAS/OpenMP helper pools multiply the CUDA worker count on a node.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

exec "$python_bin" "$project_root/scripts/nth_sweep_cuda.py" "$@"
