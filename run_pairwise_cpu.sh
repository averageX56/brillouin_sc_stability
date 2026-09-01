#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
python_bin="${PYTHON:-python3}"
exec "$python_bin" "$script_dir/scripts/nth_sweep_pairwise_cpu.py" "$@"
