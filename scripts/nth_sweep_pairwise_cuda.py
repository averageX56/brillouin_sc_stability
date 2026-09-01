#!/usr/bin/env python3
"""CUDA n_th sweep for N photons with N-1 independent phonons.

For photon amplitudes a_0,...,a_{N-1}, phonon b_j belongs only to the
adjacent link (a_j,a_{j+1}), j=0,...,N-2.  Edit the configuration below and
run cuda/run_pairwise_cuda_cluster.sh (Linux) or .bat (Windows).
"""

from __future__ import annotations

import math
import os

import nth_sweep_cuda as pipeline


# =============================================================================
# USER CONFIGURATION — EDIT THIS BLOCK
# =============================================================================

TEMPERATURES_K = [
    0.0,
    4.0,
    10.0,
    20.0,
    50.0,
    100.0,
    300.0,
]

CUDA_DEVICES = [0]
TEMPERATURE_WORKERS = len(CUDA_DEVICES)

GAMMA_OPT = 2.0 * math.pi * 83.0e6
GAMMA_PHON = 2.0 * math.pi * 13.1e6
G_COUPLING = 1.11e4
OMEGA_B = 2.0 * math.pi * 6.02e9

# Any value from 2 through 16. The solver creates exactly N_PHOTON_MODES - 1
# independent phonons and reports arrays B_det/B_mean/g2_0_phonon of that size.
N_PHOTON_MODES = 3

N_PUMP_POINTS = 50
E_MIN_OVER_E2 = 0.0
E_MAX_OVER_E2 = 10.0

N_PATHS = 100
DT = 1.0e-9
BURN_TAU = 200.0
RECORD_TAU = 1000.0
SAMPLES_PER_TAU = 10.0
RANDOM_SEED = 0
RK_SUBSTEPS = 4

GPU_MEMORY_FRACTION = 0.35
PUMP_CHUNK = 0
G1_LAGS = 64
G1_ORIGINS = 256

# =============================================================================
# END USER CONFIGURATION
# =============================================================================


def main() -> None:
    pipeline.CUDA_EXE = pipeline.CUDA_BUILD / (
        "sde_solver_pairwise_cuda.exe" if os.name == "nt"
        else "sde_solver_pairwise_cuda"
    )
    pipeline.PIPELINE_NAME = "nth_sweep_pairwise_cuda"
    pipeline.PHONON_LAYOUT = "pairwise"

    for name in (
        "TEMPERATURES_K", "CUDA_DEVICES", "TEMPERATURE_WORKERS",
        "GAMMA_OPT", "GAMMA_PHON", "G_COUPLING", "OMEGA_B",
        "N_PHOTON_MODES", "N_PUMP_POINTS", "E_MIN_OVER_E2",
        "E_MAX_OVER_E2", "N_PATHS", "DT", "BURN_TAU", "RECORD_TAU",
        "SAMPLES_PER_TAU", "RANDOM_SEED", "RK_SUBSTEPS",
        "GPU_MEMORY_FRACTION", "PUMP_CHUNK", "G1_LAGS", "G1_ORIGINS",
    ):
        setattr(pipeline, name, globals()[name])

    pipeline.main()


if __name__ == "__main__":
    main()
