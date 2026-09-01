#!/usr/bin/env python3
"""CUDA/C++ entry point for the temperature-driven Brillouin sweep.

Edit only the USER CONFIGURATION block below for an ordinary run. Command-line
options remain available for debugging, but are not required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CUDA_SOURCE = ROOT / "cuda"
CUDA_BUILD = ROOT / "build_cuda"
CUDA_EXE = CUDA_BUILD / ("sde_solver_cuda.exe" if os.name == "nt" else "sde_solver_cuda")
PIPELINE_NAME = "nth_sweep_cuda"
PHONON_LAYOUT = "shared_two"


# =============================================================================
# USER CONFIGURATION — EDIT THIS BLOCK
# =============================================================================

# Temperatures are entered manually in kelvin. The script converts every value
# to n_th(T) = 1/expm1(hbar*OMEGA_B/(k_B*T)); the solver never receives T itself.
TEMPERATURES_K = [
    0.0,
    4.0,
    10.0,
    20.0,
    50.0,
    100.0,
    300.0,
]

# Logical CUDA ids visible inside the job. The number of simultaneously used
# GPUs is len(CUDA_DEVICES). Examples: [0], [0, 1], [0, 1, 2, 3].
CUDA_DEVICES = [0]
TEMPERATURE_WORKERS = len(CUDA_DEVICES)  # one temperature queue per GPU

# Physical parameters, all in SI angular-frequency units s^-1.
GAMMA_OPT = 2.0 * math.pi * 83.0e6       # common photon decay gamma_j
GAMMA_PHON = 2.0 * math.pi * 13.1e6      # phonon decay Gamma
G_COUPLING = 1.11e4                      # Brillouin coupling g
OMEGA_B = 2.0 * math.pi * 6.02e9         # acoustic angular frequency
N_PHOTON_MODES = 3

# Pump grid. E2 is recomputed from the physical parameters above.
N_PUMP_POINTS = 50
E_MIN_OVER_E2 = 0.0
E_MAX_OVER_E2 = 10.0

# Stochastic sampling and time integration.
N_PATHS = 100
DT = 1.0e-9
BURN_TAU = 200.0
RECORD_TAU = 1000.0
SAMPLES_PER_TAU = 10.0
RANDOM_SEED = 0
RK_SUBSTEPS = 4

# CUDA memory and g1 estimator.
GPU_MEMORY_FRACTION = 0.35
PUMP_CHUNK = 0                           # 0 = select automatically
G1_LAGS = 64
G1_ORIGINS = 256

# =============================================================================
# END USER CONFIGURATION
# =============================================================================


def build_cuda() -> None:
    global CUDA_EXE
    if os.name == "nt":
        build_script = CUDA_SOURCE / "build_cuda.bat"
        if not build_script.exists():
            raise FileNotFoundError(f"CUDA build script was not found: {build_script}")
        command = ["cmd.exe", "/d", "/c", "call", str(build_script)]
        print(">>", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=str(ROOT), check=True)
        if not CUDA_EXE.exists():
            raise FileNotFoundError(
                f"CUDA build finished but {CUDA_EXE} was not produced")
        return

    configure = ["cmake", "-S", str(CUDA_SOURCE), "-B", str(CUDA_BUILD),
                 "-DCMAKE_BUILD_TYPE=Release"]
    build = ["cmake", "--build", str(CUDA_BUILD), "--config", "Release", "-j"]
    print(">>", shlex.join(configure), flush=True)
    subprocess.run(configure, check=True)
    print(">>", shlex.join(build), flush=True)
    subprocess.run(build, check=True)
    if not CUDA_EXE.exists():
        raise FileNotFoundError(f"CUDA build finished but {CUDA_EXE} was not produced")


def has_option(argv: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(name + "=") for arg in argv)


def option_value(argv: list[str], name: str) -> str | None:
    """Return the last CLI value from either --name VALUE or --name=VALUE."""
    value = None
    for i, arg in enumerate(argv):
        if arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
        elif arg == name and i + 1 < len(argv):
            value = argv[i + 1]
    return value


def main() -> None:
    cuda_parser = argparse.ArgumentParser(add_help=False)
    cuda_parser.add_argument("--cuda-device", type=int, default=None,
                             help="single-device compatibility alias")
    cuda_parser.add_argument("--cuda-devices", default=None,
                             help="override CUDA_DEVICES from this file")
    cuda_parser.add_argument("--temperature-workers", type=int, default=None,
                             help="override TEMPERATURE_WORKERS from this file")
    cuda_parser.add_argument("--g1-lags", type=int, default=G1_LAGS)
    cuda_parser.add_argument("--g1-origins", type=int, default=G1_ORIGINS)
    cuda_parser.add_argument("--pump-chunk", type=int, default=PUMP_CHUNK,
                             help="pump points resident on GPU at once; 0 = automatic")
    cuda_parser.add_argument("--rk-substeps", type=int, default=RK_SUBSTEPS,
                             help="RK4/noise substeps inside each recorded dt (default 4)")
    cuda_parser.add_argument("--gpu-memory-fraction", type=float, default=GPU_MEMORY_FRACTION,
                             help="fraction of currently free memory available to a worker")
    cuda_parser.add_argument("--no-build-cuda", action="store_true")
    cuda_args, sweep_argv = cuda_parser.parse_known_args()

    if cuda_args.cuda_device is not None and cuda_args.cuda_devices is not None:
        cuda_parser.error("use either --cuda-device or --cuda-devices, not both")
    if cuda_args.cuda_device is not None:
        devices = [cuda_args.cuda_device]
    elif cuda_args.cuda_devices is not None:
        try:
            devices = [int(x.strip()) for x in cuda_args.cuda_devices.split(",") if x.strip()]
        except ValueError:
            cuda_parser.error("--cuda-devices must be a comma-separated list of integers")
    else:
        devices = list(CUDA_DEVICES)
    if not devices or any(d < 0 for d in devices) or len(set(devices)) != len(devices):
        cuda_parser.error("CUDA device ids must be distinct non-negative integers")
    temperature_workers = (TEMPERATURE_WORKERS if cuda_args.temperature_workers is None
                           else cuda_args.temperature_workers)
    if temperature_workers < 1 or temperature_workers > len(devices):
        cuda_parser.error("--temperature-workers must be between 1 and the number of devices")
    if not 0.0 < cuda_args.gpu_memory_fraction <= 0.8:
        cuda_parser.error("--gpu-memory-fraction must be in (0, 0.8]")
    devices = devices[:temperature_workers]

    for name, value in {
        # The parent value is used by calibration; each temperature worker
        # receives its own device through --worker-devices below.
        "SDE_CUDA_DEVICE": devices[0],
        "SDE_CUDA_G1_LAGS": cuda_args.g1_lags,
        "SDE_CUDA_G1_ORIGINS": cuda_args.g1_origins,
        "SDE_CUDA_PUMP_CHUNK": cuda_args.pump_chunk,
        "SDE_CUDA_RK_SUBSTEPS": cuda_args.rk_substeps,
        "SDE_CUDA_MEMORY_FRACTION": cuda_args.gpu_memory_fraction,
    }.items():
        os.environ[name] = str(value)

    dry_run = has_option(sweep_argv, "--dry-run")
    if not dry_run and not CUDA_EXE.exists():
        if cuda_args.no_build_cuda:
            raise FileNotFoundError(f"{CUDA_EXE} does not exist and --no-build-cuda was set")
        build_cuda()

    E2 = GAMMA_OPT ** 1.5 * math.sqrt(GAMMA_PHON) / (2.0 * G_COUPLING)
    defaults = ["--exe", str(CUDA_EXE), "--no-make"]

    def add_default(option: str, value: object) -> None:
        if not has_option(sweep_argv, option):
            defaults.extend([option, str(value)])

    add_default("--gamma-opt", GAMMA_OPT)
    add_default("--Gamma", GAMMA_PHON)
    add_default("--g", G_COUPLING)
    requested_n_photons = int(option_value(sweep_argv, "--N-photons") or N_PHOTON_MODES)
    add_default("--N-photons", requested_n_photons)
    add_default("--N-phonons", 2 if PHONON_LAYOUT == "shared_two" else requested_n_photons - 1)
    add_default("--phonon-layout", PHONON_LAYOUT)
    add_default("--n-paths", N_PATHS)
    add_default("--nE", N_PUMP_POINTS)
    add_default("--E-min", E_MIN_OVER_E2 * E2)
    add_default("--E-max", E_MAX_OVER_E2 * E2)
    add_default("--dt", DT)
    add_default("--burn-tau", BURN_TAU)
    add_default("--record-tau", RECORD_TAU)
    add_default("--samples-per-tau", SAMPLES_PER_TAU)
    add_default("--seed", RANDOM_SEED)
    if not has_option(sweep_argv, "--temperature-workers"):
        defaults += ["--temperature-workers", str(temperature_workers)]
    if not has_option(sweep_argv, "--worker-devices"):
        defaults += ["--worker-devices", ",".join(map(str, devices))]
    if not has_option(sweep_argv, "--cache-dir"):
        cache_tag = (f"rk{cuda_args.rk_substeps}_lags{cuda_args.g1_lags}_"
                     f"orig{cuda_args.g1_origins}")
        defaults += ["--cache-dir", str(ROOT / "data" / f"{PIPELINE_NAME}_cache" / cache_tag)]
    if not has_option(sweep_argv, "--out"):
        defaults += ["--out", str(ROOT / "data" / f"{PIPELINE_NAME}.json")]
    if not has_option(sweep_argv, "--log") and not has_option(sweep_argv, "--no-log"):
        defaults += ["--log", str(ROOT / "data" / f"{PIPELINE_NAME}.log")]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nth_sweep

    # Make this file the single source of truth for the CUDA run.
    nth_sweep.TEMPERATURES_K = [float(T) for T in TEMPERATURES_K]
    nth_sweep.OMEGA_B = OMEGA_B
    nth_sweep.THETA_B = nth_sweep.HBAR * OMEGA_B / nth_sweep.K_B

    old_argv = sys.argv
    sys.argv = [old_argv[0], *defaults, *sweep_argv]
    try:
        nth_sweep.main()
    finally:
        sys.argv = old_argv

    if dry_run:
        return
    out_arg = next((x.split("=", 1)[1] for x in sweep_argv if x.startswith("--out=")), None)
    if out_arg is None:
        out_arg = next((sweep_argv[i + 1] for i, x in enumerate(sweep_argv[:-1])
                        if x == "--out"), None)
    out_path = Path(out_arg) if out_arg else ROOT / "data" / f"{PIPELINE_NAME}.json"
    with out_path.open(encoding="utf-8") as f:
        result = json.load(f)
    result.setdefault("meta", {})["backend"] = "cuda_cpp"
    result["meta"]["thermal_control"] = f"TEMPERATURES_K in scripts/{PIPELINE_NAME}.py"
    result["meta"]["phonon_layout"] = PHONON_LAYOUT
    result["meta"]["cuda_devices"] = devices
    result["meta"]["temperature_workers"] = temperature_workers
    result["meta"]["g1_lags"] = cuda_args.g1_lags
    result["meta"]["g1_origins"] = cuda_args.g1_origins
    result["meta"]["rk_substeps"] = cuda_args.rk_substeps
    result["meta"]["gpu_memory_fraction"] = cuda_args.gpu_memory_fraction
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
