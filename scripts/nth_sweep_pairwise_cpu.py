#!/usr/bin/env python3
"""CPU n_th sweep for N photons with N-1 independent phonons.

Edit the USER CONFIGURATION block for ordinary runs. The numerical sweep,
temperature conversion, cache format and estimators are shared with
``nth_sweep.py``; only the executable and phonon topology differ.
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
CPU_EXE = ROOT / "build" / (
    "sde_solver_pairwise.exe" if os.name == "nt" else "sde_solver_pairwise"
)

# =============================================================================
# USER CONFIGURATION — EDIT THIS BLOCK
# =============================================================================

TEMPERATURES_K = [10.0, 300.0, 3000.0]

GAMMA_OPT = 2.0 * math.pi * 83.0e6
GAMMA_PHON = 2.0 * math.pi * 13.1e6
G_COUPLING = 1.11e4
OMEGA_B = 2.0 * math.pi * 6.02e9

# Any value from 2 through 64. Exactly N_PHOTON_MODES - 1 phonons are created.
N_PHOTON_MODES = 3

N_PUMP_POINTS = 20
E_MIN_OVER_E2 = 0.0
E_MAX_OVER_E2 = 5.0

N_PATHS = 100
DT = 1.0e-8
BURN_TAU = 200.0
RECORD_TAU = 1000.0
SAMPLES_PER_TAU = 10.0
RANDOM_SEED = 0
THREADS = 0

# =============================================================================
# END USER CONFIGURATION
# =============================================================================


def has_option(argv: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(name + "=") for arg in argv)


def option_value(argv: list[str], name: str) -> str | None:
    value = None
    for i, arg in enumerate(argv):
        if arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
        elif arg == name and i + 1 < len(argv):
            value = argv[i + 1]
    return value


def build_cpu() -> None:
    if os.name == "nt":
        command = ["cmd.exe", "/d", "/c", "call", str(ROOT / "build_pairwise_cpu.bat")]
    else:
        command = ["make", "pairwise"]
    print(">>", subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command),
          flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)
    if not CPU_EXE.exists():
        raise FileNotFoundError(f"CPU build finished but {CPU_EXE} was not produced")


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--no-build-cpu", action="store_true")
    wrapper_args, sweep_argv = wrapper.parse_known_args()

    dry_run = has_option(sweep_argv, "--dry-run")
    selected_exe = Path(option_value(sweep_argv, "--exe") or CPU_EXE)
    if not dry_run and not selected_exe.exists():
        if wrapper_args.no_build_cpu:
            raise FileNotFoundError(f"{selected_exe} does not exist and --no-build-cpu was set")
        if selected_exe != CPU_EXE:
            raise FileNotFoundError(f"custom --exe does not exist: {selected_exe}")
        build_cpu()

    requested_n = int(option_value(sweep_argv, "--N-photons") or N_PHOTON_MODES)
    E2 = GAMMA_OPT ** 1.5 * math.sqrt(GAMMA_PHON) / (2.0 * G_COUPLING)
    defaults = ["--exe", str(selected_exe), "--no-make"]

    def add_default(option: str, value: object) -> None:
        if not has_option(sweep_argv, option):
            defaults.extend([option, str(value)])

    add_default("--gamma-opt", GAMMA_OPT)
    add_default("--Gamma", GAMMA_PHON)
    add_default("--g", G_COUPLING)
    add_default("--N-photons", requested_n)
    add_default("--N-phonons", requested_n - 1)
    add_default("--phonon-layout", "pairwise")
    add_default("--n-paths", N_PATHS)
    add_default("--nE", N_PUMP_POINTS)
    add_default("--E-min", E_MIN_OVER_E2 * E2)
    add_default("--E-max", E_MAX_OVER_E2 * E2)
    add_default("--dt", DT)
    add_default("--burn-tau", BURN_TAU)
    add_default("--record-tau", RECORD_TAU)
    add_default("--samples-per-tau", SAMPLES_PER_TAU)
    add_default("--seed", RANDOM_SEED)
    if THREADS > 0:
        add_default("--threads", THREADS)
    add_default("--cache-dir", ROOT / "data" / "nth_sweep_pairwise_cpu_cache")
    add_default("--out", ROOT / "data" / "nth_sweep_pairwise_cpu.json")
    if not has_option(sweep_argv, "--log") and not has_option(sweep_argv, "--no-log"):
        defaults += ["--log", str(ROOT / "data" / "nth_sweep_pairwise_cpu.log")]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nth_sweep

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

    out_path = Path(option_value(sweep_argv, "--out")
                    or ROOT / "data" / "nth_sweep_pairwise_cpu.json")
    with out_path.open(encoding="utf-8") as f:
        result = json.load(f)
    result.setdefault("meta", {})["backend"] = "cpu_cpp"
    result["meta"]["phonon_layout"] = "pairwise"
    result["meta"]["thermal_control"] = (
        "TEMPERATURES_K in scripts/nth_sweep_pairwise_cpu.py"
    )
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
