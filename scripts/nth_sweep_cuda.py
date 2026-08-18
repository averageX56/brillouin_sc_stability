#!/usr/bin/env python3
"""CUDA/C++ entry point for the temperature-driven Brillouin sweep.

The temperature grid and all physical defaults stay in ``nth_sweep.py``.  This
wrapper builds the independent CUDA solver, points the existing sweep driver at
that binary, and uses separate cache/output/log paths so CPU and CUDA results
can never be mixed accidentally.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CUDA_SOURCE = ROOT / "cuda"
CUDA_BUILD = ROOT / "build_cuda"
CUDA_EXE = CUDA_BUILD / ("sde_solver_cuda.exe" if os.name == "nt" else "sde_solver_cuda")


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


def visible_device_ids() -> list[int]:
    """Return logical CUDA ids exposed to this process by a cluster scheduler."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return [0]
    if visible in {"-1", "NoDevFiles"}:
        return []
    tokens = [x.strip() for x in visible.split(",") if x.strip()]
    # CUDA renumbers visible physical ids/UUIDs densely inside the process.
    return list(range(len(tokens))) or [0]


def main() -> None:
    cuda_parser = argparse.ArgumentParser(add_help=False)
    cuda_parser.add_argument("--cuda-device", type=int, default=None,
                             help="single-device compatibility alias")
    cuda_parser.add_argument("--cuda-devices", default=None,
                             help="comma-separated logical CUDA ids; default: all "
                                  "devices exposed through CUDA_VISIBLE_DEVICES")
    cuda_parser.add_argument("--temperature-workers", type=int, default=0,
                             help="parallel temperatures; 0 = one worker per selected GPU")
    cuda_parser.add_argument("--g1-lags", type=int, default=64)
    cuda_parser.add_argument("--g1-origins", type=int, default=256)
    cuda_parser.add_argument("--pump-chunk", type=int, default=0,
                             help="pump points resident on GPU at once; 0 = automatic")
    cuda_parser.add_argument("--rk-substeps", type=int, default=4,
                             help="RK4/noise substeps inside each recorded dt (default 4)")
    cuda_parser.add_argument("--gpu-memory-fraction", type=float, default=0.35,
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
        devices = visible_device_ids()
    if not devices or any(d < 0 for d in devices) or len(set(devices)) != len(devices):
        cuda_parser.error("CUDA device ids must be distinct non-negative integers")
    temperature_workers = cuda_args.temperature_workers or len(devices)
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

    defaults = ["--exe", str(CUDA_EXE), "--no-make"]
    if not has_option(sweep_argv, "--temperature-workers"):
        defaults += ["--temperature-workers", str(temperature_workers)]
    if not has_option(sweep_argv, "--worker-devices"):
        defaults += ["--worker-devices", ",".join(map(str, devices))]
    if not has_option(sweep_argv, "--cache-dir"):
        cache_tag = (f"rk{cuda_args.rk_substeps}_lags{cuda_args.g1_lags}_"
                     f"orig{cuda_args.g1_origins}")
        defaults += ["--cache-dir", str(ROOT / "data" / "nth_cuda_cache" / cache_tag)]
    if not has_option(sweep_argv, "--out"):
        defaults += ["--out", str(ROOT / "data" / "nth_sweep_cuda.json")]
    if not has_option(sweep_argv, "--log") and not has_option(sweep_argv, "--no-log"):
        defaults += ["--log", str(ROOT / "data" / "sweep_nth_cuda.log")]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nth_sweep

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
    out_path = Path(out_arg) if out_arg else ROOT / "data" / "nth_sweep_cuda.json"
    with out_path.open(encoding="utf-8") as f:
        result = json.load(f)
    result.setdefault("meta", {})["backend"] = "cuda_cpp"
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
