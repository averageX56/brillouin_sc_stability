#!/usr/bin/env python3
"""nth_sweep.py — sweep bath temperature T and, for every T,
run a full pump sweep with the C++ Brillouin SDE solver.

The user-facing thermal control parameter is temperature in kelvin.  For each
point it is converted to the Bose--Einstein occupation of the resonant acoustic
mode,

    n_th(T) = 1 / [exp(hbar*Omega_b/(k_B*T)) - 1],

with the exact T -> 0 limit n_th = 0.  The solver still receives ``--nth`` and
uses D0 = Gamma*nth/2, so that <|b_k|^2> = n_th exactly.

Temperatures are specified explicitly in the TEMPERATURES_K list below.
There is no command-line interface for n_th, T-min/T-max, or a temperature
grid: edit that list directly.  For each T the Bose--Einstein occupation is
computed internally and passed to the low-level solver.

Produces one aggregated JSON (default nth_sweep.json) holding every
per-E record for every temperature, plus per-temperature solver JSONs in a
cache directory.

Usage
-----
    make
    # Edit TEMPERATURES_K below, then:
    python3 scripts/nth_sweep.py --dry-run --no-log
    python3 scripts/nth_sweep.py --threads 7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:  # keep the solver usable on a minimal cluster image
    tqdm = None

# Otterstrom et al., A silicon Brillouin laser (Science 2018), physical SI rates.
# No frequency normalization is used here: every decay/coupling rate is passed to
# the solver directly in s^-1, and time is measured in seconds.
#
# Table S1: Stokes/symmetric optical decay rate gamma_1 = 2*pi*83 MHz.
# The experimental pump/anti-symmetric rate is gamma_2 = 2*pi*481 MHz, but for
# this cascade we deliberately use the Stokes value for EVERY photon mode.
# Table S2: phonon decay Gamma_0 = 2*pi*13.1 MHz and Brillouin coupling g = 11.1 kHz.
# The resonant Brillouin frequency is Omega_b = 2*pi*6.02 GHz; it does not enter
# this rotating-frame/on-resonance envelope solver explicitly.
GAMMA_OPT = 5.2150438049590564e8           # s^-1 = 2*pi*83 MHz, common gamma_j
GAMMA_PHON = 8.230972752405258e7           # s^-1 = 2*pi*13.1 MHz
G_COUPLING = 1.11e4                        # s^-1, article value g = 11.1 kHz
OMEGA_B = 3.782477554922111e10             # rad/s = 2*pi*6.02 GHz
HBAR = 1.054571817e-34                     # J*s, CODATA exact-enough numerical value
K_B = 1.380649e-23                         # J/K, exact SI value
THETA_B = HBAR * OMEGA_B / K_B             # K = hbar*Omega_b/k_B ~= 0.288914 K
N_PHOTONS = 3


# ---------------------------------------------------------------------------
# TEMPERATURE SWEEP: EDIT THIS LIST BY HAND. Values are in kelvin.
#
# Examples:
#   TEMPERATURES_K = [0.0, 4.0, 10.0, 20.0, 50.0, 100.0, 300.0]
#   TEMPERATURES_K = [4.2, 77.0, 300.0]
#
# T = 0 K is allowed and is treated exactly as n_th = 0.
# The order is preserved; duplicates are rejected in main().
TEMPERATURES_K = [
    0.0,
    4.0,
    10.0,
    20.0,
    50.0,
    100.0,
    300.0,
]


def nth_from_temperature(T_K: float) -> float:
    """Mean thermal occupation of the acoustic mode at temperature T_K.

    n_th = [exp(hbar*Omega_b/(k_B*T)) - 1]^{-1}.
    ``math.expm1`` avoids loss of precision in the classical/high-T regime.
    """
    if T_K < 0.0:
        raise ValueError(f"temperature must be >= 0 K, got {T_K}")
    if T_K == 0.0:
        return 0.0
    x = THETA_B / T_K
    # expm1 overflows only deep in the T -> 0 regime, where n_th is
    # indistinguishable from zero in double precision anyway.
    if x > 700.0:
        return 0.0
    return 1.0 / math.expm1(x)

# Second-threshold pump amplitude E2 = gamma^1.5 * sqrt(Gamma) / (2*g) (see
# main() below, where the same expression is printed for the actually-used
# --g/--Gamma/--gamma-opt). Used only to size the --E-max default at 10*E2.
_E2_DEFAULT = GAMMA_OPT ** 1.5 * math.sqrt(GAMMA_PHON) / (2.0 * G_COUPLING)
E_MAX_DEFAULT = 10.0 * _E2_DEFAULT


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="temperature-driven n_th sweep for the Brillouin cascade SDE; edit TEMPERATURES_K in the script")
    ap.add_argument("--exe", default=None, help="solver binary (default ./sde_solver[.exe])")
    ap.add_argument("--no-make", action="store_true", help="skip running make first")
    # pump grid
    ap.add_argument("--E-min", type=float, default=0.0)
    ap.add_argument("--E-max", type=float, default=E_MAX_DEFAULT)
    ap.add_argument("--nE", type=int, default=50)
    # physics
    ap.add_argument("--g", type=float, default=G_COUPLING)
    ap.add_argument("--Gamma", type=float, default=GAMMA_PHON)
    ap.add_argument("--gamma-opt", type=float, default=GAMMA_OPT)
    ap.add_argument("--N-photons", type=int, default=N_PHOTONS)
    # integration
    ap.add_argument("--scheme", default="splitting", choices=["splitting", "taylor15", "euler"])
    ap.add_argument("--noise", default="gauss", choices=["gauss", "telegraph"])
    ap.add_argument("--dt", type=float, default=1.0e-9,
                    help="integration step in seconds (default 1 ns)")
    ap.add_argument("--n-paths", type=int, default=100)
    ap.add_argument("--thin", type=int, default=None,
                    help="record every thin-th step (default: derived from "
                         "--samples-per-tau, see below)")
    ap.add_argument("--samples-per-tau", type=float, default=10.0,
                    help="recorded samples per phonon correlation time 1/Gamma "
                         "(default 10; increase for linewidth fits)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    # sampling window, expressed in phonon lifetimes 1/Gamma
    ap.add_argument("--burn-tau", type=float, default=200.0,
                    help="discarded transient in units of 1/Gamma (default 200)")
    ap.add_argument("--record-tau", type=float, default=1000.0,
                    help="recorded window in units of 1/Gamma (default 1000)")
    ap.add_argument("--n-steps", type=long_or_none, default=None,
                    help="override total steps (else derived from --record-tau)")
    ap.add_argument("--burn", type=long_or_none, default=None,
                    help="override burn steps (else derived from --burn-tau)")
    # bookkeeping
    ap.add_argument("--cache-dir", default=None,
                    help="per-temperature solver JSONs (default <repo>/data/nth_cache)")
    ap.add_argument("--out", default=None,
                    help="aggregated output (default <repo>/data/nth_sweep.json)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--log", default=None,
                    help="progress log, mirrored from the console (default "
                         "<repo>/data/sweep_nth.log). Survives a frozen console.")
    ap.add_argument("--no-log", action="store_true", help="do not write a log file")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="hard limit on a single solver call, seconds (0 = none). "
                         "Blunt: a legitimately long point and a hung one both hit it.")
    ap.add_argument("--stall-timeout", type=float, default=900.0,
                    help="kill the solver if its output stops growing for this many "
                         "seconds (default 900, 0 = disabled). Catches a process that "
                         "has stopped making progress without waiting out --timeout.")
    ap.add_argument("--quiet-solver", action="store_true",
                    help="suppress the solver's per-pump-point progress lines "
                         "(not recommended: they are the only live progress signal)")
    ap.add_argument("--verbose-solver", action="store_true",
                    help="per-phase timing markers from the solver (steady state / "
                         "integration / estimators). Use this to find out which "
                         "phase a seemingly stuck run is sitting in.")
    ap.add_argument("--calibrate", action="store_true",
                    help="time a short run first and print an estimated wall time")
    ap.add_argument("--temperature-workers", type=int, default=1,
                    help="number of temperatures evaluated concurrently (default 1)")
    ap.add_argument("--worker-devices", default="",
                    help="comma-separated logical CUDA device ids, one per worker; "
                         "empty means ordinary CPU workers")
    return ap.parse_args()


def long_or_none(s):
    return None if s is None or s == "" else int(float(s))


def linspace(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1:
        return [lo]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def choose_thin(args) -> int:
    """Recording stride.

    Successive samples are only independent on the scale of the phonon
    correlation time tau = 1/Gamma. Recording every `thin` steps gives
    tau/(thin*dt) samples per correlation time; anything far above ~10 is pure
    redundancy that costs memory (n_keep*n_paths*nvar*16 bytes) and estimator
    time (D_from_msd is O(n_lags*n_paths*n_keep) and single-threaded), while the
    number of INDEPENDENT samples stays T_record/tau per path.

    With the Otterstrom defaults Gamma = 2*pi*13.1 MHz, so tau = 1/Gamma ~= 12.15 ns.
    At dt = 1 ns, thin = 1 records about 12 samples per phonon correlation time.

    BUT the two observables differ in what they need:
      * g2(0) is a single-time statistic and is insensitive to the stride.
        Measured at Gamma=1e-2, record 100/Gamma: g2_a2 = 3.065 / 3.066 / 3.048
        at thin = 20 / 200 / 1000 — flat to the sampling error.
      * the linewidth is a fit to the DECAY of |g1| (or to the phase MSD), so it
        depends on how many lags fall inside the decay window. Same runs:
        FWHM_a2 = 0.0313 / 0.0266 / 0.0055 at thin = 20 / 200 / 1000, i.e. the
        coarsest stride is off by a factor 6. The fit R^2 there is only ~0.5,
        meaning the decay is not a single exponential at that operating point,
        which is exactly why the answer moves with the fit window.
    Hence the default targets ~10 samples per tau; this is appropriate for g2 sweeps.
    Increase --samples-per-tau if a finer time grid is required for linewidth fits.
    """
    if args.thin is not None:
        return max(1, args.thin)
    tau = 1.0 / args.Gamma
    return max(1, int(round(tau / (args.samples_per_tau * args.dt))))


def sampling_window(args) -> tuple[int, int]:
    """Return (n_steps, burn) in steps, sized from the phonon lifetime 1/Gamma."""
    tau = 1.0 / args.Gamma
    burn = args.burn if args.burn is not None else int(math.ceil(args.burn_tau * tau / args.dt))
    if args.n_steps is not None:
        n_steps = args.n_steps
    else:
        n_steps = burn + int(math.ceil(args.record_tau * tau / args.dt))
    # Warn rather than silently produce a biased/noisy g2.
    t_burn = burn * args.dt
    t_rec = (n_steps - burn) * args.dt
    if t_burn * args.Gamma < 1.0:
        print(f"warning: burn = {t_burn:.3g} time units = {t_burn * args.Gamma:.2f}/Gamma "
              f"(< 1/Gamma). Thermal phonon init makes this mostly harmless, but the "
              f"PHOTON transient may survive.", file=sys.stderr)
    if t_rec * args.Gamma < 20.0:
        print(f"warning: recorded window = {t_rec * args.Gamma:.1f}/Gamma; effective sample "
              f"N_eff ~ n_paths*Gamma*T = {args.n_paths * t_rec * args.Gamma:.0f}, so the "
              f"g2 scatter is ~{100.0 / max(math.sqrt(args.n_paths * t_rec * args.Gamma), 1e-9):.1f}%.",
              file=sys.stderr)
    return n_steps, burn


class Log:
    """Progress written to BOTH the console and a file, flushed line by line.

    Why the file matters: a Windows console in QuickEdit/mark mode (i.e. anything
    selected with the mouse in cmd.exe) BLOCKS every process that writes to it.
    The symptom is a run that looks frozen — last line stuck mid-sweep, zero CPU
    load, nothing corrupt — and it resumes the moment you press Enter or Esc.
    With a log file the real progress is visible regardless of what the console
    is doing, and `type data\\sweep_nth.log` tells you where the run actually is.
    """

    def __init__(self, path: Path | None, echo: bool = True):
        self.fh = None
        self.path = path
        self.echo = echo
        self.lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(path, "a", encoding="utf-8", buffering=1)
            self.fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"{' '.join(sys.argv)} ===\n")

    def __call__(self, msg: str = "", console: bool = True) -> None:
        with self.lock:
            if console:
                try:
                    if tqdm is not None:
                        tqdm.write(msg)
                    else:
                        print(msg, flush=True)
                except OSError:
                    pass  # console gone; the file still has it
            if self.fh:
                self.fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    def child_handle(self):
        """Raw file handle for a child process to write into (no pipe)."""
        if self.fh is None:
            return None
        self.fh.flush()
        return self.fh

    def close(self) -> None:
        if self.fh:
            self.fh.close()


def launch_solver(cmd: list[str], log: "Log", timeout: float | None,
                  stall: float | None, env: dict[str, str] | None = None,
                  progress_desc: str | None = None, progress_position: int = 0) -> int:
    """Run the solver with its output going DIRECTLY into the log file.

    Deliberately no pipe. An earlier version piped the child's stdout into this
    process and echoed it to the console from the reading loop; that couples the
    solver to the console, and a Windows console in QuickEdit/mark mode then
    blocks the reader, fills the pipe and STALLS THE SOLVER ITSELF mid-run. With
    the child writing to a real file, nothing the console does can ever stop the
    computation.

    Two watchdogs, because they catch different failures:
      * `timeout` is a hard ceiling on the whole call.
      * `stall` fires when the log file has not grown for that long while the
        process is still alive — i.e. the solver has stopped making progress.
        That is the useful one: a point that legitimately takes 10 minutes keeps
        extending the deadline, while a process wedged after its last output is
        killed promptly instead of sitting out the full ceiling.

    Live console feedback comes from a daemon thread tailing the same file. If the
    console blocks, only that thread waits — the solver keeps running.
    """
    fh = log.child_handle()
    tail_path = log.path
    temporary_fh = None
    if fh is None and progress_desc is not None:
        temporary_fh = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", prefix="brillouin_progress_",
            suffix=".log", delete=False)
        fh = temporary_fh
        tail_path = Path(temporary_fh.name)
    stop = threading.Event()

    def tail(start: int) -> None:
        pos = start
        bar = None
        shown = 0
        pending = ""
        while True:
            try:
                with open(tail_path, "r", encoding="utf-8", errors="replace") as r:
                    r.seek(pos)
                    chunk = r.read()
                    pos = r.tell()
            except OSError:
                chunk = ""
            if chunk:
                ordinary = []
                pending += chunk
                lines = pending.splitlines(keepends=True)
                if lines and not lines[-1].endswith(("\n", "\r")):
                    pending = lines.pop()
                else:
                    pending = ""
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("SDE_PROGRESS "):
                        fields = stripped.split()
                        if len(fields) == 3:
                            try:
                                current, total = int(fields[1]), int(fields[2])
                            except ValueError:
                                ordinary.append(line)
                                continue
                            if tqdm is not None:
                                if bar is None:
                                    bar = tqdm(total=total, desc=progress_desc or "CUDA",
                                               unit="step", position=progress_position,
                                               dynamic_ncols=True, leave=True)
                                if current > bar.n:
                                    bar.update(current - bar.n)
                            elif current == total or current - shown >= max(1, total // 20):
                                shown = current
                                ordinary.append(f"{progress_desc or 'CUDA'}: {current}/{total} steps\n")
                        continue
                    if log.echo:
                        ordinary.append(line)
                if ordinary:
                    try:
                        text = "".join(ordinary)
                        if tqdm is not None:
                            tqdm.write(text.rstrip("\r\n"))
                        else:
                            sys.stdout.write(text)
                            sys.stdout.flush()
                    except OSError:
                        pass  # console gone or blocked; the file still has everything
            else:
                if stop.is_set():
                    break
                stop.wait(0.5)
        if bar is not None:
            if bar.total is not None and bar.n < bar.total:
                bar.update(bar.total - bar.n)
            bar.close()

    t = None
    if fh is not None and tail_path is not None and (log.echo or progress_desc is not None):
        fh.flush()
        t = threading.Thread(target=tail, args=(tail_path.stat().st_size,), daemon=True)
        t.start()

    reason = []
    try:
        proc = subprocess.Popen(cmd, stdout=(fh if fh is not None else None),
                                stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        last_size, last_growth = -1, time.time()
        while True:
            try:
                rc = proc.wait(timeout=2.0)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.time()
            if timeout and now - t0 > timeout:
                reason.append(f"hard --timeout {timeout:g} s")
                break
            if stall and tail_path is not None:
                try:
                    size = tail_path.stat().st_size
                except OSError:
                    size = last_size
                if size != last_size:
                    last_size, last_growth = size, now
                elif now - last_growth > stall:
                    reason.append(f"no output for {stall:g} s (--stall-timeout); "
                                  f"the solver stopped making progress")
                    break
        if reason:
            log(f"    !! killing solver: {reason[0]}")
            proc.kill()
            proc.wait()
            rc = -9
    finally:
        stop.set()
        if t is not None:
            t.join(timeout=2.0)
        if fh is not None:
            fh.flush()
        if temporary_fh is not None:
            temporary_name = temporary_fh.name
            temporary_fh.close()
            try:
                Path(temporary_name).unlink()
            except OSError:
                pass
    return rc


def exe_candidates(root: Path) -> list[Path]:
    """Where the solver binary may sit. On Windows MinGW g++ appends .exe itself,
    so both spellings must be tried; build/ is the current layout and the repo
    root is accepted too for hand-built binaries."""
    names = ["sde_solver.exe", "sde_solver"] if os.name == "nt" else ["sde_solver", "sde_solver.exe"]
    return [d / n for d in (root / "build", root) for n in names]


def find_exe(root: Path) -> Path:
    for c in exe_candidates(root):
        if c.exists():
            return c
    return (root / "build" /
            ("sde_solver.exe" if os.name == "nt" else "sde_solver"))  # for the error message


def try_make(root: Path) -> bool:
    """Run whichever make is available. Returns False if none is, so the caller
    can fall back to an already-built binary instead of dying with a traceback.
    MinGW usually installs make as mingw32-make, and plenty of Windows boxes
    have no make at all (build.bat covers those)."""
    for prog in ("make", "mingw32-make", "gmake", "nmake"):
        try:
            r = subprocess.run([prog], cwd=str(root))
        except FileNotFoundError:
            continue
        except OSError as e:
            print(f"warning: {prog} could not be started ({e})", file=sys.stderr)
            continue
        if r.returncode != 0:
            sys.exit(f"error: `{prog}` failed with code {r.returncode}. Fix the build "
                     f"first, or pass --no-make if the binary is already up to date.")
        return True
    print("warning: no make program found (tried make, mingw32-make, gmake, nmake). "
          "Falling back to an existing binary; on Windows build it with build.bat.",
          file=sys.stderr)
    return False


def calibrate(exe: Path, args, n_steps: int, burn: int) -> float:
    """Measure step-paths per second on this machine with a short run.

    Only the integration part scales this way; the estimators add a roughly
    constant cost per pump point that depends on n_keep, so the ETA below is a
    lower bound. It is still far better than no estimate at all.
    """
    import tempfile
    probe_steps = min(n_steps, max(20000, burn // 4 + 20000))
    probe_burn = probe_steps // 4
    with tempfile.TemporaryDirectory() as td:
        cmd = [str(exe), "--E-list", f"{args.E_max:.10g}",
               "--N-photons", str(args.N_photons),
               "--scheme", args.scheme, "--noise", args.noise,
               "--dt", f"{args.dt:.10g}", "--n-steps", str(probe_steps),
               "--burn", str(probe_burn), "--thin", str(args.thin),
               "--n-paths", str(args.n_paths), "--seed", "0",
               "--g", f"{args.g:.10g}", "--Gamma", f"{args.Gamma:.10g}",
               "--gamma-opt", f"{args.gamma_opt:.10g}", "--nth", f"{nth_from_temperature(max(TEMPERATURES_K)):.10g}",
               "--quiet", "--out", str(Path(td) / "cal.json")]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        t0 = time.time()
        r = subprocess.run(cmd)
        el = time.time() - t0
    if r.returncode != 0 or el <= 0:
        return float("nan")
    return probe_steps * args.n_paths / el


def param_fingerprint(args, n_steps: int, burn: int) -> str:
    """Short hash of everything that changes the numbers in a sweep JSON.

    The cache filename used to depend on n_th alone, so a rerun with a different
    dt / thin / n_paths / E-grid silently REUSED files computed with the old
    settings — half the sweep at one sampling and half at another, with no
    warning. Any parameter below therefore goes into the name, and mismatched
    entries simply miss the cache instead of corrupting the result.
    """
    keys = (args.dt, n_steps, burn, args.thin, args.n_paths, args.seed,
            args.nE, args.E_min, args.E_max, args.N_photons, args.scheme,
            args.noise, args.g, args.Gamma, args.gamma_opt, OMEGA_B, HBAR, K_B, tuple(TEMPERATURES_K))
    blob = "|".join(repr(k) for k in keys)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def run_one(exe: Path, T_K: float, nth: float, args, n_steps: int, burn: int, cache: Path,
            log: "Log", solver_env: dict[str, str] | None = None,
            progress_position: int = 0) -> dict:
    """Run (or reuse) one full pump sweep at fixed bath temperature."""
    fp = param_fingerprint(args, n_steps, burn)
    out = cache / f"T_{T_K:.9g}K_{fp}.json"
    # A leftover .partial from an interrupted run is not a result — drop it, so
    # `dir` stays readable and nobody mistakes it for a corrupt output.
    partial = Path(str(out) + ".partial")
    if partial.exists():
        log(f"    discarding stale {partial.name} (incomplete write)")
        try:
            partial.unlink()
        except OSError:
            pass
    if out.exists():
        try:
            with open(out) as f:
                cached = json.load(f)
            log(f"    reusing {out.name}")
            return cached
        except Exception as e:
            log(f"    cache entry {out.name} unreadable ({e}) -> recomputing")
            out.unlink()
    cmd = [
        str(exe),
        "--E-min", f"{args.E_min:.10g}", "--E-max", f"{args.E_max:.10g}",
        "--nE", str(args.nE),
        "--N-photons", str(args.N_photons),
        "--scheme", args.scheme, "--noise", args.noise,
        "--dt", f"{args.dt:.10g}", "--n-steps", str(n_steps), "--burn", str(burn),
        "--thin", str(args.thin), "--n-paths", str(args.n_paths),
        "--seed", str(args.seed),
        "--g", f"{args.g:.10g}", "--Gamma", f"{args.Gamma:.10g}",
        "--gamma-opt", f"{args.gamma_opt:.10g}", "--nth", f"{nth:.10g}",
        "--out", str(out),
    ]
    # NOTE: deliberately NOT --quiet. The solver prints one line per pump point,
    # which is the only live progress signal during a multi-minute sweep; with
    # --quiet a long run is indistinguishable from a hang.
    if args.quiet_solver:
        cmd.append("--quiet")
    if args.verbose_solver:
        cmd.append("--verbose")
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    log(f"    launching: {shlex.join(cmd)}", console=False)
    rc = launch_solver(cmd, log,
                       args.timeout if args.timeout > 0 else None,
                       args.stall_timeout if args.stall_timeout > 0 else None,
                       env=solver_env, progress_desc=f"T={T_K:g} K",
                       progress_position=progress_position)
    if rc != 0:
        log(f"error: solver exited with code {rc}")
        sys.exit(f"error: solver exited with code {rc} on\n  " + shlex.join(cmd)
                 + "\n(run that command by hand to see the full message)")
    if not out.exists():
        sys.exit(f"error: solver reported success but {out} was not written")
    with open(out) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent          # repo root
    exe = Path(args.exe) if args.exe else find_exe(root)
    out_path = Path(args.out) if args.out else root / "data" / "nth_sweep.json"
    cache = Path(args.cache_dir) if args.cache_dir else root / "data" / "nth_cache"

    log = Log(None if args.no_log else
              (Path(args.log) if args.log else root / "data" / "sweep_nth.log"))

    T_grid = [float(T_K) for T_K in TEMPERATURES_K]
    if not T_grid:
        sys.exit("error: TEMPERATURES_K is empty; add at least one temperature in kelvin")
    if any(not math.isfinite(T_K) for T_K in T_grid):
        sys.exit("error: every entry in TEMPERATURES_K must be finite")
    if any(T_K < 0.0 for T_K in T_grid):
        sys.exit("error: every entry in TEMPERATURES_K must be >= 0 K")
    if len(set(T_grid)) != len(T_grid):
        sys.exit("error: TEMPERATURES_K contains duplicate temperatures")

    if args.g <= 0.0 or args.Gamma <= 0.0 or args.gamma_opt <= 0.0:
        sys.exit("error: g, Gamma, and gamma-opt must all be positive")
    if args.dt <= 0.0 or args.nE < 2 or args.n_paths < 1 or args.N_photons < 2:
        sys.exit("error: require dt > 0, nE >= 2, n-paths >= 1, and N-photons >= 2")
    if args.E_min < 0.0 or args.E_max <= args.E_min:
        sys.exit("error: require 0 <= E-min < E-max")
    if args.burn_tau < 0.0 or args.record_tau <= 0.0 or args.samples_per_tau <= 0.0:
        sys.exit("error: require burn-tau >= 0, record-tau > 0, and samples-per-tau > 0")
    if args.temperature_workers < 1:
        sys.exit("error: --temperature-workers must be >= 1")
    try:
        worker_devices = ([int(x.strip()) for x in args.worker_devices.split(",") if x.strip()]
                          if args.worker_devices else [])
    except ValueError:
        sys.exit("error: --worker-devices must be a comma-separated list of integers")
    if any(d < 0 for d in worker_devices) or len(set(worker_devices)) != len(worker_devices):
        sys.exit("error: --worker-devices must contain distinct non-negative ids")
    if worker_devices and args.temperature_workers > len(worker_devices):
        sys.exit("error: one concurrent temperature worker per CUDA device is allowed; "
                 "reduce --temperature-workers or provide more --worker-devices")
    args.temperature_workers = min(args.temperature_workers, len(T_grid))

    nth_grid = [nth_from_temperature(T_K) for T_K in T_grid]
    if len(set(nth_grid)) != len(nth_grid):
        sys.exit("error: two temperatures map to the same floating-point n_th; remove one")

    n_steps, burn = sampling_window(args)
    args.thin = choose_thin(args)
    tau = 1.0 / args.Gamma
    E2 = args.gamma_opt ** 1.5 * math.sqrt(args.Gamma) / (2.0 * args.g)

    print(f"T grid ({len(T_grid)}) [K]: {', '.join(f'{T_K:g}' for T_K in T_grid)}")
    print(f"n_th(T) ({len(nth_grid)}): {', '.join(f'{v:.8g}' for v in nth_grid)}")
    print(f"Bose-Einstein conversion: n_th(T) = 1/expm1((hbar*Omega_b/k_B)/T), "
          f"hbar*Omega_b/k_B = {THETA_B:.9g} K")
    print(f"E grid: {args.nE} points on [{args.E_min:g}, {args.E_max:g}]  "
          f"(2nd threshold E2 = {E2:.4g}, so E_max/E2 = {args.E_max / E2:.2f})")
    print(f"gamma = {args.gamma_opt:g} s^-1, Gamma = {args.Gamma:g} s^-1 "
          f"(1/Gamma = {tau:.4g} s), g = {args.g:g} s^-1")
    print(f"Otterstrom reference Omega_b = {OMEGA_B:g} s^-1 "
          f"(Omega_b/2pi = {OMEGA_B / (2.0 * math.pi):g} Hz); rotating-frame solver")
    print(f"scheme = {args.scheme}, noise = {args.noise}, dt = {args.dt:g} s")
    print(f"steps: burn = {burn} (t = {burn * args.dt:g} = {burn * args.dt / tau:.1f}/Gamma), "
          f"total = {n_steps} (record t = {(n_steps - burn) * args.dt:g} = "
          f"{(n_steps - burn) * args.dt / tau:.1f}/Gamma)")
    n_keep = (n_steps - burn) // args.thin
    spt = tau / (args.thin * args.dt)
    mem_mb = n_keep * args.n_paths * (args.N_photons + 2) * 16 / 2**20
    print(f"thin = {args.thin} -> n_keep = {n_keep} samples/path "
          f"({spt:.1f} per correlation time, {n_keep * args.dt * args.thin / tau:.0f} "
          f"independent), trajectory buffer {mem_mb:.0f} MB per pump point")
    if spt > 500:
        print(f"warning: {spt:.0f} samples per correlation time is heavy oversampling. "
              f"The estimators are single-threaded and scale with n_keep, so this "
              f"inflates both runtime and memory without adding independent samples. "
              f"Drop --thin and let --samples-per-tau pick it.", file=sys.stderr)
    if n_keep < 2000:
        print(f"warning: only {n_keep} samples per path. g2 is fine, but the linewidth "
              f"fit becomes window-sensitive below a few thousand samples — treat "
              f"fwhm_g1/fwhm_msd from this run as indicative only.", file=sys.stderr)
    if mem_mb > 512:
        print(f"warning: {mem_mb:.0f} MB trajectory buffer per pump point. On a "
              f"memory-tight machine this alone can make the run crawl; raise "
              f"--thin or lower --n-paths.", file=sys.stderr)
    total_sp = len(T_grid) * args.nE * n_steps * args.n_paths
    print(f"cost: {total_sp:.3g} step-paths in total "
          f"({len(T_grid)} temperatures x {args.nE} E x {n_steps} steps x {args.n_paths} paths)")
    if args.temperature_workers > 1:
        where = (f" on logical CUDA devices {worker_devices[:args.temperature_workers]}"
                 if worker_devices else "")
        print(f"temperature parallelism: {args.temperature_workers} workers{where}")
    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    if not args.no_make:
        print(">> make", flush=True)
        if try_make(root) and not args.exe:
            exe = find_exe(root)          # the build may have produced the other spelling
    if not exe.exists():
        tried = "\n  ".join(str(c) for c in exe_candidates(root))
        sys.exit(f"error: solver binary not found. Looked at:\n  {tried}\n"
                 f"Build it with `make` (or mingw32-make), or on Windows without make: "
                 f"build.bat, then rerun. Point at it explicitly with --exe PATH.")
    print(f"solver: {exe}", flush=True)

    cache.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_log:
        log(f"log file: {root / 'data' / 'sweep_nth.log' if not args.log else args.log}")

    if args.calibrate:
        log(">> calibrating ...")
        rate = calibrate(exe, args, n_steps, burn)
        if math.isfinite(rate):
            eta = total_sp / rate
            log(f"   {rate:.3g} step-paths/s here -> integration alone ~"
                f"{eta / 3600:.1f} h for the whole sweep "
                f"({eta / len(T_grid) / 60:.0f} min per temperature); estimators add more")
        else:
            print("   calibration failed, skipping the estimate", file=sys.stderr)

    entries = []
    temperature_data: list[dict | None] = [None] * len(T_grid)

    def run_temperature_queue(slot: int) -> None:
        """Run one FIFO queue per device, so two jobs never collide on one GPU."""
        device = worker_devices[slot] if worker_devices else None
        for index in range(slot, len(T_grid), args.temperature_workers):
            T_K, nth = T_grid[index], nth_grid[index]
            device_text = f" GPU {device}" if device is not None else ""
            log(f"[{index + 1}/{len(T_grid)}] T = {T_K:g} K -> "
                f"n_th = {nth:.8g}{device_text} ...")
            separate_log = args.temperature_workers > 1
            worker_log_path = cache / "worker_logs" / f"T_{T_K:.9g}K.log"
            worker_log = Log(worker_log_path, echo=False) if separate_log else log
            solver_env = os.environ.copy()
            if device is not None:
                solver_env["SDE_CUDA_DEVICE"] = str(device)
            t_pt = time.time()
            try:
                temperature_data[index] = run_one(
                    exe, T_K, nth, args, n_steps, burn, cache, worker_log, solver_env,
                    progress_position=slot)
            except SystemExit as exc:
                raise RuntimeError(f"temperature {T_K:g} K failed: {exc}") from exc
            finally:
                if separate_log:
                    worker_log.close()
            log(f"    completed T = {T_K:g} K on{device_text or ' worker'} "
                f"[{(time.time() - t_pt) / 60:.1f} min]")

    try:
        with ThreadPoolExecutor(max_workers=args.temperature_workers,
                                thread_name_prefix="temperature") as pool:
            futures = [pool.submit(run_temperature_queue, slot)
                       for slot in range(args.temperature_workers)]
            for future in futures:
                future.result()
    except Exception as exc:
        log.close()
        sys.exit(f"error: parallel temperature sweep failed: {exc}")

    # Aggregation is serial and follows TEMPERATURES_K, independent of completion order.
    for i, (T_K, nth, data) in enumerate(zip(T_grid, nth_grid, temperature_data), 1):
        if data is None:
            log.close()
            sys.exit(f"error: no result produced for T = {T_K:g} K")
        R = data["results"]

        def col(key, j):
            return [None if r[key][j] is None else float(r[key][j]) for r in R]

        n_ph = args.N_photons
        entries.append({
            "T_K": T_K,
            "nth": nth,
            "E": [float(r["E"]) for r in R],
            # generation curves: deterministic and SDE-averaged amplitudes
            "A_det": [col("A_det", j) for j in range(n_ph)],
            "A_mean": [col("A_mean", j) for j in range(n_ph)],
            "B_det": [col("B_det", k) for k in range(2)],
            "B_mean": [col("B_mean", k) for k in range(2)],
            # g2 at EVERY pump point
            "g2_0": [col("g2_0", j) for j in range(n_ph)],
            "g2_lin": [col("g2_lin", j) for j in range(n_ph)],
            "g2_0_phonon": [col("g2_0_phonon", k) for k in range(2)],
            "fwhm_g1": [col("fwhm_g1", j) for j in range(n_ph)],
            "fwhm_msd": [col("fwhm_msd", j) for j in range(n_ph)],
            "n_diverged": [int(r["n_diverged"]) for r in R],
            "steady_ok": [bool(r["steady_converged"]) for r in R],
            "D0_PHONON": data["params"]["D0_PHONON"],
            "source": str(cache / f"T_{T_K:.9g}K_{param_fingerprint(args, n_steps, burn)}.json"),
        })
        div = sum(entries[-1]["n_diverged"])
        g2a2 = entries[-1]["g2_0"][1] if n_ph > 1 else []
        finite = [v for v in g2a2 if v is not None and math.isfinite(v)]
        # Below the first threshold the photon fixed point is a_j = 0, so both the
        # amplitude and g2 there are numerical residue (g2 ~ hundreds at E = 0),
        # not physics. Exclude those points from the summary range so it reflects
        # the generating regime; the raw values stay in the JSON.
        A2 = entries[-1]["A_mean"][1] if n_ph > 1 else []
        amax = max((a for a in A2 if a is not None and math.isfinite(a)), default=0.0)
        phys = [v for v, a in zip(g2a2, A2)
                if v is not None and math.isfinite(v)
                and a is not None and math.isfinite(a) and amax > 0 and a > 1e-3 * amax]
        rng_txt = (f"[{min(phys):.4f}, {max(phys):.4f}]" if phys
                   else (f"[{min(finite):.4f}, {max(finite):.4f}] (sub-threshold only)"
                         if finite else "all NaN"))
        n_sub = len(finite) - len(phys)
        ph = entries[-1]["g2_0_phonon"]
        ph_fin = [v for v in ph[1] if v is not None and math.isfinite(v)]
        ph_txt = f"{sum(ph_fin) / len(ph_fin):.4f}" if ph_fin else "NaN"
        log(f"    g2_a2 range = {rng_txt}   <g2_b2> = {ph_txt}   diverged = {div}"
            f"{f'   sub-threshold pts skipped: {n_sub}' if n_sub else ''}")

    result = {
        "meta": {
            "kind": "nth_sweep",
            "thermal_control": "TEMPERATURES_K in scripts/nth_sweep.py",
            "T_grid_K": T_grid,
            "nth_grid": nth_grid,
            "E_grid": linspace(args.E_min, args.E_max, args.nE),
            "E_threshold2": E2,
            "gamma_opt": args.gamma_opt, "Gamma": args.Gamma, "g": args.g,
            "Omega_b_reference": OMEGA_B,
            "hbar_J_s": HBAR,
            "k_B_J_per_K": K_B,
            "theta_b_K": THETA_B,
            "temperature_to_nth": "n_th = 1 / expm1(hbar*Omega_b/(k_B*T)); n_th(0 K)=0",
            "frequency_units": "s^-1 (angular decay rates gamma, Gamma; article g used directly)",
            "N_photons": args.N_photons,
            "scheme": args.scheme, "noise": args.noise,
            "dt": args.dt, "n_steps": n_steps, "burn": burn,
            "thin": args.thin, "n_paths": args.n_paths, "seed": args.seed,
            "burn_tau": burn * args.dt * args.Gamma,
            "record_tau": (n_steps - burn) * args.dt * args.Gamma,
            "nth_convention": "D0 = Gamma*nth/2, i.e. <|b_k|^2> = nth",
            "param_fingerprint": param_fingerprint(args, n_steps, burn),
            "phonon_init": "thermal: b = b_det + CN(0, D0/Gamma)",
            "temperature_workers": args.temperature_workers,
            "worker_devices": worker_devices[:args.temperature_workers],
        },
        "entries": entries,
    }
    partial_out = Path(str(out_path) + ".partial")
    with open(partial_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial_out, out_path)
    log(f"wrote {out_path}  ({len(entries)} temperature values)")
    log.close()


if __name__ == "__main__":
    main()
