@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

if defined PYTHON (
    set "PYTHON_BIN=%PYTHON%"
) else if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%PROJECT_ROOT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=1"
if not defined OPENBLAS_NUM_THREADS set "OPENBLAS_NUM_THREADS=1"
if not defined MKL_NUM_THREADS set "MKL_NUM_THREADS=1"
if not defined PYTHONUTF8 set "PYTHONUTF8=1"
if not defined PYTHONUNBUFFERED set "PYTHONUNBUFFERED=1"

"%PYTHON_BIN%" "%PROJECT_ROOT%\scripts\nth_sweep_pairwise_cuda.py" %*
set "RUN_EXIT_CODE=%ERRORLEVEL%"
exit /b %RUN_EXIT_CODE%
