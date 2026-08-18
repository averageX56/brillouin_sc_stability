@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem CMD-safe launcher. The Python wrapper invokes cuda\build_cuda.bat, which
rem compiles directly with nvcc and does not require CMake on Windows.
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

set "NVCC=nvcc"
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" (
    set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
)
"%NVCC%" --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc.exe was not found.
    echo Install the NVIDIA CUDA Toolkit and reopen CMD.
    exit /b 2
)

"%PYTHON_BIN%" "%PROJECT_ROOT%\scripts\nth_sweep_cuda.py" %*
set "RUN_EXIT_CODE=%ERRORLEVEL%"
exit /b %RUN_EXIT_CODE%
