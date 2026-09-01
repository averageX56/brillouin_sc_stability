@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
if defined PYTHON (set "PYTHON_BIN=%PYTHON%") else (set "PYTHON_BIN=python")
"%PYTHON_BIN%" "%SCRIPT_DIR%scripts\nth_sweep_pairwise_cpu.py" %*
set "RUN_EXIT_CODE=%ERRORLEVEL%"
exit /b %RUN_EXIT_CODE%
