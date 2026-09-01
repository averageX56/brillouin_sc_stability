@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "CUDA_SOURCE=%PROJECT_ROOT%\cuda\sde_solver_cuda.cu"
set "BUILD_DIR=%PROJECT_ROOT%\build_cuda"
set "CUDA_EXE=%BUILD_DIR%\sde_solver_pairwise_cuda.exe"

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

set "NVCC=nvcc"
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
"%NVCC%" --version >nul 2>&1
if errorlevel 1 (
    echo BUILD FAILED: nvcc.exe was not found.
    exit /b 1
)

set "NVCCFLAGS=-O3 -std=c++17 -lineinfo -Xcompiler=/EHsc -DPAIRWISE_PHONONS=1"
if defined CUDA_ARCH set "NVCCFLAGS=%NVCCFLAGS% -arch=%CUDA_ARCH%"

echo "%NVCC%" %NVCCFLAGS% -o "%CUDA_EXE%" "%CUDA_SOURCE%"
"%NVCC%" %NVCCFLAGS% -o "%CUDA_EXE%" "%CUDA_SOURCE%"
if errorlevel 1 exit /b 1

echo OK -^> "%CUDA_EXE%"
endlocal
exit /b 0
