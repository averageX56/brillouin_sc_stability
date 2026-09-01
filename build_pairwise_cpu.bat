@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

set "CXXFLAGS=-O3 -std=c++17 -Wall -Wextra -fno-math-errno -DPAIRWISE_PHONONS=1 -Isrc"
set "OMPFLAGS=-fopenmp"
if /i "%~1"=="no-omp" set "OMPFLAGS="

pushd "%SCRIPT_DIR%"
echo g++ %CXXFLAGS% %OMPFLAGS% -o build\sde_solver_pairwise.exe src\main.cpp
g++ %CXXFLAGS% %OMPFLAGS% -o build\sde_solver_pairwise.exe src\main.cpp
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
popd

if not "%BUILD_EXIT_CODE%"=="0" exit /b %BUILD_EXIT_CODE%
echo OK -^> "%BUILD_DIR%\sde_solver_pairwise.exe"
endlocal
exit /b 0
