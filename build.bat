@echo off
rem Build without make (MinGW-w64 g++). Run from the repository root:
rem     build.bat
rem     build.bat no-omp     single-threaded, if OpenMP is missing
rem     build.bat probe      also build the drift_probe test helper
setlocal
if not exist build mkdir build
if not exist data  mkdir data

set CXXFLAGS=-O3 -std=c++17 -Wall -Wextra -fno-math-errno -Isrc
set OMPFLAGS=-fopenmp
if /i "%~1"=="no-omp" set OMPFLAGS=

echo g++ %CXXFLAGS% %OMPFLAGS% -o build\sde_solver.exe src\main.cpp
g++ %CXXFLAGS% %OMPFLAGS% -o build\sde_solver.exe src\main.cpp
if errorlevel 1 (
  echo.
  echo BUILD FAILED. If the error mentions -fopenmp, retry:  build.bat no-omp
  exit /b 1
)

echo g++ %CXXFLAGS% %OMPFLAGS% -DPAIRWISE_PHONONS=1 -o build\sde_solver_pairwise.exe src\main.cpp
g++ %CXXFLAGS% %OMPFLAGS% -DPAIRWISE_PHONONS=1 -o build\sde_solver_pairwise.exe src\main.cpp
if errorlevel 1 (
  echo.
  echo PAIRWISE BUILD FAILED. If the error mentions -fopenmp, retry: build.bat no-omp
  exit /b 1
)

if /i "%~1"=="probe" (
  echo g++ %CXXFLAGS% -o build\drift_probe.exe src\drift_probe.cpp
  g++ %CXXFLAGS% -o build\drift_probe.exe src\drift_probe.cpp
)

echo.
echo OK -^> build\sde_solver.exe
echo OK -^> build\sde_solver_pairwise.exe
echo Test it:  build\sde_solver.exe --help
endlocal
