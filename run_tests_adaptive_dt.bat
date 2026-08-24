@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem Adaptive-dt sweep for Brillouin paper parameter checks.
rem
rem Characteristic fastest rate:
rem     rate_max = max(gamma_opt, Gamma, abs(g))
rem
rem Time step:
rem     dt = min(DT_MAX, SAFETY / rate_max)
rem
rem For the usual amplitude equations containing -gamma*a/2,
rem SAFETY=0.20 corresponds to ~10 steps per fastest amplitude
rem relaxation time 2/gamma.
rem ============================================================

set "SAFETY=0.20"
set "DT_MAX=5.0e-9"
set "MAX_RETRIES=3"

rem Optional: uncomment to force a common upper cap closer to the
rem old setup.  The adaptive rule will still reduce dt where needed.
rem set "DT_MAX=1.0e-9"

call :run_case "Kharel 2019 - measured" ^
    "kharel_measured" ^
    4.586725274e8 ^
    5.403539364e5 ^
    1.130973355e2
if errorlevel 1 exit /b %errorlevel%

call :run_case "Kharel 2019 - theory max" ^
    "kharel_theory" ^
    4.586725274e8 ^
    5.403539364e5 ^
    1.507964474e2
if errorlevel 1 exit /b %errorlevel%

call :run_case "Doeleman 2023 - mode 1" ^
    "doeleman" ^
    1.507964474e7 ^
    3.424335992e5 ^
    5.271592473e1
if errorlevel 1 exit /b %errorlevel%

call :run_case "Kim 2017" ^
    "kim" ^
    3.204424507e7 ^
    7.853981634e4 ^
    8.796459430e1
if errorlevel 1 exit /b %errorlevel%

call :run_case "He 2020" ^
    "he" ^
    1.784424627e9 ^
    5.340707511e5 ^
    8.356636459e5
if errorlevel 1 exit /b %errorlevel%

call :run_case "Otterstrom 2018" ^
    "otterstrom" ^
    5.215043805e8 ^
    8.230972752e7 ^
    6.974335691e4
if errorlevel 1 exit /b %errorlevel%

echo.
echo ========================================
echo ALL RUNS FINISHED
echo ========================================
exit /b 0


:run_case
set "LABEL=%~1"
set "OUTNAME=%~2"
set "GAMMA_OPT=%~3"
set "GAMMA_PH=%~4"
set "G_COUPLING=%~5"

set "DT="

rem cmd.exe has no floating-point arithmetic, so use the same Python
rem interpreter as the simulation to calculate dt robustly.
for /f "usebackq delims=" %%D in (`python -c "go=float('%GAMMA_OPT%'); gp=float('%GAMMA_PH%'); g=abs(float('%G_COUPLING%')); s=float('%SAFETY%'); dmax=float('%DT_MAX%'); r=max(go,gp,g); print(format(min(dmax,s/r),'.12g'))"`) do set "DT=%%D"

if not defined DT (
    echo ERROR: failed to calculate dt for %LABEL%
    exit /b 1
)

if "%OUTNAME%"=="kim" set "DT=5e-10"

echo.
echo ========================================
echo %LABEL%
echo gamma_opt = %GAMMA_OPT% 1/s
echo Gamma     = %GAMMA_PH% 1/s
echo g         = %G_COUPLING% 1/s
echo dt        = !DT! s
echo ========================================

set /a ATTEMPT=0

:retry_case
set /a ATTEMPT+=1
echo attempt   = !ATTEMPT! / %MAX_RETRIES%

python scripts\nth_sweep.py ^
    --gamma-opt %GAMMA_OPT% ^
    --Gamma %GAMMA_PH% ^
    --g %G_COUPLING% ^
    --dt !DT!
set "RC=!errorlevel!"

if not "!RC!"=="0" (
    if !ATTEMPT! GEQ %MAX_RETRIES% (
        echo ERROR: simulation failed for %LABEL% after %MAX_RETRIES% attempts
        exit /b !RC!
    )

    for /f "usebackq delims=" %%D in (`python -c "print(format(float('!DT!')/2.0,'.12g'))"`) do set "DT=%%D"
    echo numerical failure - retrying with dt=!DT! s
    goto :retry_case
)

move /Y data\nth_sweep.json data\nth_sweep_%OUTNAME%.json >nul
if errorlevel 1 (
    echo ERROR: failed to move output for %LABEL%
    exit /b !errorlevel!
)

exit /b 0
