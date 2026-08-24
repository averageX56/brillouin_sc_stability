@echo off
setlocal

echo ========================================
echo Kharel 2019 - measured
echo ========================================

python scripts\nth_sweep.py ^
    --gamma-opt 4.586725274e8 ^
    --Gamma 5.403539364e5 ^
    --g 1.130973355e2

if errorlevel 1 exit /b %errorlevel%

move /Y data\nth_sweep.json data\nth_sweep_kharel_measured.json


echo.
echo ========================================
echo Kharel 2019 - theory max
echo ========================================

python scripts\nth_sweep.py ^
    --gamma-opt 4.586725274e8 ^
    --Gamma 5.403539364e5 ^
    --g 1.507964474e2

if errorlevel 1 exit /b %errorlevel%

move /Y data\nth_sweep.json data\nth_sweep_kharel_theory.json


echo.
echo ========================================
echo Doeleman 2023 - mode 1
echo ========================================

python scripts\nth_sweep.py ^
    --gamma-opt 1.507964474e7 ^
    --Gamma 3.424335992e5 ^
    --g 5.271592473e1

if errorlevel 1 exit /b %errorlevel%

move /Y data\nth_sweep.json data\nth_sweep_doeleman.json


echo.
echo ========================================
echo Diamandi 2025
echo ========================================

python scripts\nth_sweep.py ^
    --gamma-opt 2.513274123e7 ^
    --Gamma 3.769911184e3 ^
    --g 3.820176667e1

if errorlevel 1 exit /b %errorlevel%

move /Y data\nth_sweep.json data\nth_sweep_diamandi.json


echo.
echo ========================================
echo Kim 2017
echo ========================================

python scripts\nth_sweep.py ^
    --gamma-opt 3.204424507e7 ^
    --Gamma 7.853981634e4 ^
    --g 8.796459430e1

if errorlevel 1 exit /b %errorlevel%

move /Y data\nth_sweep.json data\nth_sweep_kim.json


echo.
echo ========================================
echo He 2020
echo ========================================

python scripts\nth_sweep.py ^
    --gamma-opt 1.784424627e9 ^
    --Gamma 5.340707511e5 ^
    --g 8.356636459e5

if errorlevel 1 exit /b %errorlevel%

move /Y data\nth_sweep.json data\nth_sweep_he.json


echo.
echo ========================================
echo Otterstrom 2018
echo ========================================

python scripts\nth_sweep.py ^
    --gamma-opt 5.215043805e8 ^
    --Gamma 8.230972752e7 ^
    --g 6.974335691e4

if errorlevel 1 exit /b %errorlevel%

move /Y data\nth_sweep.json data\nth_sweep_otterstrom.json


echo.
echo ========================================
echo ALL RUNS FINISHED
echo ========================================

endlocal