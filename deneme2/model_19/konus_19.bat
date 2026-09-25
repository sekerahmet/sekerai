@echo off
rem Modelle konus (model_19).  Cift tiklayinca acilir, pencere kapanmaz.
rem Belirli bir paket:  konus_19.bat PR_TS_ARCH_BASE_S0/t20000
chcp 65001 >nul
cd /d "%~dp0"
python konus_19.py %*
echo.
pause
