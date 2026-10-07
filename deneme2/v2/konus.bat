@echo off
rem Modelle konus (V2: Model Z + G ve transformer).  Cift tiklayinca acilir, pencere kapanmaz.
rem Belirli bir kosu:  konus.bat v2_mzl_g1_fw_d768
chcp 65001 >nul
cd /d "%~dp0"
python konus.py %*
echo.
pause
