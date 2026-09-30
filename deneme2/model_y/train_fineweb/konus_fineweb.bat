@echo off
rem Modelle konus (model_y, FineWeb-Edu).  Cift tiklayinca acilir, pencere kapanmaz.
rem Belirli bir kosu:  konus_fineweb.bat fineweb_modely_8x2_d1024_gpt2_s0
chcp 65001 >nul
cd /d "%~dp0"
python konus_fineweb.py %*
echo.
pause
