@echo off
chcp 65001 > nul
REM konus_05'i CIFT TIKLAMAYLA baslatir. Kullanici, 17 Eylul:
REM "bunu nasil calistiracagim?" -- .py dosyasina cift tiklaninca
REM pencere acilip KAPANIYOR (python bitince konsol kapanir), o yuzden
REM burada hem calisma klasoru ayarlaniyor hem de sonda pause var.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python konus_05.py %*
echo.
pause
