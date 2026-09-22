@echo off
rem Modelle konus.  Cift tiklayinca acilir, pencere kapanmaz.
chcp 65001 >nul
cd /d "%~dp0"
python konus.py %*
echo.
pause
