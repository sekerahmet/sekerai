@echo off
rem Modelle konus (model_y, TinyStories).  Cift tiklayinca acilir, pencere kapanmaz.
rem Belirli bir kosu:  konus_y.bat tinystories_modelx2_layers2_turns4_coherence_s0
chcp 65001 >nul
cd /d "%~dp0"
python konus_y.py %*
echo.
pause
