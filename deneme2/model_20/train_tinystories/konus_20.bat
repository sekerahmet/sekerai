@echo off
rem Modelle konus (model_20, TinyStories).  Cift tiklayinca acilir, pencere kapanmaz.
rem Belirli bir kosu:  konus_20.bat tinystories_modelx2_layers2_turns4_coherence_s0
chcp 65001 >nul
cd /d "%~dp0"
python konus_20.py %*
echo.
pause
