@echo off
REM model_14 ile konusma -- CIFT TIKLAYIN.
REM `.py` dosyasina cift tiklayinca Windows duzenleyiciyi aciyor;
REM calistirmak icin bu kisayol gerekiyor.
chcp 65001 > nul
title model_14 -- KONUS
cd /d "%~dp0deneme2\model_14"

where python > nul 2>&1
if errorlevel 1 (
  echo.
  echo   python bulunamadi. Python PATH'te degil.
  echo   Asagidaki satiri kendi yolunuzla degistirip tekrar deneyin:
  echo      set PY=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe
  echo.
  pause
  exit /b 1
)

python konus_14.py
echo.
echo   --- kapandi ---
pause > nul
