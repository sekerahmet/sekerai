@echo off
chcp 65001 > nul
REM konus_06'yi CIFT TIKLAMAYLA baslatir.
REM !! DOSYA ADI KLASORDEN TURETILIYOR, elle yazilmiyor. Sebep:
REM model_06 klasoru model_05'ten kopyalandiginda dosya adlari
REM degisti ama bu bat'in ICI kopyalandigi gibi kaldi ve
REM 'konus_05.py bulunamadi' hatasi verdi (kullanici, 17 Eylul).
REM Klasor adi ne ise konus_<o>.py aranir -- bir daha kaymaz.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
for %%I in ("%~dp0.") do set KLASOR=%%~nxI
set BETIK=konus_%KLASOR:~6%.py
if not exist "%BETIK%" (
  echo HATA: %BETIK% bu klasorde YOK.
  dir /b konus_*.py
  pause
  exit /b 1
)
python "%BETIK%" %*
echo.
pause
