@echo off
chcp 65001 > nul
REM Bu klasorun konus_<NN>.py betigini CIFT TIKLAMAYLA baslatir.
REM !! DOSYA ADI KLASORDEN TURETILIYOR, elle yazilmiyor. Sebep:
REM Bir kol klasoru bir digerinden kopyalandiginda dosya adlari
REM degisti ama bu bat'in ICI kopyalandigi gibi kaldi ve
REM 'betik bulunamadi' hatasi verdi (kullanici, 17 Eylul, IKI KEZ).
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
