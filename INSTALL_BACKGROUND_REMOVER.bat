@echo off
chcp 65001 >nul
cd /d C:\AI\Character_Lab_V6
python -m pip install --upgrade "rembg[cpu]>=2.0.60"
if errorlevel 1 (
  echo.
  echo INSTALL FAILED.
  pause
  exit /b 1
)
echo.
echo Background remover installed.
echo Run Character_Lab_V6.py again.
pause
