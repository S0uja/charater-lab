@echo off
setlocal
cd /d "%~dp0"

where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is not installed or is not on PATH.
  echo Install Node.js, then run this file again.
  pause
  exit /b 1
)

rem Reuse the running site instead of starting another Node server on the same port.
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:4173/' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 (
  echo The site is already running at http://127.0.0.1:4173/
  start "" http://127.0.0.1:4173/
  exit /b 0
)

echo Starting the local site at http://127.0.0.1:4173/
start "" http://127.0.0.1:4173/
node server.js
pause
