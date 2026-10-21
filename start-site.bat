@echo off
setlocal
cd /d "%~dp0"

where node >nul 2>nul
if errorlevel 1 (
  echo Node.js 18 or newer is required.
  echo Install Node.js, then run this file again.
  pause
  exit /b 1
)

rem Reuse the running site instead of starting another Node server.
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:4173/' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 (
  echo The site is already running at http://127.0.0.1:4173/
  start "" http://127.0.0.1:4173/
  exit /b 0
)

echo Starting LiveRoom server...
start "LiveRoom server" cmd /k "cd /d ""%~dp0"" && node server.js"

echo Waiting for the server to become ready...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ready=$false; for($i=0;$i -lt 30;$i++){ try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:4173/' -TimeoutSec 2 | Out-Null; $ready=$true; break } catch { Start-Sleep -Seconds 1 } }; if($ready){ exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo LiveRoom did not start in time. Check the server window for errors.
  pause
  exit /b 1
)

start "" http://127.0.0.1:4173/
