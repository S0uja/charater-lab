@echo off
setlocal
cd /d "%~dp0ComfyUI_windows_portable\ComfyUI_windows_portable"
if not exist "python_embeded\python.exe" (
  echo ComfyUI portable is not fully extracted yet.
  pause
  exit /b 1
)

rem Reuse the running server instead of opening a second copy against the same database.
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8188/system_stats' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 (
  echo ComfyUI is already running at http://127.0.0.1:8188
  start "" http://127.0.0.1:8188/
  exit /b 0
)

echo Starting local ComfyUI at http://127.0.0.1:8188
start "" http://127.0.0.1:8188/
python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --lowvram --listen 127.0.0.1 --port 8188
pause
