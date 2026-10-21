@echo off
setlocal
cd /d "%~dp0"

set "OLLAMA_URL=http://127.0.0.1:11434/api/tags"
set "COMFY_URL=http://127.0.0.1:8188/system_stats"

echo === LiveRoom AI services ===
echo.

rem Ollama normally runs in the background after installation. Start it only when needed.
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing -Uri '%OLLAMA_URL%' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
  where ollama >nul 2>nul
  if errorlevel 1 (
    echo Ollama is not running and the 'ollama' command was not found.
    echo Install Ollama from https://ollama.com/ and run this file again.
    pause
    exit /b 1
  )
  echo Starting Ollama...
  start "Ollama service" cmd /k "ollama serve"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ready=$false; for($i=0;$i -lt 30;$i++){ try { Invoke-WebRequest -UseBasicParsing -Uri '%OLLAMA_URL%' -TimeoutSec 2 | Out-Null; $ready=$true; break } catch { Start-Sleep -Seconds 1 } }; if($ready){ exit 0 } else { exit 1 }"
  if errorlevel 1 (
    echo Ollama did not become ready. Check the Ollama window.
    pause
    exit /b 1
  )
) else (
  echo Ollama is already running at http://127.0.0.1:11434
)

echo.
rem The existing launcher checks whether ComfyUI is already running before starting it.
if not exist "%~dp0start-comfyui.bat" (
  echo start-comfyui.bat was not found.
  pause
  exit /b 1
)
echo Starting or reusing ComfyUI...
start "LiveRoom ComfyUI" cmd /k "call ""%~dp0start-comfyui.bat"""

echo.
echo AI service startup requested.
echo Ollama:  http://127.0.0.1:11434
echo ComfyUI: http://127.0.0.1:8188
echo.
echo Run start-site.bat separately when you want to open LiveRoom.
pause
