@echo off
setlocal
cd /d "%~dp0"

where node >nul 2>nul
if errorlevel 1 (
  echo Node.js 18 or newer is required to run LiveRoom Monitor.
  pause
  exit /b 1
)

where git >nul 2>nul
if errorlevel 1 (
  echo Git was not found in PATH. Install Git for Windows first.
  pause
  exit /b 1
)

node "%~dp0monitor.js"
if errorlevel 1 (
  echo.
  echo LiveRoom Monitor exited with an error.
  pause
)
