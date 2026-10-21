@echo off
setlocal
cd /d "%~dp0"

rem Stop only the Node process currently serving this local site.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$listener = Get-NetTCPConnection -State Listen -LocalPort 4173 -ErrorAction SilentlyContinue | Select-Object -First 1; if ($listener) { $proc = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $listener.OwningProcess); if ($proc.Name -eq 'node.exe' -and $proc.CommandLine -match 'server\.js') { Stop-Process -Id $listener.OwningProcess } else { Write-Host 'Port 4173 belongs to another application; leaving it untouched.'; exit 2 } }"
if errorlevel 1 (
  echo Could not safely restart the site. Close its existing server window, then run start-site.bat.
  pause
  exit /b 1
)

echo Restarting the local site at http://127.0.0.1:4173/
node server.js
pause
