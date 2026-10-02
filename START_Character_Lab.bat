@echo off
chcp 65001 >nul
cd /d C:\AI\Character_Lab_V6
echo ============================================================
echo Character Lab - PC + PHONE LAN MODE
echo ============================================================
echo.
echo Adding Windows Firewall rule for ports 7860-7880...
netsh advfirewall firewall delete rule name="Character Lab LAN 7860-7880" >nul 2>&1
netsh advfirewall firewall add rule name="Character Lab LAN 7860-7880" dir=in action=allow protocol=TCP localport=7860-7880 profile=private >nul 2>&1
if errorlevel 1 (
  echo [WARNING] Could not add Firewall rule automatically.
  echo Run this BAT as Administrator if phone access is blocked.
) else (
  echo [OK] Firewall rule added for Private networks.
)
echo.
python Character_Lab_V6.py
pause
