@echo off
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Drafter off %*
set "result=%ERRORLEVEL%"
echo.
pause
exit /b %result%
