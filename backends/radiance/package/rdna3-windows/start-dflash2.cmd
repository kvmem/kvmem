@echo off
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Drafter dflash2 %*
set "result=%ERRORLEVEL%"
echo.
pause
exit /b %result%
