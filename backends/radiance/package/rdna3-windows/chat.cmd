@echo off
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0chat.ps1" %*
set "result=%ERRORLEVEL%"
echo.
pause
exit /b %result%
