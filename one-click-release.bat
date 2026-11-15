@echo off
setlocal
cd /d "%~dp0"

echo [1/2] Building frontend and Windows release package...
call npm run tauri:build
if errorlevel 1 (
  echo.
  echo Build failed. Review the error above.
  pause
  exit /b 1
)

echo.
echo [2/2] Release build completed.
echo Portable EXE: src-tauri\target\release\vanta.exe
echo Installers:   src-tauri\target\release\bundle\
pause
