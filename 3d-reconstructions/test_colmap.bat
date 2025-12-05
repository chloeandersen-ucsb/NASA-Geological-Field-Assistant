@echo off
REM Test COLMAP installation
SET COLMAP_PATH=%LOCALAPPDATA%\COLMAP\COLMAP-3.9.1-windows-cuda\bin\colmap.exe

echo Testing COLMAP at: %COLMAP_PATH%
echo.

if exist "%COLMAP_PATH%" (
    echo COLMAP found!
    echo Running version check...
    "%COLMAP_PATH%" -h
) else (
    echo ERROR: COLMAP not found at %COLMAP_PATH%
)
