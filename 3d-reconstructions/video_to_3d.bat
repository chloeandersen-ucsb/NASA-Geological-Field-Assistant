@echo off
REM Quick start batch file for video-to-3D reconstruction
REM Usage: video_to_3d.bat your_video.mp4

echo ============================================================
echo Video to 3D Reconstruction
echo ============================================================
echo.

if "%~1"=="" (
    echo ERROR: No video file specified
    echo.
    echo Usage: video_to_3d.bat your_video.mp4
    echo.
    echo Options:
    echo   video_to_3d.bat video.mp4                  - Quick preview (10 frames, DUSt3R)
    echo   video_to_3d.bat video.mp4 balanced         - Balanced quality (20 frames, MASt3R)
    echo   video_to_3d.bat video.mp4 high_quality     - High quality (50 frames, COLMAP)
    echo.
    pause
    exit /b 1
)

set VIDEO_FILE=%~1
set PRESET=%~2

if "%PRESET%"=="" (
    set PRESET=balanced
)

echo Video: %VIDEO_FILE%
echo Preset: %PRESET%
echo.

REM Check if video exists
if not exist "%VIDEO_FILE%" (
    echo ERROR: Video file not found: %VIDEO_FILE%
    pause
    exit /b 1
)

REM Set parameters based on preset
if "%PRESET%"=="fast" (
    echo Running FAST preset - Quick preview
    python scripts/generate_scene_from_video.py "%VIDEO_FILE%" ^
        --backend dust3r ^
        --max-frames 10 ^
        --blur-threshold 80
) else if "%PRESET%"=="high_quality" (
    echo Running HIGH QUALITY preset - Best results
    python scripts/generate_scene_from_video.py "%VIDEO_FILE%" ^
        --backend colmap ^
        --max-frames 50 ^
        --blur-threshold 150 ^
        --similarity-threshold 0.90 ^
        --keep-frames ^
        --device cuda
) else (
    echo Running BALANCED preset - Recommended
    python scripts/generate_scene_from_video.py "%VIDEO_FILE%" ^
        --backend mast3r ^
        --max-frames 20 ^
        --filter-quality ^
        --filter-similarity ^
        --device cuda
)

echo.
echo ============================================================
echo Reconstruction complete!
echo ============================================================
echo.
echo Check outputs folder for results.
echo Load the point cloud in viewer: http://localhost:5000/viewer/index.html
echo.
pause
