@echo off
echo ========================================
echo Running DUSt3R on Pipes Dataset
echo This will take approximately 3-4 minutes
echo ========================================
echo.

C:\Users\Bruce\3d-generation\.venv\Scripts\python.exe scripts\generate_scene_multiview_dust3r.py "C:\Users\Bruce\Downloads\pipes_dslr_undistorted\pipes\images\dslr_images_undistorted\resized" --output outputs/pipes_dust3r --model dust3r --device cpu --batch-size 1 --image-size 224

echo.
echo ========================================
echo DUSt3R processing complete!
echo Output: outputs\pipes_dust3r\
echo ========================================
pause
