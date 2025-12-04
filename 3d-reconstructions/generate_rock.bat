@echo off
.\.venv\Scripts\python external\TripoSR\run.py C:\Users\Bruce\3d-generation\outputs\rocktest\0\rock.jpg --output-dir .\outputs\rocktest_3d --bake-texture --texture-resolution 2048 --model-save-format obj --chunk-size 8192 --mc-resolution 256
echo.
echo Done! View the result at: outputs\rocktest_3d\0\mesh.obj
echo Open the viewer and use the file picker to load it.
pause
