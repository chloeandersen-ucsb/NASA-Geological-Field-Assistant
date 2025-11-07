# Video to 3D Reconstruction Pipeline (COLMAP)

This project provides a Python-based end-to-end pipeline that converts a video file into a 3D reconstruction using [COLMAP](https://colmap.github.io/).

## Features
- Extract frames from a video at a fixed temporal interval (default 0.5s; adaptive presets via `--mode`)
- Run COLMAP steps automatically:
  1. Feature extraction
  2. Exhaustive matching
  3. Sparse reconstruction (mapper)
  4. Image undistortion
  5. Dense stereo (PatchMatch)
  6. Stereo fusion -> fused point cloud
  7. (Optional) Poisson meshing
- Export fused point cloud (`fused.ply`) and optional Poisson mesh (`meshed-poisson.ply`).
- Optional OBJ export if `open3d` is installed.
- Works on Windows (PowerShell), Linux, macOS (with minor path adjustments).
- Automatic frame capping (default max 60) for predictable runtime.
- Reuse previously extracted frames with `--reuse_frames`.
- Fast vs Full mode presets (`--mode fast|full`).
- GPU index selection via `--gpu_index` or `COLMAP_GPU_INDEX` env.
- Per-step runtime timing logged.
- Per-step runtime timing logged and a summary printed at the end.

## Requirements
- Python 3.9+
- [COLMAP](https://colmap.github.io/) installed; add its binary to your PATH or pass with `--colmap_path`.
- (Recommended) [FFmpeg](https://ffmpeg.org/) for faster frame extraction.
- Python packages:
  ```
  pip install -r requirements.txt
  ```
- Optional: `open3d` for OBJ conversion.

## Quick Start (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py --video .\sample.mp4 --interval 0.5 --work_dir work --colmap_path "C:\Program Files\COLMAP\colmap.exe" --run_poisson --convert_obj
```

### Launch the Testing UI (Streamlit)
Install dependencies (includes Streamlit) then run:
```powershell
streamlit run ui_app.py
```
Features:
- Upload or reference a local video file.
- Configure mode, interval, max frames, matcher strategy, Poisson meshing.
- Real-time log streaming and downloadable output artifacts.
- Dry-run option to inspect commands without execution.

Outputs will still be written under `outputs/` and `work/` as with CLI usage.

### Desktop GUI (Tkinter)
If you prefer a desktop window instead of a browser UI:
```powershell
python desktop_ui.py
```
Features:
- File dialog for video & vocab tree.
- All pipeline flags (mode, matcher, Poisson, reuse frames, skip dense, GPU index, dense size, dry-run).
- Live scrolling log window.
- Stop button to terminate the run.
- Automatic summary and list of output files.
- Button to open the `outputs/` folder in Explorer.

Tkinter ships with standard Python on Windows; no extra dependency required.

## Command Line Arguments (key flags)
| Argument | Description |
|----------|-------------|
| `--video` | Path to input video file (required) |
| `--interval` | Base frame extraction interval in seconds (default 0.5; overridden by `--mode` unless customized) |
| `--max_frames` | Cap number of frames after extraction (default 60; 0 = no cap) |
| `--mode` | Preset: `fast` (interval≈0.7, no Poisson) or `full` (interval≈0.3, Poisson) |
| `--reuse_frames` | Skip extraction if frame JPGs already exist |
| `--work_dir` | Working directory for intermediate data (default `work`) |
| `--colmap_path` | Path to `colmap` executable if not on PATH |
| `--camera_model` | Camera model (e.g. `PINHOLE`, `SIMPLE_PINHOLE`, `OPENCV`, etc.) |
| `--single_camera` | Treat all frames as from a single camera |
| `--skip_dense` | Stop after sparse reconstruction and export sparse points3D.txt |
| `--run_poisson` | Force Poisson mesher (overrides mode fast) |
| `--convert_obj` | Convert Poisson mesh PLY to OBJ (requires open3d) |
| `--gpu_index` | Explicit GPU index for COLMAP (default from `COLMAP_GPU_INDEX` env or 0) |
| `--dense_size` | Max image size for undistortion/dense (default 2000) |
| `--dry_run` | Print planned commands without executing (no file changes) |
| `--matcher` | Matching strategy: `exhaustive` (default), `sequential`, or `vocab` |
| `--vocab_tree` | Path to vocabulary tree file when using `--matcher vocab` |
| `--overwrite` | Delete and recreate working directory |
| `--verbose` | (Reserved for future verbose logging) |

## Output Structure
```
work/
  images/            # Extracted frames
  database.db        # COLMAP feature DB
  sparse/0/          # Sparse model (cameras, images, points3D)
  dense/
    images/          # Undistorted images
    stereo/          # PatchMatch data
    fused.ply        # Dense fused point cloud
outputs/
  fused.ply
  meshed-poisson.ply (if --run_poisson)
  meshed.obj         (if --convert_obj and open3d installed)
  sparse_points3D.txt (if --skip_dense)
```

## Tips
- If reconstruction fails early, inspect logs under `work/logs/*.log`.
- Use `--mode fast` for quick iteration (fewer frames, no Poisson) or `--mode full` for higher detail.
- Increase `--max_frames` (or set to 0) for higher coverage on short clips.
- Ensure sufficient scene parallax; slow panning or orbiting helps.
- For long videos or Jetson-class devices, consider `--matcher sequential` (faster) or `--matcher vocab --vocab_tree path/to/tree.bin`.
- Set `COLMAP_GPU_INDEX` or use `--gpu_index` to choose a specific GPU.

### End-of-run summary
At the end of a run, a summary shows frames used, mode, interval, GPU index, matcher, and total time, for quick profiling.

## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| `database.db` already exists | Previous run left files | Use `--overwrite` |
| No `0` folder in `sparse` | Mapper produced no model | Check feature/matcher logs; ensure enough overlap |
| `fused.ply` missing | Stereo fusion failed | Check `stereo_fusion.log`; reduce image size or ensure enough views |
| OBJ not produced | `open3d` missing or mesh bad | Install `open3d` or inspect Poisson mesh |

## Roadmap / Possible Enhancements
- Add sequential / vocab-tree matcher options.
- Add advanced GPU utilization toggles & multi-GPU splitting.
- Integrate depth map filtering parameters via CLI.
- Provide automatic video frame sub-sampling by motion magnitude.
- Add outlier filtering & point cloud decimation post-fusion.

## License
MIT (add a LICENSE file as needed).
