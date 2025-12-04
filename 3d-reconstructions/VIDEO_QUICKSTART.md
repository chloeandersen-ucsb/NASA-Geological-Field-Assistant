# Video-to-3D Quick Reference

## Installation

```powershell
# Install dependencies
pip install opencv-python numpy pillow torch scikit-image PyYAML

# Test installation
python test_video_pipeline.py

# Generate config presets
python config.py
```

## Quick Start Commands

### Simple (Auto Mode)
```powershell
python scripts/generate_scene_from_video.py video.mp4
```
- Extracts 20 best frames
- Uses MASt3R reconstruction
- ~3-8 minutes runtime

### Presets

**Fast Preview:**
```powershell
video_to_3d.bat video.mp4 fast
# OR
python scripts/generate_scene_from_video.py video.mp4 --backend dust3r --max-frames 10
```

**Balanced (Default):**
```powershell
video_to_3d.bat video.mp4 balanced
# OR
python scripts/generate_scene_from_video.py video.mp4 --backend mast3r --max-frames 20
```

**High Quality:**
```powershell
video_to_3d.bat video.mp4 high_quality
# OR
python scripts/generate_scene_from_video.py video.mp4 --backend colmap --max-frames 50 --device cuda
```

## Frame Extraction Only

```powershell
# Extract 30 frames at 2 FPS with quality filtering
python scripts/extract_frames.py video.mp4 --fps 2 --max-frames 30 --filter-quality

# Then run reconstruction separately
python scripts/generate_scene_multiview_pycolmap.py outputs/video_frames/
```

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-frames` | 20 | Maximum frames to extract |
| `--fps` | auto | Frames per second to sample |
| `--backend` | mast3r | colmap / dust3r / mast3r |
| `--device` | cpu | cpu / cuda |
| `--blur-threshold` | 100 | Min sharpness (50-200) |
| `--similarity-threshold` | 0.85 | Max frame similarity (0-1) |
| `--filter-quality` | auto | Enable blur/exposure filter |
| `--filter-similarity` | auto | Remove similar frames |
| `--keep-frames` | false | Keep extracted frames |

## Troubleshooting

**No frames extracted:**
```powershell
# Lower thresholds
python scripts/extract_frames.py video.mp4 --blur-threshold 50 --no-quality-filter
```

**Too few frames for COLMAP:**
```powershell
# Increase frames or use DUSt3R/MASt3R
python scripts/generate_scene_from_video.py video.mp4 --max-frames 40
# OR
python scripts/generate_scene_from_video.py video.mp4 --backend mast3r
```

**Very slow:**
```powershell
# Use GPU and reduce frames
python scripts/generate_scene_from_video.py video.mp4 --device cuda --max-frames 15
```

## File Locations

**Output:**
- Point cloud: `outputs/<video>_3d/reconstruction/fused_pointcloud.ply`
- Frames: `outputs/<video>_3d/frames/` (if `--keep-frames`)
- Metadata: `outputs/<video>_3d/frames/extraction_metadata.txt`

**Config:**
- Presets: `config_fast.yaml`, `config_balanced.yaml`, `config_high_quality.yaml`
- Custom: Create your own YAML with same format

## Performance Guide

| Setup | Frames | Backend | Expected Time |
|-------|--------|---------|---------------|
| Laptop CPU | 10 | DUSt3R | 2-3 min |
| Laptop CPU | 20 | MASt3R | 6-8 min |
| Laptop CPU | 30 | COLMAP | 15-20 min |
| Desktop GPU | 10 | DUSt3R | 40s |
| Desktop GPU | 20 | MASt3R | 2 min |
| Desktop GPU | 50 | COLMAP | 8-10 min |

## Next Steps

1. Capture video (slow, steady motion, good overlap)
2. Run pipeline with default settings
3. Check output quality in viewer
4. Adjust parameters if needed (see VIDEO_PIPELINE.md)
5. Iterate with different thresholds

**Full documentation:** `VIDEO_PIPELINE.md`
