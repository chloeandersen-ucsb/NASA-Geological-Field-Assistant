# DUSt3R vs MASt3R: Backend Comparison Guide

## Quick Answer

**Use MASt3R by default** - it's the upgraded version of DUSt3R with better multi-view performance.

## Technical Differences

### DUSt3R (Geometric 3D Vision Made Easy)
- **Release**: December 2023
- **Purpose**: Joint depth and pose estimation from image pairs
- **Architecture**: Transformer-based stereo matching
- **Strengths**:
  - Works with just 2 images
  - No feature matching required
  - Handles low-texture scenes
  - Simpler, faster for small sets

**How it works:**
1. Processes image pairs through transformer
2. Predicts dense depth maps + relative poses
3. Global alignment to register all views
4. Fuses depth into point cloud

### MASt3R (Multi-view Stereo 3D Reconstruction)
- **Release**: 2024 (successor to DUSt3R)
- **Purpose**: Same as DUSt3R but optimized for multi-view
- **Architecture**: DUSt3R + local feature head + metric depth
- **Improvements over DUSt3R**:
  - ✓ Better multi-view consistency
  - ✓ Metric depth (absolute scale)
  - ✓ Local feature matching for refinement
  - ✓ More robust global alignment
  - ✓ Higher quality point clouds

**How it works:**
1. Same transformer backbone as DUSt3R
2. **PLUS** local feature extraction for better matches
3. **PLUS** metric depth prediction (not just relative)
4. Improved global alignment algorithm
5. Higher quality fusion

## Performance Comparison

| Metric | DUSt3R | MASt3R | COLMAP |
|--------|--------|--------|--------|
| **Min frames** | 2 | 2 | 10-15 |
| **Optimal frames** | 5-15 | 10-30 | 20-60 |
| **Speed (10 frames, GPU)** | 30-40s | 45-60s | 2-3 min |
| **Speed (30 frames, GPU)** | 2-3 min | 3-4 min | 8-10 min |
| **Point density** | Medium | High | Very High |
| **Accuracy** | Good | Better | Best |
| **Low-texture scenes** | Excellent | Excellent | Poor |
| **Large-scale scenes** | Limited | Good | Excellent |

## When to Use Each

### Use DUSt3R when:
- ✓ Very quick preview needed (5-10 frames)
- ✓ Testing pipeline with minimal frames
- ✓ Extremely low-texture scenes (blank walls, snow)
- ✓ Limited GPU memory
- ✓ You only have 2-5 images

**Example:**
```powershell
python scripts/generate_scene_from_video.py video.mp4 --backend dust3r --max-frames 8
```

### Use MASt3R when: (RECOMMENDED DEFAULT)
- ✓ General purpose reconstruction
- ✓ 10-30 frames available
- ✓ Balance of speed and quality needed
- ✓ Indoor and outdoor scenes
- ✓ Want metric scale (absolute depth)

**Example:**
```powershell
python scripts/generate_scene_from_video.py video.mp4 --backend mast3r
# Uses smart max_frames based on video length
```

### Use COLMAP when:
- ✓ Maximum quality needed
- ✓ Many frames available (20-60+)
- ✓ High-texture scenes
- ✓ Large environments
- ✓ Production/final results
- ✓ Have GPU and time budget

**Example:**
```powershell
python scripts/generate_scene_from_video.py video.mp4 --backend colmap --max-frames 40 --device cuda
```

## Quality Tradeoffs

### Point Cloud Density
```
COLMAP (50 frames):     █████████████████████  15-25M points
MASt3R (20 frames):     ████████████           3-5M points
DUSt3R (10 frames):     ██████                 1-2M points
```

### Reconstruction Time (30s video, GPU)
```
DUSt3R (8 frames):      ████              ~1 min
MASt3R (20 frames):     ██████████        ~3 min
COLMAP (40 frames):     ███████████████   ~10 min
```

### Scene Coverage
```
COLMAP:     ████████████  Best for large spaces
MASt3R:     ████████      Good for rooms/objects
DUSt3R:     █████         Best for small objects
```

## Model Weights

Both use HuggingFace model hub (auto-download on first use):

**DUSt3R:**
```python
model_path = 'naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt'
```

**MASt3R:**
```python
model_path = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
```

**Size:** ~1.5GB each (one-time download)

## Smart Max Frames (New Feature)

The pipeline now auto-calculates optimal frame count based on video duration:

### DUSt3R Auto Frames
- < 10s video → 8 frames
- 10-30s → 12 frames
- > 30s → 15 frames

### MASt3R Auto Frames (Default)
- < 10s video → 12 frames
- 10-30s → 20 frames
- 30-60s → 30 frames
- > 60s → 40 frames

### COLMAP Auto Frames
- < 10s video → 15 frames
- 10-30s → 25 frames
- 30-60s → 40 frames
- > 60s → 60 frames

**Override anytime:**
```powershell
python scripts/generate_scene_from_video.py video.mp4 --max-frames 25
```

## Real-World Examples

### Indoor Room Scan (20s video)
```powershell
# Fast preview
python scripts/generate_scene_from_video.py room.mp4 --backend dust3r
# Auto: 12 frames, ~1 min

# Balanced (recommended)
python scripts/generate_scene_from_video.py room.mp4 --backend mast3r
# Auto: 20 frames, ~3 min

# High quality
python scripts/generate_scene_from_video.py room.mp4 --backend colmap --device cuda
# Auto: 25 frames, ~8 min
```

### Object 360° Capture (10s video)
```powershell
# MASt3R is perfect
python scripts/generate_scene_from_video.py object.mp4
# Auto: 12 frames, ~2 min
```

### Large Outdoor Scene (60s video)
```powershell
# COLMAP recommended
python scripts/generate_scene_from_video.py landscape.mp4 --backend colmap --device cuda
# Auto: 60 frames, ~15 min
```

## Common Issues

### DUSt3R/MASt3R: "Model download failed"
- Ensure internet connection
- First run downloads ~1.5GB
- Models cached in `~/.cache/huggingface/`

### MASt3R: "Out of memory"
- Reduce `--max-frames`
- Use `--device cpu` (slower)
- Or switch to DUSt3R (lighter)

### Point cloud too sparse
- Use COLMAP instead
- Increase `--max-frames`
- Lower `--voxel-size` (e.g., 0.02)

### Reconstruction too slow
- Use DUSt3R instead of MASt3R
- Reduce `--max-frames`
- Enable GPU: `--device cuda`

## Summary Decision Tree

```
Need results in < 2 min?
  └─> DUSt3R (8-12 frames)

Have 10-30 frames, want good quality?
  └─> MASt3R (default, recommended)

Have 30+ frames, want best quality?
  └─> COLMAP (if textured scene)

Low-texture scene (walls, snow)?
  └─> MASt3R or DUSt3R (NOT COLMAP)

Large outdoor environment?
  └─> COLMAP (if time permits)
      └─> MASt3R (if need speed)
```

## References

- DUSt3R Paper: https://arxiv.org/abs/2312.14132
- MASt3R: https://github.com/naver/mast3r
- COLMAP: https://colmap.github.io/

---

**TL;DR:** Use **MASt3R** for most cases. It's the sweet spot between DUSt3R's speed and COLMAP's quality.
