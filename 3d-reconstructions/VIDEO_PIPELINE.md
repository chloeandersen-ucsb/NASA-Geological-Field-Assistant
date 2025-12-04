# Video to 3D Reconstruction Pipeline

Complete guide for converting video files to 3D point clouds and meshes.

## Quick Start

```powershell
# Simplest usage - auto frame filtering + backend decision tree
python scripts/generate_scene_from_video.py your_video.mp4

# View results
# Load outputs/your_video_3d/reconstruction in viewer
```

**Auto Backend Logic (NEW):**
- Single dominant object detected → TripoSR (single-image textured mesh)
- Else if filtered frames < 15 → DUSt3R (robust few-view reconstruction)
- Else → MASt3R (balanced multiview quality)
- COLMAP only when explicitly requested (`--backend colmap`)

**Expected runtime:** 3-8 minutes for ~15-30 filtered frames (device dependent)

---

## Overview

The video-to-3D pipeline consists of three steps:

1. **Frame Extraction** - Sample + quality filter frames
2. **Auto Backend Selection** - Decision tree chooses TripoSR / DUSt3R / MASt3R
3. **Reconstruction** - Single-image mesh or multiview point cloud

```
Video (MP4/MOV/AVI)
    ↓
Frame Extraction (extract_frames.py)
    ↓ Quality filtering (blur, exposure, similarity)
Selected Frames (JPG/PNG)
    ↓
Backend Auto Decision (TripoSR / DUSt3R / MASt3R or explicit COLMAP)
    ↓
Point Cloud (PLY)
```

---

## Frame Extraction Strategies

### Strategy 1: Time-Based Sampling

Extract frames at fixed time intervals:

```powershell
# Extract 2 frames per second
python scripts/extract_frames.py video.mp4 --fps 2

# Extract every 10th frame
python scripts/extract_frames.py video.mp4 --every-n-frames 10
```

**When to use:**
- Steady camera motion
- Uniform scene coverage
- Quick preview

### Strategy 2: Quality-Filtered Sampling

Extract frames meeting quality criteria:

```powershell
# Extract up to 30 frames, keep only sharp, well-exposed ones
python scripts/extract_frames.py video.mp4 \
    --max-frames 30 \
    --filter-quality \
    --blur-threshold 150 \
    --exposure-min 40 \
    --exposure-max 200
```

**Quality metrics:**
- **Blur score** (Laplacian variance): Rejects blurry frames
  - Typical values: 50-500
  - Threshold 100 = moderate sharpness
  - Threshold 150 = high sharpness
- **Exposure**: Rejects under/overexposed frames
  - Mean brightness: 0-255 (aim for 80-180)
  - Overexposure: % of pixels > 240 (keep < 10%)
  - Underexposure: % of pixels < 20 (keep < 10%)

**When to use:**
- Variable video quality
- Handheld/shaky footage
- Outdoor scenes with lighting changes

### Strategy 3: Similarity-Filtered Sampling

Remove redundant frames with similar content:

```powershell
# Remove consecutive frames that are >85% similar
python scripts/extract_frames.py video.mp4 \
    --filter-similarity \
    --similarity-threshold 0.85
```

**Similarity metric (SSIM):**
- 1.0 = identical frames
- 0.85 = very similar (default threshold)
- 0.70 = moderately similar
- < 0.50 = different

**When to use:**
- Static scenes with slow camera movement
- Videos with pauses/stops
- Maximizing frame diversity

### Strategy 4: Combined (Recommended)

Use all filters for best results:

```powershell
python scripts/extract_frames.py video.mp4 \
    --fps 2 \
    --max-frames 30 \
    --filter-quality \
    --filter-similarity \
    --blur-threshold 120 \
    --similarity-threshold 0.85
```

**Process flow:**
1. Sample at 2 FPS → ~60 candidate frames (30s video)
2. Filter blur/exposure → ~40 frames pass
3. Filter similarity → ~30 unique frames
4. Keep max 30 frames

---

## Quality Threshold Guidelines

### Blur Threshold (`--blur-threshold`)

| Value | Quality | Use Case |
|-------|---------|----------|
| 50-80 | Low | Accept most frames, fast extraction |
| 100-120 | Medium | **Recommended default** |
| 150-200 | High | Only very sharp frames |
| 200+ | Very High | Professional video, tripod-mounted |

**How to tune:**
- Too high → No frames extracted
- Too low → Blurry frames cause poor reconstruction

### Exposure Range (`--exposure-min`, `--exposure-max`)

| Scenario | Min | Max | Notes |
|----------|-----|-----|-------|
| Indoor | 30 | 220 | Default range |
| Outdoor sunny | 50 | 200 | Avoid sky overexposure |
| Low light | 20 | 180 | Accept darker frames |
| HDR video | 40 | 210 | More controlled exposure |

### Similarity Threshold (`--similarity-threshold`)

| Value | Effect | Frame Diversity |
|-------|--------|-----------------|
| 0.95 | Keep almost identical frames | Low diversity |
| 0.85 | **Recommended default** | Balanced |
| 0.75 | Aggressive deduplication | High diversity |
| 0.65 | Very aggressive | May skip important frames |

---

## Backend Selection

### Auto Mode (Default)

When `--backend auto` (default) the pipeline performs:

1. Samples the middle extracted frame and runs a contour-based single-object heuristic.
2. If a single dominant object is present → selects TripoSR (sharpest frame chosen by blur score).
3. Else, if `num_filtered_frames < 15` → selects DUSt3R (few-view robustness).
4. Else → selects MASt3R (balanced multiview quality).

Run output includes reasoning:
```
BACKEND SELECTION
Requested: auto
Selected:  mast3r
Reason:    24 frames (>=15) → default multiview quality
```

Override with explicit `--backend dust3r`, `--backend mast3r`, or `--backend colmap`.

### COLMAP (Explicit Only)

**Best for:**
- Many frames (20-100+)
- High-texture scenes
- Maximum accuracy
- Large-scale environments

**Runtime:** Slowest (5-20 minutes for 30 frames)

```powershell
python scripts/generate_scene_from_video.py video.mp4 \
    --backend colmap \
    --max-frames 50 \
    --device cuda
```

**Requirements:**
- Minimum 10-15 frames
- Good feature matches (textured surfaces)
- Sufficient overlap (60-70%)

### DUSt3R (Few-View Fallback / Explicit)

**Best for:**
- Few frames (5-15)
- Low-texture scenes (walls, floors)
- Quick preview
- Indoor environments

**Runtime:** Fast (2-5 minutes for 10 frames)

```powershell
python scripts/generate_scene_from_video.py video.mp4 \
    --backend dust3r \
    --max-frames 10
```

**Requirements:**
- Minimum 2 frames
- Works on low-texture scenes
- Less accurate at scale

### MASt3R (Auto Default for >=15 Frames)

**Best for:**
- General purpose
- Balanced speed/quality
- 10-30 frames
- Indoor + outdoor
- **Successor to DUSt3R with better multi-view performance**

**Runtime:** Medium (3-8 minutes for 20 frames)

```powershell
python scripts/generate_scene_from_video.py video.mp4 --backend mast3r --max-frames 30
```

**Requirements:**
- Minimum 2 frames
- Better than DUSt3R for multi-view
- Faster than COLMAP
- Metric depth (absolute scale)

**See [docs/dust3r-vs-mast3r.md](docs/dust3r-vs-mast3r.md) for detailed comparison**

---

## Complete Examples

### Example 1: Indoor Room Scan

**Scenario:** Walking around a room with phone camera

```powershell
python scripts/generate_scene_from_video.py room_scan.mp4 \
    --backend mast3r \
    --max-frames 25 \
    --fps 2 \
    --filter-quality \
    --filter-similarity \
    --blur-threshold 100 \
    --device cuda
```

**Expected output:** 25 frames, 3-4 million points, 3-5 min runtime

### Example 2: Outdoor Landscape

**Scenario:** Drone/handheld video of outdoor scene

```powershell
python scripts/generate_scene_from_video.py landscape.mp4 \
    --backend colmap \
    --max-frames 40 \
    --fps 3 \
    --filter-quality \
    --blur-threshold 150 \
    --exposure-min 50 \
    --exposure-max 200 \
    --device cuda
```

**Expected output:** 40 frames, 8-12 million points, 10-15 min runtime

### Example 3: Quick Preview

**Scenario:** Fast test of video before full processing

```powershell
python scripts/generate_scene_from_video.py test.mp4 \
    --backend dust3r \
    --max-frames 8 \
    --fps 1 \
    --no-similarity-filter
```

**Expected output:** 8 frames, 1-2 million points, 1-2 min runtime

### Example 4: High-Quality Reconstruction

**Scenario:** Professional-grade reconstruction

```powershell
python scripts/generate_scene_from_video.py scene.mp4 \
    --backend colmap \
    --max-frames 60 \
    --fps 4 \
    --filter-quality \
    --filter-similarity \
    --blur-threshold 180 \
    --similarity-threshold 0.90 \
    --voxel-size 0.02 \
    --keep-frames \
    --device cuda
```

**Expected output:** 60 frames, 15-25 million points, 20-30 min runtime

---

## Troubleshooting

### Problem: No frames extracted

**Symptoms:**
```
[WARNING] No frames extracted!
```

**Solutions:**
1. Lower blur threshold: `--blur-threshold 50`
2. Disable quality filter: `--no-quality-filter`
3. Increase max frames: `--max-frames 50`
4. Check video file is valid: `ffmpeg -i video.mp4`

### Problem: Too few frames for reconstruction

**Symptoms:**
```
[ERROR] colmap requires at least 10 frames
        Only extracted 5 frames
```

**Solutions:**
1. Increase `--max-frames`
2. Lower `--blur-threshold`
3. Increase `--fps` or decrease `--every-n-frames`
4. Use DUSt3R/MASt3R (requires only 2 frames)

### Problem: Reconstruction fails (COLMAP)

**Symptoms:**
```
[ERROR] COLMAP reconstruction failed
```

**Solutions:**
1. Check frame overlap (need 60-70%)
2. Ensure textured surfaces (not blank walls)
3. Try DUSt3R/MASt3R instead
4. Verify camera motion is not too fast
5. Check lighting consistency

### Problem: Poor quality point cloud

**Symptoms:** Sparse, noisy, or incomplete point cloud

**Solutions:**
1. **Increase frames:** `--max-frames 40`
2. **Stricter filtering:** `--blur-threshold 150`
3. **Better backend:** Switch to COLMAP
4. **Camera technique:** Slower, steadier video capture
5. **Lighting:** Consistent, bright lighting

### Problem: Very slow reconstruction

**Symptoms:** Taking > 30 minutes

**Solutions:**
1. Enable CUDA: `--device cuda`
2. Reduce frames: `--max-frames 15`
3. Use faster backend: `--backend dust3r`
4. Increase voxel size: `--voxel-size 0.1`

---

## Performance Benchmarks

| Backend | Frames | Device | Time | Point Count |
|---------|--------|--------|------|-------------|
| DUSt3R | 10 | CPU | 2 min | 1.5M |
| DUSt3R | 10 | GPU | 40s | 1.5M |
| MASt3R | 20 | CPU | 6 min | 3.5M |
| MASt3R | 20 | GPU | 2 min | 3.5M |
| COLMAP | 30 | CPU | 18 min | 8M |
| COLMAP | 30 | GPU | 8 min | 8M |
| COLMAP | 50 | GPU | 15 min | 15M |

*Tested on: Intel i7-12700K, RTX 3060, 1080p video*

---

## Advanced: Configuration Files

Create reusable presets with YAML:

```yaml
# config_room_scan.yaml
video:
  max_frames: 25
  fps: 2
  filter_quality: true
  filter_similarity: true
  blur_threshold: 100.0
  similarity_threshold: 0.85

reconstruction:
  backend: mast3r
  device: cuda
  voxel_size: 0.05

keep_frames: false
```

**Usage:**
```powershell
python scripts/generate_scene_from_video.py video.mp4 --config config_room_scan.yaml
```

**Generate sample configs:**
```powershell
python config.py
# Creates: config_fast.yaml, config_balanced.yaml, config_high_quality.yaml
```

---

## Tips for Best Results

### Video Capture Guidelines

1. **Camera Motion:**
   - Move slowly and steadily
   - Maintain 60-70% overlap between frames
   - Circle around object/scene
   - Avoid sudden movements

2. **Lighting:**
   - Consistent, diffuse lighting
   - Avoid direct sunlight (harsh shadows)
   - No flickering lights
   - Avoid reflective surfaces

3. **Camera Settings:**
   - Disable auto-exposure if possible
   - Fixed focus (not continuous AF)
   - 1080p or higher resolution
   - 30-60 FPS (smoother for extraction)

4. **Scene Coverage:**
   - Capture all sides of object
   - Multiple heights/angles
   - Overlap viewpoints
   - Avoid large gaps

### Frame Extraction Best Practices

1. **Start with defaults:**
   ```powershell
   python scripts/generate_scene_from_video.py video.mp4
   ```

2. **Check extracted frames:**
   - Verify frame quality in output directory
   - Ensure good coverage of scene
   - Adjust thresholds if needed

3. **Iterate parameters:**
   - If too few frames → lower thresholds
   - If too many similar frames → increase similarity threshold
   - If blurry frames → increase blur threshold

### Reconstruction Optimization

1. **Backend selection:**
    - Use auto mode first (object / frame-count aware)
    - Explicit DUSt3R for rapid few-view tests
    - Explicit COLMAP for traditional photogrammetry

2. **Frame count:**
    - Minimum: 10-15 for COLMAP (explicit), 2 for DUSt3R/MASt3R
    - Auto fallback to DUSt3R when <15 frames extracted
    - Optimal (MASt3R): 15-30 frames
    - Large environments (explicit COLMAP): 40-60 frames

3. **Performance:**
   - Use `--device cuda` if available
   - Adjust `--voxel-size` to control density
   - Use `--keep-frames` to debug extraction

---

## Integration with Existing Pipelines

The video pipeline integrates seamlessly with existing scripts:

```powershell
# Step 1: Extract frames manually
python scripts/extract_frames.py video.mp4 --output my_frames/

# Step 2: Run any reconstruction pipeline
python scripts/generate_scene_multiview_pycolmap.py my_frames/
python scripts/generate_scene_multiview_dust3r.py my_frames/ --model mast3r

# Step 3: View results
python scripts/viewer_3d.py outputs/my_frames_multiview/fused_pointcloud.ply
```

---

## Next Steps

- Run default auto mode and inspect backend reasoning
- Adjust `--max-frames` (default 30) to control coverage
- Use explicit `--backend colmap` only when you need classic SfM
- Tune quality thresholds for your capture conditions
- Create config presets per environment (indoor, outdoor, object)

---

## Related Documentation

- [Main README](README.md) - Overall project documentation
- [Multi-View README](README_MULTIVIEW.md) - COLMAP setup and usage
- [COLMAP Setup](COLMAP_SETUP.md) - COLMAP installation
- [Jetson Setup](docs/jetson-orin-setup.md) - Edge deployment
