# Scene Reconstruction Workflows

This guide covers landscape and scene reconstruction using ZoeDepth depth estimation and COLMAP multi-view geometry.

## Overview

**Use Cases:**
- **Single Image → 3D Scene**: Fast field reconstruction (2-4 seconds) for landscapes, terrain, environments
- **Multi-Image → High-Quality Scene**: Mission-grade reconstruction (10-20 seconds) with accurate geometry and scale
- **Object Scans**: Use TripoSR (see main README)

---

## Single-Image Scene Reconstruction

Generate a 3D point cloud from a single photo using ZoeDepth metric depth estimation.

### Quick Start

```bash
# Activate environment
.\.venv\Scripts\python

# Run on single image
python scripts/generate_scene_zoedepth.py landscape.jpg

# With GPU acceleration
python scripts/generate_scene_zoedepth.py landscape.jpg --device cuda

# Custom camera FOV and depth range
python scripts/generate_scene_zoedepth.py landscape.jpg --camera-fov 70 --max-depth 15.0
```

### Output

Creates `outputs/<image_name>_scene/`:
- `pointcloud.ply` - 3D colored point cloud
- `depth.png` - Depth map visualization
- `input.png` - Copy of input image

### Performance

| Device | Resolution | Time |
|--------|-----------|------|
| CPU (Intel i7) | 1080p | ~3-5s |
| CUDA (RTX 3060) | 1080p | ~0.5-1s |
| Jetson Orin NX | 1080p | ~2-4s (TensorRT) |

### Parameters

- `--device` - `cpu` or `cuda` (default: `cpu`)
- `--camera-fov` - Field of view in degrees (default: `60`)
- `--max-depth` - Maximum depth in meters (default: `10.0`)
- `--output` - Custom output directory

---

## Multi-View Scene Reconstruction

Reconstruct high-quality 3D scenes from multiple images using COLMAP + ZoeDepth depth fusion.

### Prerequisites

Install COLMAP:

**Windows:**
```powershell
# Download from https://github.com/colmap/colmap/releases
# Add to PATH or use --colmap-path
```

**Linux/Jetson:**
```bash
sudo apt install colmap
```

### Quick Start

```bash
# From directory of images
python scripts/generate_scene_multiview.py images/

# From glob pattern
python scripts/generate_scene_multiview.py "photos/*.jpg" --device cuda

# Custom COLMAP path
python scripts/generate_scene_multiview.py images/ --colmap-path C:\colmap\bin\colmap.exe
```

### Output

Creates `outputs/<folder>_multiview/`:
- `fused_pointcloud.ply` - Dense fused point cloud with color
- `database.db` - COLMAP feature database
- `sparse/` - COLMAP sparse reconstruction
- `colmap_sparse/` - Copy of COLMAP output for reference

### Workflow

1. **COLMAP Feature Extraction** - SIFT keypoints from all images (~2-5s)
2. **COLMAP Feature Matching** - Match correspondences between images (~3-8s)
3. **COLMAP Sparse Mapper** - Compute camera poses and sparse 3D points (~2-5s)
4. **ZoeDepth Per-Frame** - Estimate metric depth for each image (~0.05s per frame)
5. **Depth Fusion** - Backproject and merge into dense cloud (~0.2s)

### Performance

| Images | Resolution | Device | Time |
|--------|-----------|--------|------|
| 10 | 1080p | CPU | ~25-35s |
| 10 | 1080p | CUDA | ~12-18s |
| 20 | 1080p | CUDA | ~20-30s |
| 50 | 1080p | Jetson Orin | ~45-60s |

### Best Practices

**Image Capture:**
- Take 10-30 images with 60-80% overlap
- Move in arc or circle around scene
- Avoid pure rotation (causes scale ambiguity)
- Ensure good lighting and minimal motion blur

**For Large Scenes:**
- Use wider camera FOV
- Increase `--max-depth` if needed
- Consider sequential matcher instead of exhaustive

---

## Gaussian Splatting Training

The point clouds generated can be used as initialization for Gaussian Splatting training.

### Using Nerfstudio (Recommended)

```bash
# Install nerfstudio
pip install nerfstudio

# Train on COLMAP output
ns-train splatfacto --data outputs/landscape_multiview/colmap_sparse/

# View results
ns-viewer --load-config outputs/.../config.yml
```

### Using Original Gaussian Splatting

```bash
# Clone repo to external/
cd external
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive

# Install
cd gaussian-splatting
pip install -e .

# Train
python train.py -s ../outputs/landscape_multiview --iterations 7000
```

---

## Viewing Point Clouds

### In Browser Viewer

Our Three.js viewer supports PLY point clouds:

```bash
# Start server
python -m http.server 8000

# Open http://localhost:8000/viewer/index.html
# Click "Load Folder" and select output directory
```

### External Tools

**CloudCompare** (Windows/Linux):
- Free, powerful point cloud viewer
- Download: https://www.cloudcompare.org/

**MeshLab** (Cross-platform):
- Supports PLY, mesh processing
- Download: https://www.meshlab.net/

**Open3D Viewer** (Python):
```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("pointcloud.ply")
o3d.visualization.draw_geometries([pcd])
```

---

## Jetson Optimization

See `docs/jetson-optimization.md` for:
- TensorRT conversion for ZoeDepth (5-10× speedup)
- cuSIFT for GPU-accelerated COLMAP
- Memory management tips
- JetPack 5 vs 6 considerations

---

## Troubleshooting

**COLMAP fails with "Insufficient matches":**
- Ensure 60%+ overlap between adjacent images
- Check lighting consistency
- Try `--ImageReader.single_camera 0` if using multiple camera types

**Depth maps look noisy:**
- ZoeDepth works best on well-lit scenes
- Avoid highly reflective surfaces (water, glass)
- Try different `--camera-fov` if scale seems wrong

**Point cloud has holes:**
- Single-image mode has no multi-view consistency - holes are normal
- Use multi-view mode for complete reconstruction
- Increase image count for better coverage

**Out of memory:**
- Reduce image resolution before processing
- Use `--max-depth` to limit point cloud size
- On Jetson, close unnecessary processes

---

## Examples

### Landscape Scan (Single Image)
```bash
python scripts/generate_scene_zoedepth.py mars_terrain.jpg --max-depth 20.0
# Output: outputs/mars_terrain_scene/pointcloud.ply
```

### Indoor Environment (Multi-View)
```bash
python scripts/generate_scene_multiview.py habitat_photos/ --device cuda
# Output: outputs/habitat_photos_multiview/fused_pointcloud.ply
```

### Rock Formation (Use TripoSR Instead)
```bash
# For isolated objects, TripoSR gives better geometry
python scripts/generate_3d_triposr.py rock_sample.jpg
# Output: outputs/rock_sample_3d/0/mesh.obj + texture.png
```
