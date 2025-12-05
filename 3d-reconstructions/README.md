# 3D Generation: Multi-Modal Reconstruction Pipeline

This workspace provides **NASA GFA field-ready 3D reconstruction** using state-of-the-art AI models. It supports:

- **Object reconstruction** - TripoSR for single-image mesh generation
- **Scene reconstruction** - ZoeDepth + Gaussian Splatting for landscapes/environments
- **Multi-view reconstruction** - COLMAP + depth fusion for mission-grade quality

Designed for deployment on **Jetson Orin NX SUPER** and Windows laptops.

---

## 🚀 Quick Start

### Objects (Rocks, Equipment, Small Items)

```powershell
# Generate 3D mesh from single image using TripoSR
python scripts/generate_3d_triposr.py rock.jpg --device cuda

# Output: outputs/rock_3d/0/mesh.obj + texture.png
```

**Runtime:** ~2 min CPU, ~10s GPU

### Landscapes/Scenes (Single Image)

```powershell
# Generate 3D point cloud from single photo using ZoeDepth
python scripts/generate_scene_zoedepth.py terrain.jpg --device cuda

# Output: outputs/terrain_scene/pointcloud.ply
```

**Runtime:** ~2-4s on Jetson Orin NX

### Landscapes/Scenes (Multi-View, Mission-Grade)

```powershell
# Reconstruct scene from multiple photos using COLMAP + ZoeDepth
python scripts/generate_scene_multiview.py landscape_photos/ --device cuda

# Output: outputs/landscape_photos_multiview/fused_pointcloud.ply
```

**Runtime:** ~10-20s for 10-20 images on Jetson Orin NX

### Video to 3D (Automatic Frame Extraction + Auto Backend)

```powershell
# One-step video to 3D (auto backend selection + frame filtering)
python scripts/generate_scene_from_video.py video.mp4 --device cuda

# Output: outputs/video_3d/reconstruction/fused_pointcloud.ply
```

**Runtime:** ~3-8 minutes (default 30 max frames, filtered)

See [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md) for detailed video reconstruction guide.

---

## 📋 Use Cases

| Scenario | Method | Runtime | Output |
|----------|--------|---------|--------|
| **Rock sample scan** | TripoSR | ~10s GPU | Textured mesh (OBJ/GLB) |
| **Terrain snapshot** | ZoeDepth single-image | ~2-4s | Point cloud (PLY) |
| **Habitat panorama** | COLMAP + ZoeDepth | ~10-20s | Dense point cloud (PLY) |
| **Equipment catalog** | TripoSR batch | ~10s per item | Mesh collection |
| **Video walkaround** | Video → frames → MASt3R | ~3-8 min | Dense point cloud (PLY) |

---

## 🛠️ Setup

### Windows Laptop

See `docs/windows-setup.md` for:
- Python virtual environment setup
- PyTorch installation (CPU or CUDA)
- TripoSR and ZoeDepth installation
- COLMAP installation

### Jetson Orin NX

See `docs/jetson-orin-setup.md` for:
- JetPack 5/6 compatibility
- aarch64 PyTorch wheels
- TensorRT optimization
- COLMAP GPU acceleration (cuSIFT)

---

## 📁 Repository Structure

```
3d-generation/
├── scripts/
│   ├── generate_3d_triposr.py      # Object mesh generation
│   ├── generate_scene_zoedepth.py  # Single-image scene reconstruction
│   ├── generate_scene_multiview.py # Multi-view scene reconstruction
│   ├── extract_frames.py           # Video frame extraction with quality filtering
│   ├── generate_scene_from_video.py # End-to-end video to 3D pipeline
│   ├── convert_to_glb.py           # Mesh format conversion
│   └── viewer_3d.py                # Native PyOpenGL mesh viewer
├── docs/
│   ├── windows-setup.md            # Windows installation guide
│   ├── jetson-orin-setup.md        # Jetson Orin setup
│   ├── scene-reconstruction.md     # Scene reconstruction workflows
│   └── jetson-optimization.md      # TensorRT and performance tuning
├── VIDEO_PIPELINE.md               # Comprehensive video-to-3D guide
├── config.py                       # Configuration management
├── viewer/
│   └── index.html                  # Browser-based 3D viewer (Three.js)
├── external/
│   └── TripoSR/                    # TripoSR repository (cloned)
├── outputs/                        # Generated 3D models
└── requirements.txt                # Python dependencies
```

---

## 🎯 Workflows

### Object Reconstruction (TripoSR)

Best for: isolated objects, rocks, equipment, samples

```bash
# Single image
python scripts/generate_3d_triposr.py image.jpg

# Batch processing
python scripts/generate_3d_triposr.py images/*.jpg

# Custom output format
python scripts/generate_3d_triposr.py rock.jpg --model-save-format glb
```

**Outputs:** Textured mesh (OBJ + PNG or GLB)

### Single-Image Scene Reconstruction

Best for: quick terrain scans, field snapshots, zero setup time

```bash
# Basic usage
python scripts/generate_scene_zoedepth.py landscape.jpg

# With custom parameters
python scripts/generate_scene_zoedepth.py mars_terrain.jpg \
  --device cuda \
  --camera-fov 70 \
  --max-depth 20.0
```

**Outputs:** Point cloud (PLY), depth map (PNG)

### Multi-View Scene Reconstruction

Best for: accurate geometry, large scenes, mission documentation

```bash
# From directory
python scripts/generate_scene_multiview.py photos/

# From glob pattern
python scripts/generate_scene_multiview.py "sweep_*.jpg"
```

**Outputs:** Dense point cloud (PLY), COLMAP sparse reconstruction

### Video to 3D Reconstruction

Best for: easy capture, automatic frame selection, walkarounds

```bash
# Automatic - extracts best frames and reconstructs
python scripts/generate_scene_from_video.py room_scan.mp4

# Custom parameters
python scripts/generate_scene_from_video.py scene.mp4 \
  --backend mast3r \
  --max-frames 30 \
  --filter-quality \
  --device cuda
```

**Features:**
- Intelligent frame extraction (blur, exposure, similarity filtering)
- Auto backend decision tree (single-object → TripoSR, <15 frames → DUSt3R, else MASt3R)
- Explicit COLMAP option for traditional photogrammetry
- Similarity-based deduplication
- Quality filtering presets

**Outputs:** Dense point cloud (PLY), camera poses, filtered frames

See [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md) for detailed guide and auto backend logic.
See [docs/dust3r-vs-mast3r.md](docs/dust3r-vs-mast3r.md) for backend comparison (used when not in TripoSR single-object branch).

See `docs/scene-reconstruction.md` for detailed workflows.

---

## 🖥️ Viewing Results

### Browser Viewer (Recommended)

```powershell
# Start HTTP server
python -m http.server 8000

# Open http://localhost:8000/viewer/index.html
# Click "Load Folder" and select output directory
```

**Features:**
- Loads OBJ meshes with textures
- Loads PLY point clouds
- Interactive camera controls
- Wireframe/environment toggles
- Mesh statistics display

### Native Viewer (PyOpenGL)

```bash
python scripts/viewer_3d.py outputs/rock_3d/0/mesh.obj
```

### External Tools

- **CloudCompare** - Point cloud inspection
- **MeshLab** - Mesh editing and analysis
- **Blender** - Full 3D modeling suite

---

## 📊 Performance Benchmarks

| Device | Model | Resolution | Time |
|--------|-------|------------|------|
| **RTX 3060** | TripoSR | 1024×1024 | ~8s |
| **RTX 3060** | ZoeDepth | 1080p | ~0.5s |
| **Jetson Orin NX** | TripoSR | 1024×1024 | ~45s |
| **Jetson Orin NX** | ZoeDepth (TRT) | 1080p | ~0.05s |
| **i7-12700K (CPU)** | TripoSR | 512×512 | ~2min |

*TRT = TensorRT optimized (see `docs/jetson-optimization.md`)*

---

## 🔬 Alternatives & Comparisons

### For Objects:
- **TripoSR** ✅ - Fast, clean meshes (this project)
- **InstantMesh** - Higher quality, slower (~30s)
- **Zero123++** - Multi-view synthesis, needs refinement

### For Scenes:
- **ZoeDepth + GS** ✅ - Fast single/multi-view (this project)
- **COLMAP Dense** - High accuracy, very slow (hours)
- **Dust3R** - No pose estimation needed, experimental
- **NeRF** - Photorealistic, slow training (30min+)

---

## 🐛 Troubleshooting

**"CUDA out of memory":**
- Reduce `--mc-resolution` for TripoSR
- Lower image resolution before processing
- Close other GPU applications

**"COLMAP reconstruction failed":**
- Ensure 60%+ overlap between images
- Use `--ImageReader.single_camera 1` for same camera
- Check lighting consistency

**Viewer not loading textures:**
- Ensure HTTP server runs from project root: `python -m http.server 8000`
- Check browser console (F12) for path errors

**Depth maps look wrong:**
- Adjust `--camera-fov` to match your camera
- ZoeDepth works best on well-lit outdoor scenes
- Avoid highly reflective surfaces (water, glass)

---

## 📄 License

This repository is MIT licensed. Individual dependencies (TripoSR, ZoeDepth, COLMAP, Gaussian Splatting) have their own licenses - check their respective repositories.

---

## 🙏 Acknowledgments

- **TripoSR** - Stability AI & Tripo AI (https://github.com/VAST-AI-Research/TripoSR)
- **ZoeDepth** - Intel ISL (https://github.com/isl-org/ZoeDepth)
- **COLMAP** - Johannes Schönberger (https://colmap.github.io/)
- **Gaussian Splatting** - INRIA (https://github.com/graphdeco-inria/gaussian-splatting)

---

## 🚀 NASA GFA Deployment Notes

This pipeline is optimized for **NASA Goddard Flight Robotics**' field requirements:

- **Jetson Orin NX SUPER** target platform
- **2-4s single-image reconstruction** for instant feedback
- **10-20s multi-view reconstruction** for mission-grade accuracy
- **Offline operation** - no cloud dependencies
- **Robust to field conditions** - works in poor lighting, cluttered backgrounds

For TensorRT optimization and deployment packaging, see `docs/jetson-optimization.md`.
