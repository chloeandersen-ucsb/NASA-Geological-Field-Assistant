# Multi-View Scene Reconstruction Setup

## Installation

### Install COLMAP (Windows)

1. **Download COLMAP:**
   - Visit: https://github.com/colmap/colmap/releases/latest
   - Download: `COLMAP-3.X-windows-cuda.zip` (or `-no-cuda` if no NVIDIA GPU)

2. **Extract:**
   ```
   Extract to: C:\Program Files\COLMAP\
   ```

3. **Add to PATH:**
   - Open "Environment Variables" in Windows Settings
   - Edit System PATH variable
   - Add: `C:\Program Files\COLMAP\bin`
   - Restart terminal

4. **Verify:**
   ```powershell
   colmap -h
   ```

## Usage

### Prepare Test Images

Capture 10-20 overlapping images of a scene:
- Walk around object/scene in a circle
- Keep ~60-70% overlap between consecutive images
- Maintain similar distance from subject
- Use consistent lighting

Place images in a folder (e.g., `test_multiview/`)

### Run Multi-View Reconstruction

#### Using COLMAP + ZoeDepth (Traditional Pipeline)

```powershell
# Basic usage (CPU)
python scripts/generate_scene_multiview.py test_multiview/ --output outputs/scene_multi

# With CUDA acceleration (if available)
python scripts/generate_scene_multiview.py test_multiview/ --device cuda

# Custom COLMAP path
python scripts/generate_scene_multiview.py test_multiview/ --colmap-path "C:\Program Files\COLMAP\bin\colmap.exe"
```

#### Using DUSt3R/MASt3R (Alternative Pipeline - No COLMAP needed)

**Installation:**

```powershell
# Clone dust3r repository
git clone --recursive https://github.com/naver/dust3r
cd dust3r
pip install -r requirements.txt
pip install -e .

# Return to your project
cd ..
```

**Usage:**

```powershell
# Basic usage with DUSt3R
python scripts/generate_scene_multiview_dust3r.py test_multiview/ --output outputs/scene_dust3r

# Use MASt3R (better for multi-view)
python scripts/generate_scene_multiview_dust3r.py test_multiview/ --model mast3r --device cuda

# Custom voxel size for downsampling
python scripts/generate_scene_multiview_dust3r.py test_multiview/ --model mast3r --voxel-size 0.02
```

**Which pipeline to use?**
- **DUSt3R/MASt3R**: Low-texture scenes, fewer images (<10), faster setup
- **COLMAP + ZoeDepth**: Many images (20+), high detail, proven accuracy

### Expected Output

#### COLMAP + ZoeDepth Output

```
outputs/scene_multi/
├── fused_pointcloud.ply    # Final dense point cloud
├── colmap_sparse/          # COLMAP reconstruction
│   ├── cameras.txt         # Camera intrinsics
│   ├── images.txt          # Camera poses
│   └── points3D.txt        # Sparse points
├── database.db             # COLMAP feature database
└── sparse/                 # COLMAP workspace
```

#### DUSt3R/MASt3R Output

```
outputs/scene_dust3r/
├── fused_pointcloud.ply    # Downsampled point cloud
├── pointcloud_full.ply     # Full resolution point cloud
└── cameras.txt             # Camera poses and intrinsics
```

### View Results

Load `fused_pointcloud.ply` in the viewer:
```
http://localhost:5000/viewer/index.html
```
Use "Load Folder" → select `outputs/scene_multi`

## Pipeline Details

### COLMAP + ZoeDepth Pipeline

1. **COLMAP Sparse Reconstruction** (~30-60s for 20 images)
   - SIFT feature extraction
   - Feature matching
   - Structure-from-Motion (camera pose estimation)

2. **ZoeDepth Estimation** (~15-20s per image)
   - Metric depth for each image
   - Uses existing camera poses

3. **Depth Fusion** (~5-10s)
   - Backproject depth maps to 3D
   - Transform using COLMAP poses
   - Merge all points

**Total Time:** 5-10 minutes for 20 images (CPU), ~2-3 minutes with CUDA

### DUSt3R / MASt3R Pipeline (Alternative)

1. **Joint Depth + Pose Estimation** (~2-5 minutes for 20 images)
   - Processes image pairs with transformer model
   - Directly predicts dense depth and relative poses
   - No feature extraction or matching needed

2. **Global Alignment** (~10-30s)
   - Registers all views into common coordinate system
   - Optimizes camera poses globally

3. **Point Cloud Fusion** (~5-10s)
   - Fuses depth from all views
   - Filters by confidence threshold
   - Voxel downsampling

**Total Time:** 3-6 minutes for 20 images (GPU recommended)

**Advantages over COLMAP:**
- Works better with low-texture scenes (snow, sand, etc.)
- Needs fewer images (2+ vs 10+ for COLMAP)
- Faster for small image sets
- No manual feature tuning

**When to use COLMAP instead:**
- Very large image sets (100+ images)
- High-resolution images (>2K)
- Scenes with good texture/features
- Need traditional SfM outputs

## Troubleshooting

**"COLMAP not found"**
- Check PATH: `echo $env:PATH` should include COLMAP bin folder
- Verify: `colmap -h` should work
- Try full path: `--colmap-path "C:\Program Files\COLMAP\bin\colmap.exe"`

**"No images reconstructed"**
- Ensure sufficient overlap (60-70%)
- Use more images (minimum 10, recommended 15-20)
- Check image quality (not blurry, good lighting)
- Avoid textureless surfaces

**CUDA errors**
- Use `--device cpu` to disable GPU
- Check CUDA toolkit compatibility with PyTorch version
