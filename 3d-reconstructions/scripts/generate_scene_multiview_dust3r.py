#!/usr/bin/env python3
"""
Multi-View Scene Reconstruction with DUSt3R/MASt3R

Alternative pipeline that uses DUSt3R or MASt3R for joint depth and pose estimation.
Unlike COLMAP+ZoeDepth, dust3r/mast3r directly predicts both camera poses and dense depth
in a single forward pass, making it faster and more robust to low-texture scenes.

Key differences from COLMAP pipeline:
- No SIFT features or sparse reconstruction needed
- Joint depth + pose estimation
- Works better with fewer images and low-texture scenes
- Generally faster (especially for small image sets)
- No need for separate depth estimation model

Usage:
    python generate_scene_multiview_dust3r.py test_multiview/ --output outputs/my_scene_dust3r
    python generate_scene_multiview_dust3r.py images/*.jpg --device cuda
    python generate_scene_multiview_dust3r.py test_multiview/ --model mast3r
"""

import argparse
import sys
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from PIL import Image

print("[INFO] Imports successful")

# Add dust3r to path if it exists in external/
dust3r_path = Path(__file__).resolve().parents[1] / "external" / "dust3r"
if dust3r_path.exists():
    sys.path.insert(0, str(dust3r_path))
    print(f"[INFO] Added dust3r to path: {dust3r_path}")

# Try to import dust3r
try:
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.device import to_numpy
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.image import load_images as dust3r_load_images
    HAS_DUST3R = True
except ImportError as e:
    HAS_DUST3R = False
    print(f"[WARNING] dust3r not installed. Error: {e}")
    print("[INFO] Install with:")
    print("  git clone --recursive https://github.com/naver/dust3r external/dust3r")
    print("  cd external/dust3r")
    print("  pip install -r requirements.txt")


def load_dust3r_model(model_name='dust3r', device='cpu'):
    """
    Load DUSt3R or MASt3R model.
    
    Args:
        model_name: 'dust3r' or 'mast3r'
        device: 'cpu' or 'cuda'
    
    Returns:
        model: DUSt3R/MASt3R model
    """
    if not HAS_DUST3R:
        raise RuntimeError("dust3r not installed")
    
    print(f"[INFO] Loading {model_name.upper()} model...")
    start_time = time.time()
    
    # Model weights paths
    if model_name.lower() == 'mast3r':
        # MASt3R is generally better for multi-view reconstruction
        model_path = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
    else:
        # DUSt3R baseline
        model_path = 'naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt'
    
    try:
        model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    except Exception as e:
        print(f"[ERROR] Failed to load {model_name} from HuggingFace: {e}")
        print("[INFO] Make sure you have internet connection for first-time download")
        raise
    
    load_time = time.time() - start_time
    print(f"[TIMING] Model loading: {load_time:.2f}s")
    
    return model


def load_images(image_paths, model, max_size=512):
    """
    Load and prepare images for dust3r processing using dust3r's native loader.
    
    Args:
        image_paths: list of image file paths
        model: dust3r model (needed for patch_size)
        max_size: maximum dimension (dust3r works best with 512x512)
    
    Returns:
        images: list of dicts with 'img', 'true_shape', 'idx', 'instance'
        image_names: list of image filenames
    """
    print(f"[INFO] Loading {len(image_paths)} images using dust3r's loader...")
    
    # Convert Path objects to strings
    image_paths_str = [str(p) for p in image_paths]
    image_names = [Path(p).name for p in image_paths]
    
    # Use dust3r's native image loader
    images = dust3r_load_images(
        image_paths_str, 
        size=max_size, 
        square_ok=False, 
        verbose=True,
        patch_size=model.patch_size
    )
    
    return images, image_names


def run_dust3r_reconstruction(model, images, device='cpu', batch_size=1):
    """
    Run DUSt3R/MASt3R inference on image pairs to get depth and poses.
    
    Args:
        model: DUSt3R/MASt3R model
        images: list of image dicts from dust3r's load_images
        device: 'cpu' or 'cuda'
        batch_size: number of image pairs to process at once
    
    Returns:
        pairs: list of predictions from dust3r
    """
    print("[INFO] Running dust3r inference on image pairs...")
    start_time = time.time()
    
    # Create image pairs - dust3r expects all pairwise combinations
    # Images are already in the correct format from load_images
    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            pairs.append((images[i], images[j]))
    
    print(f"[INFO] Processing {len(pairs)} image pairs...")
    
    # Run inference
    with torch.no_grad():
        predictions = inference(pairs, model, device, batch_size=batch_size)
    
    inference_time = time.time() - start_time
    print(f"[TIMING] DUSt3R inference: {inference_time:.2f}s ({len(pairs)} pairs)")
    
    return predictions


def align_and_fuse_pointclouds(predictions, images, device='cpu', mode='GlobalAlignerMode.PointCloudOptimizer'):
    """
    Globally align all views and extract fused point cloud.
    
    Args:
        predictions: output from dust3r inference
        images: original input images
        device: 'cpu' or 'cuda'
        mode: alignment mode (PointCloudOptimizer or PairViewer)
    
    Returns:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors
        cameras: camera parameters and poses
    """
    print("[INFO] Running global alignment...")
    start_time = time.time()
    
    # Global alignment to register all views into a common coordinate system
    scene = global_aligner(
        predictions, 
        device=device, 
        mode=GlobalAlignerMode.PointCloudOptimizer
    )
    
    # Optimize camera poses
    loss = scene.compute_global_alignment(
        init='mst',  # Minimum spanning tree initialization
        niter=300,    # Number of optimization iterations
        schedule='cosine',
        lr=0.01
    )
    
    print(f"[INFO] Global alignment loss: {loss:.6f}")
    
    # Extract point cloud from all views
    print("[INFO] Extracting fused point cloud...")
    
    # Get 3D points, images, and confidence for all views
    pts3d_list = scene.get_pts3d()  # List of (H, W, 3) tensors
    conf_list = scene.get_conf(mode='log')  # List of (H, W) tensors
    
    all_points = []
    all_colors = []
    
    for view_idx in range(len(images)):
        # Get depth map and RGB for this view
        pts3d = to_numpy(pts3d_list[view_idx])  # (H, W, 3)
        rgb = to_numpy(scene.imgs[view_idx])  # (H, W, 3)
        
        # Get confidence mask (filter uncertain predictions)
        conf = to_numpy(conf_list[view_idx])  # (H, W)
        
        # Apply confidence threshold
        conf_threshold = 1.5  # Lower = more permissive
        mask = conf > conf_threshold
        
        # Reshape and filter
        pts3d_flat = pts3d.reshape(-1, 3)
        rgb_flat = (rgb.reshape(-1, 3) * 255).astype(np.uint8)
        mask_flat = mask.reshape(-1)
        
        # Apply mask
        pts3d_valid = pts3d_flat[mask_flat]
        rgb_valid = rgb_flat[mask_flat]
        
        all_points.append(pts3d_valid)
        all_colors.append(rgb_valid)
        
        print(f"  View {view_idx+1}/{len(images)}: {len(pts3d_valid):,} confident points")
    
    # Concatenate all points
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    align_time = time.time() - start_time
    print(f"[TIMING] Alignment and fusion: {align_time:.2f}s")
    print(f"[INFO] Total points before voxel fusion: {len(all_points):,}")
    
    return all_points, all_colors, scene


def voxel_downsample(points, colors, voxel_size=0.05):
    """
    Downsample point cloud using voxel grid.
    
    Args:
        points: (N, 3) array
        colors: (N, 3) array
        voxel_size: size of voxel in meters
    
    Returns:
        downsampled_points, downsampled_colors
    """
    print(f"[INFO] Voxel downsampling (voxel size: {voxel_size}m)...")
    
    # Discretize to voxel coordinates
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    
    # Create unique voxel keys and average points within each voxel
    voxel_dict = {}
    for i in range(len(voxel_coords)):
        key = tuple(voxel_coords[i])
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(i)
    
    downsampled_points = []
    downsampled_colors = []
    
    for indices in voxel_dict.values():
        avg_point = points[indices].mean(axis=0)
        avg_color = colors[indices].mean(axis=0)
        downsampled_points.append(avg_point)
        downsampled_colors.append(avg_color)
    
    downsampled_points = np.array(downsampled_points)
    downsampled_colors = np.array(downsampled_colors)
    
    print(f"[INFO] Downsampled from {len(points):,} to {len(downsampled_points):,} points")
    
    return downsampled_points, downsampled_colors


def save_pointcloud_ply(points, colors, output_path):
    """Save point cloud as PLY file."""
    print(f"[INFO] Saving point cloud to {output_path}...")
    
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    
    print(f"[OK] Saved {len(points):,} points")


def save_cameras(scene, image_names, output_path):
    """
    Save camera poses and intrinsics to text file.
    
    Args:
        scene: GlobalAligner scene object
        image_names: list of image filenames
        output_path: path to save cameras.txt
    """
    print(f"[INFO] Saving camera parameters to {output_path}...")
    
    with open(output_path, 'w') as f:
        f.write("# Camera poses and intrinsics from DUSt3R\n")
        f.write("# Format: image_name focal_length cx cy qw qx qy qz tx ty tz\n\n")
        
        for i, img_name in enumerate(image_names):
            # Get camera pose (world to camera transform)
            cam_pose = to_numpy(scene.get_im_poses()[i])  # 4x4 matrix
            
            # Extract rotation (convert to quaternion) and translation
            R = cam_pose[:3, :3]
            t = cam_pose[:3, 3]
            
            # Convert rotation matrix to quaternion (w, x, y, z)
            from scipy.spatial.transform import Rotation
            quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            
            # Get focal length (approximate - dust3r uses internal focal)
            focal = to_numpy(scene.get_focals()[i])[0]
            
            # Get principal point (approximate)
            h, w = scene.imgs[i].shape[:2]
            cx, cy = w / 2, h / 2
            
            f.write(f"{img_name} {focal:.2f} {cx:.2f} {cy:.2f} "
                   f"{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
                   f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n")
    
    print(f"[OK] Saved {len(image_names)} camera poses")


def main():
    parser = argparse.ArgumentParser(description="Multi-view scene reconstruction with DUSt3R/MASt3R")
    parser.add_argument('input', type=str, help="Input images directory or glob pattern")
    parser.add_argument('--output', type=str, default=None, help="Output directory")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--model', type=str, default='dust3r', choices=['dust3r', 'mast3r'],
                       help="Model to use (mast3r is generally better for multi-view)")
    parser.add_argument('--image-size', type=int, default=512, 
                       help="Maximum image dimension (512 recommended)")
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help="Voxel size for downsampling in meters (0.05 = 5cm)")
    parser.add_argument('--batch-size', type=int, default=1,
                       help="Batch size for inference (increase if you have GPU memory)")
    
    args = parser.parse_args()
    
    # Check if dust3r is installed
    if not HAS_DUST3R:
        print("[ERROR] dust3r is not installed. See installation instructions above.")
        sys.exit(1)
    
    # Parse input
    input_path = Path(args.input)
    if input_path.is_dir():
        image_dir = input_path
        image_paths = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    else:
        # Glob pattern
        import glob
        image_paths = sorted([Path(p) for p in glob.glob(args.input)])
        image_dir = image_paths[0].parent if image_paths else Path('.')
    
    if not image_paths:
        print("[ERROR] No images found")
        sys.exit(1)
    
    if len(image_paths) < 2:
        print("[ERROR] Need at least 2 images for multi-view reconstruction")
        sys.exit(1)
    
    print(f"[INFO] Found {len(image_paths)} images")
    
    # Setup output
    if args.output is None:
        output_dir = Path('outputs') / f"{image_dir.name}_dust3r"
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    total_start = time.time()
    
    # Step 1: Load model
    model = load_dust3r_model(args.model, device=args.device)
    
    # Step 2: Load and prepare images using dust3r's loader
    images, image_names = load_images(image_paths, model, max_size=args.image_size)
    
    # Step 3: Run dust3r reconstruction
    predictions = run_dust3r_reconstruction(
        model, 
        images, 
        device=args.device, 
        batch_size=args.batch_size
    )
    
    # Step 4: Global alignment and point cloud extraction
    points, colors, scene = align_and_fuse_pointclouds(
        predictions, 
        images, 
        device=args.device
    )
    
    # Step 5: Voxel downsampling
    points_downsampled, colors_downsampled = voxel_downsample(
        points, 
        colors, 
        voxel_size=args.voxel_size
    )
    
    # Step 6: Save outputs
    pointcloud_path = output_dir / 'fused_pointcloud.ply'
    save_pointcloud_ply(points_downsampled, colors_downsampled, pointcloud_path)
    
    # Save camera poses
    cameras_path = output_dir / 'cameras.txt'
    save_cameras(scene, image_names, cameras_path)
    
    # Also save full resolution point cloud (no downsampling)
    if len(points) != len(points_downsampled):
        full_res_path = output_dir / 'pointcloud_full.ply'
        save_pointcloud_ply(points, colors, full_res_path)
        print(f"[INFO] Also saved full resolution point cloud: {full_res_path}")
    
    total_time = time.time() - total_start
    print(f"\n[COMPLETE] Total runtime: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"[COMPLETE] Output saved to: {output_dir}")
    print(f"[COMPLETE] Point cloud: {pointcloud_path}")
    print(f"[COMPLETE] Camera poses: {cameras_path}")
    print(f"\n[NEXT] Load result in viewer:")
    print(f"  http://localhost:5000/viewer/index.html")
    print(f"  Use 'Load Folder' -> select: {output_dir}")
    print(f"\n[INFO] Pipeline used: {args.model.upper()}")
    print(f"[INFO] Total images: {len(images)}")
    print(f"[INFO] Final point count: {len(points_downsampled):,}")


if __name__ == '__main__':
    main()
