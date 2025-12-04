#!/usr/bin/env python3
"""
Multi-View Scene Reconstruction with COLMAP + ZoeDepth

Mission-grade landscape reconstruction for NASA GFA.
Takes multiple images, computes camera poses with COLMAP sparse reconstruction,
estimates per-frame depth with ZoeDepth, fuses into dense point cloud.

Usage:
    python generate_scene_multiview.py images/ --output landscape_scene/
    python generate_scene_multiview.py images/*.jpg --device cuda --colmap-path /usr/local/bin/colmap
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import time
import shutil

import numpy as np
import torch
from PIL import Image


def check_colmap():
    """Check if COLMAP is available."""
    # Try common paths first
    colmap_paths = [
        'colmap',
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'COLMAP', 'COLMAP-3.8-windows-no-cuda', 'bin', 'colmap.exe'),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'COLMAP', 'COLMAP-3.9.1-windows-cuda', 'bin', 'colmap.exe'),
        r'C:\Program Files\COLMAP\bin\colmap.exe',
    ]
    
    for colmap_cmd in colmap_paths:
        try:
            result = subprocess.run([colmap_cmd, '-h'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True, colmap_cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    return False, None


def run_colmap_sparse(image_dir, output_dir, colmap_path='colmap'):
    """
    Run COLMAP sparse reconstruction.
    
    Steps:
    1. Feature extraction (SIFT)
    2. Feature matching
    3. Sparse reconstruction (mapper)
    
    Returns database and sparse model paths.
    """
    print("[INFO] Running COLMAP sparse reconstruction...")
    start_time = time.time()
    
    # Setup paths
    database_path = output_dir / 'database.db'
    sparse_dir = output_dir / 'sparse'
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature extraction
    print("[COLMAP] Extracting features...")
    subprocess.run([
        colmap_path, 'feature_extractor',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--ImageReader.single_camera', '1',
        '--SiftExtraction.use_gpu', '1'  # Use GPU if available
    ], check=True)
    
    # 2. Feature matching
    print("[COLMAP] Matching features...")
    subprocess.run([
        colmap_path, 'exhaustive_matcher',
        '--database_path', str(database_path),
        '--SiftMatching.use_gpu', '1'
    ], check=True)
    
    # 3. Sparse reconstruction
    print("[COLMAP] Running mapper...")
    subprocess.run([
        colmap_path, 'mapper',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--output_path', str(sparse_dir)
    ], check=True)
    
    colmap_time = time.time() - start_time
    print(f"[TIMING] COLMAP sparse reconstruction: {colmap_time:.2f}s")
    
    # Find the reconstruction (usually in sparse/0/)
    reconstruction_dir = sparse_dir / '0'
    if not reconstruction_dir.exists():
        raise RuntimeError("COLMAP reconstruction failed - no output found")
    
    return database_path, reconstruction_dir


def load_colmap_cameras(reconstruction_dir):
    """
    Parse COLMAP cameras.txt to extract intrinsics.
    
    Returns:
        dict of camera_id -> (model, width, height, params)
    """
    cameras_file = reconstruction_dir / 'cameras.txt'
    cameras = {}
    
    with open(cameras_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    
    return cameras


def load_colmap_images(reconstruction_dir):
    """
    Parse COLMAP images.txt to extract camera poses.
    
    Returns:
        list of (image_name, camera_id, qw, qx, qy, qz, tx, ty, tz)
    """
    images_file = reconstruction_dir / 'images.txt'
    images = []
    
    with open(images_file, 'r') as f:
        lines = [l for l in f if not l.startswith('#')]
        
        # Images.txt has pairs of lines: image info, then point data
        for i in range(0, len(lines), 2):
            parts = lines[i].strip().split()
            if len(parts) < 10:
                continue
            
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            image_name = parts[9]
            
            images.append({
                'name': image_name,
                'camera_id': camera_id,
                'quat': np.array([qw, qx, qy, qz]),
                'trans': np.array([tx, ty, tz])
            })
    
    return images


def estimate_depth_batch(model, image_paths, device='cpu'):
    """
    Run ZoeDepth on multiple images.
    
    Returns:
        list of (depth_map, rgb_image) tuples
    """
    print(f"[INFO] Estimating depth for {len(image_paths)} images...")
    model = model.to(device)
    model.eval()
    
    results = []
    start_time = time.time()
    
    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Processing {Path(img_path).name}...")
        
        image = Image.open(img_path).convert('RGB')
        
        with torch.no_grad():
            depth = model.infer_pil(image)
        
        results.append((depth, image, img_path))
    
    total_time = time.time() - start_time
    print(f"[TIMING] Depth estimation ({len(image_paths)} images): {total_time:.2f}s ({total_time/len(image_paths):.3f}s per image)")
    
    return results


def fuse_depth_maps(depth_results, colmap_cameras, colmap_images, image_dir):
    """
    Fuse multiple depth maps into a single dense point cloud using COLMAP poses.
    
    Simple approach: backproject each depth map to 3D using camera pose,
    merge all points (with optional voxel downsampling).
    """
    print("[INFO] Fusing depth maps into point cloud...")
    start_time = time.time()
    
    all_points = []
    all_colors = []
    
    # Create lookup from image name to pose
    image_poses = {img['name']: img for img in colmap_images}
    
    for depth_map, rgb_image, img_path in depth_results:
        img_name = Path(img_path).name
        
        if img_name not in image_poses:
            print(f"[WARNING] No pose found for {img_name}, skipping")
            continue
        
        pose = image_poses[img_name]
        camera = colmap_cameras[pose['camera_id']]
        
        # Get camera intrinsics (assume PINHOLE or SIMPLE_PINHOLE)
        if camera['model'] == 'PINHOLE':
            fx, fy, cx, cy = camera['params'][:4]
        elif camera['model'] == 'SIMPLE_PINHOLE':
            f, cx, cy = camera['params'][:3]
            fx = fy = f
        else:
            print(f"[WARNING] Unsupported camera model {camera['model']}, skipping {img_name}")
            continue
        
        H, W = depth_map.shape
        
        # Create pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Backproject to camera space
        z = depth_map
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points_camera = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # Transform to world space using COLMAP pose
        # COLMAP stores world-to-camera, we need camera-to-world
        # R_w2c and t_w2c -> R_c2w = R_w2c^T, t_c2w = -R_c2w * t_w2c
        
        # Convert quaternion to rotation matrix
        qw, qx, qy, qz = pose['quat']
        R_w2c = quat_to_rotation_matrix(qw, qx, qy, qz)
        t_w2c = pose['trans']
        
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        
        # Transform points
        points_world = (R_c2w @ points_camera.T).T + t_c2w
        
        # Get colors
        rgb_array = np.array(rgb_image).reshape(-1, 3)
        
        # Filter valid depths
        valid_mask = (z.reshape(-1) > 0) & (z.reshape(-1) < 50)  # Max 50m depth
        
        all_points.append(points_world[valid_mask])
        all_colors.append(rgb_array[valid_mask])
    
    # Concatenate all points
    fused_points = np.vstack(all_points)
    fused_colors = np.vstack(all_colors)
    
    fusion_time = time.time() - start_time
    print(f"[TIMING] Depth fusion: {fusion_time:.2f}s")
    print(f"[INFO] Fused point cloud: {len(fused_points):,} points")
    
    return fused_points, fused_colors


def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


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


def main():
    parser = argparse.ArgumentParser(description="Multi-view scene reconstruction with COLMAP + ZoeDepth")
    parser.add_argument('input', type=str, help="Input images directory or glob pattern (e.g., images/*.jpg)")
    parser.add_argument('--output', type=str, default=None, help="Output directory")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--colmap-path', type=str, default='colmap', help="Path to COLMAP executable")
    
    args = parser.parse_args()
    
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
    
    print(f"[INFO] Found {len(image_paths)} images")
    
    # Setup output
    if args.output is None:
        output_dir = Path('outputs') / f"{image_dir.name}_multiview"
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Check COLMAP
    colmap_available, colmap_cmd = check_colmap()
    if not colmap_available:
        print(f"[ERROR] COLMAP not found. Please install COLMAP:")
        print(f"  Download: https://github.com/colmap/colmap/releases/download/3.9.1/COLMAP-3.9.1-windows-cuda.zip")
        print(f"  Extract to: %LOCALAPPDATA%\\COLMAP")
        sys.exit(1)
    
    print(f"[INFO] Using COLMAP: {colmap_cmd}")
    
    # Override colmap path if auto-detected
    if args.colmap_path == 'colmap':
        args.colmap_path = colmap_cmd
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    total_start = time.time()
    
    # Step 1: COLMAP sparse reconstruction
    database_path, reconstruction_dir = run_colmap_sparse(image_dir, output_dir, args.colmap_path)
    
    # Step 2: Load COLMAP results
    cameras = load_colmap_cameras(reconstruction_dir)
    images = load_colmap_images(reconstruction_dir)
    print(f"[INFO] COLMAP reconstructed {len(images)} images with {len(cameras)} camera(s)")
    
    # Step 3: Load ZoeDepth
    print("[INFO] Loading ZoeDepth model...")
    zoedepth_model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    
    # Step 4: Estimate depth for each image
    depth_results = estimate_depth_batch(zoedepth_model, image_paths, device=args.device)
    
    # Step 5: Fuse depth maps using COLMAP poses
    fused_points, fused_colors = fuse_depth_maps(depth_results, cameras, images, image_dir)
    
    # Step 6: Save output
    pointcloud_path = output_dir / 'fused_pointcloud.ply'
    save_pointcloud_ply(fused_points, fused_colors, pointcloud_path)
    
    # Copy COLMAP reconstruction for reference
    colmap_output = output_dir / 'colmap_sparse'
    if colmap_output.exists():
        shutil.rmtree(colmap_output)
    shutil.copytree(reconstruction_dir, colmap_output)
    
    total_time = time.time() - total_start
    print(f"\n[COMPLETE] Total runtime: {total_time:.2f}s")
    print(f"[COMPLETE] Output saved to: {output_dir}")
    print(f"[COMPLETE] Point cloud: {pointcloud_path}")


if __name__ == '__main__':
    main()
