#!/usr/bin/env python3
"""
Multi-View Scene Reconstruction with PyColmap + ZoeDepth

Mission-grade landscape reconstruction for NASA GFA.
Takes multiple images, computes camera poses with COLMAP sparse reconstruction,
estimates per-frame depth with ZoeDepth, fuses into dense point cloud.

Usage:
    python generate_scene_multiview_pycolmap.py test_multiview/ --output outputs/my_scene
    python generate_scene_multiview_pycolmap.py images/*.jpg --device cuda
"""

import argparse
import sys
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
import pycolmap
from scipy.spatial import cKDTree
from sklearn.linear_model import RANSACRegressor

print("[INFO] Imports successful")


def get_zoe_intrinsics(camera):
    """
    Get camera intrinsics directly from COLMAP (NO SCALING).
    
    ZoeDepth internally resizes to its network resolution but upsamples
    the output back to the ORIGINAL input resolution. Therefore, we use
    COLMAP intrinsics as-is without any scaling.
    
    Args:
        camera: pycolmap camera object
    
    Returns:
        fx, fy, cx, cy at original COLMAP resolution
    """
    if camera.model in [pycolmap.CameraModelId.SIMPLE_RADIAL, pycolmap.CameraModelId.SIMPLE_PINHOLE]:
        # SIMPLE_RADIAL: [f, cx, cy, k1]
        # SIMPLE_PINHOLE: [f, cx, cy]
        f = camera.params[0]
        cx = camera.params[1]
        cy = camera.params[2]
        
        return f, f, cx, cy
    elif camera.model == pycolmap.CameraModelId.PINHOLE:
        # PINHOLE: [fx, fy, cx, cy]
        fx = camera.params[0]
        fy = camera.params[1]
        cx = camera.params[2]
        cy = camera.params[3]
        
        return fx, fy, cx, cy
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")


def run_colmap_sparse(image_dir, output_dir):
    """
    Run COLMAP sparse reconstruction using PyColmap.
    
    Steps:
    1. Feature extraction (SIFT)
    2. Feature matching
    3. Sparse reconstruction (incremental mapper)
    
    Returns reconstruction object.
    """
    print("[INFO] Running COLMAP sparse reconstruction...")
    start_time = time.time()
    
    # Setup paths
    database_path = output_dir / 'database.db'
    sparse_dir = output_dir / 'sparse'
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image list
    image_list = sorted([str(p) for p in image_dir.glob('*.jpg')] + 
                       [str(p) for p in image_dir.glob('*.png')])
    
    if not image_list:
        raise RuntimeError(f"No images found in {image_dir}")
    
    print(f"[COLMAP] Processing {len(image_list)} images...")
    
    # Extract features
    print("[COLMAP] Extracting SIFT features...")
    extraction_options = pycolmap.FeatureExtractionOptions()
    
    pycolmap.extract_features(
        str(database_path),
        str(image_dir),
        extraction_options=extraction_options
    )
    
    # Match features
    print("[COLMAP] Matching features...")
    pycolmap.match_exhaustive(str(database_path))
    
    # Incremental mapping
    print("[COLMAP] Running incremental mapper...")
    maps = pycolmap.incremental_mapping(
        str(database_path),
        str(image_dir),
        str(sparse_dir)
    )
    
    if not maps or len(maps) == 0:
        raise RuntimeError("COLMAP reconstruction failed - no models created")
    
    # Get the largest reconstruction
    reconstruction = maps[max(maps.keys(), key=lambda k: len(maps[k].images))]
    
    colmap_time = time.time() - start_time
    print(f"[TIMING] COLMAP sparse reconstruction: {colmap_time:.2f}s")
    print(f"[INFO] Reconstructed {len(reconstruction.images)} images, {len(reconstruction.points3D)} points")
    
    return reconstruction, sparse_dir / str(max(maps.keys()))


def estimate_depth_batch(model, image_paths, device='cpu'):
    """
    Run ZoeDepth on multiple images.
    
    Returns:
        list of (depth_map, rgb_image, img_path) tuples
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
    print(f"[TIMING] Depth estimation ({len(image_paths)} images): {total_time:.2f}s ({total_time/len(image_paths):.2f}s per image)")
    
    return results


def create_depth_mask(depth, max_depth_percentile=95):
    """
    Create validity mask to filter out sky, invalid depths, and extreme outliers.
    
    Args:
        depth: depth map (H, W)
        max_depth_percentile: percentile threshold for max depth (filters sky)
    
    Returns:
        mask: boolean array same shape as depth
    """
    mask = np.ones_like(depth, dtype=bool)
    
    # Remove invalid depths
    mask &= depth > 0.01
    
    # Remove far outliers (likely sky)
    depth_threshold = np.percentile(depth[depth > 0.01], max_depth_percentile)
    mask &= depth < depth_threshold
    
    return mask


def reprojection_consistency_check(points_world, depth_neighbor, camera_neighbor, 
                                   colmap_img_neighbor, threshold_relative=0.05, threshold_abs=0.1):
    """
    Check if world points are consistent when reprojected into a neighboring view.
    
    Args:
        points_world: (N, 3) array of 3D points in world coordinates
        depth_neighbor: (H, W) metric depth map of neighbor view
        camera_neighbor: COLMAP camera object for neighbor
        colmap_img_neighbor: COLMAP image object for neighbor (contains pose)
        threshold_relative: relative depth agreement threshold (5% default - tightened)
        threshold_abs: absolute depth agreement threshold (0.1m default - tightened)
    
    Returns:
        keep: (N,) boolean mask of points that pass consistency check
    """
    if depth_neighbor is None:
        # No neighbor available, keep all points
        return np.ones(len(points_world), dtype=bool)
    
    # Get neighbor camera intrinsics (NO SCALING - ZoeDepth returns original resolution)
    fx, fy, cx, cy = get_zoe_intrinsics(camera_neighbor)
    
    # Get neighbor cam_from_world transform (use directly)
    cam_from_world = colmap_img_neighbor.cam_from_world()
    R_cw = cam_from_world.rotation.matrix()
    t_cw = cam_from_world.translation
    
    # Project world points into neighbor camera frame
    P_cam = (R_cw @ points_world.T).T + t_cw
    
    z_expected = P_cam[:, 2]
    u = (P_cam[:, 0] / z_expected) * fx + cx
    v = (P_cam[:, 1] / z_expected) * fy + cy
    
    H, W = depth_neighbor.shape
    
    # Check if projection is within image bounds
    valid_proj = (
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H) &
        (z_expected > 0.1)
    )
    
    keep = np.zeros(len(points_world), dtype=bool)
    
    # For valid projections, check depth agreement
    valid_idxs = np.where(valid_proj)[0]
    if len(valid_idxs) == 0:
        return keep
    
    u_int = u[valid_idxs].astype(int)
    v_int = v[valid_idxs].astype(int)
    
    # Sample neighbor depth at projected locations
    d_neighbor = depth_neighbor[v_int, u_int]
    z_exp = z_expected[valid_idxs]
    
    # Check agreement: difference < max(threshold_abs, threshold_relative * depth)
    diff = np.abs(d_neighbor - z_exp)
    threshold = np.maximum(threshold_abs, threshold_relative * z_exp)
    
    consistent = diff < threshold
    keep[valid_idxs[consistent]] = True
    
    return keep


def align_depth_to_colmap(depth_zoe, reconstruction, colmap_img, camera):
    """
    Align ZoeDepth scale to COLMAP metric scale using sparse 3D points.
    
    Returns:
        scale, bias such that depth_metric = scale * depth_zoe + bias
    """
    # Collect 2D-3D correspondences
    points2D_list = []
    points3D_world_list = []
    
    for i in range(colmap_img.num_points2D()):
        point2D = colmap_img.points2D[i]
        if point2D.has_point3D():
            # Get 2D position
            points2D_list.append(point2D.xy)
            # Get corresponding 3D point
            point3D = reconstruction.points3D[point2D.point3D_id]
            points3D_world_list.append(point3D.xyz)
    
    if len(points2D_list) < 10:
        print(f"[WARNING] Too few COLMAP correspondences ({len(points2D_list)}), using default scale")
        return 1.0, 0.0
    
    points2D = np.array(points2D_list)
    points3D_world = np.array(points3D_world_list)
    
    # Transform to camera frame
    # cam_from_world() returns cam_from_world transform (use directly)
    cam_from_world = colmap_img.cam_from_world()
    R_cw = cam_from_world.rotation.matrix()
    t_cw = cam_from_world.translation
    
    # Transform P_world → P_cam (direct application)
    points3D_cam = (R_cw @ points3D_world.T).T + t_cw
    
    # Get depth from COLMAP (z component in camera frame)
    z_colmap = points3D_cam[:, 2]
    
    # Sample ZoeDepth at 2D keypoint locations with strict masking
    H, W = depth_zoe.shape
    u = np.clip(points2D[:, 0].astype(int), 0, W - 1)
    v = np.clip(points2D[:, 1].astype(int), 0, H - 1)
    d_zoe = depth_zoe[v, u]
    
    # Apply strict depth masking to remove outliers
    depth_median = np.median(depth_zoe[depth_zoe > 0])
    depth_std = np.std(depth_zoe[depth_zoe > 0])
    
    # Filter out invalid/outlier correspondences with strict thresholds
    valid = (
        (z_colmap > 0.1) &  # Near clipping
        (z_colmap < 20.0) &  # Far clipping (20m max for indoor)
        (d_zoe > 0.01) &
        (d_zoe < 20.0) &
        (d_zoe < depth_median + 2 * depth_std)  # Statistical outlier removal
    )
    z_colmap = z_colmap[valid]
    d_zoe = d_zoe[valid]
    
    if len(z_colmap) < 10:
        print(f"[WARNING] Too few valid correspondences for alignment")
        return 1.0, 0.0
    
    # Robust RANSAC fitting: z_colmap = scale * d_zoe + bias
    # RANSAC is robust to outliers in low-texture regions (trees, walls, etc.)
    X = d_zoe.reshape(-1, 1)
    y = z_colmap
    
    ransac = RANSACRegressor(
        residual_threshold=0.2,  # 20cm inlier threshold
        max_trials=1000,
        random_state=42
    )
    ransac.fit(X, y)
    
    scale = ransac.estimator_.coef_[0]
    bias = ransac.estimator_.intercept_
    inlier_mask = ransac.inlier_mask_
    num_inliers = np.sum(inlier_mask)
    
    print(f"  Aligned: scale={scale:.3f}, bias={bias:.3f}, {num_inliers}/{len(y)} inliers (RANSAC)")
    
    return scale, bias


def fuse_depth_maps(depth_results, reconstruction, image_dir):
    """
    Fuse multiple depth maps into a single dense point cloud using COLMAP poses.
    
    Now includes:
    - Scale alignment to COLMAP sparse points
    - Voxel-based fusion with averaging
    - Multi-view consistency checking
    """
    print("[INFO] Fusing depth maps into point cloud...")
    start_time = time.time()
    
    # Create lookup from image name to COLMAP image
    image_lookup = {img.name: img for img in reconstruction.images.values()}
    
    # Step 1: Align all depth maps to COLMAP scale
    print("[INFO] Step 1: Aligning depth maps to COLMAP metric scale...")
    aligned_depths = []
    
    for depth_map, rgb_image, img_path in depth_results:
        img_name = Path(img_path).name
        
        if img_name not in image_lookup:
            print(f"[WARNING] No pose found for {img_name}, skipping")
            continue
        
        colmap_img = image_lookup[img_name]
        camera = reconstruction.cameras[colmap_img.camera_id]
        
        # Align depth to COLMAP scale
        scale, bias = align_depth_to_colmap(depth_map, reconstruction, colmap_img, camera)
        depth_metric = scale * depth_map + bias
        
        # Apply depth masking to remove sky and outliers
        depth_mask = create_depth_mask(depth_metric)
        
        aligned_depths.append((depth_metric, depth_mask, rgb_image, img_path, colmap_img, camera))
    
    print(f"[INFO] Aligned {len(aligned_depths)} depth maps")
    
    # Step 2: Backproject all points to world coordinates with reprojection consistency
    print("[INFO] Step 2: Backprojecting points with multi-view consistency checking...")
    all_points = []
    all_colors = []
    
    for view_id, (depth_metric, depth_mask, rgb_image, img_path, colmap_img, camera) in enumerate(aligned_depths):
        # Get camera intrinsics (NO SCALING - ZoeDepth returns original resolution)
        H, W = depth_metric.shape
        fx, fy, cx, cy = get_zoe_intrinsics(camera)
        
        # Create pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Backproject to camera space
        z = depth_metric
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points_camera = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # Transform to world space
        # cam_from_world() returns cam_from_world, invert to get world_from_cam
        cam_from_world = colmap_img.cam_from_world()
        R_cw = cam_from_world.rotation.matrix()
        t_cw = cam_from_world.translation
        
        # Invert to get world_from_cam transform
        R_wc = R_cw.T
        t_wc = -R_cw.T @ t_cw
        
        # Transform P_cam → P_world
        points_world = (R_wc @ points_camera.T).T + t_wc
        
        # Get colors
        rgb_array = np.array(rgb_image).reshape(-1, 3)
        
        # Apply combined mask: valid depth + depth mask
        valid_mask = (z.reshape(-1) > 0.1) & (z.reshape(-1) < 100) & depth_mask.reshape(-1)
        
        points_world_valid = points_world[valid_mask]
        colors_valid = rgb_array[valid_mask]
        
        # Apply reprojection consistency check with next neighbor
        if view_id + 1 < len(aligned_depths):
            depth_neighbor, _, _, _, colmap_img_neighbor, camera_neighbor = aligned_depths[view_id + 1]
            consistency_mask = reprojection_consistency_check(
                points_world_valid, 
                depth_neighbor, 
                camera_neighbor, 
                colmap_img_neighbor
            )
            points_world_valid = points_world_valid[consistency_mask]
            colors_valid = colors_valid[consistency_mask]
            consistent_count = consistency_mask.sum()
        else:
            consistent_count = len(points_world_valid)
        
        all_points.append(points_world_valid)
        all_colors.append(colors_valid)
        
        print(f"  View {view_id+1}/{len(aligned_depths)}: {valid_mask.sum():,} points → {consistent_count:,} after consistency check")
    
    # Concatenate all points
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    print(f"[INFO] Total raw points: {len(all_points):,}")
    
    # Step 3: Voxel-based fusion with averaging
    print("[INFO] Step 3: Fusing points with voxel grid (5cm resolution)...")
    voxel_size = 0.05  # 5cm voxels
    
    # Discretize points into voxel coordinates
    voxel_coords = np.floor(all_points / voxel_size).astype(np.int32)
    
    # Create unique voxel keys
    voxel_keys = {}
    for i in range(len(voxel_coords)):
        key = tuple(voxel_coords[i])
        if key not in voxel_keys:
            voxel_keys[key] = []
        voxel_keys[key].append(i)
    
    # Average points within each voxel
    fused_points = []
    fused_colors = []
    
    for key, indices in voxel_keys.items():
        # Average position and color
        avg_point = all_points[indices].mean(axis=0)
        avg_color = all_colors[indices].mean(axis=0)
        
        fused_points.append(avg_point)
        fused_colors.append(avg_color)
    
    fused_points = np.array(fused_points)
    fused_colors = np.array(fused_colors)
    
    fusion_time = time.time() - start_time
    print(f"[TIMING] Depth fusion: {fusion_time:.2f}s")
    print(f"[INFO] Fused point cloud: {len(fused_points):,} points (downsampled from {len(all_points):,})")
    
    return fused_points, fused_colors


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
    parser = argparse.ArgumentParser(description="Multi-view scene reconstruction with PyColmap + ZoeDepth")
    parser.add_argument('input', type=str, help="Input images directory or glob pattern")
    parser.add_argument('--output', type=str, default=None, help="Output directory")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--downsample', type=float, default=None, 
                       help="Further downsample voxel size in meters (e.g., 0.1 for 10cm) to reduce final point count")
    
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
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    total_start = time.time()
    
    # Step 1: COLMAP sparse reconstruction
    try:
        reconstruction, sparse_path = run_colmap_sparse(image_dir, output_dir)
    except Exception as e:
        print(f"[ERROR] COLMAP reconstruction failed: {e}")
        sys.exit(1)
    
    # Step 2: Load ZoeDepth
    print("[INFO] Loading ZoeDepth model...")
    zoedepth_model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    
    # Step 3: Estimate depth for each image
    depth_results = estimate_depth_batch(zoedepth_model, image_paths, device=args.device)
    
    # Step 4: Fuse depth maps using COLMAP poses
    fused_points, fused_colors = fuse_depth_maps(depth_results, reconstruction, image_dir)
    
    # Step 5: Optional further downsampling
    if args.downsample:
        print(f"[INFO] Applying additional downsampling (voxel size: {args.downsample}m)...")
        voxel_size = args.downsample
        
        # Discretize to voxel grid
        voxel_coords = np.floor(fused_points / voxel_size).astype(np.int32)
        
        # Average within each voxel
        voxel_dict = {}
        for i in range(len(voxel_coords)):
            key = tuple(voxel_coords[i])
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)
        
        downsampled_points = []
        downsampled_colors = []
        for indices in voxel_dict.values():
            downsampled_points.append(fused_points[indices].mean(axis=0))
            downsampled_colors.append(fused_colors[indices].mean(axis=0))
        
        fused_points = np.array(downsampled_points)
        fused_colors = np.array(downsampled_colors)
        
        print(f"[INFO] Downsampled to {len(fused_points):,} points")
    
    # Step 6: Save output
    pointcloud_path = output_dir / 'fused_pointcloud.ply'
    save_pointcloud_ply(fused_points, fused_colors, pointcloud_path)
    
    # Save COLMAP model for reference
    colmap_output = output_dir / 'colmap_sparse'
    colmap_output.mkdir(exist_ok=True)
    reconstruction.write(str(colmap_output))
    
    total_time = time.time() - total_start
    print(f"\n[COMPLETE] Total runtime: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"[COMPLETE] Output saved to: {output_dir}")
    print(f"[COMPLETE] Point cloud: {pointcloud_path}")
    print(f"\n[NEXT] Load result in viewer:")
    print(f"  http://localhost:5000/viewer/index.html")
    print(f"  Use 'Load Folder' -> select: {output_dir}")


if __name__ == '__main__':
    main()
