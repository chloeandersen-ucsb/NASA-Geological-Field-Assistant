#!/usr/bin/env python3
"""
ZoeDepth Single-Image Scene Reconstruction

Generates 3D point cloud from a single RGB image using ZoeDepth for metric depth estimation.
Designed for NASA GFA field deployment on Jetson Orin NX.

Usage:
    python generate_scene_zoedepth.py input.jpg --output output_dir/
    python generate_scene_zoedepth.py input.jpg --device cuda --camera-fov 60
"""

import argparse
import os
import sys
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
import cv2

# Optional: Open3D for mesh reconstruction
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def setup_zoedepth():
    """Load ZoeDepth model from torch hub."""
    print("[INFO] Loading ZoeDepth model...")
    # ZoeDepth NK (NYU + KITTI) - best for mixed indoor/outdoor
    # Use ZoeD_N (NYU only) to avoid compatibility issues
    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
    return model


def estimate_depth(model, image_path, device='cpu'):
    """
    Run depth estimation on image.
    
    Returns:
        depth_map: numpy array (H, W) with metric depth values in meters
    """
    print(f"[INFO] Processing {image_path}...")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        depth = model.infer_pil(image)
    inference_time = time.time() - start_time
    
    print(f"[TIMING] Depth estimation: {inference_time:.3f}s")
    
    return depth, image


def inpaint_depth(depth_map, method='telea', inpaint_radius=3, detect_edges=False):
    """
    Fill gaps/holes in depth map using inpainting.
    
    Args:
        depth_map: (H, W) depth array with invalid regions (0, inf, or nan)
        method: 'telea' (fast, smooth) or 'ns' (Navier-Stokes, slower, more detail)
        inpaint_radius: Radius of circular neighborhood for inpainting
        detect_edges: If True, also inpaint sharp depth discontinuities (helps with object boundaries)
    
    Returns:
        inpainted_depth: (H, W) depth array with filled gaps
    """
    print("[INFO] Inpainting depth map to fill gaps...")
    start_time = time.time()
    
    # Create mask of invalid pixels (0, inf, nan)
    invalid_mask = np.logical_or(
        np.logical_or(depth_map <= 0, np.isinf(depth_map)),
        np.isnan(depth_map)
    ).astype(np.uint8)
    
    # Optionally detect edges/discontinuities to inpaint
    if detect_edges:
        # Normalize depth for edge detection
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        depth_8u = (depth_norm * 255).astype(np.uint8)
        
        # Detect edges using Sobel
        sobelx = cv2.Sobel(depth_8u, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth_8u, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Threshold for strong edges (depth discontinuities)
        edge_threshold = np.percentile(gradient_mag, 95)  # Top 5% of gradients
        edge_mask = (gradient_mag > edge_threshold).astype(np.uint8)
        
        # Dilate edges slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)
        
        # Combine with invalid mask
        invalid_mask = np.logical_or(invalid_mask, edge_mask).astype(np.uint8)
    
    # Normalize depth to 0-255 for OpenCV
    valid_depths = depth_map[invalid_mask == 0]
    if len(valid_depths) == 0:
        print("[WARNING] No valid depth values found!")
        return depth_map
    
    depth_min, depth_max = valid_depths.min(), valid_depths.max()
    depth_norm = np.zeros_like(depth_map, dtype=np.float32)
    depth_norm[invalid_mask == 0] = (depth_map[invalid_mask == 0] - depth_min) / (depth_max - depth_min + 1e-6)
    depth_norm = (depth_norm * 255).astype(np.uint8)
    
    # Apply inpainting
    if method == 'telea':
        inpainted = cv2.inpaint(depth_norm, invalid_mask, inpaint_radius, cv2.INPAINT_TELEA)
    elif method == 'ns':
        inpainted = cv2.inpaint(depth_norm, invalid_mask, inpaint_radius, cv2.INPAINT_NS)
    else:
        raise ValueError(f"Unknown inpaint method: {method}")
    
    # Denormalize back to original depth range
    inpainted_depth = (inpainted.astype(np.float32) / 255.0) * (depth_max - depth_min) + depth_min
    
    # Preserve original valid values
    inpainted_depth[invalid_mask == 0] = depth_map[invalid_mask == 0]
    
    inpaint_time = time.time() - start_time
    num_inpainted = invalid_mask.sum()
    total_pixels = invalid_mask.size
    print(f"[TIMING] Depth inpainting: {inpaint_time:.3f}s")
    print(f"[INFO] Inpainted {num_inpainted:,} pixels ({num_inpainted/total_pixels*100:.1f}%)")
    
    return inpainted_depth


def depth_to_pointcloud(depth_map, rgb_image, camera_fov=60, max_depth=10.0):
    """
    Convert depth map and RGB image to 3D point cloud.
    
    Args:
        depth_map: (H, W) depth in meters
        rgb_image: PIL Image
        camera_fov: Field of view in degrees
        max_depth: Maximum depth to include (meters)
    
    Returns:
        points: (N, 3) XYZ coordinates
        colors: (N, 3) RGB colors [0-255]
    """
    print("[INFO] Converting depth map to point cloud...")
    start_time = time.time()
    
    H, W = depth_map.shape
    
    # Camera intrinsics (estimated from FOV)
    fov_rad = np.deg2rad(camera_fov)
    focal_length = (W / 2.0) / np.tan(fov_rad / 2.0)
    cx, cy = W / 2.0, H / 2.0
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Backproject to 3D
    z = depth_map
    x = (u - cx) * z / focal_length
    y = (v - cy) * z / focal_length
    
    # Filter by max depth
    valid_mask = (z > 0) & (z < max_depth)
    
    points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)
    
    # Get colors
    rgb_array = np.array(rgb_image)
    colors = rgb_array[valid_mask]
    
    conversion_time = time.time() - start_time
    print(f"[TIMING] Point cloud conversion: {conversion_time:.3f}s")
    print(f"[INFO] Generated {len(points):,} points")
    
    return points, colors


def save_pointcloud_ply(points, colors, output_path):
    """Save point cloud as PLY file."""
    print(f"[INFO] Saving point cloud to {output_path}...")
    
    with open(output_path, 'w') as f:
        # PLY header
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
        
        # Vertex data
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    
    print(f"[OK] Saved {len(points):,} points")


def reconstruct_mesh(points, colors, depth=9, output_path=None):
    """
    Reconstruct a mesh surface from point cloud using Poisson reconstruction.
    Requires open3d to be installed.
    
    Args:
        points: (N, 3) point coordinates
        colors: (N, 3) RGB colors [0-255]
        depth: Octree depth for Poisson (higher = more detail, 8-10 typical)
        output_path: If provided, save mesh to this path
    
    Returns:
        mesh: Open3D TriangleMesh or None if Open3D not available
    """
    if not HAS_OPEN3D:
        print("[WARNING] Open3D not installed, skipping mesh reconstruction")
        print("[INFO] Install with: pip install open3d")
        return None
    
    print("[INFO] Reconstructing mesh surface from point cloud...")
    start_time = time.time()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize to [0, 1]
    
    # Estimate normals (required for Poisson)
    print("[INFO] Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # Poisson surface reconstruction
    print(f"[INFO] Running Poisson reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=False
    )
    
    # Remove low-density vertices (noise)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.01)  # Remove bottom 1%
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Clean up mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    reconstruction_time = time.time() - start_time
    print(f"[TIMING] Mesh reconstruction: {reconstruction_time:.3f}s")
    print(f"[INFO] Generated mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    
    # Save if requested
    if output_path:
        # Save as PLY to preserve vertex colors (OBJ vertex colors not well supported in Three.js)
        output_path_str = str(output_path)
        if output_path_str.endswith('.obj'):
            output_path_ply = output_path_str.replace('.obj', '.ply')
            o3d.io.write_triangle_mesh(output_path_ply, mesh)
            print(f"[OK] Saved mesh to {output_path_ply}")
        else:
            o3d.io.write_triangle_mesh(output_path_str, mesh)
            print(f"[OK] Saved mesh to {output_path}")
    
    return mesh


def save_depth_visualization(depth_map, output_path):
    """Save depth map as grayscale image."""
    # Normalize to 0-255
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_image = (depth_normalized * 255).astype(np.uint8)
    
    Image.fromarray(depth_image).save(output_path)
    print(f"[OK] Saved depth visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ZoeDepth single-image scene reconstruction")
    parser.add_argument('input', type=str, help="Input image path")
    parser.add_argument('--output', type=str, default=None, help="Output directory (default: outputs/<input_name>_scene/)")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to run on")
    parser.add_argument('--camera-fov', type=float, default=60.0, help="Camera field of view in degrees")
    parser.add_argument('--max-depth', type=float, default=10.0, help="Maximum depth in meters")
    parser.add_argument('--inpaint', action='store_true', help="Apply depth inpainting to fill gaps")
    parser.add_argument('--inpaint-method', type=str, default='telea', choices=['telea', 'ns'], 
                        help="Inpainting method: telea (fast, smooth) or ns (Navier-Stokes, detailed)")
    parser.add_argument('--inpaint-radius', type=int, default=5, help="Inpainting radius in pixels")
    parser.add_argument('--inpaint-edges', action='store_true', help="Also inpaint sharp depth discontinuities")
    parser.add_argument('--mesh', action='store_true', help="Generate mesh surface (requires open3d)")
    parser.add_argument('--mesh-depth', type=int, default=9, help="Poisson octree depth for meshing (8-10 typical)")
    
    args = parser.parse_args()
    
    # Check input
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)
    
    # Setup output directory
    if args.output is None:
        input_name = Path(args.input).stem
        output_dir = Path('outputs') / f"{input_name}_scene"
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    total_start = time.time()
    
    # Load model
    model = setup_zoedepth()
    
    # Estimate depth
    depth_map, rgb_image = estimate_depth(model, args.input, device=args.device)
    
    # Apply inpainting if requested
    if args.inpaint:
        depth_map = inpaint_depth(
            depth_map, 
            method=args.inpaint_method, 
            inpaint_radius=args.inpaint_radius,
            detect_edges=args.inpaint_edges
        )
    
    # Save depth visualization
    depth_viz_path = output_dir / 'depth.png'
    save_depth_visualization(depth_map, depth_viz_path)
    
    # Convert to point cloud
    points, colors = depth_to_pointcloud(
        depth_map, 
        rgb_image, 
        camera_fov=args.camera_fov,
        max_depth=args.max_depth
    )
    
    # Save point cloud
    pointcloud_path = output_dir / 'pointcloud.ply'
    save_pointcloud_ply(points, colors, pointcloud_path)
    
    # Optionally reconstruct mesh
    if args.mesh:
        mesh_path = output_dir / 'mesh.ply'  # Changed from .obj to .ply for better color support
        reconstruct_mesh(points, colors, depth=args.mesh_depth, output_path=mesh_path)
    
    # Copy input image
    import shutil
    input_copy_path = output_dir / 'input.png'
    shutil.copy(args.input, input_copy_path)
    
    total_time = time.time() - total_start
    print(f"\n[COMPLETE] Total runtime: {total_time:.2f}s")
    print(f"[COMPLETE] Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
