#!/usr/bin/env python3
"""Quick test to verify COLMAP pose convention"""

import pycolmap
from pathlib import Path
import numpy as np

# Load existing reconstruction
sparse_path = Path("outputs/boulders_corrected/colmap_sparse")
reconstruction = pycolmap.Reconstruction(str(sparse_path))

# Get first image with poses
for img_id, colmap_img in list(reconstruction.images.items())[:3]:
    print(f"\n{'='*70}")
    print(f"Image: {colmap_img.name}")
    print(f"Image ID: {img_id}")
    
    # Get the pose
    pose = colmap_img.cam_from_world()
    R = pose.rotation.matrix()
    t = pose.translation
    
    print(f"\nPose object type: {type(pose)}")
    print(f"Pose: {pose}")
    
    print(f"\nRotation matrix (3x3):")
    print(R)
    
    print(f"\nTranslation vector:")
    print(t)
    
    print(f"\nTranslation norm (camera distance from origin): {np.linalg.norm(t):.2f}")
    
    # Get a 3D point visible in this image
    for i in range(colmap_img.num_points2D()):
        point2D = colmap_img.points2D[i]
        if point2D.has_point3D():
            point3D = reconstruction.points3D[point2D.point3D_id]
            P_world = point3D.xyz
            
            print(f"\nExample 3D point:")
            print(f"  P_world = {P_world}")
            
            # Test transform: P_world → P_cam (assuming pose is world_from_cam)
            # If pose is world_from_cam, we need to invert
            R_wc = R
            t_wc = t
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
            
            P_cam_inverted = R_cw @ P_world + t_cw
            print(f"  P_cam (using inversion) = {P_cam_inverted}")
            print(f"  Depth (z) = {P_cam_inverted[2]:.2f}")
            
            # Test transform: P_world → P_cam (assuming pose is cam_from_world)
            # If pose is cam_from_world, use directly
            R_cw_direct = R
            t_cw_direct = t
            
            P_cam_direct = R_cw_direct @ P_world + t_cw_direct
            print(f"  P_cam (using direct) = {P_cam_direct}")
            print(f"  Depth (z) = {P_cam_direct[2]:.2f}")
            
            print(f"\n  --> If depth is POSITIVE and reasonable (1-100m), that method is CORRECT")
            print(f"  --> If depth is NEGATIVE, that method is WRONG")
            
            # Now check which projects correctly to the 2D point
            camera = reconstruction.cameras[colmap_img.camera_id]
            fx = fy = camera.params[0]  # SIMPLE_RADIAL
            cx = camera.params[1]
            cy = camera.params[2]
            
            # Project using inverted pose
            u_inv = (P_cam_inverted[0] / P_cam_inverted[2]) * fx + cx
            v_inv = (P_cam_inverted[1] / P_cam_inverted[2]) * fy + cy
            
            # Project using direct pose
            u_dir = (P_cam_direct[0] / P_cam_direct[2]) * fx + cx
            v_dir = (P_cam_direct[1] / P_cam_direct[2]) * fy + cy
            
            # Actual 2D point
            u_actual = point2D.xy[0]
            v_actual = point2D.xy[1]
            
            print(f"\n  2D Projection Test:")
            print(f"    Actual 2D point: ({u_actual:.1f}, {v_actual:.1f})")
            print(f"    Projected (inverted): ({u_inv:.1f}, {v_inv:.1f}) - error: {np.sqrt((u_inv-u_actual)**2 + (v_inv-v_actual)**2):.2f} px")
            print(f"    Projected (direct): ({u_dir:.1f}, {v_dir:.1f}) - error: {np.sqrt((u_dir-u_actual)**2 + (v_dir-v_actual)**2):.2f} px")
            print(f"\n  --> WHICHEVER HAS LOWER ERROR IS THE CORRECT TRANSFORM")
            
            break  # Only test first point
    
    break  # Only test first image

print(f"\n{'='*70}")
print("CONCLUSION:")
print("If 'using inversion' gives positive depth → pose is world_from_cam")
print("If 'using direct' gives positive depth → pose is cam_from_world")
