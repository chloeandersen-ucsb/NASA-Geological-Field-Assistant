import cv2
from datetime import datetime
import json
import os
from dataclasses import dataclass

import numpy as np

# Dual-view volume estimation
from volume_estimation_dual_view import estimate_volume_dual_view


@dataclass
class MeasureConfig:
    # Physical size of the AprilTag (side length) in centimeters
    reference_real_size_cm: float = 10.0  # 100 mm tag36h11
    density_min_g_cm3: float = 2.5
    density_max_g_cm3: float = 3.0


def save_json(out_path, data):
    """Save data to JSON file, converting numpy types to native Python types."""
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        return obj
    
    data = convert_to_native(data)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    import argparse

    # Generate timestamp for default output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = os.path.join("outputs", f"{timestamp}-measurement.json")

    parser = argparse.ArgumentParser(description="Automatic dual-view rock volume measurement")
    parser.add_argument("--top", required=True, help="Top-down image path (required)")
    parser.add_argument("--side", required=True, help="Side image path (required)")
    parser.add_argument("--ref_cm", type=float, default=10.0, help="AprilTag side length in cm (tag36h11, default 100mm=10cm).")
    parser.add_argument("--dens_min", type=float, default=2.5, help="Min density in g/cm^3.")
    parser.add_argument("--dens_max", type=float, default=3.0, help="Max density in g/cm^3.")
    parser.add_argument("--out", default=default_output, help="Output JSON path (default: outputs/TIMESTAMP-measurement.json).")
    args = parser.parse_args()

    cfg = MeasureConfig(
        reference_real_size_cm=args.ref_cm,
        density_min_g_cm3=args.dens_min,
        density_max_g_cm3=args.dens_max,
    )

    # Check for required dependencies on first run
    sam_model_path = os.path.expanduser("~/.sam_models/sam_vit_b.pth")
    intrinsics_path = "camera_intrinsics.json"
    
    if not os.path.exists(sam_model_path):
        print("\n" + "="*70)
        print("SETUP REQUIRED: SAM Model Not Found")
        print("="*70)
        print("\nTo use automatic segmentation, download the SAM model:")
        print("  python download_sam.py")
        print("\nThis will download ~375 MB to ~/.sam_models/")
        print("="*70 + "\n")
        raise RuntimeError(f"SAM model not found at {sam_model_path}")
    
    if not os.path.exists(intrinsics_path):
        print("\n" + "="*70)
        print("OPTIONAL: Camera Calibration Not Found")
        print("="*70)
        print(f"\nFor best results, calibrate your camera:")
        print("  1. Print the ChArUco board: python charuco_calibration.py --board_only")
        print("  2. Capture 20+ calibration images of the board at different angles")
        print("  3. Run: python charuco_calibration.py --images <calib_dir>")
        print(f"\nThis will create {intrinsics_path}")
        print("Proceeding with default scaling (AprilTag only)...")
        print("="*70 + "\n")

    # ====================================================================
    # AUTOMATIC DUAL-VIEW MODE
    # ====================================================================
    print("\n[AUTO] Loading top and side images...")
    top_image = cv2.imread(args.top)
    side_image = cv2.imread(args.side)
    
    if top_image is None:
        raise RuntimeError(f"Could not read top image: {args.top}")
    if side_image is None:
        raise RuntimeError(f"Could not read side image: {args.side}")
    
    print("[AUTO] Running dual-view volume estimation...")
    volume_result = estimate_volume_dual_view(
        top_image=top_image,
        side_image=side_image,
        tag_side_cm=cfg.reference_real_size_cm,
        intrinsics_path=intrinsics_path if os.path.exists(intrinsics_path) else None,
        method="adaptive"
    )
    
    L_cm = volume_result.length_cm
    W_cm = volume_result.width_cm
    H_cm = volume_result.height_cm
    V_cm3 = volume_result.estimated_volume_cm3
    shape_factor = volume_result.shape_factor
    volume_method = volume_result.volume_method
    
    scale_info = {
        "mode": "auto_stereo_dual_view",
        "reference_real_size_cm": cfg.reference_real_size_cm,
        "volume_method": volume_method,
        "shape_factor": float(shape_factor),
        "segmentation_method": "SAM_center_prompt",
    }
    inputs_info = {
        "top_image": args.top,
        "side_image": args.side,
        "reference_real_size_cm": cfg.reference_real_size_cm,
        "density_min_g_cm3": cfg.density_min_g_cm3,
        "density_max_g_cm3": cfg.density_max_g_cm3,
    }

    mass_min_g = V_cm3 * cfg.density_min_g_cm3
    mass_max_g = V_cm3 * cfg.density_max_g_cm3

    results = {
        "inputs": inputs_info,
        "scale": scale_info,
        "dimensions_cm": {"L": L_cm, "W": W_cm, "H": H_cm},
        "volume_cm3": V_cm3,
        "mass_range": {
            "min_g": mass_min_g,
            "max_g": mass_max_g,
            "min_kg": mass_min_g / 1000.0,
            "max_kg": mass_max_g / 1000.0,
        },
        "model": volume_method,
    }

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_json(args.out, results)

    print("\n" + "="*70)
    print("MEASUREMENT COMPLETE")
    print("="*70)
    print(f"L × W × H (cm): {L_cm:.2f} × {W_cm:.2f} × {H_cm:.2f}")
    print(f"Volume (cm³): {V_cm3:.1f}")
    print(f"Mass range (kg): {mass_min_g/1000.0:.3f} to {mass_max_g/1000.0:.3f}")
    print(f"Saved: {args.out}")
    print("="*70)


if __name__ == "__main__":
    main()
