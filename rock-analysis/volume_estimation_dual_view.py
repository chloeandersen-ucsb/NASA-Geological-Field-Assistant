"""
Dual View Volume Estimation Module

A fully isolated module for estimating object volume using two orthogonal views
(top and side) with AprilTag-based metric scaling. No global state dependencies.
"""

from dataclasses import dataclass
import json
from typing import Optional, Tuple

import cv2
import numpy as np

# Import only the existing functions (no modification)
from measure_rock import detect_apriltag_side_px, segment_object


@dataclass
class ViewMeasurement:
    """Result of processing a single view (top or side)."""
    mask: np.ndarray              # Binary mask (uint8, 0 or 1)
    scale_cm_per_pixel: float     # Metric scale derived from AprilTag
    rectified_image: Optional[np.ndarray]  # Rectified image for visualization


@dataclass
class DualViewVolumeResult:
    """Final volume estimation result."""
    footprint_area_cm2: float
    height_cm: float
    shape_factor: float
    estimated_volume_cm3: float
    volume_method: str = "adaptive"  # Method used for volume calculation
    length_cm: Optional[float] = None  # Extracted from top view
    width_cm: Optional[float] = None   # Extracted from top view


# ============================================================================
# VIEW PROCESSING
# ============================================================================

def _get_tag_corners_pixels(image: np.ndarray, tag_dict_name: str = "DICT_APRILTAG_36h11") -> tuple:
    """
    Detect AprilTag and return corners in pixels.
    
    Args:
        image: BGR image
        tag_dict_name: ArUco dictionary name
    
    Returns:
        Tuple of (avg_side_px, corners_list, tag_id)
        Raises RuntimeError if tag not detected
    """
    avg_side_px, corners, tag_id = detect_apriltag_side_px(image, tag_dict_name)
    return avg_side_px, corners, tag_id


def _load_intrinsics(intrinsics_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    """
    Load camera intrinsics from a JSON file.

    Args:
        intrinsics_path: Path to JSON with "camera_matrix" and "dist_coeffs"

    Returns:
        Tuple of (camera_matrix, dist_coeffs, image_size)
    """
    with open(intrinsics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    camera_matrix = np.asarray(data.get("camera_matrix"), dtype=np.float32)
    dist_coeffs = np.asarray(data.get("dist_coeffs"), dtype=np.float32).reshape(-1)

    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must be 3x3")
    if dist_coeffs.size < 4:
        raise ValueError("dist_coeffs must have at least 4 values")

    image_size = None
    size_info = data.get("image_size")
    if isinstance(size_info, dict) and "width" in size_info and "height" in size_info:
        image_size = (int(size_info["width"]), int(size_info["height"]))

    return camera_matrix, dist_coeffs, image_size


def _rescale_intrinsics(
    camera_matrix: np.ndarray,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Rescale intrinsics when the calibration image size differs from the target image size.

    Args:
        camera_matrix: 3x3 camera matrix
        source_size: (width, height) used for calibration
        target_size: (width, height) of current image

    Returns:
        Rescaled 3x3 camera matrix
    """
    src_w, src_h = source_size
    tgt_w, tgt_h = target_size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("source_size must be positive")

    scale_x = float(tgt_w) / float(src_w)
    scale_y = float(tgt_h) / float(src_h)

    K = camera_matrix.copy().astype(np.float32)
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y
    return K


def _rotate_to_match_calib(
    image: np.ndarray,
    calib_size: Optional[Tuple[int, int]]
) -> np.ndarray:
    """
    Rotate the image to match calibration orientation, if needed.

    Args:
        image: Input image
        calib_size: (width, height) of calibration images

    Returns:
        Rotated image (if needed) aligned to calibration orientation
    """
    if calib_size is None:
        return image

    h, w = image.shape[:2]
    calib_w, calib_h = calib_size
    image_landscape = w >= h
    calib_landscape = calib_w >= calib_h

    if image_landscape == calib_landscape:
        return image

    if calib_landscape:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def _undistort_image(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Undistort an image using camera intrinsics.

    Args:
        image: Input BGR image
        camera_matrix: 3x3 camera matrix
        dist_coeffs: Distortion coefficients

    Returns:
        Undistorted image
    """
    return cv2.undistort(image, camera_matrix, dist_coeffs)


def _compute_homography_to_plane(tag_corners_pixels: list, tag_side_cm: float) -> np.ndarray:
    """
    Compute homography matrix from image space to rectified (metric) space.

    Assumes the AprilTag lies in the plane of interest (object base for top view,
    vertical plane for side view).

    Args:
        tag_corners_pixels: List of 4 corner points [[x0,y0], [x1,y1], ...]
        tag_side_cm: Physical side length of the tag in cm

    Returns:
        Homography matrix (3x3) mapping image pixels to metric plane
    """
    image_pts = np.asarray(tag_corners_pixels, dtype=np.float32)
    
    if image_pts.shape != (4, 2):
        raise ValueError("Expected 4 corner points for AprilTag")
    
    # Define canonical square in metric space centered at origin
    # Corners: top-left, top-right, bottom-right, bottom-left
    s = tag_side_cm
    metric_pts = np.array(
        [
            [-s / 2, s / 2],       # top-left
            [s / 2, s / 2],        # top-right
            [s / 2, -s / 2],       # bottom-right
            [-s / 2, -s / 2],      # bottom-left
        ],
        dtype=np.float32,
    )
    
    H, _ = cv2.findHomography(image_pts, metric_pts)
    if H is None:
        raise RuntimeError("Homography estimation failed")
    
    return H


def _compute_homography_to_square(tag_corners_pixels: list, square_center: np.ndarray, square_size_px: float) -> np.ndarray:
    """
    Compute homography that maps the detected tag corners to a square in pixel space.

    Args:
        tag_corners_pixels: List of 4 corner points [[x0,y0], [x1,y1], ...]
        square_center: Center of the output square (x, y)
        square_size_px: Side length of the output square in pixels

    Returns:
        Homography matrix (3x3) mapping image pixels to rectified pixel plane
    """
    image_pts = np.asarray(tag_corners_pixels, dtype=np.float32)
    if image_pts.shape != (4, 2):
        raise ValueError("Expected 4 corner points for AprilTag")

    half = float(square_size_px) / 2.0
    cx, cy = float(square_center[0]), float(square_center[1])
    dest_pts = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(image_pts, dest_pts)
    if H is None:
        raise RuntimeError("Homography estimation failed")

    return H


def _rectify_image(image: np.ndarray, homography: np.ndarray, output_size: int = 256) -> np.ndarray:
    """
    Apply perspective transform using homography.
    
    Args:
        image: Input image
        homography: Homography matrix
        output_size: Size of output square image
    
    Returns:
        Rectified image (for visualization only; not used for measurements)
    """
    # Note: This is for visualization. Actual measurements use perspectiveTransform on points.
    # We use the inverse homography for backward mapping (warpPerspective requirement)
    H_inv = np.linalg.inv(homography)
    
    rectified = cv2.warpPerspective(
        image,
        H_inv,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return rectified


def _warp_mask(mask: np.ndarray, homography: np.ndarray, output_size: int = 256) -> np.ndarray:
    """
    Apply perspective transform to a binary mask.
    
    Args:
        mask: Binary mask (0 or 1)
        homography: Homography matrix
        output_size: Size of output square image
    
    Returns:
        Rectified mask (binary, for visualization)
    """
    H_inv = np.linalg.inv(homography)
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
    warped = cv2.warpPerspective(
        mask_uint8,
        H_inv,
        (output_size, output_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    # Convert back to binary
    return (warped > 127).astype(np.uint8)


def process_top_view(
    image: np.ndarray,
    tag_side_cm: float = 10.0,
    intrinsics_path: Optional[str] = None
) -> ViewMeasurement:
    """
    Process top view image to extract footprint.
    
    Steps:
    1. Rotate to portrait if landscape
    2. Undistort image using camera intrinsics
    3. Detect AprilTag and extract corners
    4. Segment object mask
    5. Compute scale in cm/pixel from detected tag size
    
    Args:
        image: BGR image (top-down view)
        tag_side_cm: Physical side length of AprilTag in cm (default 10.0 for tag36h11)
    
    Returns:
        ViewMeasurement with mask and scale
    
    Raises:
        RuntimeError if AprilTag not detected
    """
    # Step 1: Rotate to portrait if needed
    h, w = image.shape[:2]
    if w > h:
        image_oriented = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        image_oriented = image

    # Step 2: Undistort
    if intrinsics_path is not None:
        camera_matrix, dist_coeffs, calib_size = _load_intrinsics(intrinsics_path)
        h0, w0 = image_oriented.shape[:2]
        if calib_size is not None and calib_size != (w0, h0):
            camera_matrix = _rescale_intrinsics(camera_matrix, calib_size, (w0, h0))
        image_proc = _undistort_image(image_oriented, camera_matrix, dist_coeffs)
    else:
        image_proc = image_oriented

    # Step 3: Detect tag
    avg_side_px, tag_corners, tag_id = _get_tag_corners_pixels(image_proc)

    # Step 4: Segment object
    mask = segment_object(image_proc, tag_corners=tag_corners)

    # Step 5: Compute scale from detected tag size
    scale_cm_per_pixel = tag_side_cm / avg_side_px

    return ViewMeasurement(
        mask=mask,
        scale_cm_per_pixel=scale_cm_per_pixel,
        rectified_image=image_proc
    )


def process_side_view(
    image: np.ndarray,
    tag_side_cm: float = 10.0,
    intrinsics_path: Optional[str] = None
) -> ViewMeasurement:
    """
    Process side view image to extract height profile.
    
    Steps:
    1. Rotate to portrait if landscape
    2. Undistort image using camera intrinsics
    3. Detect AprilTag and extract corners
    4. Segment object mask
    5. Compute scale in cm/pixel from detected tag size
    
    Args:
        image: BGR image (side view)
        tag_side_cm: Physical side length of AprilTag in cm (default 10.0 for tag36h11)
    
    Returns:
        ViewMeasurement with mask and scale
    
    Raises:
        RuntimeError if AprilTag not detected
    """
    # Step 1: Rotate to portrait if needed
    h, w = image.shape[:2]
    if w > h:
        image_oriented = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        image_oriented = image

    # Step 2: Undistort
    if intrinsics_path is not None:
        camera_matrix, dist_coeffs, calib_size = _load_intrinsics(intrinsics_path)
        h0, w0 = image_oriented.shape[:2]
        if calib_size is not None and calib_size != (w0, h0):
            camera_matrix = _rescale_intrinsics(camera_matrix, calib_size, (w0, h0))
        image_proc = _undistort_image(image_oriented, camera_matrix, dist_coeffs)
    else:
        image_proc = image_oriented

    # Step 3: Detect tag
    avg_side_px, tag_corners, tag_id = _get_tag_corners_pixels(image_proc)

    # Step 4: Segment object
    mask = segment_object(image_proc, tag_corners=tag_corners)

    # Step 5: Compute scale from detected tag size
    scale_cm_per_pixel = tag_side_cm / avg_side_px

    return ViewMeasurement(
        mask=mask,
        scale_cm_per_pixel=scale_cm_per_pixel,
        rectified_image=image_proc
    )


# ============================================================================
# MEASUREMENT FUNCTIONS
# ============================================================================

def compute_footprint_area(mask: np.ndarray, scale_cm_per_pixel: float) -> float:
    """
    Compute 2D footprint area from mask.
    
    Args:
        mask: Binary segmentation mask (0 or 1)
        scale_cm_per_pixel: Metric scale factor
    
    Returns:
        Area in cm²
    """
    pixel_area = int(np.count_nonzero(mask))
    area_cm2 = pixel_area * (scale_cm_per_pixel ** 2)
    return float(area_cm2)


def compute_object_height(mask: np.ndarray, scale_cm_per_pixel: float) -> float:
    """
    Compute object height from side view mask.
    
    Finds the vertical extent (top to bottom pixels) and scales to metric space.
    
    Args:
        mask: Binary segmentation mask from side view
        scale_cm_per_pixel: Metric scale factor
    
    Returns:
        Height in cm
    """
    ys, xs = np.where(mask > 0)
    
    if len(ys) == 0:
        raise ValueError("No object pixels in mask")
    
    # Find topmost and bottommost pixels
    y_min = int(ys.min())
    y_max = int(ys.max())
    
    pixel_height = y_max - y_min + 1
    height_cm = float(pixel_height * scale_cm_per_pixel)
    
    return height_cm


def extract_principal_axes(mask: np.ndarray, scale_cm_per_pixel: float) -> Tuple[float, float]:
    """
    Extract length and width from mask using PCA on the contour points.
    
    Args:
        mask: Binary segmentation mask
        scale_cm_per_pixel: Metric scale factor
    
    Returns:
        Tuple of (length_cm, width_cm) where length >= width
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")
    
    contour = max(contours, key=cv2.contourArea)
    
    # Use minimum area rectangle for axis-aligned dimensions
    rect = cv2.minAreaRect(contour)
    (_, _), (w_px, h_px), _ = rect
    
    # Ensure length >= width
    length_px = max(w_px, h_px)
    width_px = min(w_px, h_px)
    
    length_cm = float(length_px * scale_cm_per_pixel)
    width_cm = float(width_px * scale_cm_per_pixel)
    
    return length_cm, width_cm


def compute_ellipsoid_volume(length_cm: float, width_cm: float, height_cm: float) -> float:
    """
    Compute volume assuming ellipsoid shape.
    
    Formula: V = (4/3) * π * (L/2) * (W/2) * (H/2)
    
    Args:
        length_cm: Length (longest axis)
        width_cm: Width (intermediate axis)
        height_cm: Height (shortest axis)
    
    Returns:
        Volume in cm³
    """
    a = length_cm / 2.0
    b = width_cm / 2.0
    c = height_cm / 2.0
    
    volume = (4.0 / 3.0) * np.pi * a * b * c
    return float(volume)


def estimate_shape_factor(mask_top: np.ndarray, mask_side: np.ndarray) -> float:
    """
    Estimate shape factor to account for non-rectangular objects.
    
    Analyzes circularity and solidity from both views to estimate
    how the object's volume relates to its bounding box.
    
    A shape factor of 1.0 means perfect rectangular box.
    Real objects (spheres, ellipsoids, etc.) have lower factors.
    
    Args:
        mask_top: Top view binary mask
        mask_side: Side view binary mask
    
    Returns:
        Shape factor (0 < factor <= 1)
    """
    # Analyze top view geometry
    def analyze_mask(mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.65, 0.8  # Defaults
        
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity: 4π * area / perimeter²
        # Perfect circle = 1.0, more irregular shapes < 1.0
        circularity = 0.0
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            circularity = min(circularity, 1.0)
        
        # Solidity: contour area / convex hull area
        # Measures how "full" the shape is (vs concave)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.8
        
        return circularity, solidity
    
    circ_top, solid_top = analyze_mask(mask_top)
    circ_side, solid_side = analyze_mask(mask_side)
    
    # Combine metrics to estimate shape factor
    # High circularity + high solidity → ellipsoid-like → factor ≈ π/6 ≈ 0.524 (sphere)
    # Low circularity (even with high solidity) → more boxy → higher factor
    
    avg_circularity = (circ_top + circ_side) / 2
    avg_solidity = (solid_top + solid_side) / 2
    
    # Heuristic mapping with strong emphasis on circularity:
    # - Perfect ellipsoid: high circ (>0.85) → 0.52
    # - Rounded ellipsoid: medium-high circ (0.7-0.85) → 0.54-0.60
    # - Rounded irregular: medium circ (0.55-0.7) → 0.60-0.68
    # - Boxy/rectangular: low circ (<0.55) → 0.68-0.80
    
    if avg_circularity > 0.85:
        # Nearly perfect ellipsoid (circularity of circle/ellipse is ~1.0)
        shape_factor = 0.52 + (1.0 - avg_circularity) * 0.2
    elif avg_circularity > 0.70:
        # Rounded ellipsoidal object
        shape_factor = 0.54 + (1.0 - avg_circularity) * 0.4
    elif avg_circularity > 0.55:
        # Transition from rounded to irregular
        shape_factor = 0.60 + (1.0 - avg_circularity) * 0.5
    else:
        # Low circularity = rectangular or very irregular
        # Use solidity to distinguish: high solidity = box, low solidity = irregular rock
        if avg_solidity > 0.90:
            # Rectangular/boxy (low circ, high solidity)
            shape_factor = 0.72 + (1.0 - avg_solidity) * 0.2
        else:
            # Irregular rock (low circ, medium-low solidity)
            shape_factor = 0.68 + (1.0 - avg_solidity) * 0.15
    
    # Clamp to reasonable range
    shape_factor = np.clip(shape_factor, 0.50, 0.85)
    
    return float(shape_factor)


# ============================================================================
# MAIN VOLUME ESTIMATION
# ============================================================================

def estimate_volume_dual_view(
    top_image: np.ndarray,
    side_image: np.ndarray,
    tag_side_cm: float = 10.0,
    intrinsics_path: Optional[str] = None,
    method: str = "adaptive"
) -> DualViewVolumeResult:
    """
    Estimate object volume from dual orthogonal views.
    
    High-level pipeline:
    1. Process top view -> extract footprint and scale
    2. Process side view -> extract height and scale
    3. Extract principal dimensions (L, W, H)
    4. Choose volume calculation method:
       - "ellipsoid": Use ellipsoid formula with L, W, H
       - "shape_adaptive": Use area * height * computed shape factor
       - "adaptive": Choose best method based on shape analysis (default)
    
    Args:
        top_image: BGR image from top-down view
        side_image: BGR image from side view
        tag_side_cm: AprilTag physical side length in cm (default 10.0)
        intrinsics_path: Optional path to camera intrinsics JSON
        method: Volume calculation method ("ellipsoid", "shape_adaptive", "adaptive")
    
    Returns:
        DualViewVolumeResult with all measurements and final volume
    
    Raises:
        RuntimeError if AprilTags not detected in either view
        ValueError if segmentation fails
    """
    # Step 1: Process top view
    top_meas = process_top_view(top_image, tag_side_cm=tag_side_cm, intrinsics_path=intrinsics_path)
    
    # Step 2: Process side view
    side_meas = process_side_view(side_image, tag_side_cm=tag_side_cm, intrinsics_path=intrinsics_path)
    
    # Step 3: Extract dimensions
    length_cm, width_cm = extract_principal_axes(top_meas.mask, top_meas.scale_cm_per_pixel)
    # Side view PCA gives two axes. One will be close to L (redundant), the other is H.
    # Pick whichever axis is NOT closest to L as the true height.
    side_major_cm, side_minor_cm = extract_principal_axes(side_meas.mask, side_meas.scale_cm_per_pixel)
    if abs(side_major_cm - length_cm) < abs(side_minor_cm - length_cm):
        # Major is closer to L -> minor is H
        height_cm = side_minor_cm
    else:
        # Minor is closer to L -> major is H
        height_cm = side_major_cm
    print(f"  Side PCA: major={side_major_cm:.2f} cm, minor={side_minor_cm:.2f} cm -> H={height_cm:.2f} cm")
    footprint_area_cm2 = compute_footprint_area(top_meas.mask, top_meas.scale_cm_per_pixel)
    
    # Step 4: Estimate shape characteristics
    shape_factor = estimate_shape_factor(top_meas.mask, side_meas.mask)
    
    # Step 5: Choose and compute volume
    if method == "ellipsoid":
        # Pure ellipsoid model
        estimated_volume_cm3 = compute_ellipsoid_volume(length_cm, width_cm, height_cm)
        volume_method = "ellipsoid"
        
    elif method == "shape_adaptive":
        # Area-based with improved shape factor
        estimated_volume_cm3 = footprint_area_cm2 * height_cm * shape_factor
        volume_method = "shape_adaptive"
        
    else:  # method == "adaptive" (default)
        # Choose best method based on shape analysis
        # Compute both methods
        volume_ellipsoid = compute_ellipsoid_volume(length_cm, width_cm, height_cm)
        volume_area_based = footprint_area_cm2 * height_cm * shape_factor
        
        # Adaptive blending based on shape factor:
        # - Shape factor 0.50-0.55: Very ellipsoidal → mostly use ellipsoid formula
        # - Shape factor 0.55-0.65: Moderately rounded → blend methods
        # - Shape factor 0.65-0.85: Irregular/boxy → mostly use area-based
        
        if shape_factor < 0.55:
            # Highly ellipsoidal - use pure ellipsoid formula
            estimated_volume_cm3 = volume_ellipsoid
            volume_method = "adaptive_ellipsoid"
        elif shape_factor > 0.68:
            # More irregular/boxy - use shape-adaptive area method
            estimated_volume_cm3 = volume_area_based
            volume_method = "adaptive_area"
        else:
            # Transition zone - blend both methods
            # Linear interpolation: sf=0.55 → 100% ellipsoid, sf=0.68 → 100% area
            weight_ellipsoid = (0.68 - shape_factor) / (0.68 - 0.55)
            weight_area = 1.0 - weight_ellipsoid
            estimated_volume_cm3 = (weight_ellipsoid * volume_ellipsoid + 
                                   weight_area * volume_area_based)
            volume_method = f"adaptive_blend_{weight_ellipsoid:.2f}e_{weight_area:.2f}a"
    
    return DualViewVolumeResult(
        footprint_area_cm2=footprint_area_cm2,
        height_cm=height_cm,
        shape_factor=shape_factor,
        estimated_volume_cm3=estimated_volume_cm3,
        volume_method=volume_method,
        length_cm=length_cm,
        width_cm=width_cm
    )
