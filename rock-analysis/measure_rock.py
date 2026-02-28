import cv2
from datetime import datetime
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


@dataclass
class MeasureConfig:
    # Physical size of the AprilTag (side length) in centimeters
    reference_real_size_cm: float = 10.0  # 100 mm tag36h11
    density_min_g_cm3: float = 2.5
    density_max_g_cm3: float = 3.0

# Global SAM objects (initialized once on first use)
_sam_model = None
_sam_predictor = None
_sam_mask_generator = None

def _get_sam_model():
    """Lazy-load SAM model on first use."""
    global _sam_model
    if _sam_model is None:
        if not SAM_AVAILABLE:
            raise RuntimeError("SAM not available. Install: pip install segment-anything torch torchvision")

        model_path = os.path.expanduser("~/.sam_models/sam_vit_b.pth")
        if not os.path.exists(model_path):
            raise RuntimeError(f"SAM model not found at {model_path}. Run: python download_sam.py")

        print("Loading SAM model...")
        _sam_model = sam_model_registry["vit_b"](checkpoint=model_path)
        _sam_model.to("cpu")  # Use CPU
    return _sam_model

def get_sam_predictor():
    """Create a SAM predictor using the shared model."""
    global _sam_predictor
    if _sam_predictor is None:
        _sam_predictor = SamPredictor(_get_sam_model())
    return _sam_predictor

def get_sam_mask_generator():
    """Create a SAM automatic mask generator using the shared model."""
    global _sam_mask_generator
    if _sam_mask_generator is None:
        _sam_mask_generator = SamAutomaticMaskGenerator(
            _get_sam_model(),
            points_per_side=8,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=0,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200,
        )
    return _sam_mask_generator

# Global MiDaS model (lazy loaded)
_midas_model = None
_midas_transform = None

def _get_midas():
    """Lazy-load MiDaS small model for monocular depth estimation."""
    global _midas_model, _midas_transform
    if _midas_model is None:
        _midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
        _midas_model.eval()
        transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
        _midas_transform = transforms.small_transform
    return _midas_model, _midas_transform


def estimate_depth_map(image_rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Estimate relative depth map using MiDaS small.
    Higher values = closer to camera.
    
    Args:
        image_rgb: RGB uint8 image (any size)
        target_h, target_w: Output depth map dimensions
    
    Returns:
        float32 depth map of shape (target_h, target_w)
    """
    midas, transform = _get_midas()
    input_batch = transform(image_rgb)
    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
    return depth


def distance(p1, p2):
    dx = float(p1[0] - p2[0])
    dy = float(p1[1] - p2[1])
    return math.sqrt(dx * dx + dy * dy)

def ellipsoid_volume_cm3(L_cm, W_cm, H_cm):
    return (4.0 / 3.0) * math.pi * (L_cm / 2.0) * (W_cm / 2.0) * (H_cm / 2.0)


def rectified_distance_cm(point_a_px, point_b_px, tag_corners_px, tag_side_cm):
    """Distance between two clicked pixel points measured in the ArUco tag plane (cm)."""
    image_pts = np.asarray(tag_corners_px, dtype=np.float32)
    if image_pts.shape != (4, 2):
        raise ValueError("tag_corners_px must be 4x2")

    s = float(tag_side_cm)
    tag_plane_pts = np.array(
        [
            [-s / 2.0, s / 2.0],
            [s / 2.0, s / 2.0],
            [s / 2.0, -s / 2.0],
            [-s / 2.0, -s / 2.0],
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(image_pts, tag_plane_pts)
    if H is None:
        raise RuntimeError("Homography estimation failed")

    pts = np.asarray([point_a_px, point_b_px], dtype=np.float32).reshape(-1, 1, 2)
    rectified = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    return float(np.linalg.norm(rectified[0] - rectified[1]))


def measure_rectified_length_width(corners, aruco_side_length, avocado_contour, units="cm"):
    """Measure length/width on the ArUco tag plane via homography.

    Assumes `corners` are in OpenCV ArUco order:
    top-left, top-right, bottom-right, bottom-left.
    """
    if corners is None or avocado_contour is None:
        raise ValueError("corners and avocado_contour are required")

    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape == (4, 2):
        image_pts = corners
    elif corners.shape == (1, 4, 2):
        image_pts = corners[0]
    else:
        raise ValueError("corners must have shape (1, 4, 2) or (4, 2)")

    s = float(aruco_side_length)
    tag_plane_pts = np.array(
        [
            [-s / 2, s / 2],
            [s / 2, s / 2],
            [s / 2, -s / 2],
            [-s / 2, -s / 2],
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(image_pts, tag_plane_pts)
    if H is None:
        raise RuntimeError("Homography estimation failed")

    contour_pts = np.asarray(avocado_contour, dtype=np.float32)
    if contour_pts.ndim != 2 or contour_pts.shape[1] != 2:
        raise ValueError("avocado_contour must be an Nx2 array")
    if contour_pts.shape[0] < 2:
        raise ValueError("avocado_contour must have at least 2 points")

    contour_pts = contour_pts.reshape(-1, 1, 2)
    rectified_pts = cv2.perspectiveTransform(contour_pts, H).reshape(-1, 2)

    mean = np.mean(rectified_pts, axis=0)
    centered = rectified_pts - mean

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    principal_axis = eigvecs[:, np.argmax(eigvals)]
    secondary_axis = eigvecs[:, np.argmin(eigvals)]

    proj_main = centered @ principal_axis
    proj_perp = centered @ secondary_axis

    length = proj_main.max() - proj_main.min()
    width = proj_perp.max() - proj_perp.min()

    return {
        "length": float(length),
        "width": float(width),
        "units": units,
    }


def detect_apriltag_side_px(image, tag_dict_name="DICT_APRILTAG_36h11"):
    """Detect an AprilTag and return the average side length in pixels.

    We try a few preprocessing variants to make detection more robust to
    lighting/contrast. If no tag is found, a RuntimeError is raised.
    """
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module not available; install opencv-contrib-python")

    aruco = cv2.aruco
    dictionary = getattr(aruco, "getPredefinedDictionary")(getattr(aruco, tag_dict_name))

    # Build a list of grayscale candidates (raw, CLAHE, slightly blurred)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = [gray]
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        candidates.append(clahe.apply(gray))
    except Exception:
        pass
    candidates.append(cv2.GaussianBlur(gray, (5, 5), 0))

    # If the image is very large, also test a downscaled copy to reduce noise
    h, w = gray.shape[:2]
    if max(h, w) > 2000:
        scale = 1600.0 / float(max(h, w))
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        candidates.append(small)

    def _make_params_new():
        p = aruco.DetectorParameters()
        if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
            p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 23
        p.adaptiveThreshWinSizeStep = 10
        return p

    def _make_params_legacy():
        p = aruco.DetectorParameters_create()
        if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
            try:
                p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            except Exception:
                pass
        try:
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 23
            p.adaptiveThreshWinSizeStep = 10
        except Exception:
            pass
        return p

    corners = ids = None
    used_idx = None

    for idx, cand in enumerate(candidates):
        try:
            parameters = _make_params_new()
            detector = aruco.ArucoDetector(dictionary, parameters)
            corners, ids, _ = detector.detectMarkers(cand)
        except AttributeError:
            parameters = _make_params_legacy()
            corners, ids, _ = aruco.detectMarkers(cand, dictionary, parameters=parameters)

        if corners is not None and len(corners) > 0:
            used_idx = idx
            break

    if corners is None or len(corners) == 0:
        raise RuntimeError("No AprilTag (tag36h11) detected; ensure the tag is fully in frame, sharp, and high contrast")

    # Use the largest detected tag to stay stable when multiple are visible
    if len(corners) > 1:
        areas = [cv2.contourArea(c.astype(float)) for c in corners]
        best_idx = int(max(range(len(areas)), key=lambda i: areas[i]))
    else:
        best_idx = 0

    pts = corners[best_idx].reshape(-1, 2)

    side_lengths = [distance(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    avg_side = sum(side_lengths) / 4.0

    detected_id = int(ids[best_idx][0]) if ids is not None and len(ids) > best_idx else None

    return avg_side, pts.tolist(), detected_id


# ============================================================================
# AUTOMATIC SEGMENTATION & DIMENSION DETECTION (No User Input)
# ============================================================================

def segment_object(image, tag_corners=None):
    """
    Segment the object using center-pixel SAM prompt.
    
    Strategy:
    1. User places the object at the center of the frame
    2. Downscale image to ~1024px for SAM efficiency
    3. Prompt SAM with center point (Method A) and center point+box (Method C)
    4. Pick the best non-empty mask with reasonable size
    5. Exclude the AprilTag region
    6. Keep only the largest connected component near the center
       to remove disconnected fragments
    
    Args:
        image: BGR image (from cv2.imread)
        tag_corners: Optional 4x2 array of AprilTag corners (used to exclude tag)
    
    Returns:
        Binary mask where object pixels == 1 (foreground), background == 0
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    # Downscale BGR first, then convert to RGB (matches SAM's expected input)
    max_sam_dim = 1024
    scale = min(1.0, float(max_sam_dim) / float(max(h, w)))
    sam_w = int(w * scale)
    sam_h = int(h * scale)
    if scale < 1.0:
        image_bgr_sam = cv2.resize(image, (sam_w, sam_h))
    else:
        sam_h, sam_w = h, w
        image_bgr_sam = image
    image_rgb_sam = cv2.cvtColor(image_bgr_sam, cv2.COLOR_BGR2RGB)

    scx, scy = sam_w // 2, sam_h // 2
    print(f"  Center-pixel SAM: image {w}x{h}, SAM input {sam_w}x{sam_h}, center=({scx},{scy})")

    predictor = get_sam_predictor()
    predictor.set_image(image_rgb_sam)

    # Collect candidate masks from multiple prompt strategies
    all_candidates = []  # (mask_small, score, method_label)

    # Method A: Center point only
    masks_a, scores_a, _ = predictor.predict(
        point_coords=np.array([[scx, scy]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    for i, (m, s) in enumerate(zip(masks_a, scores_a)):
        all_candidates.append((m, float(s), f"A:pt#{i}"))

    # Method C: Center point + center box (50% of image)
    bx1, by1 = sam_w // 4, sam_h // 4
    bx2, by2 = 3 * sam_w // 4, 3 * sam_h // 4
    masks_c, scores_c, _ = predictor.predict(
        point_coords=np.array([[scx, scy]]),
        point_labels=np.array([1]),
        box=np.array([bx1, by1, bx2, by2]),
        multimask_output=True,
    )
    for i, (m, s) in enumerate(zip(masks_c, scores_c)):
        all_candidates.append((m, float(s), f"C:pt+box#{i}"))

    # Evaluate all candidates: upscale, exclude tag, keep reasonable size
    total_px = h * w
    evaluated = []
    for mask_small, score, label in all_candidates:
        m_up = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        m_clean = _exclude_tag_from_mask(m_up, tag_corners)
        px = int(m_clean.sum())
        pct = px / total_px * 100
        print(f"    {label}: {px:>10,}px ({pct:5.1f}%), score={score:.3f}")
        evaluated.append((m_clean, px, score, label))

    # Filter: must have pixels, size between 0.1% and 30% of image
    min_pct, max_pct = 0.1, 30.0
    valid = [(mc, px, s, lbl) for mc, px, s, lbl in evaluated
             if px > 0 and (px / total_px * 100) >= min_pct and (px / total_px * 100) <= max_pct]

    if valid:
        # Among valid masks, pick highest score
        valid.sort(key=lambda x: x[2], reverse=True)
        best_mask, best_px, best_score, best_label = valid[0]
        print(f"    -> Selected {best_label} (score={best_score:.3f}, {best_px:,}px)")
    elif any(px > 0 for _, px, _, _ in evaluated):
        # No mask in size range but some have pixels — pick smallest non-empty
        non_empty = [(mc, px, s, lbl) for mc, px, s, lbl in evaluated if px > 0]
        non_empty.sort(key=lambda x: x[1])
        best_mask, best_px, best_score, best_label = non_empty[0]
        print(f"    -> Fallback: {best_label} (smallest non-empty, {best_px:,}px)")
    else:
        # All empty — try SAM auto mask generator
        print("  [WARN] All center-pixel SAM masks empty, falling back to SAM auto")
        generator = get_sam_mask_generator()
        auto_masks = generator.generate(image_rgb_sam)
        if auto_masks:
            # Pick mask closest to center
            best_auto = min(auto_masks, key=lambda m: np.sqrt(
                (m['bbox'][0] + m['bbox'][2] / 2 - scx)**2 +
                (m['bbox'][1] + m['bbox'][3] / 2 - scy)**2
            ))
            fallback = best_auto['segmentation'].astype(np.uint8)
            fallback = cv2.resize(fallback, (w, h), interpolation=cv2.INTER_NEAREST)
            fallback = _exclude_tag_from_mask(fallback, tag_corners)
            return _keep_largest_cc_near_center(fallback, cx, cy)
        return np.zeros((h, w), dtype=np.uint8)

    # Keep only the largest connected component near the center
    mask_out = _keep_largest_cc_near_center(best_mask, cx, cy)

    final_px = int(mask_out.sum())
    print(f"    -> After CC cleanup: {final_px:,}px ({final_px / total_px * 100:.1f}%)")
    return mask_out


def _keep_largest_cc_near_center(mask, cx, cy):
    """
    Keep only the connected component that contains or is closest to (cx, cy).
    Removes disconnected fragments from the SAM mask.
    
    Args:
        mask: Binary mask (H, W) with values 0/1
        cx, cy: Center coordinates to anchor selection
    Returns:
        Cleaned mask with only the main object CC
    """
    if mask is None or mask.sum() == 0:
        return mask
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    
    if num_labels <= 2:
        # Only background + 1 component — already clean
        return mask
    
    # First: check if center pixel belongs to a CC
    if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
        center_label = labels[cy, cx]
        if center_label > 0:
            # The center pixel is on a foreground CC — keep that one
            result = (labels == center_label).astype(np.uint8)
            dropped = num_labels - 2  # minus background and kept
            if dropped > 0:
                print(f"    CC cleanup: kept CC at center, dropped {dropped} disconnected region(s)")
            return result
    
    # Fallback: pick CC closest to center with area > 1% of largest CC
    largest_area = max(stats[lid, cv2.CC_STAT_AREA] for lid in range(1, num_labels))
    min_area = max(100, int(largest_area * 0.01))
    
    best_label = -1
    best_dist = float('inf')
    for lid in range(1, num_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        ccx, ccy = centroids[lid]
        dist = np.sqrt((ccx - cx)**2 + (ccy - cy)**2)
        if dist < best_dist:
            best_dist = dist
            best_label = lid
    
    if best_label > 0:
        result = (labels == best_label).astype(np.uint8)
        dropped = sum(1 for lid in range(1, num_labels) if lid != best_label)
        if dropped > 0:
            print(f"    CC cleanup: kept CC {best_label} (dist={best_dist:.0f}), dropped {dropped} region(s)")
        return result
    
    return mask


def _exclude_tag_from_mask(mask, tag_corners, margin_px=80):
    """
    Zero out the AprilTag region (with a fixed pixel margin) from any mask.
    Uses a dilated convex hull of the tag corners rather than a large circle,
    so the exclusion zone stays tight around the actual tag.

    Args:
        mask: Binary mask (H, W) with values 0/1
        tag_corners: 4x2 array of tag corner coordinates, or None
        margin_px: Fixed pixel margin to expand beyond the tag corners
    Returns:
        Mask with tag region zeroed out
    """
    if tag_corners is None or len(tag_corners) == 0:
        return mask

    h, w = mask.shape[:2]
    tag_pts = np.array(tag_corners, dtype=np.float32)
    tag_center = tag_pts.mean(axis=0)

    # Expand each corner outward from center by margin_px
    expanded = []
    for pt in tag_pts:
        direction = pt - tag_center
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
        expanded.append(pt + direction * margin_px)
    expanded = np.array(expanded, dtype=np.int32)

    mask_out = mask.copy()
    cv2.fillConvexPoly(mask_out, expanded, 0)

    return mask_out


def extract_largest_contour(mask):
    """
    Extract the largest contour from a binary mask.
    
    Args:
        mask: Binary mask (0 or 1)
    
    Returns:
        Contour as Nx2 float array, or None if no contours found
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    # Select largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Convert to Nx2 float array
    contour_pts = largest_contour.reshape(-1, 2).astype(np.float32)
    
    return contour_pts


def compute_LW_from_points(points_cm):
    """
    Computes length and width using PCA projections on rectified contour points.
    
    Args:
        points_cm: Nx2 array of contour points in centimeters (rectified to tag plane)
    
    Returns:
        dict with 'length_cm' and 'width_cm'
    """
    if points_cm is None or len(points_cm) < 2:
        raise ValueError("points_cm must have at least 2 points")
    
    points_cm = np.asarray(points_cm, dtype=np.float32)
    
    # Subtract mean
    mean = np.mean(points_cm, axis=0)
    centered = points_cm - mean
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eig(cov)
    
    # Principal axis corresponds to largest eigenvalue (length)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    # Secondary axis corresponds to smallest eigenvalue (width)
    secondary_axis = eigvecs[:, np.argmin(eigvals)]
    
    # Project points onto both axes
    proj_main = centered @ principal_axis
    proj_perp = centered @ secondary_axis
    
    # Compute extents
    length_cm = float(proj_main.max() - proj_main.min())
    width_cm = float(proj_perp.max() - proj_perp.min())
    
    return {
        "length_cm": length_cm,
        "width_cm": width_cm,
    }


def compute_height_cm(side_image, tag_corners_px, tag_size_cm):
    """
    Compute height automatically from side-view image.
    
    Steps:
    1. Segment object in side image using SAM
    2. Extract largest contour
    3. Find vertical extent (y-axis) in pixel space
    4. Convert to cm using AprilTag pixel scale from side view
    
    Args:
        side_image: BGR image (side view)
        tag_corners_px: 4x2 array of AprilTag corners in pixels
        tag_size_cm: Physical size of AprilTag in cm
    
    Returns:
        Height in centimeters
    """
    # Segment object
    mask = segment_object(side_image, tag_corners_px)
    
    # Extract largest contour
    contour_pts = extract_largest_contour(mask)
    if contour_pts is None or len(contour_pts) < 2:
        raise ValueError("Could not extract contour from side image")
    
    # Compute pixel-space scale: cm per pixel
    # Use AprilTag corners to compute average side length in pixels
    tag_corners = np.asarray(tag_corners_px, dtype=np.float32)
    side_lengths = [distance(tag_corners[i], tag_corners[(i + 1) % 4]) for i in range(4)]
    tag_px = sum(side_lengths) / 4.0
    
    cm_per_px = float(tag_size_cm) / float(tag_px)
    
    # Find vertical extent (y-axis) of contour
    y_min = contour_pts[:, 1].min()
    y_max = contour_pts[:, 1].max()
    height_px = float(y_max - y_min)
    
    # Convert to cm
    height_cm = height_px * cm_per_px
    
    return height_cm


class ClickMeasurer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        self.display_scale = 1.0
        self.max_display_dim_px = self._compute_max_display_dim()
        self.display = self._make_display_image(self.image)
        self.points = []  # points stored in original image coordinates
        self.window_name = "click_to_measure"

    def _compute_max_display_dim(self, fallback=1200):
        """Pick a display size that fits ~90% of the current screen."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)
            return int(max(600, min(screen_w, screen_h) * 0.9))
        except Exception:
            return fallback

    def _make_display_image(self, image):
        """Create a resized display image while tracking scale factor."""
        h, w = image.shape[:2]
        max_dim = float(max(h, w))
        if max_dim > self.max_display_dim_px:
            self.display_scale = self.max_display_dim_px / max_dim
            interp = cv2.INTER_AREA
        else:
            self.display_scale = 1.0
            interp = cv2.INTER_LINEAR
        if self.display_scale != 1.0:
            new_w = int(round(w * self.display_scale))
            new_h = int(round(h * self.display_scale))
            return cv2.resize(image, (new_w, new_h), interpolation=interp)
        return image.copy()

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map display coords back to original image coords
            orig_pt = (int(round(x / self.display_scale)), int(round(y / self.display_scale)))
            self.points.append(orig_pt)
            disp_pt = (int(round(orig_pt[0] * self.display_scale)), int(round(orig_pt[1] * self.display_scale)))
            cv2.circle(self.display, disp_pt, 4, (0, 255, 0), -1)
            if len(self.points) % 2 == 0:
                p1 = self.points[-2]
                p2 = self.points[-1]
                p1d = (int(round(p1[0] * self.display_scale)), int(round(p1[1] * self.display_scale)))
                p2d = (int(round(p2[0] * self.display_scale)), int(round(p2[1] * self.display_scale)))
                cv2.line(self.display, p1d, p2d, (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.display)

    def collect_pairs(self, prompts):
        print("\nInstructions:")
        print("- Left click to place points")
        print("- You will click 2 points per measurement (a line)")
        print("- Press 'u' to undo last point, 'r' to reset, 'q' to quit\n")

        # Lock aspect; we render a downscaled copy if needed to fit screen.
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)
        cv2.imshow(self.window_name, self.display)

        needed_points = 2 * len(prompts)
        prompt_index = 0

        print(f"Image: {self.image_path}")
        print(f"Next: {prompts[prompt_index]} (need 2 clicks)")

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                raise RuntimeError("Quit by user")

            if key == ord("u"):
                if len(self.points) > 0:
                    self.points.pop()
                    self.display = self._make_display_image(self.image)
                    for i, p in enumerate(self.points):
                        pd = (int(round(p[0] * self.display_scale)), int(round(p[1] * self.display_scale)))
                        cv2.circle(self.display, pd, 4, (0, 255, 0), -1)
                        if i % 2 == 1:
                            p_prev = self.points[i - 1]
                            ppd = (int(round(p_prev[0] * self.display_scale)), int(round(p_prev[1] * self.display_scale)))
                            cv2.line(self.display, ppd, pd, (0, 255, 0), 2)
                    cv2.imshow(self.window_name, self.display)

            if key == ord("r"):
                self.points = []
                self.display = self._make_display_image(self.image)
                cv2.imshow(self.window_name, self.display)
                prompt_index = 0
                print(f"Reset. Next: {prompts[prompt_index]} (need 2 clicks)")

            if len(self.points) >= needed_points:
                break

            new_prompt_index = len(self.points) // 2
            if new_prompt_index != prompt_index:
                prompt_index = new_prompt_index
                if prompt_index < len(prompts):
                    print(f"Next: {prompts[prompt_index]} (need 2 clicks)")

        cv2.destroyAllWindows()

        pairs = []
        for i in range(0, needed_points, 2):
            pairs.append((self.points[i], self.points[i + 1]))
        return pairs

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "single", "stereo"], default="auto", 
                        help="Measurement mode: 'auto' (automatic segmentation), 'single' (one image, click L/W/H), or 'stereo' (two images for better accuracy)")
    parser.add_argument("--image", help="Image path (for --mode single or --mode auto)")
    parser.add_argument("--top", help="Top-down image path (for --mode stereo or --mode auto)")
    parser.add_argument("--side", help="Side image path (for --mode stereo or --mode auto)")
    parser.add_argument("--ref_cm", type=float, default=10.0, help="AprilTag side length in cm (tag36h11, default 100mm=10cm).")
    parser.add_argument("--dens_min", type=float, default=2.5, help="Min density in g/cm^3.")
    parser.add_argument("--dens_max", type=float, default=3.0, help="Max density in g/cm^3.")
    parser.add_argument("--out", default=default_output, help="Output JSON path (default: outputs/TIMESTAMP-measurement.json).")
    args = parser.parse_args()

    if args.mode == "single" and not args.image:
        parser.error("--mode single requires --image argument")
    if args.mode == "stereo" and (not args.top or not args.side):
        parser.error("--mode stereo requires both --top and --side arguments")
    if args.mode == "auto" and not args.image and not (args.top and args.side):
        parser.error("--mode auto requires either --image or both --top and --side")

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

    if args.mode == "auto":
        # ====================================================================
        # AUTOMATIC MODE: Stereo (preferred) or Single
        # ====================================================================
        if args.top and args.side:
            # Stereo automatic mode
            print("\n[AUTO] Loading top and side images...")
            top_image = cv2.imread(args.top)
            side_image = cv2.imread(args.side)
            
            if top_image is None:
                raise RuntimeError(f"Could not read top image: {args.top}")
            if side_image is None:
                raise RuntimeError(f"Could not read side image: {args.side}")
            
            print("[AUTO] Detecting AprilTag in top image...")
            tag_px_top, tag_corners_top, tag_id_top = detect_apriltag_side_px(top_image)
            
            print("[AUTO] Detecting AprilTag in side image...")
            tag_px_side, tag_corners_side, tag_id_side = detect_apriltag_side_px(side_image)
            
            print("[AUTO] Segmenting object in top image...")
            mask_top = segment_object(top_image, tag_corners_top)
            contour_top = extract_largest_contour(mask_top)
            if contour_top is None:
                raise RuntimeError("Failed to extract contour from top image")
            
            print("[AUTO] Computing L/W from top view (PCA on rectified contour)...")
            lw_result = measure_rectified_length_width(
                tag_corners_top,
                cfg.reference_real_size_cm,
                contour_top,
                units="cm"
            )
            L_cm = lw_result["length"]
            W_cm = lw_result["width"]
            
            print("[AUTO] Computing H from side view...")
            H_cm = compute_height_cm(side_image, tag_corners_side, cfg.reference_real_size_cm)
            
            scale_info = {
                "mode": "auto_stereo",
                "tag_px_top": tag_px_top,
                "tag_px_side": tag_px_side,
                "tag_id_top": tag_id_top,
                "tag_id_side": tag_id_side,
                "tag_corners_top_px": tag_corners_top,
                "tag_corners_side_px": tag_corners_side,
                "rectification_applied": "top_view_homography",
                "segmentation_method": "GrabCut",
            }
            inputs_info = {
                "top_image": args.top,
                "side_image": args.side,
                "reference_real_size_cm": cfg.reference_real_size_cm,
                "density_min_g_cm3": cfg.density_min_g_cm3,
                "density_max_g_cm3": cfg.density_max_g_cm3,
            }
            
        else:
            # Single image automatic mode
            print("\n[AUTO] Loading single image...")
            image = cv2.imread(args.image)
            
            if image is None:
                raise RuntimeError(f"Could not read image: {args.image}")
            
            print("[AUTO] Detecting AprilTag...")
            tag_px, tag_corners, tag_id = detect_apriltag_side_px(image)
            
            print("[AUTO] Segmenting object...")
            mask = segment_object(image, tag_corners)
            contour = extract_largest_contour(mask)
            if contour is None:
                raise RuntimeError("Failed to extract contour from image")
            
            print("[AUTO] Computing L/W/H from contour (PCA)...")
            lw_result = measure_rectified_length_width(
                tag_corners,
                cfg.reference_real_size_cm,
                contour,
                units="cm"
            )
            L_cm = lw_result["length"]
            W_cm = lw_result["width"]
            
            # For single image, estimate height from contour extent (simplified)
            h_px = contour[:, 1].max() - contour[:, 1].min()
            cm_per_px = cfg.reference_real_size_cm / tag_px
            H_cm = h_px * cm_per_px
            
            scale_info = {
                "mode": "auto_single",
                "tag_px": tag_px,
                "tag_id": tag_id,
                "tag_corners_px": tag_corners,
                "cm_per_px": cm_per_px,
                "segmentation_method": "GrabCut",
            }
            inputs_info = {
                "image": args.image,
                "reference_real_size_cm": cfg.reference_real_size_cm,
                "density_min_g_cm3": cfg.density_min_g_cm3,
                "density_max_g_cm3": cfg.density_max_g_cm3,
            }

    elif args.mode == "single":
        # ====================================================================
        # CLICK-BASED SINGLE MODE (legacy, with user input)
        # ====================================================================
        measurer = ClickMeasurer(args.image)
        pairs = measurer.collect_pairs([
            "Click rock length endpoints",
            "Click rock width endpoints",
            "Click rock height endpoints",
        ])

        tag_px, tag_corners, tag_id = detect_apriltag_side_px(measurer.image)
        cm_per_px = cfg.reference_real_size_cm / tag_px

        L_cm = distance(pairs[0][0], pairs[0][1]) * cm_per_px
        W_cm = distance(pairs[1][0], pairs[1][1]) * cm_per_px
        H_cm = distance(pairs[2][0], pairs[2][1]) * cm_per_px

        scale_info = {
            "mode": "single_click",
            "tag_px": tag_px,
            "tag_id": tag_id,
            "tag_corners_px": tag_corners,
            "cm_per_px": cm_per_px,
        }
        inputs_info = {
            "image": args.image,
            "reference_real_size_cm": cfg.reference_real_size_cm,
            "density_min_g_cm3": cfg.density_min_g_cm3,
            "density_max_g_cm3": cfg.density_max_g_cm3,
        }

    else:  # stereo (click-based)
        # ====================================================================
        # CLICK-BASED STEREO MODE (legacy, with user input)
        # ====================================================================
        top_measurer = ClickMeasurer(args.top)
        top_pairs = top_measurer.collect_pairs([
            "TOP: Click rock length endpoints",
            "TOP: Click rock width endpoints",
        ])

        side_measurer = ClickMeasurer(args.side)
        side_pairs = side_measurer.collect_pairs([
            "SIDE: Click rock height endpoints",
        ])

        tag_px_top, tag_corners_top, tag_id_top = detect_apriltag_side_px(top_measurer.image)
        tag_px_side, tag_corners_side, tag_id_side = detect_apriltag_side_px(side_measurer.image)

        cm_per_px_side = cfg.reference_real_size_cm / tag_px_side

        L_cm = rectified_distance_cm(
            top_pairs[0][0],
            top_pairs[0][1],
            tag_corners_top,
            cfg.reference_real_size_cm,
        )
        W_cm = rectified_distance_cm(
            top_pairs[1][0],
            top_pairs[1][1],
            tag_corners_top,
            cfg.reference_real_size_cm,
        )
        H_cm = distance(side_pairs[0][0], side_pairs[0][1]) * cm_per_px_side

        scale_info = {
            "mode": "stereo_click",
            "tag_px_top": tag_px_top,
            "tag_px_side": tag_px_side,
            "tag_id_top": tag_id_top,
            "tag_id_side": tag_id_side,
            "tag_corners_top_px": tag_corners_top,
            "tag_corners_side_px": tag_corners_side,
            "cm_per_px_side": cm_per_px_side,
            "rectification_applied": "top_view_homography",
        }
        inputs_info = {
            "top_image": args.top,
            "side_image": args.side,
            "reference_real_size_cm": cfg.reference_real_size_cm,
            "density_min_g_cm3": cfg.density_min_g_cm3,
            "density_max_g_cm3": cfg.density_max_g_cm3,
        }

    V_cm3 = ellipsoid_volume_cm3(L_cm, W_cm, H_cm)

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
        "model": "ellipsoid",
    }

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_json(args.out, results)

    print("\n" + "="*70)
    print(f"Measurements (mode: {scale_info.get('mode', 'unknown')})")
    print("="*70)
    print(f"L × W × H (cm): {L_cm:.2f} × {W_cm:.2f} × {H_cm:.2f}")
    print(f"Volume (cm³): {V_cm3:.1f}")
    print(f"Mass range (kg): {mass_min_g/1000.0:.3f} to {mass_max_g/1000.0:.3f}")
    print(f"Saved: {args.out}")
    print("="*70)


if __name__ == "__main__":
    main()
