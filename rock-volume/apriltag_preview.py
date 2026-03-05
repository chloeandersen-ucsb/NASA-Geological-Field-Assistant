"""
Lightweight AprilTag detection for live camera preview.
No SAM or other heavy dependencies; uses only OpenCV aruco.
Returns tag corners (and optional in-focus flag) for overlay drawing.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

TAG_DICT_NAME = "DICT_APRILTAG_36h11"
# Laplacian variance above this is considered "in focus" for the tag ROI
FOCUS_VARIANCE_THRESHOLD = 100.0


def _detect_corners_single_grayscale(
    gray: np.ndarray,
) -> Optional[list[tuple[float, float]]]:
    """Run ArUco AprilTag detection on one grayscale image. Returns 4 corners or None."""
    if not hasattr(cv2, "aruco"):
        return None
    aruco = cv2.aruco
    try:
        dictionary = getattr(aruco, "getPredefinedDictionary")(
            getattr(aruco, TAG_DICT_NAME)
        )
    except (AttributeError, TypeError):
        return None

    def _make_params():
        try:
            p = aruco.DetectorParameters()
            if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
                p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 23
            p.adaptiveThreshWinSizeStep = 10
            return p
        except AttributeError:
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

    try:
        parameters = _make_params()
        try:
            detector = aruco.ArucoDetector(dictionary, parameters)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    except Exception:
        return None

    if corners is None or len(corners) == 0:
        return None

    # Use largest tag if multiple
    if len(corners) > 1:
        areas = [cv2.contourArea(c.astype(float)) for c in corners]
        best_idx = int(max(range(len(areas)), key=lambda i: areas[i]))
    else:
        best_idx = 0

    pts = corners[best_idx].reshape(-1, 2)
    return [(float(pts[i][0]), float(pts[i][1])) for i in range(4)]


def _is_tag_in_focus(bgr: np.ndarray, corners: list[tuple[float, float]]) -> bool:
    """Compute Laplacian variance on the tag ROI; above threshold = in focus."""
    if len(corners) != 4:
        return False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    xs = [int(round(c[0])) for c in corners]
    ys = [int(round(c[1])) for c in corners]
    x1 = max(0, min(xs) - 5)
    x2 = min(gray.shape[1], max(xs) + 5)
    y1 = max(0, min(ys) - 5)
    y2 = min(gray.shape[0], max(ys) + 5)
    if x2 <= x1 or y2 <= y1:
        return False
    roi = gray[y1:y2, x1:x2]
    lap = cv2.Laplacian(roi, cv2.CV_64F)
    variance = lap.var()
    return variance >= FOCUS_VARIANCE_THRESHOLD


def detect_corners_for_preview(
    bgr_image: np.ndarray,
) -> tuple[Optional[list[tuple[float, float]]], bool]:
    """
    Detect one AprilTag (36h11) for live preview overlay.

    Args:
        bgr_image: BGR image (e.g. from OpenCV VideoCapture).

    Returns:
        (corners, in_focus):
        - corners: None if no tag detected, else list of 4 (x, y) in image coordinates.
        - in_focus: True only when corners is not None and Laplacian variance
          on the tag ROI is above threshold (tag is sharp).
    """
    if bgr_image is None or bgr_image.size == 0:
        return None, False
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    corners = _detect_corners_single_grayscale(gray)
    if corners is None:
        return None, False
    in_focus = _is_tag_in_focus(bgr_image, corners)
    return corners, in_focus
