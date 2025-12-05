#!/usr/bin/env python3
"""
Simple foreground single-object detection helper.

Area-based heuristic (stable for real-world scenes):
1. Convert frame to grayscale
2. Apply Gaussian blur to reduce noise
3. Use adaptive threshold + morphology to get foreground mask
4. Find contours; keep those above minimum noise threshold
5. Accept if largest contour occupies 15-70% of frame (well-framed primary object)
6. Reject if <15% (too small/distant) or >70% (too zoomed/clipped)

This handles scenes with multiple objects as long as one is clearly dominant/central.

Usage:
    from object_select import is_single_object
    is_obj = is_single_object('path/to/frame.jpg')
"""
from pathlib import Path
import cv2
import numpy as np

def is_single_object(image_path:str, min_primary_area:float=0.15, max_primary_area:float=0.70, min_noise_filter:float=0.02) -> bool:
    """
    Area-based object detection using simple background modeling.
    Returns True if dominant foreground object occupies 15-70% of frame.
    """
    p = Path(image_path)
    if not p.exists():
        return False
    img = cv2.imread(str(p))
    if img is None:
        return False
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's thresholding for better object separation
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological cleanup
    kernel = np.ones((7,7), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    
    # Calculate areas and filter noise
    frame_area = h * w
    areas = [cv2.contourArea(c) for c in contours]
    filtered = [(c, a) for c, a in zip(contours, areas) if a/frame_area >= min_noise_filter]
    
    if not filtered:
        return False
    
    # Sort by area (largest first)
    filtered.sort(key=lambda x: x[1], reverse=True)
    largest_area = filtered[0][1]
    largest_ratio = largest_area / frame_area
    
    # Area-based acceptance: well-framed primary object
    # 15-70% of frame = perfect single-object framing
    if min_primary_area <= largest_ratio <= max_primary_area:
        return True
    
    return False

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Single object heuristic detector')
    ap.add_argument('image', type=str)
    args = ap.parse_args()
    print('SINGLE_OBJECT' if is_single_object(args.image) else 'MULTI_OBJECT_OR_SCENE')
