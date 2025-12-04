#!/usr/bin/env python3
"""Debug object detection - show contour areas"""
import sys
from pathlib import Path
import cv2
import numpy as np

def debug_detection(image_path: str):
    p = Path(image_path)
    if not p.exists():
        print(f"File not found: {image_path}")
        return
    
    img = cv2.imread(str(p))
    if img is None:
        print(f"Cannot read image: {image_path}")
        return
    
    h, w = img.shape[:2]
    frame_area = h * w
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's thresholding
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((7, 7), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found")
        return
    
    areas = [(cv2.contourArea(c), c) for c in contours]
    areas.sort(reverse=True, key=lambda x: x[0])
    
    print(f"\nImage: {p.name}")
    print(f"Resolution: {w}x{h} ({frame_area:,} pixels)")
    print(f"Found {len(contours)} contours\n")
    print("Top 5 contours:")
    for i, (area, _) in enumerate(areas[:5]):
        ratio = area / frame_area
        print(f"  {i+1}. Area: {area:>10.0f} pixels ({ratio*100:>5.2f}% of frame)")
    
    largest_ratio = areas[0][0] / frame_area
    print(f"\nLargest contour: {largest_ratio*100:.2f}%")
    print(f"Status: {'SINGLE_OBJECT' if 0.15 <= largest_ratio <= 0.70 else 'MULTI_OBJECT_OR_SCENE'}")
    print(f"  (Accept range: 15-70%)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_object_detection.py <image_path>")
        sys.exit(1)
    debug_detection(sys.argv[1])
