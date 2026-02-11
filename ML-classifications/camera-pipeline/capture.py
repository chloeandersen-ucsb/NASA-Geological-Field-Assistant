#!/usr/bin/env python3
"""
Camera capture for Jetson (Arducam IMX477 on CAM0).
Saves one image to captures/ with a unique name and prints the absolute path to stdout.
Used by the LED display app; no arguments (CameraService runs this script as-is).
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
CAPTURES_DIR = SCRIPT_DIR / "captures"


def generate_capture_path() -> Path:
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return CAPTURES_DIR / f"capture_{timestamp}_{short_uuid}.jpg"


def main() -> int:
    out_path = generate_capture_path()
    # nvgstcapture uses --file-name as base name and saves in cwd; it may add extension
    out_basename = out_path.stem
    out_dir = out_path.parent

    # Still mode (-m 1), automate one capture, start after 1s, quit after 4s
    # Higher resolutions available: 5=2104x1560, 6=2592x1944, 8=3840x2160
    cmd = [
        "nvgstcapture-1.0",
        "-m", "1",
        "--image-res", "4",  # 1920x1080 - I think high enough quality for ML classification
        "--file-name", out_basename,
        "-A",
        "--capture-auto",
        "-C", "1",
        "-S", "1",
        "-Q", "4",
    ]

    try:
        # Note: nvgstcapture may open a GUI window, but camera should still capture
        # The main fix is ensuring the preview stream is stopped before capture
        result = subprocess.run(
            cmd,
            cwd=str(out_dir),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        print("ERROR: nvgstcapture-1.0 not found (run on Jetson with camera)", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print("ERROR: Camera capture timed out", file=sys.stderr)
        return 1

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        print(f"ERROR: {err or 'nvgstcapture failed'}", file=sys.stderr)
        return 1

    # nvgstcapture may write .jpg or no extension; prefer .jpg then any file with basename
    candidate = out_dir / f"{out_basename}.jpg"
    if not candidate.exists():
        candidates = list(out_dir.glob(f"{out_basename}*"))
        candidate = candidates[0] if candidates else None
    if not candidate or not candidate.is_file():
        print("ERROR: Capture completed but output file not found", file=sys.stderr)
        return 1

    print(candidate.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
