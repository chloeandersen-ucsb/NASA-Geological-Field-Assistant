#!/usr/bin/env python3
"""
mock camera capture: returns the path to a test image.
In the future, this will trigger actual camera capture.
"""

import argparse
import sys
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# Mock image path
MOCK_IMAGE = SCRIPT_DIR / "S76-23285_10069.jpg"


def main():
    parser = argparse.ArgumentParser(description="Camera capture script")
    parser.add_argument(
        "--out",
        help="Output image path (optional, not used in mock version)"
    )
    args = parser.parse_args()
    
    # For mock version, return the path to the test image
    if not MOCK_IMAGE.exists():
        print(f"ERROR: Mock image not found: {MOCK_IMAGE}", file=sys.stderr)
        sys.exit(1)
    
    # Output the image path to stdout (required by CameraService)
    print(str(MOCK_IMAGE.absolute()))
    sys.exit(0)


if __name__ == "__main__":
    main()
