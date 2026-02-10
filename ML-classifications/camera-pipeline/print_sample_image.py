#!/usr/bin/env python3
"""Print path to sample image for run-mock-cam (real ML, no camera)"""

from pathlib import Path

if __name__ == "__main__":
    path = Path(__file__).parent.resolve() / "S76-23285_10069.jpg"
    print(path)
