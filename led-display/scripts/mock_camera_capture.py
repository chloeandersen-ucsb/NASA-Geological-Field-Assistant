import argparse
import os
import sys
import tempfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="Output image path (optional)")
    args = parser.parse_args()
    
    # Create a dummy image file
    if args.out:
        output_path = Path(args.out)
    else:
        # Use temp directory if no output specified
        temp_dir = tempfile.gettempdir()
        output_path = Path(temp_dir) / "sage_capture.jpg"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a minimal dummy image file (just a placeholder for UI testing)
    output_path.write_bytes(b"FAKE_IMAGE_DATA_FOR_UI_TESTING")
    
    # Print the path to stdout (as expected by CameraService)
    print(str(output_path.absolute()))
    sys.exit(0)

if __name__ == "__main__":
    main()
