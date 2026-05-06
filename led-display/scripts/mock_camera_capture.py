import argparse
import os
import sys
import tempfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="Output image path (optional)")
    args = parser.parse_args()
    
    if args.out:
        output_path = Path(args.out)
    else:
        temp_dir = tempfile.gettempdir()
        output_path = Path(temp_dir) / "sage_capture.jpg"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy the bundled mock image so the UI renders a real preview
    project_root = Path(__file__).resolve().parent.parent.parent
    mock_src = project_root / "ML-classifications" / "scripts" / "images" / "mock_image.jpg"
    if mock_src.exists():
        import shutil
        shutil.copy2(mock_src, output_path)
    else:
        output_path.write_bytes(b"FAKE_IMAGE_DATA_FOR_UI_TESTING")

    print(str(output_path.absolute()))
    sys.exit(0)

if __name__ == "__main__":
    main()
