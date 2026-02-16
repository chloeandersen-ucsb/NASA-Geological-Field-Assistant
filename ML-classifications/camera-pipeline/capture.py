import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import uuid

_current_dir = Path(__file__).parent
_process = None
_file_name = None
_mission_dir = None


def start_preview():
    """Start nvgstcapture (1080p preview, 4K image capture). Returns image path."""
    global _process, _file_name, _mission_dir
    _file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    _mission_dir = _current_dir / "images" / datetime.now().strftime("%Y%m%d")
    _mission_dir.mkdir(parents=True, exist_ok=True)
    _process = subprocess.Popen(
        [
            "nvgstcapture-1.0",
            "--prev-res=4",
            "--image-res=8",
            "-m", "1",
            f"--file-name={_file_name}",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=_mission_dir,
    )
    return str(_mission_dir / f"{_file_name}.jpg")


def capture():
    """Send 'j' to nvgstcapture; return image path or None if not running."""
    global _process, _file_name, _mission_dir
    if _process is None or _process.poll() is not None:
        return None
    _process.stdin.write("j\n")
    _process.stdin.flush()
    time.sleep(1)
    return str(_mission_dir / f"{_file_name}.jpg")


def _run_preview_server():
    """--preview: print path, then read CAPTURE (print path) / QUIT from stdin."""
    path = start_preview()
    print(path)
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        cmd = line.strip().upper()
        if cmd == "QUIT":
            break
        if cmd == "CAPTURE":
            print(capture() or "")
            sys.stdout.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()
    if args.preview:
        _run_preview_server()
    else:
        start_preview()
        time.sleep(2)
        print(capture())
    sys.exit(0)
