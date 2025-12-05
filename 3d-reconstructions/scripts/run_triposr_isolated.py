#!/usr/bin/env python3
"""
Isolated TripoSR runner that temporarily downgrades huggingface-hub
to be compatible with transformers 4.35.0 requirement.
"""
import argparse
import sys
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run TripoSR with isolated dependencies")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--mc-resolution", type=int, default=256, help="Marching cubes resolution (default: 256)")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Chunk size for processing (default: 8192)")
    args = parser.parse_args()
    
    # Get the venv python path
    venv_python = Path(sys.executable)
    triposr_script = Path(__file__).parent.parent / "external" / "TripoSR" / "run.py"
    
    if not triposr_script.exists():
        print(f"[ERROR] TripoSR not found at {triposr_script}")
        return 1
    
    print("[INFO] Temporarily adjusting huggingface-hub for TripoSR compatibility...")
    
    # Store current version
    result = subprocess.run(
        [str(venv_python), "-m", "pip", "show", "huggingface-hub"],
        capture_output=True,
        text=True
    )
    current_version = None
    for line in result.stdout.split('\n'):
        if line.startswith('Version:'):
            current_version = line.split(':', 1)[1].strip()
            break
    
    print(f"[INFO] Current huggingface-hub version: {current_version}")
    
    # Downgrade to <1.0
    print("[INFO] Installing huggingface-hub<1.0 for TripoSR...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-q", "huggingface-hub<1.0"],
        check=True
    )
    
    try:
        # Run TripoSR
        print(f"[INFO] Running TripoSR on {args.input} with mc-resolution={args.mc_resolution}...")
        cmd = [
            str(venv_python),
            str(triposr_script),
            args.input,
            "--output-dir", args.output_dir,
            "--device", args.device,
            "--mc-resolution", str(args.mc_resolution),
            "--chunk-size", str(args.chunk_size)
        ]
        
        result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            print(f"[ERROR] TripoSR failed with exit code {result.returncode}")
            return result.returncode
        
        print(f"[OK] TripoSR completed successfully")
        
    finally:
        # Restore original version for dust3r
        if current_version and current_version != "0.26.5":  # 0.26.5 is latest <1.0
            print(f"[INFO] Restoring huggingface-hub to version {current_version} for dust3r...")
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "-q", f"huggingface-hub=={current_version}"],
                check=True
            )
            print("[OK] Dependencies restored")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
