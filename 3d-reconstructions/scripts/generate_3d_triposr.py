#!/usr/bin/env python3
"""
TripoSR wrapper: run single-image 3D reconstruction with consistent flags
across Windows and Jetson. Relies on TripoSR being installed either as a
console script `triposr` or a module `python -m triposr`.

Usage examples:
  python scripts/generate_3d_triposr.py --input path/to/image.jpg --out outputs --device auto
  python scripts/generate_3d_triposr.py --input path/to/folder --out outputs --device cuda --extra "--save-obj --save-ply"

Notes:
- This script does not ship TripoSR; install it via pip or from source.
- If both CLI and module invocations fail, we print install guidance.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def which(cmd: str) -> str | None:
    """Return full path to command if it exists in PATH, else None."""
    from shutil import which as _which

    return _which(cmd)


def build_triposr_command(input_arg: list[str], out_dir: Path, device: str, extra: str | None) -> list[str]:
    """Build the command to run TripoSR.

    Resolution order:
      1) `triposr infer ...` (if console script available)
      2) `python -m triposr infer ...` (if module available)
      3) `python external/TripoSR/run.py ...` (VAST-AI-Research repo layout)
    """
    # 1) Try console script
    cli = which("triposr")
    if cli:
        cmd = [cli, "infer", "--output", str(out_dir), "--device", device] + input_arg
        if extra:
            cmd += shlex.split(extra)
        return cmd

    # 2) Try python -m triposr
    module_cmd = [sys.executable, "-m", "triposr", "infer", "--output", str(out_dir), "--device", device]
    if extra:
        module_cmd += shlex.split(extra)
    module_cmd += input_arg

    # We can't easily verify module presence without importing; attempt later if needed.
    # 3) Fallback to VAST repo's run.py
    vast_repo = Path(__file__).resolve().parents[1] / "external" / "TripoSR"
    run_py = vast_repo / "run.py"
    if run_py.exists():
        # VAST run.py expects: python run.py <img1> <img2> ... --output-dir <out>
        # It doesn't use a --device flag; device selection is internal to torch.
        cmd = [sys.executable, str(run_py)] + input_arg + ["--output-dir", str(out_dir)]
        if extra:
            cmd += shlex.split(extra)
        return cmd

    # If no repo found, return the module invocation by default; errors will surface at runtime.
    return module_cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TripoSR on an image or folder of images.")
    parser.add_argument("--input", required=True, help="Path to input image (jpg/png) or a folder containing images")
    parser.add_argument("--out", required=True, help="Output folder for generated meshes")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use. 'auto' picks CUDA if available.",
    )
    parser.add_argument(
        "--extra",
        default=None,
        help="Extra raw flags to pass to TripoSR (quoted string), e.g. --extra \"--save-obj --save-ply\"",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        return 2

    # If device is auto, choose cuda when torch.cuda.is_available()
    device = args.device
    if device == "auto":
        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            # If torch isn't installed, we can't probe; default to cpu
            device = "cpu"

    # Support folder input by expanding to image list
    input_arg: list[str]
    if input_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        imgs = sorted(str(p) for p in input_path.iterdir() if p.suffix.lower() in exts)
        if not imgs:
            print(f"[ERROR] No images found in folder: {input_path}", file=sys.stderr)
            return 2
        input_arg = imgs
    else:
        input_arg = [str(input_path)]

    cmd = build_triposr_command(input_arg, out_dir, device, args.extra)
    print("[INFO] Running:")
    print(" ", " ".join(shlex.quote(c) for c in cmd))

    try:
        completed = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print(
            "[ERROR] Could not find TripoSR CLI or module.\n"
            "Install via: pip install triposr (if available) OR clone the TripoSR repo and run `python -m triposr`.\n"
            "See docs/windows-setup.md or docs/jetson-orin-setup.md for platform-specific steps.",
            file=sys.stderr,
        )
        return 3

    if completed.returncode != 0:
        print(f"[ERROR] TripoSR exited with code {completed.returncode}", file=sys.stderr)
        return completed.returncode

    print(f"[OK] Finished. Outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
