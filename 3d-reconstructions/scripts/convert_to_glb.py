#!/usr/bin/env python3
"""Convert a mesh (OBJ/PLY/STL/GLTF/GLB) to GLB using trimesh.

Example:
  python scripts/convert_to_glb.py --input outputs/sample/mesh.obj
  python scripts/convert_to_glb.py --input outputs/sample/mesh.ply --output converted/sample.glb

Requires `trimesh` (see requirements.txt). Large meshes may take a few seconds.
"""

import argparse
from pathlib import Path
import sys

import trimesh  # type: ignore

SUPPORTED_EXT = {".obj", ".ply", ".stl", ".glb", ".gltf"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a mesh file to GLB format")
    parser.add_argument("--input", required=True, help="Path to source mesh (OBJ/PLY/STL/GLTF/GLB)")
    parser.add_argument("--output", help="Optional explicit output path (defaults to <stem>.glb in same dir)")
    parser.add_argument("--merge", action="store_true", help="Merge geometries into single scene before export")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] Input mesh not found: {in_path}", file=sys.stderr)
        return 2

    ext = in_path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        print(f"[ERROR] Unsupported extension '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXT))}", file=sys.stderr)
        return 3

    out_path = Path(args.output) if args.output else in_path.with_suffix(".glb")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading mesh: {in_path}")
    mesh_or_scene = trimesh.load(in_path, force='scene')

    if isinstance(mesh_or_scene, trimesh.Scene):
        scene = mesh_or_scene
        if args.merge:
            print("[INFO] Merging scene geometries into a single mesh")
            merged = trimesh.util.concatenate(tuple(scene.dump()))
            export_obj = merged
        else:
            export_obj = scene
    else:
        export_obj = mesh_or_scene

    print(f"[INFO] Exporting GLB -> {out_path}")
    try:
        export_obj.export(out_path, file_type="glb")
    except Exception as e:
        print(f"[ERROR] Export failed: {e}", file=sys.stderr)
        return 4

    print(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
