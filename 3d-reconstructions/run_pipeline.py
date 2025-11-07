#!/usr/bin/env python3
"""
Video-to-3D Reconstruction Pipeline using COLMAP.

Steps:
1. Extract frames from input video at a fixed interval (seconds).
2. Run COLMAP feature extraction & matching.
3. Sparse reconstruction (mapper).
4. Undistort images.
5. Dense stereo + fusion.
6. (Optional) Poisson meshing.
7. Export PLY (and OBJ if Open3D available).

Requirements:
- COLMAP installed and colmap executable on PATH or provided via --colmap_path.
- ffmpeg (optional) for faster frame extraction; fallback to OpenCV.

Outputs are organized under a working directory structure:
work/
    images/                Extracted frames
    database.db            COLMAP database
    sparse/                Sparse reconstruction (0/ folder)
    dense/                 Undistorted + stereo + fusion results
    logs/                  Individual step logs
outputs/
    fused.ply              Fused dense point cloud
    meshed-poisson.ply     Poisson surface mesh (if generated)
    meshed.obj             OBJ converted from Poisson mesh (if Open3D installed)

Usage Example (PowerShell):
python run_pipeline.py --video sample.mp4 --interval 0.5 --work_dir work --colmap_path "C:\\Program Files\\COLMAP\\colmap.exe" --run_poisson --convert_obj
"""
import argparse
import subprocess
import shutil
import sys
import os
import math
import time
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm
from colorama import Fore, Style

try:
    import open3d as o3d  # optional
    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False


def log(msg: str, color: str = Fore.CYAN):
    print(color + msg + Style.RESET_ALL)


def run_cmd(cmd, log_file: Path):
    log(f"Running: {' '.join(cmd)}", Fore.YELLOW)
    with log_file.open('w', encoding='utf-8') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed (exit {ret}): {' '.join(cmd)}. See {log_file}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def extract_frames(video_path: Path, images_dir: Path, interval: float):
    ensure_dir(images_dir)
    # Try ffmpeg first for speed
    ffmpeg_exe = shutil.which('ffmpeg')
    if ffmpeg_exe:
        log("Extracting frames using ffmpeg...")
        # fps = 1/interval; use video timestamp-based extraction
        # select=not(mod(t,interval)) is approximate; instead use -vf fps=1/interval
        fps = 1.0 / interval
        output_pattern = str(images_dir / 'frame_%06d.jpg')
        cmd = [ffmpeg_exe, '-i', str(video_path), '-vf', f'fps={fps}', '-qscale:v', '2', output_pattern]
        subprocess.check_call(cmd)
        return
    # Fallback to OpenCV
    log("ffmpeg not found, falling back to OpenCV frame extraction (slower)...", Fore.MAGENTA)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(fps * interval)), 1)
    idx = 0
    saved = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    with tqdm(total=total, desc='Extracting frames') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                out_path = images_dir / f'frame_{saved:06d}.jpg'
                cv2.imwrite(str(out_path), frame)
                saved += 1
            idx += 1
            pbar.update(1)
    cap.release()


def colmap_feature_extractor(colmap: str, db: Path, images: Path, work: Path, camera_model: str, single_camera: bool, log_dir: Path, gpu_index: Optional[str]):
    cmd = [colmap, 'feature_extractor', '--database_path', str(db), '--image_path', str(images), '--ImageReader.camera_model', camera_model]
    if single_camera:
        cmd += ['--ImageReader.single_camera', '1']
    # GPU settings for SIFT
    if gpu_index is not None:
        cmd += ['--SiftExtraction.use_gpu', '1', '--SiftExtraction.gpu_index', gpu_index]
    run_cmd(cmd, log_dir / 'feature_extractor.log')


def colmap_exhaustive_matcher(colmap: str, db: Path, log_dir: Path):
    cmd = [colmap, 'exhaustive_matcher', '--database_path', str(db)]
    run_cmd(cmd, log_dir / 'exhaustive_matcher.log')

def colmap_sequential_matcher(colmap: str, db: Path, images: Path, log_dir: Path):
    # Sequential matcher assumes temporal ordering
    cmd = [colmap, 'sequential_matcher', '--database_path', str(db), '--image_path', str(images), '--SequentialMatching.overlap', '5']
    run_cmd(cmd, log_dir / 'sequential_matcher.log')

def colmap_vocab_tree_matcher(colmap: str, db: Path, images: Path, vocab_tree_path: Path, log_dir: Path):
    cmd = [colmap, 'vocab_tree_matcher', '--database_path', str(db), '--image_path', str(images), '--VocabTreeMatching.vocab_tree_path', str(vocab_tree_path)]
    run_cmd(cmd, log_dir / 'vocab_tree_matcher.log')


def colmap_mapper(colmap: str, db: Path, images: Path, sparse_dir: Path, log_dir: Path):
    ensure_dir(sparse_dir)
    cmd = [colmap, 'mapper', '--database_path', str(db), '--image_path', str(images), '--output_path', str(sparse_dir)]
    run_cmd(cmd, log_dir / 'mapper.log')


def colmap_image_undistorter(colmap: str, images: Path, sparse_model: Path, dense_dir: Path, log_dir: Path, max_image_size: int = 2000):
    ensure_dir(dense_dir)
    cmd = [colmap, 'image_undistorter', '--image_path', str(images), '--input_path', str(sparse_model), '--output_path', str(dense_dir), '--output_type', 'COLMAP', '--max_image_size', str(max_image_size)]
    run_cmd(cmd, log_dir / 'image_undistorter.log')


def colmap_patch_match_stereo(colmap: str, dense_dir: Path, log_dir: Path, gpu_index: Optional[str]):
    cmd = [colmap, 'patch_match_stereo', '--workspace_path', str(dense_dir), '--workspace_format', 'COLMAP', '--PatchMatchStereo.geom_consistency', '1']
    if gpu_index is not None:
        cmd += ['--PatchMatchStereo.gpu_index', gpu_index]
    run_cmd(cmd, log_dir / 'patch_match_stereo.log')


def colmap_stereo_fusion(colmap: str, dense_dir: Path, output_ply: Path, log_dir: Path):
    cmd = [colmap, 'stereo_fusion', '--workspace_path', str(dense_dir), '--workspace_format', 'COLMAP', '--input_type', 'geometric', '--output_path', str(output_ply)]
    run_cmd(cmd, log_dir / 'stereo_fusion.log')


def colmap_poisson_mesher(colmap: str, dense_dir: Path, output_ply: Path, log_dir: Path):
    cmd = [colmap, 'poisson_mesher', '--input_path', str(dense_dir / 'fused.ply'), '--output_path', str(output_ply)]
    run_cmd(cmd, log_dir / 'poisson_mesher.log')


def convert_ply_to_obj(ply_path: Path, obj_path: Path):
    if not HAS_OPEN3D:
        log("Open3D not installed; skipping OBJ conversion", Fore.MAGENTA)
        return False
    log("Converting PLY to OBJ using Open3D...")
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    if not mesh.has_triangles():
        log("PLY mesh has no triangles; skipping OBJ conversion", Fore.RED)
        return False
    o3d.io.write_triangle_mesh(str(obj_path), mesh)
    return True


def main():
    parser = argparse.ArgumentParser(description="Video to 3D reconstruction pipeline using COLMAP")
    parser.add_argument('--video', required=True, help='Input video file (.mp4, .mov, etc.)')
    parser.add_argument('--interval', type=float, default=0.5, help='Frame extraction interval in seconds (default 0.5; overridden by --mode unless custom)')
    parser.add_argument('--max_frames', type=int, default=60, help='Cap number of frames after extraction (default 60; 0 = no cap)')
    parser.add_argument('--work_dir', default='work', help='Working directory for intermediate outputs')
    parser.add_argument('--colmap_path', default='colmap', help='Path to colmap executable if not in PATH')
    parser.add_argument('--camera_model', default='PINHOLE', choices=['PINHOLE','SIMPLE_PINHOLE','RADIAL','SIMPLE_RADIAL','OPENCV','OPENCV_FISHEYE'], help='Camera model for feature extraction')
    parser.add_argument('--single_camera', action='store_true', help='Treat all images as from a single camera')
    parser.add_argument('--run_poisson', action='store_true', help='Run Poisson meshing step')
    parser.add_argument('--convert_obj', action='store_true', help='Convert meshed PLY to OBJ if Open3D available')
    parser.add_argument('--skip_dense', action='store_true', help='Skip dense reconstruction (fast sparse only)')
    parser.add_argument('--mode', choices=['fast','full'], help='Preset quality/runtime mode: fast or full (overrides interval & run_poisson defaults)')
    parser.add_argument('--reuse_frames', action='store_true', help='Reuse existing extracted frames if present')
    parser.add_argument('--gpu_index', help='Override GPU index for COLMAP (defaults to COLMAP_GPU_INDEX env or 0 if available)')
    parser.add_argument('--dense_size', type=int, default=2000, help='Max image size for dense undistortion (default 2000)')
    parser.add_argument('--dry_run', action='store_true', help='Print planned commands without executing pipeline steps')
    parser.add_argument('--matcher', choices=['exhaustive','sequential','vocab'], default='exhaustive', help='Image matching strategy')
    parser.add_argument('--vocab_tree', type=str, help='Path to vocabulary tree file for vocab matcher')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing work directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        log(f"Video not found: {video_path}", Fore.RED)
        sys.exit(1)

    # Resolve COLMAP executable
    colmap_exec = args.colmap_path
    if colmap_exec == 'colmap':
        found = shutil.which('colmap') or shutil.which('colmap.exe')
        if not found:
            log("COLMAP executable not found in PATH. Provide with --colmap_path.", Fore.RED)
            sys.exit(1)
        colmap_exec = found
    else:
        if not Path(colmap_exec).exists():
            # Still allow PATH resolution
            found = shutil.which(colmap_exec)
            if not found:
                log(f"COLMAP not found at {colmap_exec}", Fore.RED)
                sys.exit(1)
            colmap_exec = found

    work_dir = Path(args.work_dir)
    images_dir = work_dir / 'images'
    db_path = work_dir / 'database.db'
    sparse_dir = work_dir / 'sparse'
    dense_dir = work_dir / 'dense'
    log_dir = work_dir / 'logs'
    outputs_dir = Path('outputs')

    global_start = time.time()

    if args.dry_run:
        log("Dry-run mode: no commands will be executed.", Fore.MAGENTA)

    if args.overwrite and work_dir.exists() and not args.dry_run:
        shutil.rmtree(work_dir)
    ensure_dir(work_dir)
    ensure_dir(log_dir)
    ensure_dir(outputs_dir)

    if db_path.exists() and not args.overwrite and not args.dry_run:
        log("Database already exists. Use --overwrite to rebuild.", Fore.RED)
        sys.exit(1)

    # Determine mode adjustments
    effective_interval = args.interval
    effective_run_poisson = args.run_poisson
    if args.mode == 'fast':
        effective_interval = 0.7 if args.interval == 0.5 else args.interval  # allow user custom override
        effective_run_poisson = False
    elif args.mode == 'full':
        effective_interval = 0.3 if args.interval == 0.5 else args.interval
        effective_run_poisson = True if not args.run_poisson else args.run_poisson

    # Resolve GPU index
    gpu_index = args.gpu_index if args.gpu_index else os.environ.get('COLMAP_GPU_INDEX')
    if gpu_index is None:
        gpu_index = '0'  # default assumption

    # Step 1: Extract frames (unless reusing)
    if args.reuse_frames and images_dir.exists() and any(images_dir.glob('*.jpg')):
        log("Step 1: Reusing existing frames", Fore.MAGENTA)
    else:
        log("Step 1: Extracting frames")
        t0 = time.time()
        if not args.dry_run:
            extract_frames(video_path, images_dir, effective_interval)
        log(f"Step 1 took {time.time() - t0:.1f}s", Fore.CYAN)

    # Frame capping
    if args.max_frames != 0:
        frame_list = sorted(images_dir.glob('*.jpg'))
        num_frames = len(frame_list)
        if num_frames > args.max_frames:
            log(f"Capping frames to {args.max_frames} (currently {num_frames}) for speed...", Fore.MAGENTA)
            if not args.dry_run:
                for f in frame_list[args.max_frames:]:
                    f.unlink()

    # Step 2: COLMAP feature extraction
    log("Step 2: Feature extraction")
    t0 = time.time()
    if args.dry_run:
        log(f"DRY-RUN: feature_extractor on {len(list(images_dir.glob('*.jpg')))} images", Fore.YELLOW)
    else:
        colmap_feature_extractor(colmap_exec, db_path, images_dir, work_dir, args.camera_model, args.single_camera, log_dir, gpu_index)
    log(f"Step 2 took {time.time() - t0:.1f}s", Fore.CYAN)

    # Step 3: Matching
    # Step 3: Matching (exhaustive / sequential / vocab)
    log(f"Step 3: {args.matcher.capitalize()} matching")
    t0 = time.time()
    if args.matcher == 'exhaustive':
        if args.dry_run:
            log("DRY-RUN: exhaustive_matcher", Fore.YELLOW)
        else:
            colmap_exhaustive_matcher(colmap_exec, db_path, log_dir)
    elif args.matcher == 'sequential':
        if args.dry_run:
            log("DRY-RUN: sequential_matcher", Fore.YELLOW)
        else:
            colmap_sequential_matcher(colmap_exec, db_path, images_dir, log_dir)
    elif args.matcher == 'vocab':
        if not args.vocab_tree:
            log("Vocab matcher selected but --vocab_tree not provided.", Fore.RED)
            sys.exit(1)
        vocab_path = Path(args.vocab_tree)
        if not vocab_path.exists():
            log(f"Vocabulary tree file not found: {vocab_path}", Fore.RED)
            sys.exit(1)
        if args.dry_run:
            log(f"DRY-RUN: vocab_tree_matcher using {vocab_path}", Fore.YELLOW)
        else:
            colmap_vocab_tree_matcher(colmap_exec, db_path, images_dir, vocab_path, log_dir)
    log(f"Step 3 took {time.time() - t0:.1f}s", Fore.CYAN)

    # Step 4: Sparse reconstruction (mapper)
    log("Step 4: Sparse reconstruction (mapper)")
    t0 = time.time()
    if args.dry_run:
        log("DRY-RUN: mapper", Fore.YELLOW)
    else:
        colmap_mapper(colmap_exec, db_path, images_dir, sparse_dir, log_dir)
    log(f"Step 4 took {time.time() - t0:.1f}s", Fore.CYAN)

    # Expect sparse_dir/0 as first model
    first_model = sparse_dir / '0'
    if not first_model.exists() and not args.dry_run:
        log("Sparse model directory not found. Mapper may have failed.", Fore.RED)
        sys.exit(1)

    if args.skip_dense:
        log("Skipping dense reconstruction per --skip_dense. Exporting sparse points.", Fore.MAGENTA)
        if not args.dry_run:
            sparse_points = first_model / 'points3D.txt'
            if sparse_points.exists():
                out_sparse = outputs_dir / 'sparse_points3D.txt'
                shutil.copy2(sparse_points, out_sparse)
                log(f"Sparse reconstruction saved to {out_sparse}", Fore.GREEN)
            else:
                log("Sparse points3D.txt not found; mapper may have failed to produce points.", Fore.RED)
        # Summary before exit
        log("------ SUMMARY ------", Fore.YELLOW)
        log(f"Frames used: {len(list(images_dir.glob('*.jpg')))}")
        log(f"Mode: {args.mode or 'custom'} | Interval: {effective_interval:.2f}s")
        log(f"GPU Index: {gpu_index}")
        log(f"Matcher: {args.matcher}")
        log(f"Total time: {time.time() - global_start:.1f}s")
        sys.exit(0)

    # Step 5: Image undistortion
    log("Step 5: Undistorting images")
    t0 = time.time()
    if args.dry_run:
        log("DRY-RUN: image_undistorter", Fore.YELLOW)
    else:
        colmap_image_undistorter(colmap_exec, images_dir, first_model, dense_dir, log_dir, args.dense_size)
    log(f"Step 5 took {time.time() - t0:.1f}s", Fore.CYAN)

    # Step 6: Patch-match stereo
    log("Step 6: PatchMatch stereo")
    t0 = time.time()
    if args.dry_run:
        log("DRY-RUN: patch_match_stereo", Fore.YELLOW)
    else:
        colmap_patch_match_stereo(colmap_exec, dense_dir, log_dir, gpu_index)
    log(f"Step 6 took {time.time() - t0:.1f}s", Fore.CYAN)

    # Step 7: Stereo fusion -> fused.ply
    log("Step 7: Stereo fusion")
    fused_ply = dense_dir / 'fused.ply'
    t0 = time.time()
    if args.dry_run:
        log("DRY-RUN: stereo_fusion", Fore.YELLOW)
    else:
        colmap_stereo_fusion(colmap_exec, dense_dir, fused_ply, log_dir)
    log(f"Step 7 took {time.time() - t0:.1f}s", Fore.CYAN)

    # Copy fused result to outputs
    if fused_ply.exists() and not args.dry_run:
        out_fused = outputs_dir / 'fused.ply'
        shutil.copy2(fused_ply, out_fused)
        log(f"Fused point cloud saved to {out_fused}", Fore.GREEN)
    else:
        if not args.dry_run:
            log("Fused point cloud not found; stereo fusion may have failed.", Fore.RED)

    meshed_poisson = None
    if effective_run_poisson and fused_ply.exists() and not args.dry_run:
        # Step 8: Poisson mesher
        log("Step 8: Poisson meshing")
        meshed_poisson = outputs_dir / 'meshed-poisson.ply'
        t0 = time.time()
        colmap_poisson_mesher(colmap_exec, dense_dir, meshed_poisson, log_dir)
        log(f"Step 8 took {time.time() - t0:.1f}s", Fore.CYAN)
        if meshed_poisson.exists():
            log(f"Poisson mesh saved to {meshed_poisson}", Fore.GREEN)
        else:
            log("Poisson mesh was not generated.", Fore.RED)

    if args.convert_obj and meshed_poisson and meshed_poisson.exists() and not args.dry_run:
        obj_path = outputs_dir / 'meshed.obj'
        ok = convert_ply_to_obj(meshed_poisson, obj_path)
        if ok:
            log(f"OBJ mesh saved to {obj_path}", Fore.GREEN)

    log("Pipeline complete.")
    # Summary block
    log("------ SUMMARY ------", Fore.YELLOW)
    log(f"Frames used: {len(list(images_dir.glob('*.jpg')))}")
    log(f"Mode: {args.mode or 'custom'} | Interval: {effective_interval:.2f}s")
    log(f"GPU Index: {gpu_index}")
    log(f"Matcher: {args.matcher}")
    log(f"Total time: {time.time() - global_start:.1f}s")


if __name__ == '__main__':
    main()
