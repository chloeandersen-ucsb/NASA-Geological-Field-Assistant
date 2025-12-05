#!/usr/bin/env python3
"""
End-to-End Video to 3D Scene Reconstruction

Simplified automated pipeline:
1. Extract frames from video (sampling + quality filtering)
2. Auto-select backend unless user explicitly chooses COLMAP / DUSt3R / MASt3R
     Decision tree (backend=auto):
         - If single dominant object detected -> TripoSR (single-image mesh)
         - Else if filtered frame count < 15 -> DUSt3R (robust with few views)
         - Else -> MASt3R (default multiview quality)
     (COLMAP only when explicitly requested via --backend colmap)
3. Run reconstruction producing point cloud / mesh

Usage:
        # Auto backend (recommended):
        python scripts/generate_scene_from_video.py video.mp4

        # Force COLMAP (more frames recommended):
        python scripts/generate_scene_from_video.py video.mp4 --backend colmap --max-frames 60 --device cuda

        # Force DUSt3R quick preview:
        python scripts/generate_scene_from_video.py video.mp4 --backend dust3r --max-frames 12

        # Force MASt3R multiview:
        python scripts/generate_scene_from_video.py video.mp4 --backend mast3r --max-frames 30

        # Custom extraction parameters:
        python scripts/generate_scene_from_video.py video.mp4 --fps 3 --blur-threshold 150 --no-similarity-filter

        # Keep extracted frames:
        python scripts/generate_scene_from_video.py video.mp4 --keep-frames --output custom_output/
"""

import argparse
import sys
import subprocess
from pathlib import Path
import shutil
import time
import cv2
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Single-object heuristic
from object_select import is_single_object


def compute_blur_score(image_path: Path) -> float:
    """Compute Laplacian variance (sharpness proxy)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def calculate_adaptive_fps(video_path: Path) -> float:
    """Calculate adaptive FPS based on video duration.
    
    Short videos need higher sampling to capture enough frames.
    Long videos can use lower sampling to avoid redundancy.
    
    Returns:
        fps: adaptive frames per second
    """
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    # Adaptive FPS based on duration
    if duration < 10:
        fps = 4.0
    elif duration < 20:
        fps = 3.0
    elif duration < 60:
        fps = 2.0
    else:
        fps = 1.0
    
    return fps


def extract_frames_from_video(
    video_path,
    frames_dir,
    fps=None,
    every_n_frames=None,
    max_frames=20,
    filter_quality=True,
    filter_similarity=True,
    blur_threshold=100.0,
    exposure_min=30,
    exposure_max=220,
    similarity_threshold=0.85
):
    """
    Extract frames from video using extract_frames.py
    
    Returns:
        frames_dir: Path to extracted frames
        num_frames: number of extracted frames
    """
    print("=" * 80)
    print("STEP 1: EXTRACTING FRAMES FROM VIDEO")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        'scripts/extract_frames.py',
        str(video_path),
        '--output', str(frames_dir)
    ]
    
    # Sampling
    if fps is not None:
        cmd.extend(['--fps', str(fps)])
    elif every_n_frames is not None:
        cmd.extend(['--every-n-frames', str(every_n_frames)])
    
    if max_frames is not None:
        cmd.extend(['--max-frames', str(max_frames)])
    
    # Quality filtering
    if filter_quality:
        cmd.append('--filter-quality')
        cmd.extend(['--blur-threshold', str(blur_threshold)])
        cmd.extend(['--exposure-min', str(exposure_min)])
        cmd.extend(['--exposure-max', str(exposure_max)])
    
    if filter_similarity:
        cmd.append('--filter-similarity')
        cmd.extend(['--similarity-threshold', str(similarity_threshold)])
    
    print(f"[CMD] {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        raise RuntimeError("Frame extraction failed")
    
    # Count extracted frames
    frames = list(frames_dir.glob('*.jpg')) + list(frames_dir.glob('*.png'))
    num_frames = len(frames)
    
    if num_frames == 0:
        raise RuntimeError("No frames were extracted")
    
    print(f"\n[OK] Extracted {num_frames} frames to {frames_dir}")
    
    return frames_dir, num_frames


def run_reconstruction(
    frames_dir,
    output_dir,
    backend='mast3r',
    device='cpu',
    voxel_size=None
):
    """
    Run 3D reconstruction on extracted frames.
    
    Args:
        frames_dir: directory containing extracted frames
        output_dir: output directory for reconstruction
        backend: 'colmap', 'dust3r', or 'mast3r'
        device: 'cpu' or 'cuda'
        voxel_size: optional voxel downsampling size
    
    Returns:
        output_dir: Path to reconstruction output
    """
    print("\n" + "=" * 80)
    print(f"STEP 2: 3D RECONSTRUCTION ({backend.upper()})")
    print("=" * 80)
    
    if backend == 'colmap':
        script = 'scripts/generate_scene_multiview_pycolmap.py'
        cmd = [
            sys.executable,
            script,
            str(frames_dir),
            '--output', str(output_dir),
            '--device', device
        ]
        if voxel_size:
            cmd.extend(['--downsample', str(voxel_size)])
    
    elif backend in ['dust3r', 'mast3r']:
        script = 'scripts/generate_scene_multiview_dust3r.py'
        cmd = [
            sys.executable,
            script,
            str(frames_dir),
            '--output', str(output_dir),
            '--device', device,
            '--model', backend
        ]
        if voxel_size:
            cmd.extend(['--voxel-size', str(voxel_size)])
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'colmap', 'dust3r', or 'mast3r'")
    
    print(f"[CMD] {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"{backend} reconstruction failed")
    
    print(f"\n[OK] Reconstruction complete: {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end video to 3D scene reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick preview (10 frames, MASt3R)
  python scripts/generate_scene_from_video.py video.mp4 --max-frames 10
  
  # High quality (50 frames, COLMAP, GPU)
  python scripts/generate_scene_from_video.py video.mp4 --max-frames 50 --backend colmap --device cuda
  
  # Custom quality thresholds
  python scripts/generate_scene_from_video.py video.mp4 --blur-threshold 150 --fps 2
        """
    )
    
    # Input/output
    parser.add_argument('video', type=str, help="Input video file (MP4, MOV, AVI, etc.)")
    parser.add_argument('--output', type=str, default=None,
                       help="Output directory (default: outputs/<video_name>_3d)")
    parser.add_argument('--keep-frames', action='store_true',
                       help="Keep extracted frames after reconstruction")
    
    # Frame extraction
    parser.add_argument('--fps', type=float, default=None,
                       help="Extract frames at this FPS (default: adaptive based on video length)")
    parser.add_argument('--every-n-frames', type=int, default=None,
                       help="Extract every N frames (alternative to --fps)")
    parser.add_argument('--max-frames', type=int, default=30,
                       help="Maximum frames to extract (default: 30)")
    
    # Quality filtering
    parser.add_argument('--no-quality-filter', action='store_true',
                       help="Disable blur/exposure filtering")
    parser.add_argument('--no-similarity-filter', action='store_true',
                       help="Disable similarity-based deduplication")
    parser.add_argument('--blur-threshold', type=float, default=60.0,
                       help="Minimum blur score (default: 60)")
    parser.add_argument('--exposure-min', type=float, default=20,
                       help="Minimum brightness (default: 20)")
    parser.add_argument('--exposure-max', type=float, default=240,
                       help="Maximum brightness (default: 240)")
    parser.add_argument('--similarity-threshold', type=float, default=0.90,
                       help="Max frame similarity to keep (default: 0.90)")
    
    # Reconstruction backend
    parser.add_argument('--backend', type=str, default='auto',
                       choices=['auto', 'colmap', 'dust3r', 'mast3r'],
                       help="Backend selection. 'auto' applies decision tree (default: auto)")
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help="Device for reconstruction (default: cpu)")
    parser.add_argument('--voxel-size', type=float, default=None,
                       help="Voxel downsampling size in meters (default: 0.05)")
    
    args = parser.parse_args()
    
    # Validate input
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)
    
    # Setup output directory
    if args.output is None:
        output_base = Path('outputs') / f"{video_path.stem}_3d"
    else:
        output_base = Path(args.output)
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    frames_dir = output_base / 'frames'
    reconstruction_dir = output_base / 'reconstruction'
    
    # Calculate adaptive FPS if not specified
    if args.fps is None:
        args.fps = calculate_adaptive_fps(video_path)
        print(f"[INFO] Using adaptive FPS: {args.fps} for video length")
    
    print("=" * 80)
    print("VIDEO TO 3D RECONSTRUCTION PIPELINE")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Output: {output_base}")
    print(f"Backend request: {args.backend}")
    print(f"Device: {args.device}")
    print(f"Max frames: {args.max_frames}")
    print("=" * 80)
    
    total_start = time.time()
    
    try:
        # Step 1: Extract frames
        frames_dir, num_frames = extract_frames_from_video(
            video_path,
            frames_dir,
            fps=args.fps,
            every_n_frames=args.every_n_frames,
            max_frames=args.max_frames,
            filter_quality=not args.no_quality_filter,
            filter_similarity=not args.no_similarity_filter,
            blur_threshold=args.blur_threshold,
            exposure_min=args.exposure_min,
            exposure_max=args.exposure_max,
            similarity_threshold=args.similarity_threshold
        )
        
        # Global minimum frame count enforcement
        GLOBAL_MIN_FRAMES = 8
        if num_frames < GLOBAL_MIN_FRAMES:
            print(f"\n{'='*80}")
            print("INSUFFICIENT FRAMES FOR RECONSTRUCTION")
            print("=" * 80)
            print(f"Extracted frames: {num_frames}")
            print(f"Required minimum: {GLOBAL_MIN_FRAMES}")
            print(f"\n[SOLUTION] Move the camera around the scene to provide more parallax.")
            print(f"\nTips for better frame capture:")
            print(f"  - Circle around the object/scene")
            print(f"  - Capture from multiple heights and angles")
            print(f"  - Ensure steady, well-lit footage")
            print(f"  - Record longer video (15-30 seconds recommended)")
            print(f"\nOr adjust extraction parameters:")
            print(f"  --blur-threshold 40")
            print(f"  --no-quality-filter")
            print(f"  --max-frames 50")
            sys.exit(1)
        
        # Decide backend
        requested_backend = args.backend
        selected_backend = requested_backend
        reason = "explicit user choice"

        if requested_backend == 'auto':
            # Single object test (sample mid frame for speed)
            frame_list = sorted(list(frames_dir.glob('*.jpg')) + list(frames_dir.glob('*.png')))
            sample_frame = frame_list[len(frame_list)//2] if frame_list else None
            single_object = False
            if sample_frame:
                try:
                    single_object = is_single_object(str(sample_frame))
                except Exception as _e:
                    print(f"[WARN] Single-object detection failed ({_e}); proceeding without object shortcut")
            if single_object:
                selected_backend = 'tripo'
                reason = 'single dominant object detected'
            elif num_frames < 15:
                selected_backend = 'dust3r'
                reason = f'only {num_frames} frames (<15) → robust few-view backend'
            else:
                selected_backend = 'mast3r'
                reason = f'{num_frames} frames (>=15) → default multiview quality'

        print("\n" + "=" * 80)
        print("BACKEND SELECTION")
        print("=" * 80)
        print(f"Requested: {requested_backend}")
        print(f"Selected:  {selected_backend}")
        print(f"Reason:    {reason}")

        if selected_backend == 'tripo':
            # Choose best frame (sharpest) among extracted frames
            frame_candidates = sorted(list(frames_dir.glob('*.jpg')) + list(frames_dir.glob('*.png')))
            if not frame_candidates:
                raise RuntimeError("No frames found for TripoSR path")
            sharpest = max(frame_candidates, key=compute_blur_score)
            tripo_out = output_base / 'tripo_mesh'
            tripo_out.mkdir(parents=True, exist_ok=True)
            print("\n" + "=" * 80)
            print("STEP 2: SINGLE-IMAGE MESH (TRIPOSR)")
            print("=" * 80)
            cmd = [
                sys.executable,
                'scripts/run_triposr_isolated.py',
                str(sharpest),
                '--output-dir', str(tripo_out),
                '--device', args.device if args.device != 'cpu' else 'cpu',
                '--mc-resolution', '256'
            ]
            print(f"[CMD] {' '.join(cmd)}\n")
            r = subprocess.run(cmd, capture_output=False, text=True)
            if r.returncode != 0:
                raise RuntimeError("TripoSR reconstruction failed")
            reconstruction_dir = tripo_out
        else:
            # Check minimum frame count for selected multiview backend
            min_frames_required = {
                'colmap': 10,
                'dust3r': 2,
                'mast3r': 2
            }
            if selected_backend in min_frames_required and num_frames < min_frames_required[selected_backend]:
                print(f"\n[ERROR] {selected_backend} requires at least {min_frames_required[selected_backend]} frames")
                print(f"        Only extracted {num_frames} frames")
                print(f"[HINT] Try:")
                print(f"  - Increase --max-frames")
                print(f"  - Lower --blur-threshold")
                print(f"  - Disable filtering with --no-quality-filter")
                sys.exit(1)
            reconstruction_dir = run_reconstruction(
                frames_dir,
                reconstruction_dir,
                backend=selected_backend,
                device=args.device,
                voxel_size=args.voxel_size
            )
        
        # Cleanup frames if requested
        if not args.keep_frames:
            print(f"\n[INFO] Cleaning up extracted frames...")
            shutil.rmtree(frames_dir)
            print(f"[OK] Removed {frames_dir}")
        
        # Summary
        total_time = time.time() - total_start
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Frames extracted: {num_frames}")
        print(f"Output directory: {output_base}")
        print(f"Selected backend: {selected_backend} ({reason})")
        print(f"\nReconstruction results:")
        if selected_backend == 'tripo':
            print(f"  {reconstruction_dir} (mesh outputs)")
        else:
            print(f"  {reconstruction_dir}/fused_pointcloud.ply")
            if selected_backend == 'colmap':
                print(f"  {reconstruction_dir}/colmap_sparse/")
            else:
                print(f"  {reconstruction_dir}/cameras.txt")
        
        print(f"\n[NEXT] View results:")
        print(f"  Load {reconstruction_dir} in 3D viewer")
        print(f"  http://localhost:5000/viewer/index.html")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
