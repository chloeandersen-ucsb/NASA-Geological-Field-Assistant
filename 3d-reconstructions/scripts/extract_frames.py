#!/usr/bin/env python3
"""
Video Frame Extraction and Quality Filtering for 3D Reconstruction

Extracts frames from video files and applies intelligent filtering to select
the best frames for multi-view 3D reconstruction.

Quality metrics:
- Blur detection (Laplacian variance)
- Exposure analysis (brightness histogram)
- Motion estimation (optical flow magnitude)
- Frame similarity (SSIM-based deduplication)

Usage:
    # Extract every 10th frame
    python scripts/extract_frames.py video.mp4 --output frames/ --every-n-frames 10
    
    # Extract at 2 FPS with quality filtering
    python scripts/extract_frames.py video.mp4 --fps 2 --filter-quality
    
    # Extract best 20 frames automatically
    python scripts/extract_frames.py video.mp4 --max-frames 20 --filter-quality --filter-similarity
    
    # Custom quality thresholds
    python scripts/extract_frames.py video.mp4 --blur-threshold 150 --exposure-min 30 --exposure-max 220
"""

import argparse
import sys
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image


class VideoFrameExtractor:
    """Extract and filter frames from video files."""
    
    def __init__(self, video_path, output_dir=None):
        """
        Initialize frame extractor.
        
        Args:
            video_path: path to video file
            output_dir: directory to save extracted frames (auto-created if None)
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path('outputs') / f"{self.video_path.stem}_frames"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        print(f"[INFO] Video: {self.video_path.name}")
        print(f"[INFO] Resolution: {self.width}x{self.height}")
        print(f"[INFO] FPS: {self.fps:.2f}")
        print(f"[INFO] Total frames: {self.total_frames:,}")
        print(f"[INFO] Duration: {self.duration:.2f}s")
    
    def __del__(self):
        """Release video capture."""
        if hasattr(self, 'cap'):
            self.cap.release()
    
    def compute_blur_score(self, frame):
        """
        Compute blur score using Laplacian variance.
        Higher values = sharper image.
        
        Args:
            frame: BGR image
        
        Returns:
            blur_score: float (typically 50-500 for normal images)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score = laplacian.var()
        return score
    
    def compute_exposure_score(self, frame):
        """
        Compute exposure quality score.
        Returns (mean_brightness, overexposure_ratio, underexposure_ratio).
        
        Args:
            frame: BGR image
        
        Returns:
            mean: average brightness (0-255)
            overexposed: ratio of pixels > 240
            underexposed: ratio of pixels < 20
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = gray.mean()
        overexposed = (gray > 240).sum() / gray.size
        underexposed = (gray < 20).sum() / gray.size
        return mean, overexposed, underexposed
    
    def compute_motion_score(self, prev_frame, curr_frame):
        """
        Compute optical flow magnitude between consecutive frames.
        Higher values = more motion.
        
        Args:
            prev_frame: previous BGR frame
            curr_frame: current BGR frame
        
        Returns:
            motion_score: average optical flow magnitude
        """
        if prev_frame is None:
            return 0.0
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Compute magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        motion_score = magnitude.mean()
        
        return motion_score
    
    def compute_similarity(self, frame1, frame2):
        """
        Compute structural similarity between two frames.
        Returns SSIM value (0-1, higher = more similar).
        
        Args:
            frame1, frame2: BGR images
        
        Returns:
            similarity: float 0-1
        """
        # Resize for faster computation
        h, w = frame1.shape[:2]
        max_dim = 256
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame1 = cv2.resize(frame1, (new_w, new_h))
            frame2 = cv2.resize(frame2, (new_w, new_h))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM
        from skimage.metrics import structural_similarity
        similarity = structural_similarity(gray1, gray2)
        
        return similarity
    
    def extract_frames(
        self,
        fps=None,
        every_n_frames=None,
        max_frames=None,
        filter_quality=False,
        filter_similarity=False,
        blur_threshold=60.0,
        exposure_min=20,
        exposure_max=240,
        overexposure_threshold=0.20,
        underexposure_threshold=0.20,
        motion_threshold=None,
        similarity_threshold=0.90,
        save_format='jpg',
        save_quality=95
    ):
        """
        Extract frames from video with optional quality filtering.
        
        Args:
            fps: extract at this FPS (None = use every_n_frames)
            every_n_frames: extract every N frames (overridden by fps)
            max_frames: maximum number of frames to extract
            filter_quality: apply blur/exposure filtering
            filter_similarity: remove very similar frames
            blur_threshold: minimum Laplacian variance (higher = sharper required)
            exposure_min: minimum acceptable mean brightness
            exposure_max: maximum acceptable mean brightness
            overexposure_threshold: max ratio of overexposed pixels
            underexposure_threshold: max ratio of underexposed pixels
            motion_threshold: min motion score (None = no filtering)
            similarity_threshold: max SSIM for keeping frames (0.85 = keep if <85% similar)
            save_format: 'jpg' or 'png'
            save_quality: JPEG quality (1-100)
        
        Returns:
            extracted_frames: list of paths to saved frames
        """
        start_time = time.time()
        
        # Determine frame sampling interval
        if fps is not None:
            frame_interval = max(1, int(self.fps / fps))
            print(f"[INFO] Extracting at {fps} FPS (every {frame_interval} frames)")
        elif every_n_frames is not None:
            frame_interval = every_n_frames
            effective_fps = self.fps / frame_interval
            print(f"[INFO] Extracting every {frame_interval} frames (~{effective_fps:.2f} FPS)")
        else:
            frame_interval = 1
            print(f"[INFO] Extracting all frames")
        
        # Tracking
        extracted_frames = []
        frame_metadata = []
        
        frame_idx = 0
        extracted_count = 0
        skipped_blur = 0
        skipped_exposure = 0
        skipped_motion = 0
        skipped_similarity = 0
        
        prev_frame = None
        last_kept_frame = None
        
        print(f"\n[INFO] Processing frames...")
        if filter_quality:
            print(f"  Quality filters:")
            print(f"    Blur threshold: {blur_threshold}")
            print(f"    Exposure range: {exposure_min}-{exposure_max}")
            print(f"    Max overexposure: {overexposure_threshold*100:.1f}%")
            print(f"    Max underexposure: {underexposure_threshold*100:.1f}%")
            if motion_threshold is not None:
                print(f"    Min motion: {motion_threshold}")
        if filter_similarity:
            print(f"  Similarity filter: {similarity_threshold}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Skip frames based on interval
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                prev_frame = frame
                continue
            
            # Check max frames limit
            if max_frames and extracted_count >= max_frames:
                break
            
            # Quality filtering
            if filter_quality:
                # Blur detection
                blur_score = self.compute_blur_score(frame)
                if blur_score < blur_threshold:
                    skipped_blur += 1
                    frame_idx += 1
                    prev_frame = frame
                    continue
                
                # Exposure analysis
                mean_brightness, overexposed, underexposed = self.compute_exposure_score(frame)
                if (mean_brightness < exposure_min or 
                    mean_brightness > exposure_max or
                    overexposed > overexposure_threshold or
                    underexposed > underexposure_threshold):
                    skipped_exposure += 1
                    frame_idx += 1
                    prev_frame = frame
                    continue
                
                # Motion filtering (optional)
                if motion_threshold is not None:
                    motion_score = self.compute_motion_score(prev_frame, frame)
                    if motion_score < motion_threshold:
                        skipped_motion += 1
                        frame_idx += 1
                        prev_frame = frame
                        continue
            else:
                blur_score = 0
                mean_brightness = 0
                overexposed = 0
                underexposed = 0
                motion_score = 0
            
            # Similarity filtering
            if filter_similarity and last_kept_frame is not None:
                similarity = self.compute_similarity(last_kept_frame, frame)
                if similarity > similarity_threshold:
                    skipped_similarity += 1
                    frame_idx += 1
                    prev_frame = frame
                    continue
            
            # Save frame
            timestamp = frame_idx / self.fps if self.fps > 0 else frame_idx
            frame_filename = f"frame_{extracted_count:05d}_t{timestamp:.2f}s.{save_format}"
            frame_path = self.output_dir / frame_filename
            
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            if save_format.lower() == 'jpg':
                pil_image.save(frame_path, 'JPEG', quality=save_quality)
            else:
                pil_image.save(frame_path, 'PNG')
            
            extracted_frames.append(frame_path)
            frame_metadata.append({
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'blur_score': blur_score,
                'brightness': mean_brightness,
                'path': frame_path
            })
            
            extracted_count += 1
            last_kept_frame = frame.copy()
            
            # Progress
            if extracted_count % 10 == 0:
                print(f"  Extracted: {extracted_count} frames")
            
            frame_idx += 1
            prev_frame = frame
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n[COMPLETE] Frame extraction finished in {total_time:.2f}s")
        print(f"  Total processed: {frame_idx:,} frames")
        print(f"  Extracted: {extracted_count} frames")
        
        if filter_quality or filter_similarity:
            print(f"\n  Filtered out:")
            if filter_quality:
                print(f"    Blur: {skipped_blur}")
                print(f"    Exposure: {skipped_exposure}")
                if motion_threshold is not None:
                    print(f"    Motion: {skipped_motion}")
            if filter_similarity:
                print(f"    Similarity: {skipped_similarity}")
        
        print(f"\n[INFO] Frames saved to: {self.output_dir}")
        
        # Save metadata
        metadata_path = self.output_dir / 'extraction_metadata.txt'
        with open(metadata_path, 'w') as f:
            f.write(f"Video: {self.video_path}\n")
            f.write(f"Resolution: {self.width}x{self.height}\n")
            f.write(f"Original FPS: {self.fps:.2f}\n")
            f.write(f"Extraction interval: every {frame_interval} frames\n")
            f.write(f"Total extracted: {extracted_count}\n")
            f.write(f"\nFrame details:\n")
            for meta in frame_metadata:
                f.write(f"  {meta['path'].name}: blur={meta['blur_score']:.1f}, brightness={meta['brightness']:.1f}\n")
        
        print(f"[INFO] Metadata saved to: {metadata_path}")
        
        return extracted_frames


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video for 3D reconstruction")
    parser.add_argument('video', type=str, help="Input video file")
    parser.add_argument('--output', type=str, default=None, help="Output directory for frames")
    
    # Sampling options
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument('--fps', type=float, default=None, 
                             help="Extract frames at this FPS (e.g., 2 = 2 frames per second)")
    sample_group.add_argument('--every-n-frames', type=int, default=None,
                             help="Extract every N frames (e.g., 10 = every 10th frame)")
    
    parser.add_argument('--max-frames', type=int, default=None,
                       help="Maximum number of frames to extract")
    
    # Quality filtering
    parser.add_argument('--filter-quality', action='store_true',
                       help="Enable blur and exposure filtering")
    parser.add_argument('--blur-threshold', type=float, default=60.0,
                       help="Minimum blur score (Laplacian variance, default 60)")
    parser.add_argument('--exposure-min', type=float, default=20,
                       help="Minimum mean brightness (0-255, default 20)")
    parser.add_argument('--exposure-max', type=float, default=240,
                       help="Maximum mean brightness (0-255, default 240)")
    parser.add_argument('--overexposure-threshold', type=float, default=0.20,
                       help="Max ratio of overexposed pixels (default 0.20 = 20%%)")
    parser.add_argument('--underexposure-threshold', type=float, default=0.20,
                       help="Max ratio of underexposed pixels (default 0.20 = 20%%)")
    parser.add_argument('--motion-threshold', type=float, default=None,
                       help="Minimum motion score (optical flow, default None = no filtering)")
    
    # Similarity filtering
    parser.add_argument('--filter-similarity', action='store_true',
                       help="Remove very similar consecutive frames")
    parser.add_argument('--similarity-threshold', type=float, default=0.90,
                       help="Max SSIM for keeping frames (0-1, default 0.90)")
    
    # Output options
    parser.add_argument('--format', type=str, default='jpg', choices=['jpg', 'png'],
                       help="Output image format")
    parser.add_argument('--quality', type=int, default=95,
                       help="JPEG quality (1-100, default 95)")
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import cv2
        import numpy
        from PIL import Image
        if args.filter_similarity:
            from skimage.metrics import structural_similarity
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install opencv-python numpy pillow")
        if args.filter_similarity:
            print("  pip install scikit-image")
        sys.exit(1)
    
    # Set defaults if no sampling specified - adaptive FPS based on video length
    if args.fps is None and args.every_n_frames is None:
        extractor = VideoFrameExtractor(args.video, args.output)
        duration = extractor.duration
        
        # Adaptive FPS: shorter videos need higher sampling
        if duration < 10:
            args.fps = 4.0
        elif duration < 20:
            args.fps = 3.0
        elif duration < 60:
            args.fps = 2.0
        else:
            args.fps = 1.0
        
        print(f"[INFO] No sampling rate specified, using adaptive FPS: {args.fps} FPS for {duration:.1f}s video")
        del extractor  # Release video capture before recreating
    
    # Create extractor
    try:
        extractor = VideoFrameExtractor(args.video, args.output)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # Extract frames
    extracted = extractor.extract_frames(
        fps=args.fps,
        every_n_frames=args.every_n_frames,
        max_frames=args.max_frames,
        filter_quality=args.filter_quality,
        filter_similarity=args.filter_similarity,
        blur_threshold=args.blur_threshold,
        exposure_min=args.exposure_min,
        exposure_max=args.exposure_max,
        overexposure_threshold=args.overexposure_threshold,
        underexposure_threshold=args.underexposure_threshold,
        motion_threshold=args.motion_threshold,
        similarity_threshold=args.similarity_threshold,
        save_format=args.format,
        save_quality=args.quality
    )
    
    if not extracted:
        print("[WARNING] No frames extracted!")
        sys.exit(1)
    
    # Global minimum frame count check
    GLOBAL_MIN_FRAMES = 8
    if len(extracted) < GLOBAL_MIN_FRAMES:
        print(f"\n[ERROR] Insufficient usable frames for reconstruction.")
        print(f"  Extracted: {len(extracted)} frames")
        print(f"  Required: {GLOBAL_MIN_FRAMES} frames minimum")
        print(f"\n[SOLUTION] Move the camera around the scene to provide more parallax.")
        print(f"  Tips:")
        print(f"    - Circle around objects/scene")
        print(f"    - Capture from multiple heights")
        print(f"    - Ensure steady, well-lit footage")
        print(f"    - OR lower quality thresholds: --blur-threshold 40 --no-quality-filter")
        sys.exit(1)
    
    print(f"\n[SUCCESS] Extracted {len(extracted)} frames")
    print(f"\n[NEXT STEPS] Run 3D reconstruction:")
    print(f"  python scripts/generate_scene_multiview_pycolmap.py {extractor.output_dir}")
    print(f"  python scripts/generate_scene_multiview_dust3r.py {extractor.output_dir} --model mast3r")


if __name__ == '__main__':
    main()
