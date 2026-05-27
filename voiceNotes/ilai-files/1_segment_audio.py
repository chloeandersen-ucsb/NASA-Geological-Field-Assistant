#!/usr/bin/env python3
"""
STEP 1: Segment raw audio into clean clips using ffmpeg silence detection.

Cuts at natural silences, targeting 5-15s clips (sweet spot for CTC models).
Outputs 16kHz mono WAV files ready for transcription.

Usage:
    python 1_segment_audio.py --input_dir ./raw_audio --output_dir ./segments
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


# ── Tunable parameters ────────────────────────────────────────────────────────
SILENCE_THRESH_DB   = -35   # dB below which is considered silence
SILENCE_MIN_DUR     = 0.5   # minimum silence gap to cut on (seconds)
TARGET_MIN_S        = 4.0   # discard clips shorter than this
TARGET_MAX_S        = 15.0  # hard-split clips longer than this
SAMPLE_RATE         = 16000 # NeMo FastConformer expects 16kHz mono
# ─────────────────────────────────────────────────────────────────────────────


def detect_silences(audio_path: str) -> list[tuple[float, float]]:
    """Return list of (silence_start, silence_end) pairs using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", f"silencedetect=noise={SILENCE_THRESH_DB}dB:d={SILENCE_MIN_DUR}",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr  # ffmpeg writes filter output to stderr

    starts, ends = [], []
    for line in output.splitlines():
        m = re.search(r"silence_start: ([0-9.]+)", line)
        if m:
            starts.append(float(m.group(1)))
        m = re.search(r"silence_end: ([0-9.]+)", line)
        if m:
            ends.append(float(m.group(1)))

    # Pair up starts and ends
    silences = list(zip(starts, ends[:len(starts)]))
    return silences


def get_duration(audio_path: str) -> float:
    """Get total duration of audio file in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def silences_to_cut_points(silences: list, total_duration: float) -> list[float]:
    """
    Convert silence ranges to cut points (midpoints of each silence gap).
    Always includes 0.0 and total_duration as boundaries.
    """
    cut_points = [0.0]
    for start, end in silences:
        midpoint = (start + end) / 2.0
        cut_points.append(midpoint)
    cut_points.append(total_duration)
    return sorted(set(cut_points))


def merge_short_segments(cut_points: list[float]) -> list[tuple[float, float]]:
    """
    Convert cut points into (start, end) segments, merging any that are
    too short and hard-splitting any that are too long.
    """
    raw_segments = [(cut_points[i], cut_points[i+1])
                    for i in range(len(cut_points) - 1)]

    merged = []
    buffer_start = raw_segments[0][0]
    buffer_end   = raw_segments[0][1]

    for seg_start, seg_end in raw_segments[1:]:
        current_len = buffer_end - buffer_start
        addition    = seg_end - seg_start

        if current_len < TARGET_MIN_S:
            # Segment too short — absorb the next one
            buffer_end = seg_end
        elif current_len + addition <= TARGET_MAX_S:
            # Still fits — keep merging
            buffer_end = seg_end
        else:
            # Would exceed max — flush and start fresh
            merged.append((buffer_start, buffer_end))
            buffer_start = seg_start
            buffer_end   = seg_end

    # Flush the last buffer
    if buffer_end - buffer_start >= TARGET_MIN_S:
        merged.append((buffer_start, buffer_end))

    # Hard-split any segments still over TARGET_MAX_S (rare edge case)
    final = []
    for start, end in merged:
        duration = end - start
        if duration <= TARGET_MAX_S:
            final.append((start, end))
        else:
            # Split evenly
            n_parts = int(duration // TARGET_MAX_S) + 1
            part_dur = duration / n_parts
            for i in range(n_parts):
                seg_start = start + i * part_dur
                seg_end   = min(start + (i + 1) * part_dur, end)
                if seg_end - seg_start >= TARGET_MIN_S:
                    final.append((seg_start, seg_end))

    return final


def extract_segment(audio_path: str, start: float, end: float,
                    output_path: str) -> bool:
    """Extract a single segment as 16kHz mono WAV using ffmpeg."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start),
        "-t",  str(duration),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",            # mono
        "-acodec", "pcm_s16le", # 16-bit PCM WAV
        "-loglevel", "error",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def process_file(audio_path: str, output_dir: str, file_stem: str) -> int:
    """Process a single audio file. Returns number of segments created."""
    print(f"\n{'─'*60}")
    print(f"Processing: {Path(audio_path).name}")

    total_dur = get_duration(audio_path)
    print(f"  Duration: {total_dur:.1f}s ({total_dur/60:.1f} min)")

    silences = detect_silences(audio_path)
    print(f"  Found {len(silences)} silence gaps")

    cut_points = silences_to_cut_points(silences, total_dur)
    segments   = merge_short_segments(cut_points)

    durations = [e - s for s, e in segments]
    avg_dur   = sum(durations) / len(durations) if durations else 0
    print(f"  Segments: {len(segments)}  |  avg {avg_dur:.1f}s  |  "
          f"range {min(durations):.1f}–{max(durations):.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    created = 0
    for i, (start, end) in enumerate(segments):
        out_name = f"{file_stem}_{i:04d}.wav"
        out_path = os.path.join(output_dir, out_name)
        if extract_segment(audio_path, start, end, out_path):
            created += 1
        else:
            print(f"  WARNING: Failed to extract segment {i}")

    print(f"  ✓ Created {created} WAV segments → {output_dir}")
    return created


def main():
    parser = argparse.ArgumentParser(description="Segment geology audio for ASR fine-tuning")
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing raw .mp3/.wav/.m4a files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write segmented .wav files")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    audio_files = sorted(
        list(input_dir.glob("*.mp3")) +
        list(input_dir.glob("*.wav")) +
        list(input_dir.glob("*.m4a")) +
        list(input_dir.glob("*.flac"))
    )

    if not audio_files:
        print(f"ERROR: No audio files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files to process")

    total_segments = 0
    for audio_path in audio_files:
        file_stem  = audio_path.stem.replace(" ", "_")
        file_outdir = output_dir / file_stem
        n = process_file(str(audio_path), str(file_outdir), file_stem)
        total_segments += n

    print(f"\n{'='*60}")
    print(f"Done. Total segments created: {total_segments}")
    print(f"Estimated training audio: {total_segments * 9 / 3600:.1f}h "
          f"(assuming ~9s avg)")
    print(f"\nNext step: run  2_transcribe_and_clean.py  on {output_dir}")


if __name__ == "__main__":
    main()
