#!/usr/bin/env python3
"""
SAGE Camera Freeze Diagnostic Script
=====================================
Reproduces the camera-freeze bug WITHOUT PySide6 or QThread.

What it does
------------
1. Opens the exact GStreamer pipeline that process_service.py uses.
2. Reads frames in the main thread, printing the wall-clock time of every
   cap.read() call so a block shows up as a gap in the log.
3. In a background *subprocess* (matching the real app), runs rocknet_daemon.py
   exactly as ClassificationService does -- loading the PyTorch model, emitting
   "ready", then running the dummy-CUDA warm-up.
4. Optionally (--transcriber) also starts transcriber_fine_tuned.py in a second
   subprocess, same as TranscriptionService / boot_model().

Usage
-----
# Test 1 -- camera only (should run forever):
python3 scripts/diag_camera_freeze.py --camera-only

# Test 2 -- camera + rocknet daemon (the prime suspect):
python3 scripts/diag_camera_freeze.py

# Test 3 -- camera + rocknet daemon + transcriber:
python3 scripts/diag_camera_freeze.py --transcriber

Diagnosis
---------
If cap.read() blocks, you will see a timestamped "FREEZE" line.
The gap between the last "OK" and the "FREEZE" line identifies which
subprocess event (model-load "ready" vs. CUDA-sync completion) caused it.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
CONNECTOR    = REPO_ROOT / "connector.py"
ROCKNET_DAEMON = REPO_ROOT / "ML-classifications" / "scripts" / "rocknet_daemon.py"
ROCKNET_WEIGHTS = REPO_ROOT / "ML-classifications" / "models" / "best_rocknet_v2.pt"
TRANSCRIBER    = REPO_ROOT / "voiceNotes" / "ilai-files" / "transcriber_fine_tuned.py"
PYTHON         = sys.executable

GSTREAMER_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv ! video/x-raw, format=(string)BGRx ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
)

# ── State shared between threads ───────────────────────────────────────────
_freeze_detected = threading.Event()
_stop            = threading.Event()


def ts() -> str:
    """Wall-clock timestamp: [HH:MM:SS.mmm]"""
    t = time.time()
    ms = int((t % 1) * 1000)
    return time.strftime(f"[%H:%M:%S.{ms:03d}]", time.localtime(t))


# ── Camera thread ──────────────────────────────────────────────────────────

def camera_loop(timeout_s: float = 5.0) -> None:
    print(f"{ts()} [CAMERA] Opening GStreamer pipeline...", flush=True)
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"{ts()} [CAMERA] FATAL -- cap.isOpened() returned False. "
              "Check nvargus-daemon (sudo systemctl status nvargus-daemon).", flush=True)
        _stop.set()
        return

    print(f"{ts()} [CAMERA] Pipeline open. Reading frames ...", flush=True)

    frame_count  = 0
    fail_count   = 0
    t_last_frame = time.monotonic()
    t_start      = time.monotonic()

    while not _stop.is_set():
        t_before = time.monotonic()
        ret, frame = cap.read()
        elapsed   = time.monotonic() - t_before

        now = time.monotonic()

        if elapsed > 1.0:
            # cap.read() took >1 s -- this IS the freeze
            print(
                f"{ts()} [CAMERA] *** FREEZE DETECTED ***  "
                f"cap.read() blocked for {elapsed:.3f}s  "
                f"(frames so far: {frame_count})",
                flush=True,
            )
            _freeze_detected.set()

        if not ret:
            fail_count += 1
            if elapsed < 0.05:
                # Fast failure -- pipeline returned immediately; cap exhausted
                print(f"{ts()} [CAMERA] cap.read() returned False in {elapsed*1000:.1f}ms "
                      f"(fail #{fail_count})", flush=True)
                if fail_count > 30:
                    print(f"{ts()} [CAMERA] Too many consecutive failures. Pipeline dead.", flush=True)
                    _freeze_detected.set()
                    break
            continue

        fail_count = 0
        frame_count += 1
        gap = now - t_last_frame
        t_last_frame = now

        # Print one line per second (not every frame -- 30fps is too noisy)
        if frame_count % 30 == 0:
            uptime = now - t_start
            print(
                f"{ts()} [CAMERA] OK  frame={frame_count:5d}  "
                f"read={elapsed*1000:5.1f}ms  gap={gap*1000:5.1f}ms  "
                f"uptime={uptime:.1f}s",
                flush=True,
            )

        if now - t_start > timeout_s:
            print(f"{ts()} [CAMERA] Ran cleanly for {timeout_s:.0f}s. No freeze.", flush=True)
            break

    cap.release()
    print(f"{ts()} [CAMERA] Released. total_frames={frame_count}", flush=True)
    _stop.set()


# ── Subprocess runner ──────────────────────────────────────────────────────

def _drain(label: str, proc: subprocess.Popen) -> None:
    """Thread: drain stdout from a subprocess and annotate with timestamps."""
    for raw in proc.stdout:
        line = raw.rstrip()
        if line:
            print(f"{ts()} [{label}] {line}", flush=True)


def start_rocknet_daemon(delay_s: float = 0.5) -> subprocess.Popen | None:
    if not ROCKNET_WEIGHTS.exists():
        print(f"{ts()} [ROCKNET] WARNING: weights not found at {ROCKNET_WEIGHTS}", flush=True)
        print(f"{ts()} [ROCKNET] Skipping daemon -- set correct path or run with --camera-only", flush=True)
        return None

    print(f"{ts()} [ROCKNET] Waiting {delay_s}s before starting daemon "
          "(same as app init order)...", flush=True)
    time.sleep(delay_s)
    print(f"{ts()} [ROCKNET] Launching rocknet_daemon.py ...", flush=True)

    proc = subprocess.Popen(
        [PYTHON, str(ROCKNET_DAEMON), "--weights", str(ROCKNET_WEIGHTS)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Drain thread for stdout (which carries the {"status":"ready"} JSON)
    threading.Thread(target=_drain, args=("ROCKNET", proc), daemon=True).start()

    # Drain stderr in background so it doesn't fill the pipe buffer
    threading.Thread(
        target=lambda: [print(f"{ts()} [ROCKNET-ERR] {l.rstrip()}", flush=True)
                        for l in proc.stderr if l.strip()],
        daemon=True,
    ).start()

    return proc


def start_transcriber(delay_s: float = 0.0) -> subprocess.Popen | None:
    if not TRANSCRIBER.exists():
        print(f"{ts()} [TRANSCRIBER] WARNING: script not found at {TRANSCRIBER}", flush=True)
        return None

    print(f"{ts()} [TRANSCRIBER] Waiting {delay_s}s before starting ...", flush=True)
    time.sleep(delay_s)
    print(f"{ts()} [TRANSCRIBER] Launching transcriber_fine_tuned.py ...", flush=True)

    proc = subprocess.Popen(
        [PYTHON, str(TRANSCRIBER)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_drain, args=("TRANSCRIBER", proc), daemon=True).start()
    return proc


# ── Memory monitor ─────────────────────────────────────────────────────────

def memory_monitor(interval_s: float = 1.0) -> None:
    """Polls /proc/meminfo and prints MemAvailable + SwapFree."""
    while not _stop.is_set():
        try:
            with open("/proc/meminfo") as f:
                info = {k.rstrip(":"): int(v.split()[0])
                        for k, *v in (l.split() for l in f) if v}
            mem_avail = info.get("MemAvailable", 0) // 1024
            swap_free = info.get("SwapFree",     0) // 1024
            print(
                f"{ts()} [MEM]  MemAvailable={mem_avail:6d}MB  SwapFree={swap_free:6d}MB",
                flush=True,
            )
        except Exception:
            pass
        time.sleep(interval_s)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE camera-freeze diagnostic")
    parser.add_argument(
        "--camera-only", action="store_true",
        help="Skip all ML subprocesses. Camera should run indefinitely.",
    )
    parser.add_argument(
        "--transcriber", action="store_true",
        help="Also launch transcriber_fine_tuned.py (Test 3).",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="How many seconds to run before declaring success (default 60).",
    )
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("SAGE Camera Freeze Diagnostic", flush=True)
    print(f"  camera-only : {args.camera_only}", flush=True)
    print(f"  transcriber : {args.transcriber}", flush=True)
    print(f"  duration    : {args.duration}s", flush=True)
    print("=" * 70, flush=True)

    # Handle Ctrl-C gracefully
    def _sigint(*_):
        print(f"\n{ts()} [MAIN] Interrupted.", flush=True)
        _stop.set()
    signal.signal(signal.SIGINT, _sigint)

    procs: list[subprocess.Popen] = []

    # 1. Memory monitor
    threading.Thread(target=memory_monitor, args=(2.0,), daemon=True).start()

    # 2. Camera (runs in its own thread so cap.read() doesn't block the timer)
    cam_thread = threading.Thread(
        target=camera_loop, args=(args.duration,), daemon=True
    )
    cam_thread.start()

    # Give the camera pipeline ~2 seconds to initialise before firing ML loads
    time.sleep(2.0)

    if not args.camera_only:
        # 3. rocknet_daemon subprocess (primary suspect: loads CUDA model,
        #    then runs ~6s of dummy inference AFTER emitting "ready")
        p = start_rocknet_daemon(delay_s=0.0)
        if p:
            procs.append(p)

        if args.transcriber:
            # 4. Transcriber subprocess (loads NeMo ASR + optional LLM)
            p2 = start_transcriber(delay_s=0.0)
            if p2:
                procs.append(p2)

    # Wait for camera thread to finish (freeze or timeout)
    cam_thread.join(timeout=args.duration + 30)

    # Cleanup
    for p in procs:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    print("=" * 70, flush=True)
    if _freeze_detected.is_set():
        print("RESULT: FREEZE DETECTED", flush=True)
        print("", flush=True)
        print("Correlate the timestamp of the last [CAMERA] OK line and the", flush=True)
        print("[ROCKNET] / [TRANSCRIBER] output just before it to identify", flush=True)
        print("which subprocess event (model load vs CUDA sync) caused the stall.", flush=True)
    else:
        print("RESULT: No freeze in the test window.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
