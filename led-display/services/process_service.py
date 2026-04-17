from __future__ import annotations

import json
import os
import re
import signal
import sys
from pathlib import Path

import cv2
import datetime
import uuid
from PySide6.QtCore import QThread
from PySide6.QtGui import QImage

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtCore import QObject, Signal, QProcess

import connector


class ProcessService(QObject):
    failed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc = QProcess(self)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
    
    def kill(self) -> None:
        if self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
    
    def _on_finished(self, exit_code: int, _status) -> None:
        pass
    
    def _on_error(self, _err) -> None:
        self.failed.emit("Process error")


class CameraService(QThread):
    frame_ready = Signal(QImage)
    capture_finished = Signal(str)
    capture_failed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag = False
        self._is_capturing = False
        self._stop_after_capture = True  # False = two-step mode: keep preview after first capture

        self.save_dir = project_root / "ML-classifications" / "camera-pipeline" / "images"

    def set_stop_after_capture(self, stop: bool) -> None:
        self._stop_after_capture = stop

    def start_preview(self, x=0, y=0, w=0, h=0) -> None:
        if not self.isRunning():
            self._run_flag = True
            self._is_capturing = False
            self.start()

    def trigger_capture(self) -> None:
        if self.isRunning():
            print("[CAMERA] Capture triggered via OpenCV...", file=sys.stderr)
            self._is_capturing = True

    def stop_preview(self) -> None:
        self._run_flag = False
        self.wait()
        
    def kill(self) -> None:
        self.stop_preview()
        
    def run(self) -> None:
        # --- Hardware Detection ---
        if connector.is_jetson():
            # Original Jetson CSI hardware pipeline
            pipeline = (
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 ! "
                "nvvidconv ! video/x-raw, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            # MacBook built-in FaceTime camera (Standard index 0)
            print("[CAMERA] Mac detected: Using default camera index 0", file=sys.stderr)
            cap = cv2.VideoCapture(0)
        
        # --- Connection Check ---
        if not cap.isOpened():
            # If standard Mac index 0 is busy, try index 1 (common for external monitors/iPhone)
            if not connector.is_jetson():
                cap = cv2.VideoCapture(1)
            
            if not cap.isOpened():
                self.capture_failed.emit("Failed to open camera via OpenCV")
                return
            
        print("[CAMERA] Native Preview started", file=sys.stderr)
        
        # --- Main Loop (Preserved Logic) ---
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue
                
            if self._is_capturing:
                self._is_capturing = False
                if self._stop_after_capture:
                    self._run_flag = False

                today_dir = self.save_dir / datetime.datetime.now().strftime("%Y%m%d")
                today_dir.mkdir(parents=True, exist_ok=True)

                filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.jpg"
                filepath = str(today_dir / filename)

                cv2.imwrite(filepath, frame)
                print(f"[CAMERA] Image saved to {filepath}", file=sys.stderr)

                if self._stop_after_capture:
                    cap.release()
                    self.capture_finished.emit(filepath)
                    return
                self.capture_finished.emit(filepath)
                
            # Convert OpenCV frame to PySide6 QImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            
            # Create the UI image and copy it so memory is managed safely
            qt_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qt_img)
            
        cap.release()
        print("[CAMERA] Native Preview stopped", file=sys.stderr)


class ClassificationService(ProcessService):
    finished = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        use_mocks = os.environ.get("SAGE_USE_MOCKS", "").lower() in ("1", "true", "yes")
        use_mock_ml = os.environ.get("SAGE_USE_MOCK_ML", "").lower() in ("1", "true", "yes")
        
        if use_mocks or use_mock_ml:
            self.python = connector.get_python_executable()
            self.rocknet_script = connector.get_mock_rocknet_script_path()
            self.default_weights = "mock_weights.pt"
        else:
            self.python = connector.get_python_executable()
            self.rocknet_script = connector.get_rocknet_script_path()
            self.default_weights = connector.get_rocknet_weights_path()
            
            connector.validate_ml_paths()
        
        self._expected_json_path: str | None = None
    
    def classify(self, image_path: str, weights_path: str | None = None) -> None:
        if self.proc.state() != QProcess.NotRunning:
            self.failed.emit("Classifier already running")
            return
        
        weights = weights_path or self.default_weights
        
        base, _ = os.path.splitext(image_path)
        out_json = base + "_prediction.json"
        self._expected_json_path = out_json
        
        cmd = [
            self.python,
            str(self.rocknet_script),
            "--weights", str(weights),
            "--image", image_path,
            "--output-json", out_json,
        ]
        self.proc.start(cmd[0], cmd[1:])
    
    def _on_finished(self, exit_code: int, _status) -> None:
        err = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="ignore").strip()
        if exit_code != 0:
            self.failed.emit(err or f"Classification failed (exit {exit_code})")
            return
        
        if not self._expected_json_path:
            self.failed.emit("Internal error: expected output JSON path not set")
            return
        
        if not os.path.exists(self._expected_json_path):
            self.failed.emit(f"RockNet output JSON not found: {self._expected_json_path}")
            return
        
        try:
            with open(self._expected_json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            self.failed.emit(f"Failed to read RockNet output JSON: {e}")
            return
        
        # Detect v2 schema, legacy list format, and old dict format
        if isinstance(payload, dict) and payload.get("schema_version") == "rocknet_v2.0":
            primary = payload.get("primary", {})
            if "family" not in primary or "confidence" not in primary:
                self.failed.emit("RockNet v2 output missing primary.family/confidence")
                return
            scores = primary.get("scores", {})
            top3 = [
                {"label": k.capitalize(), "confidence": v}
                for k, v in sorted(scores.items(), key=lambda x: -x[1])
            ]
            payload = {
                "label": primary["family"].capitalize(),
                "confidence": primary["confidence"],
                "top3": top3,
                "tier": primary.get("tier"),
                "features": payload.get("features"),
                "geology_notes": payload.get("geology_notes"),
                "ui": payload.get("ui"),
            }
        elif isinstance(payload, list):
            if len(payload) == 0:
                self.failed.emit("RockNet output JSON is empty list")
                return
            top_result = payload[0]
            if "label" not in top_result or "confidence" not in top_result:
                self.failed.emit("RockNet output JSON missing label/confidence in top result")
                return
            payload = {
                "label": top_result["label"],
                "confidence": top_result["confidence"],
                "top3": payload,
            }
        elif isinstance(payload, dict):
            if "label" not in payload or "confidence" not in payload:
                self.failed.emit("RockNet output JSON missing label/confidence")
                return
        else:
            self.failed.emit(f"RockNet output JSON has unexpected type: {type(payload)}")
            return
        
        self.finished.emit(payload)
    
    def _on_error(self, _err) -> None:
        self.failed.emit("Classifier process error")


class VolumeService(ProcessService):
    finished = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.python = connector.get_python_executable()
        self.script = project_root / "rock-volume" / "measure_rock_volume.py"
        self.work_dir = project_root / "rock-volume"
        self._expected_json_path: str | None = None

    def estimate(self, top_image_path: str, side_image_path: str) -> None:
        if self.proc.state() != QProcess.NotRunning:
            print("[VOLUME] Volume estimation already running, skipping", file=sys.stderr)
            self.failed.emit("Volume estimation already running")
            return
        base, _ = os.path.splitext(top_image_path)
        out_json = base + "_volume.json"
        self._expected_json_path = out_json
        out_dir = os.path.dirname(out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        print(f"[VOLUME] Starting volume calculation: top={top_image_path!r}, side={side_image_path!r}, out={out_json!r}", file=sys.stderr)
        cmd = [
            self.python,
            str(self.script),
            "--top", top_image_path,
            "--side", side_image_path,
            "--out", out_json,
        ]
        self.proc.setWorkingDirectory(str(self.work_dir))
        self.proc.start(cmd[0], cmd[1:])

    def _on_finished(self, exit_code: int, _status) -> None:
        err = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="ignore").strip()
        if exit_code != 0:
            msg = err or f"Volume estimation failed (exit {exit_code})"
            print(f"[VOLUME] Volume calculation failed: {msg}", file=sys.stderr)
            self.failed.emit(msg)
            return
        if not self._expected_json_path or not os.path.exists(self._expected_json_path):
            print("[VOLUME] Volume output JSON not found", file=sys.stderr)
            self.failed.emit("Volume output JSON not found")
            return
        try:
            with open(self._expected_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[VOLUME] Failed to read volume JSON: {e}", file=sys.stderr)
            self.failed.emit(f"Failed to read volume JSON: {e}")
            return
        volume_cm3 = data.get("volume_cm3")
        mass_range = data.get("mass_range") or {}
        if volume_cm3 is None:
            print("[VOLUME] Volume JSON missing volume_cm3", file=sys.stderr)
            self.failed.emit("Volume JSON missing volume_cm3")
            return
        payload = {
            "volume_cm3": float(volume_cm3),
            "mass_min_g": mass_range.get("min_g"),
            "mass_max_g": mass_range.get("max_g"),
            "min_kg": mass_range.get("min_kg"),
            "max_kg": mass_range.get("max_kg"),
        }
        print(f"[VOLUME] Volume calculation finished: volume_cm3={payload['volume_cm3']}", file=sys.stderr)
        self.finished.emit(payload)

    def _on_error(self, _err) -> None:
        print(f"[VOLUME] Volume process error: {_err}", file=sys.stderr)
        self.failed.emit("Volume process error")


class TranscriptionService(ProcessService):
    token = Signal(str)
    completed = Signal(str)
    ready = Signal()
    
    # transcriber_fine_tuned.py output — [HH:MM:SS] phrase
    _PHRASE_ALT_RE = re.compile(r"\[\d{2}:\d{2}:\d{2}\]\s+(.+)$")
    # Mock format: [HH:MM:SS] ['phrase']
    _PHRASE_RE = re.compile(r"\[\d{2}:\d{2}:\d{2}\]\s*\[\s*'(.+?)'\s*\]")
    # skip these when in final-dump
    _FINAL_DUMP_SKIP_RE = re.compile(r"^[\s=\-]*$|^FINAL TRANSCRIPT:\s*$", re.IGNORECASE)
    
    @property
    def is_recording(self) -> bool:
        from PySide6.QtCore import QProcess
        return (not self._user_stopped) and (self.proc.state() == QProcess.Running)

    @staticmethod
    def _is_chatter_line(line: str) -> bool:
        """Ignore transcriber_fine_tuned.py startup/shutdown and mock chatter (not phrase lines)."""
        if not line:
            return True
        if line.startswith("Using ") or line.startswith("Using device:"):
            return True
        if "RECORDING" in line or "RECORDING NOW" in line:
            return True
        if line.startswith("Searching for ") or line.startswith("Warning: "):
            return True
        if line.startswith("Loading ") or "model loaded" in line.lower():
            return True
        if "Stopping stream" in line or "Processing full audio" in line:
            return True
        if line == "No audio recorded." or line.startswith("ERROR: "):
            return True
        return False
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc.readyReadStandardOutput.connect(self._on_stdout)
        
        if os.environ.get("SAGE_USE_MOCKS", "").lower() in ("1", "true", "yes"):
            self.python = connector.get_python_executable()
            self.script = connector.get_mock_voice_to_text_script_path()
        else:
            self.python = connector.get_python_executable()
            self.script = connector.get_voice_to_text_script_path()
            
            connector.validate_voice_to_text_paths()
        
        self._text_parts: list[str] = []
        self._final_phrases: list[str] = []
        self._active = False
        self._in_final_dump = False
        self._user_stopped = False
        # self.process.stateChanged.connect(self._on_state_changed)
        # self.proc.stateChanged.connect(self._on_state_changed)
        
    def boot_model(self) -> None:
        """Starts the Python script and blocks until 'Model loaded' is detected.
        Raises RuntimeError if the process exits before the model is ready."""
        if self.proc.state() != QProcess.NotRunning:
            return

        from PySide6.QtCore import QEventLoop
        loop = QEventLoop()
        boot_failed = [False]

        def _on_ready():
            loop.quit()

        def _on_early_exit(exit_code, exit_status):
            boot_failed[0] = True
            print(f"[VOICE-TO-TEXT] Process exited unexpectedly during boot (code={exit_code})", file=sys.stderr)
            loop.quit()

        self.ready.connect(_on_ready)
        self.proc.finished.connect(_on_early_exit)

        cmd = [self.python, str(self.script)]
        print(f"[VOICE-TO-TEXT] Booting model: {' '.join(cmd)}", file=sys.stderr)

        self.proc.start(cmd[0], cmd[1:])

        loop.exec()

        self.ready.disconnect(_on_ready)
        self.proc.finished.disconnect(_on_early_exit)

        if boot_failed[0]:
            raise RuntimeError(
                "Transcription process exited before model was ready. "
                "Check that newest_model.nemo exists and all dependencies are installed."
            )

        print("[VOICE-TO-TEXT] Boot complete. Model is in memory.", file=sys.stderr)

    def _on_state_changed(self, state):
        if state == QProcess.Running:
            print("[VOICE-TO-TEXT] Process running → emitting ready", file=sys.stderr)
            self.ready.emit()
    
   # def start(self) -> None:
    #    if self.proc.state() != QProcess.NotRunning:
     #       self.failed.emit("Transcription already running")
      #      return
        
#        self._text_parts = []
 #       self._final_phrases = []
  #      self._active = True
   #     self._in_final_dump = False
    #    self._user_stopped = False
        
     #   cmd = [self.python, str(self.script)]
      #  print(f"[VOICE-TO-TEXT] Starting transcription process: {' '.join(cmd)}", file=sys.stderr)
       # self.proc.start(cmd[0], cmd[1:])
        #print(f"[VOICE-TO-TEXT] Process started, PID: {self.proc.processId()}", file=sys.stderr)
        
    def start_recording(self) -> None:
        """Sends the 'start' command to the already-running engine."""
        self._text_parts = []
        self._final_phrases = []
        self._in_final_dump = False
        self._user_stopped = False
        
        if self.proc.state() == QProcess.Running:
            print("[VOICE-TO-TEXT] Sending START command...", file=sys.stderr)
            self.proc.write(b"start\n")
        else:
            self.boot_model()
            self.proc.write(b"start\n")
    
#    def stop(self) -> None:
 #       print(f"[VOICE-TO-TEXT] stop() called, process state: {self.proc.state()}", file=sys.stderr)
  #      print(f"[VOICE-TO-TEXT] Current text parts count: {len(self._text_parts)}", file=sys.stderr)
   #     print(f"[VOICE-TO-TEXT] Current accumulated text: '{self.full_text()}'", file=sys.stderr)
        
    #    if self.proc.state() == QProcess.NotRunning:
     #       print("[VOICE-TO-TEXT] Process already stopped, emitting completed signal", file=sys.stderr)
      #      self.completed.emit(self.full_text())
       #     return
        
        #self._user_stopped = True
#        self._stopping = True  # NEW FLAG
        
 #       pid = int(self.proc.processId())
  #      if pid > 0:
   #         try:
    #            print(f"[VOICE-TO-TEXT] Sending SIGINT to process {pid}", file=sys.stderr)
     #           os.kill(pid, signal.SIGINT)
      #      except Exception as e:
       #         print(f"[VOICE-TO-TEXT] Failed to send SIGINT: {e}, falling back to terminate", file=sys.stderr)
        #        self.proc.terminate()
#        else:
 #           print("[VOICE-TO-TEXT] No valid PID, calling terminate()", file=sys.stderr)
  #          self.proc.terminate()
  
    def stop_recording(self) -> None:
        """Sends the 'stop' command to the engine."""
        if self.proc.state() == QProcess.Running:
            print("[VOICE-TO-TEXT] Sending STOP command...", file=sys.stderr)
            self._user_stopped = True
            self.proc.write(b"stop\n")
    
    def full_text(self) -> str:
        return "".join(self._text_parts).strip()
    
    def _on_stdout(self) -> None:
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if not data:
            return
        
        print(f"[VOICE-TO-TEXT] Received stdout data ({len(data)} bytes)", file=sys.stderr)
        
        for raw_line in data.splitlines():
            line = raw_line.strip()
            if "model loaded successfully!" in line.lower():
                print("[VOICE-TO-TEXT] Model ready detected", file=sys.stderr)
                self.ready.emit()
                continue

            if not line:
                continue
            
            print(f"[VOICE-TO-TEXT] Processing line: {line[:100]}", file=sys.stderr)
            
            if "FINAL TRANSCRIPT" in line:
                print("[VOICE-TO-TEXT] Detected final transcript marker", file=sys.stderr)
                self._in_final_dump = True
                continue
                
            if "STREAMING COMPLETE" in line:
                print("[VOICE-TO-TEXT] Engine finished processing. Emitting completed.", file=sys.stderr)
                self._in_final_dump = False
                
                final_text = " ".join(self._final_phrases) if self._final_phrases else self.full_text()
                self.completed.emit(final_text)
                
                self._text_parts = []
                self._final_phrases = []
                continue
            
            # transcriber_fine_tuned.py startup/shutdown chatter (and mock "Using device:", "RECORDING NOW")
            if self._is_chatter_line(line):
                continue
            
            m = self._PHRASE_RE.search(line)
            if m:
                phrase = m.group(1).strip()
            else:
                m_alt = self._PHRASE_ALT_RE.search(line)
                if m_alt:
                    phrase = m_alt.group(1).strip()
                    if phrase == "(silence)":
                        continue
                else:
                    if self._in_final_dump and line:
                        if not self._FINAL_DUMP_SKIP_RE.match(line):
                            self._final_phrases.append(line.strip())
                        continue
                    print(f"[VOICE-TO-TEXT] Line did not match phrase pattern: {line}", file=sys.stderr)
                    continue
            
            if not phrase:
                print("[VOICE-TO-TEXT] Extracted phrase is empty", file=sys.stderr)
                continue
            
            if self._in_final_dump:
                print(f"[VOICE-TO-TEXT] Collecting phrase from final dump: '{phrase}'", file=sys.stderr)
                self._final_phrases.append(phrase)
                continue
            
            chunk = phrase + " "
            self._text_parts.append(chunk)
            print(f"[VOICE-TO-TEXT] Emitting token: '{chunk}'", file=sys.stderr)
            print(f"[VOICE-TO-TEXT] LIVE TRANSCRIPTION: {phrase}", file=sys.stdout)
            self.token.emit(chunk)
    
    def _on_finished(self, exit_code: int, _status) -> None:
        print(f"[VOICE-TO-TEXT] Process finished, exit_code: {exit_code}, status: {_status}", file=sys.stderr)
        print(f"[VOICE-TO-TEXT] User stopped: {getattr(self, '_user_stopped', False)}", file=sys.stderr)
        print(f"[VOICE-TO-TEXT] Active: {self._active}", file=sys.stderr)
        print(f"[VOICE-TO-TEXT] Final text parts count: {len(self._text_parts)}", file=sys.stderr)
        
        remaining_stdout = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        remaining_stderr = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="ignore")
        
        if remaining_stdout:
            print(f"[VOICE-TO-TEXT] Processing remaining stdout ({len(remaining_stdout)} bytes): {remaining_stdout[:500]}", file=sys.stderr)
            for raw_line in remaining_stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                
                if "FINAL TRANSCRIPT" in line or "STREAMING COMPLETE" in line:
                    self._in_final_dump = True
                    continue
                
                if self._is_chatter_line(line):
                    continue
                
                m = self._PHRASE_RE.search(line)
                if m:
                    phrase = m.group(1).strip()
                else:
                    m_alt = self._PHRASE_ALT_RE.search(line)
                    if m_alt:
                        phrase = m_alt.group(1).strip()
                        if phrase == "(silence)":
                            continue
                    else:
                        if self._in_final_dump and line and not self._FINAL_DUMP_SKIP_RE.match(line):
                            self._final_phrases.append(line.strip())
                        continue
                
                if not phrase:
                    continue
                
                if self._in_final_dump:
                    self._final_phrases.append(phrase)
                    continue
                
                chunk = phrase + " "
                self._text_parts.append(chunk)
                print(f"[VOICE-TO-TEXT] Emitting token from remaining stdout: '{chunk}'", file=sys.stderr)
                print(f"[VOICE-TO-TEXT] LIVE TRANSCRIPTION: {phrase}", file=sys.stdout)
                self.token.emit(chunk)
        
        if remaining_stderr:
            print(f"[VOICE-TO-TEXT] Remaining stderr: {remaining_stderr[:500]}", file=sys.stderr)
        
        if self._final_phrases:
            final_text = " ".join(self._final_phrases)
            print(f"[VOICE-TO-TEXT] Using final transcript phrases ({len(self._final_phrases)} phrases): '{final_text}'", file=sys.stderr)
        else:
            final_text = self.full_text()
            print(f"[VOICE-TO-TEXT] Using accumulated streaming text: '{final_text}'", file=sys.stderr)
        
        if exit_code != 0 and self._active and not getattr(self, '_user_stopped', False):
            err = remaining_stderr or f"Transcription failed (exit {exit_code})"
            print(f"[VOICE-TO-TEXT] Emitting failed signal: {err}", file=sys.stderr)
            self.failed.emit(err)
            return
        
        print(f"[VOICE-TO-TEXT] Emitting completed signal with text: '{final_text}'", file=sys.stderr)
        self.completed.emit(final_text)
    
    # def _on_error(self, _err) -> None:
    #     self.failed.emit("Transcription process error")
    def _on_error(self, _err) -> None:
        if getattr(self, "_user_stopped", False):
            # User stopped process intentionally; ignore the error
            print("[VOICE-TO-TEXT] Ignored process error after user stop", file=sys.stderr)
            return
        self.failed.emit("Transcription process error")

