from __future__ import annotations

import json
import os
import re
import signal
import sys
from pathlib import Path

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


class CameraService(ProcessService):
    capture_finished = Signal(str)
    capture_failed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.capture_failed.connect(self.failed.emit)
        
        script_path = connector.get_camera_script_path()
        python_exe = connector.get_python_executable()
        self.cmd = [python_exe, str(script_path)]
    
    def capture(self) -> None:
        self.proc.start(self.cmd[0], self.cmd[1:])
    
    def _on_finished(self, exit_code: int, _status) -> None:
        out = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="ignore").strip()
        err = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="ignore").strip()
        
        if exit_code != 0:
            self.capture_failed.emit(err or f"Camera script failed (exit {exit_code})")
            return
        
        image_path = out.splitlines()[-1].strip() if out else ""
        if not image_path:
            self.capture_failed.emit("Camera script did not output image path")
            return
        
        self.capture_finished.emit(image_path)
    
    def _on_error(self, _err) -> None:
        self.capture_failed.emit("Camera process error")


class ClassificationService(ProcessService):
    finished = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.python = connector.get_python_executable()
        if connector.use_mocks() or connector.use_mock_ml():
            self.rocknet_script = connector.get_mock_rocknet_script_path()
            self.default_weights = "mock_weights.pt"
        else:
            self.rocknet_script = connector.get_rocknet_script_path()
            self.default_weights = connector.get_rocknet_weights_path()
            connector.validate_ml_paths()
        self.temperature = 1.0
        self._expected_json_path: str | None = None
        self._current_image_path: str | None = None
    
    def classify(self, image_path: str, weights_path: str | None = None) -> None:
        if self.proc.state() != QProcess.NotRunning:
            self.failed.emit("Classifier already running")
            return
        
        weights = weights_path or self.default_weights
        self._current_image_path = image_path
        
        base, _ = os.path.splitext(image_path)
        out_json = base + "_prediction.json"
        self._expected_json_path = out_json
        
        cmd = [
            self.python,
            str(self.rocknet_script),
            "--weights", str(weights),
            "--image", image_path,
            "--temperature", str(self.temperature),
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
        
        # Handle both list format (top3) and dict format (backward compatibility)
        if isinstance(payload, list):
            # New format: list of 3 classifications [{"label": "...", "confidence": 0.xx}, ...]
            if len(payload) == 0:
                self.failed.emit("RockNet output JSON is empty list")
                return
            # Extract top result and include full top3 list
            top_result = payload[0]
            if "label" not in top_result or "confidence" not in top_result:
                self.failed.emit("RockNet output JSON missing label/confidence in top result")
                return
            # Convert to dict format expected by display
            payload = {
                "label": top_result["label"],
                "confidence": top_result["confidence"],
                "top3": payload,
            }
        elif isinstance(payload, dict):
            # Old format: dict with label/confidence (backward compatibility)
            if "label" not in payload or "confidence" not in payload:
                self.failed.emit("RockNet output JSON missing label/confidence")
                return
        else:
            self.failed.emit(f"RockNet output JSON has unexpected type: {type(payload)}")
            return
        
        if self._current_image_path:
            payload["image_path"] = self._current_image_path
        self.finished.emit(payload)
    
    def _on_error(self, _err) -> None:
        self.failed.emit("Classifier process error")


class TranscriptionService(ProcessService):
    token = Signal(str)
    completed = Signal(str)
    
    # Mock format: [HH:MM:SS] ['phrase']
    _PHRASE_RE = re.compile(r"\[\d{2}:\d{2}:\d{2}\]\s*\[\s*'(.+?)'\s*\]")
    # rtt_lav format: [HH:MM:SS] phrase (no brackets/quotes)
    _PHRASE_ALT_RE = re.compile(r"\[\d{2}:\d{2}:\d{2}\]\s+(.+)$")
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc.readyReadStandardOutput.connect(self._on_stdout)
        self.python = connector.get_python_executable()
        if connector.use_mocks():
            self.script = connector.get_mock_voice_to_text_script_path()
        else:
            self.script = connector.get_voice_to_text_script_path()
            connector.validate_voice_to_text_paths()
        self._text_parts: list[str] = []
        self._final_phrases: list[str] = []
        self._active = False
        self._in_final_dump = False
        self._user_stopped = False
    
    def start(self) -> None:
        if self.proc.state() != QProcess.NotRunning:
            self.failed.emit("Transcription already running")
            return
        
        # Verify script exists before starting
        if not self.script.exists():
            self.failed.emit(f"Voice transcription script not found: {self.script}")
            return
        
        self._text_parts = []
        self._final_phrases = []
        self._active = True
        self._in_final_dump = False
        self._user_stopped = False
        
        try:
            cmd = [self.python, str(self.script)]
            self.proc.start(cmd[0], cmd[1:])
        except Exception as e:
            self.failed.emit(f"Failed to start transcription process: {e}")
    
    def stop(self) -> None:
        if self.proc.state() == QProcess.NotRunning:
            self.completed.emit(self.full_text())
            return
        self._user_stopped = True
        pid = int(self.proc.processId())
        if pid > 0:
            try:
                os.kill(pid, signal.SIGINT)
            except Exception:
                self.proc.terminate()
        else:
            self.proc.terminate()
    
    def full_text(self) -> str:
        return "".join(self._text_parts).strip()

    def _process_stdout_lines(self, data: str) -> None:
        for raw_line in data.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "FINAL TRANSCRIPT" in line or "STREAMING COMPLETE" in line:
                self._in_final_dump = True
                continue
            if line.startswith("Using ") or line.startswith("Using device:") or "RECORDING NOW" in line:
                continue
            phrase = None
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
                        self._final_phrases.append(line.strip())
                    continue
            if not phrase:
                continue
            if self._in_final_dump:
                self._final_phrases.append(phrase)
                continue
            chunk = phrase + " "
            self._text_parts.append(chunk)
            self.token.emit(chunk)
    
    def _on_stdout(self) -> None:
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if data:
            self._process_stdout_lines(data)
    
    def _on_finished(self, exit_code: int, _status) -> None:
        remaining_stdout = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        remaining_stderr = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="ignore")
        if remaining_stdout:
            self._process_stdout_lines(remaining_stdout)
        if self._final_phrases:
            final_text = " ".join(self._final_phrases)
        else:
            final_text = self.full_text()
        if exit_code != 0 and self._active and not getattr(self, "_user_stopped", False):
            self.failed.emit(remaining_stderr or f"Transcription failed (exit {exit_code})")
            return
        self.completed.emit(final_text)
    
    def _on_error(self, _err) -> None:
        self.failed.emit("Transcription process error")
