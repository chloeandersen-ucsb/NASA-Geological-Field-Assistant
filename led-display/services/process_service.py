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
        
        self.temperature = 1.0
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
        
        if "label" not in payload or "confidence" not in payload:
            self.failed.emit("RockNet output JSON missing label/confidence")
            return
        
        self.finished.emit(payload)
    
    def _on_error(self, _err) -> None:
        self.failed.emit("Classifier process error")


class TranscriptionService(ProcessService):
    token = Signal(str)
    completed = Signal(str)
    
    _PHRASE_RE = re.compile(r"\[\d{2}:\d{2}:\d{2}\]\s*\[\s*'(.+?)'\s*\]")
    
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
    
    def start(self) -> None:
        if self.proc.state() != QProcess.NotRunning:
            self.failed.emit("Transcription already running")
            return
        
        self._text_parts = []
        self._final_phrases = []
        self._active = True
        self._in_final_dump = False
        self._user_stopped = False
        
        cmd = [self.python, str(self.script)]
        print(f"[VOICE-TO-TEXT] Starting transcription process: {' '.join(cmd)}", file=sys.stderr)
        self.proc.start(cmd[0], cmd[1:])
        print(f"[VOICE-TO-TEXT] Process started, PID: {self.proc.processId()}", file=sys.stderr)
    
    def stop(self) -> None:
        print(f"[VOICE-TO-TEXT] stop() called, process state: {self.proc.state()}", file=sys.stderr)
        print(f"[VOICE-TO-TEXT] Current text parts count: {len(self._text_parts)}", file=sys.stderr)
        print(f"[VOICE-TO-TEXT] Current accumulated text: '{self.full_text()}'", file=sys.stderr)
        
        if self.proc.state() == QProcess.NotRunning:
            print("[VOICE-TO-TEXT] Process already stopped, emitting completed signal", file=sys.stderr)
            self.completed.emit(self.full_text())
            return
        
        self._user_stopped = True
        
        pid = int(self.proc.processId())
        if pid > 0:
            try:
                print(f"[VOICE-TO-TEXT] Sending SIGINT to process {pid}", file=sys.stderr)
                os.kill(pid, signal.SIGINT)
            except Exception as e:
                print(f"[VOICE-TO-TEXT] Failed to send SIGINT: {e}, falling back to terminate", file=sys.stderr)
                self.proc.terminate()
        else:
            print("[VOICE-TO-TEXT] No valid PID, calling terminate()", file=sys.stderr)
            self.proc.terminate()
    
    def full_text(self) -> str:
        return "".join(self._text_parts).strip()
    
    def _on_stdout(self) -> None:
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if not data:
            return
        
        print(f"[VOICE-TO-TEXT] Received stdout data ({len(data)} bytes)", file=sys.stderr)
        
        for raw_line in data.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            
            print(f"[VOICE-TO-TEXT] Processing line: {line[:100]}", file=sys.stderr)
            
            if "FINAL TRANSCRIPT" in line or "STREAMING COMPLETE" in line:
                print("[VOICE-TO-TEXT] Detected final transcript marker", file=sys.stderr)
                self._in_final_dump = True
                continue
            
            if line.startswith("Using device:") or "RECORDING NOW" in line:
                print(f"[VOICE-TO-TEXT] Ignoring chatter line: {line}", file=sys.stderr)
                continue
            
            m = self._PHRASE_RE.search(line)
            if not m:
                print(f"[VOICE-TO-TEXT] Line did not match phrase pattern: {line}", file=sys.stderr)
                continue
            
            phrase = m.group(1).strip()
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
                
                if line.startswith("Using device:") or "RECORDING NOW" in line:
                    continue
                
                m = self._PHRASE_RE.search(line)
                if not m:
                    continue
                
                phrase = m.group(1).strip()
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
    
    def _on_error(self, _err) -> None:
        self.failed.emit("Transcription process error")
