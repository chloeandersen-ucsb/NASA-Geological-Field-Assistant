from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Optional, List
import json
import os
import time
import uuid

from PySide6.QtCore import QObject, Signal, QTimer

from services.process_service import CameraService, ClassificationService, TranscriptionService


class AppStateType(Enum):
    HOME = auto()
    CAMERA_PREVIEW = auto()
    CLASSIFYING = auto()
    CLASSIFIED = auto()
    VOICE_TO_TEXT_LOADING = auto()
    VOICE_TO_TEXT = auto()
    TRIP_LOAD = auto()


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: float
    image_path: Optional[str] = None
    estimated_volume: Optional[float] = None
    estimated_weight: Optional[float] = None
    raw: Optional[dict] = None


@dataclass(frozen=True)
class RockEntry:
    rock_id: str
    ts: float
    result: ClassificationResult


@dataclass(frozen=True)
class TripSummary:
    rocks: List[RockEntry]
    total_volume: float
    total_weight: float
    voice_notes: List[dict]


class Store:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.rocks_path = os.path.join(self.base_dir, "rocks.jsonl")
        self.voice_path = os.path.join(self.base_dir, "voice_notes.jsonl")

    def save_rock(self, result: ClassificationResult) -> RockEntry:
        entry = RockEntry(
            rock_id=str(uuid.uuid4()),
            ts=time.time(),
            result=result,
        )
        rec = {
            "type": "rock",
            "rock_id": entry.rock_id,
            "ts": entry.ts,
            "result": asdict(entry.result),
        }
        with open(self.rocks_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return entry

    def list_rocks(self) -> List[RockEntry]:
        if not os.path.exists(self.rocks_path):
            return []
        rocks: List[RockEntry] = []
        with open(self.rocks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") != "rock":
                    continue
                result = ClassificationResult(**rec["result"])
                rocks.append(RockEntry(rock_id=rec["rock_id"], ts=rec["ts"], result=result))
        return rocks

    def save_voice_note(self, transcript: str, cleaned: Optional[str] = None) -> None:
        rec = {
            "type": "voice",
            "ts": time.time(),
            "transcript": transcript,
            "cleaned": cleaned or transcript,
        }
        with open(self.voice_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    
    def list_voice_notes(self) -> List[dict]:
        if not os.path.exists(self.voice_path):
            return []
        notes: List[dict] = []
        with open(self.voice_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") != "voice":
                    continue
                notes.append(rec)
        notes.sort(key=lambda x: x.get("ts", 0), reverse=True)
        return notes


class ViewModel(QObject):
    state_changed = Signal(object)
    classification_changed = Signal(object)
    transcription_changed = Signal(str)
    trip_changed = Signal(object)
    error = Signal(str)

    def __init__(self, store_dir: str, parent=None):
        super().__init__(parent)
        
        self._last_captured_path: str | None = None
        
        try:
            self.store = Store(store_dir)
        except Exception as e:
            import sys
            print(f"ERROR: Failed to create Store: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
        
        try:
            self.camera = CameraService(self)
        except Exception as e:
            import sys
            print(f"ERROR: Failed to create CameraService: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
        
        try:
            self.classifier = ClassificationService(self)
        except Exception as e:
            import sys
            print(f"ERROR: Failed to create ClassificationService: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
        
        try:
            self.transcriber = TranscriptionService(self)
        except Exception as e:
            import sys
            print(f"ERROR: Failed to create TranscriptionService: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
            
        self.transcriber.boot_model()
        
        self.state = AppStateType.HOME
        
        self.state = AppStateType.HOME
        self.current_classification: ClassificationResult | None = None
        self.transcription_text = ""

        self.classification_timeout_ms = 20_000
        self._classify_timeout = QTimer(self)
        self._classify_timeout.setSingleShot(True)
        self._classify_timeout.timeout.connect(self._on_classify_timeout)

        #self.camera.preview_started.connect(self._on_preview_started)
        self.camera.capture_finished.connect(self._on_photo_captured)
        self.camera.capture_failed.connect(self._fail)
        self.classifier.finished.connect(self._on_classified)
        self.classifier.failed.connect(self._fail)
        self.transcriber.token.connect(self._on_transcription_token)
        self.transcriber.completed.connect(self._on_transcription_completed)
        self.transcriber.failed.connect(self._fail)

        self.transcriber.ready.connect(self._on_transcriber_ready)


    def _set_state(self, new_state: AppStateType) -> None:
        self.state = new_state
        self.state_changed.emit(new_state)

    def go_home(self) -> None:
        if self.state == AppStateType.VOICE_TO_TEXT:
            self.stop_voice_to_text()
        if self.state == AppStateType.CAMERA_PREVIEW:
            self.camera.kill()
        self._set_state(AppStateType.HOME)

    def _on_transcriber_ready(self) -> None:
        import sys
        print("[VIEWMODEL] Transcriber ready signal received", file=sys.stderr)
        if self.state == AppStateType.VOICE_TO_TEXT_LOADING:
            self._set_state(AppStateType.VOICE_TO_TEXT)


    def open_trip_load(self) -> None:
        self._publish_trip()
        self._set_state(AppStateType.TRIP_LOAD)
        
    def open_camera_preview(self) -> None:
        self._set_state(AppStateType.CAMERA_PREVIEW)
        
    def start_camera_stream(self, x: int, y: int, w: int, h: int) -> None:
        self.camera.start_preview(x, y, w, h)
       
    def trigger_capture(self) -> None:
        self._set_state(AppStateType.CLASSIFYING)
        self.camera.trigger_capture()
        
    def cancel_camera(self) -> None:
        self.camera.stop_preview()
        self.go_home()

    def start_classification(self) -> None:
        self.current_classification = None
        self._set_state(AppStateType.CLASSIFYING)
        self.camera.start_preview()

 #   def _on_preview_started(self, _path: str) -> None:
  #      self._set_state(AppStateType.CAMERA_PREVIEW)

    def capture_photo(self) -> None:
        """Call when user clicks Capture on the camera preview screen."""
        self.camera.capture()

    def reclassify(self) -> None:
        self.start_classification()

    def _on_photo_captured(self, image_path: str) -> None:
        self._last_captured_path = image_path
        self._set_state(AppStateType.CLASSIFYING)
        self._classify_timeout.start(self.classification_timeout_ms)
        self.classifier.classify(image_path)

    def _on_classified(self, payload: dict) -> None:
        self._classify_timeout.stop()
        result = ClassificationResult(
            label=str(payload.get("label", "UNKNOWN")),
            confidence=float(payload.get("confidence", 0.0)),
            image_path=self._last_captured_path, # <--- Use the stored path
            estimated_volume=payload.get("estimated_volume"),
            estimated_weight=payload.get("estimated_weight"),
            raw=payload,
        )
        self.current_classification = result
        self.classification_changed.emit(result)
        self._set_state(AppStateType.CLASSIFIED)

    def save_classification(self) -> None:
        if self.current_classification:
            self.store.save_rock(self.current_classification)
        self.go_home()

    def delete_classification(self) -> None:
        self.current_classification = None
        self.go_home()

    def _on_classify_timeout(self) -> None:
        self.classifier.kill()
        self._fail("Classification timeout (20s)")

    def start_voice_to_text(self) -> None:
        import sys
        print("[VIEWMODEL] start_voice_to_text() called", file=sys.stderr)
        self.transcription_text = ""
        self.transcription_changed.emit("")
        
        self.transcriber.start_recording()
        
        self._set_state(AppStateType.VOICE_TO_TEXT)
        
       # self._set_state(AppStateType.VOICE_TO_TEXT_LOADING)
       # self.transcriber.start()

    def stop_voice_to_text(self) -> None:
        import sys
        print("[VIEWMODEL] stop_voice_to_text() called", file=sys.stderr)
        #self.transcriber.stop()
        self.transcriber.stop_recording()

    def redo_voice_to_text(self) -> None:
        self.transcriber.kill()
        self.start_voice_to_text()

    def save_transcription(self) -> None:
        text = self.transcription_text.strip()
        if text:
            self.store.save_voice_note(text, cleaned=text)
        self.transcription_text = ""
        self.go_home()

    def delete_transcription(self) -> None:
        self.transcription_text = ""
        self.go_home()

    def _on_transcription_token(self, chunk: str) -> None:
        import sys
        print(f"[VIEWMODEL] Received transcription token: '{chunk}'", file=sys.stderr)
        # if self.state == AppStateType.VOICE_TO_TEXT_LOADING:
        #     self._set_state(AppStateType.VOICE_TO_TEXT)
        self.transcription_text += chunk
        print(f"[VIEWMODEL] Updated transcription_text (length: {len(self.transcription_text)}): '{self.transcription_text[:200]}'", file=sys.stderr)
        self.transcription_changed.emit(self.transcription_text)

    def _on_transcription_completed(self, final_text: str) -> None:
        import sys
        print(f"[VIEWMODEL] Received transcription completed signal", file=sys.stderr)
        print(f"[VIEWMODEL] Final text received: '{final_text}'", file=sys.stderr)
        print(f"[VIEWMODEL] Final text length: {len(final_text)}", file=sys.stderr)
        if final_text.strip():
            self.transcription_text = final_text
            print(f"[VIEWMODEL] Setting transcription_text and emitting signal", file=sys.stderr)
            self.transcription_changed.emit(self.transcription_text)
            if self.state == AppStateType.VOICE_TO_TEXT_LOADING:
                self._set_state(AppStateType.VOICE_TO_TEXT)
        else:
            print("[VIEWMODEL] Final text is empty, not updating", file=sys.stderr)

    def _publish_trip(self) -> None:
        rocks = self.store.list_rocks()
        voice_notes = self.store.list_voice_notes()
        total_volume = 0.0
        total_weight = 0.0
        for r in rocks:
            if r.result.estimated_volume is not None:
                total_volume += float(r.result.estimated_volume)
            if r.result.estimated_weight is not None:
                total_weight += float(r.result.estimated_weight)
        summary = TripSummary(rocks=rocks, total_volume=total_volume, total_weight=total_weight, voice_notes=voice_notes)
        self.trip_changed.emit(summary)

    def _fail(self, message: str) -> None:
        self._classify_timeout.stop()
        self.error.emit(message)
        self.go_home()
