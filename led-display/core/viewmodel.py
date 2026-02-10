from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Optional, List
import json
import os
import sys
import time
import traceback
import uuid

from PySide6.QtCore import QObject, Signal, QTimer

import connector
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

    def _init_component(self, name: str, factory):
        try:
            return factory()
        except Exception as e:
            print(f"ERROR: Failed to create {name}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise

    def __init__(self, store_dir: str, parent=None):
        super().__init__(parent)
        self.store = self._init_component("Store", lambda: Store(store_dir))
        self.camera = self._init_component("CameraService", lambda: CameraService(self))
        self.classifier = self._init_component("ClassificationService", lambda: ClassificationService(self))
        self.transcriber = self._init_component("TranscriptionService", lambda: TranscriptionService(self))
        self.state = AppStateType.HOME
        self.current_classification: ClassificationResult | None = None
        self.transcription_text = ""

        self.classification_timeout_ms = 20_000
        self._classify_timeout = QTimer(self)
        self._classify_timeout.setSingleShot(True)
        self._classify_timeout.timeout.connect(self._on_classify_timeout)

        self.camera.capture_finished.connect(self._on_photo_captured)
        self.camera.capture_failed.connect(self._fail)
        self.classifier.finished.connect(self._on_classified)
        self.classifier.failed.connect(self._fail)
        self.transcriber.token.connect(self._on_transcription_token)
        self.transcriber.completed.connect(self._on_transcription_completed)
        self.transcriber.failed.connect(self._fail)

    def _set_state(self, new_state: AppStateType) -> None:
        self.state = new_state
        self.state_changed.emit(new_state)

    def go_home(self) -> None:
        if self.state == AppStateType.VOICE_TO_TEXT:
            self.stop_voice_to_text()
        self._set_state(AppStateType.HOME)

    def open_trip_load(self) -> None:
        self._publish_trip()
        self._set_state(AppStateType.TRIP_LOAD)

    def _use_mock_camera(self) -> bool:
        return connector.use_mocks() or connector.use_mock_ml()

    def start_classification(self) -> None:
        self.current_classification = None
        if self._use_mock_camera():
            self._set_state(AppStateType.CLASSIFYING)
            self._classify_timeout.start(self.classification_timeout_ms)
            self.camera.capture()
        else:
            self._set_state(AppStateType.CAMERA_PREVIEW)

    def start_camera_preview(self) -> None:
        """Called when entering camera preview (real camera). Preview is embedded in the same window."""
        self._set_state(AppStateType.CAMERA_PREVIEW)

    def capture_photo(self) -> None:
        """Called when user clicks Capture on camera preview page."""
        self._set_state(AppStateType.CLASSIFYING)
        self._classify_timeout.start(self.classification_timeout_ms)
        self.camera.capture()

    def cancel_camera_preview(self) -> None:
        """Called when user clicks Cancel on camera preview page."""
        self.go_home()

    def reclassify(self) -> None:
        self.start_classification()

    def _on_photo_captured(self, image_path: str) -> None:
        self.classifier.classify(image_path)

    def _on_classified(self, payload: dict) -> None:
        self._classify_timeout.stop()
        result = ClassificationResult(
            label=str(payload.get("label", "UNKNOWN")),
            confidence=float(payload.get("confidence", 0.0)),
            image_path=payload.get("image_path"),
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
        self.transcription_text = ""
        self.transcription_changed.emit("")
        self._set_state(AppStateType.VOICE_TO_TEXT_LOADING)
        self.transcriber.start()

    def stop_voice_to_text(self) -> None:
        self.transcriber.stop()

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
        if self.state == AppStateType.VOICE_TO_TEXT_LOADING:
            self._set_state(AppStateType.VOICE_TO_TEXT)
        self.transcription_text += chunk
        self.transcription_changed.emit(self.transcription_text)

    def _on_transcription_completed(self, final_text: str) -> None:
        if final_text.strip():
            self.transcription_text = final_text
            self.transcription_changed.emit(self.transcription_text)
            if self.state == AppStateType.VOICE_TO_TEXT_LOADING:
                self._set_state(AppStateType.VOICE_TO_TEXT)

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
