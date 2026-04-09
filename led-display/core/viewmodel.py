from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Union
import datetime
import json
import os
import sys
import time
import uuid

from PySide6.QtCore import QObject, Signal, QTimer

from services.process_service import CameraService, ClassificationService, TranscriptionService, VolumeService
#from voiceNotes.ilai_files import transcriber_fine_tuned


class AppStateType(Enum):
    HOME = auto()
    CAMERA_PREVIEW = auto()
    CONFIRM_CAPTURES = auto()
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
    side_image_path: Optional[str] = None
    estimated_volume: Optional[float] = None
    estimated_weight: Optional[Union[float, str]] = None
    raw: Optional[dict] = None


@dataclass(frozen=True)
class RockEntry:
    rock_id: str
    ts: float
    result: ClassificationResult
    session_id: Optional[str] = None


@dataclass(frozen=True)
class TripSummary:
    rocks: List[RockEntry]
    total_volume: float
    total_weight: float
    voice_notes: List[dict]


class Store:
    def __init__(self, base_dir: str, voice_notes_data_dir: str):
        self.base_dir = base_dir
        self.voice_notes_data_dir = voice_notes_data_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.rocks_path = os.path.join(self.base_dir, "rocks.jsonl")
        self.voice_path = os.path.join(self.base_dir, "voice_notes.jsonl")
        
        # --- NEW: Generate a permanent ID for this specific app launch ---
        self.session_id = str(uuid.uuid4()) 

    def update_voice_note_rock_id(self, ts: float, rock_id: str) -> None:
        """Updates an existing voice note to link it to a specific rock."""
        if not os.path.isdir(self.voice_notes_data_dir): return
        for root, _dirs, files in os.walk(self.voice_notes_data_dir, topdown=True):
            for f in files:
                if not f.endswith(".json"): continue
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        rec = json.load(fp)
                    # Find the exact file matching this timestamp
                    if rec.get("ts") == ts:
                        rec["rock_id"] = rock_id
                        with open(path, "w", encoding="utf-8") as fp:
                            json.dump(rec, fp, ensure_ascii=False)
                        return # Done!
                except (json.JSONDecodeError, OSError):
                    continue
    
    def delete_all_data(self) -> None:
        """Empties both the rocks and voice notes database files."""
        if os.path.exists(self.rocks_path):
            open(self.rocks_path, 'w').close()
            
        if os.path.isdir(self.voice_notes_data_dir):
            import shutil
            for item in os.listdir(self.voice_notes_data_dir):
                item_path = os.path.join(self.voice_notes_data_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

    def save_rock(self, result: ClassificationResult) -> RockEntry:
        entry = RockEntry(
            rock_id=str(uuid.uuid4()),
            ts=time.time(),
            result=result,
            session_id=self.session_id # Attach session ID
        )
        rec = {
            "type": "rock",
            "rock_id": entry.rock_id,
            "ts": entry.ts,
            "session_id": entry.session_id, # Save to JSONL
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
                r = rec["result"]
                result = ClassificationResult(
                    label=r["label"], confidence=float(r["confidence"]),
                    image_path=r.get("image_path"), side_image_path=r.get("side_image_path"),
                    estimated_volume=r.get("estimated_volume"), estimated_weight=r.get("estimated_weight"),
                    raw=r.get("raw"),
                )
                rocks.append(RockEntry(
                    rock_id=rec["rock_id"], 
                    ts=rec["ts"], 
                    result=result,
                    session_id=rec.get("session_id") # Read from JSONL
                ))
        return rocks

    def save_voice_note(self, transcript: str, cleaned: Optional[str] = None, rock_id: Optional[str] = None) -> None:
        ts = time.time()
        now = datetime.datetime.fromtimestamp(ts)
        date_dir = os.path.join(self.voice_notes_data_dir, now.strftime("%Y%m%d"))
        os.makedirs(date_dir, exist_ok=True)
        filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.json"
        filepath = os.path.join(date_dir, filename)
        rec = {
            "type": "voice",
            "ts": time.time(),
            "session_id": getattr(self, "session_id", None),
            "rock_id": rock_id,
            "transcript": transcript,
            "cleaned": cleaned or transcript,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)

    def delete_rock(self, rock_id: str) -> None:
        if not os.path.exists(self.rocks_path): return
        lines = []
        with open(self.rocks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("rock_id") != rock_id:
                        lines.append(rec)
        with open(self.rocks_path, "w", encoding="utf-8") as f:
            for rec in lines:
                f.write(json.dumps(rec) + "\n")

    def delete_voice_note(self, ts: float) -> None:
        if not os.path.isdir(self.voice_notes_data_dir): return
        for root, _dirs, files in os.walk(self.voice_notes_data_dir, topdown=True):
            for f in files:
                if not f.endswith(".json"): continue
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        rec = json.load(fp)
                    # Find the exact file matching this timestamp and delete it
                    if rec.get("ts") == ts:
                        os.remove(path)
                        return # Done!
                except (json.JSONDecodeError, OSError):
                    continue

    def update_rock_volume(self, rock_id: str, volume: Optional[float], weight: Optional[str]) -> None:
        if not os.path.exists(self.rocks_path):
            return
        lines = []
        with open(self.rocks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") == "rock" and rec.get("rock_id") == rock_id:
                    r = rec["result"]
                    r["estimated_volume"] = volume
                    r["estimated_weight"] = weight
                lines.append(rec)
        with open(self.rocks_path, "w", encoding="utf-8") as f:
            for rec in lines:
                f.write(json.dumps(rec) + "\n")

    def list_voice_notes(self) -> List[dict]:
        if not os.path.isdir(self.voice_notes_data_dir):
            return []
        notes: List[dict] = []
        for root, _dirs, files in os.walk(self.voice_notes_data_dir, topdown=True):
            for f in files:
                if not f.endswith(".json"):
                    continue
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        rec = json.load(fp)
                    if "ts" in rec and "transcript" in rec:
                        notes.append(rec)
                except (json.JSONDecodeError, OSError):
                    continue
        notes.sort(key=lambda x: x.get("ts", 0), reverse=True)
        return notes


class ViewModel(QObject):
    state_changed = Signal(object)
    classification_changed = Signal(object)
    volume_display_changed = Signal(str)
    transcription_changed = Signal(str)
    recording_status_changed = Signal(bool)
    trip_changed = Signal(object)
    error = Signal(str)
    two_step_capture_message = Signal(str)

    def __init__(self, store_dir: str, voice_notes_data_dir: str, parent=None):
        super().__init__(parent)
        
        self._last_captured_path: str | None = None
        
        try:
            self.store = Store(store_dir, voice_notes_data_dir)
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
            print(f"ERROR: Failed to create TranscriptionService: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
        try:
            self.volume_service = VolumeService(self)
        except Exception as e:
            print(f"ERROR: Failed to create VolumeService: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

        self.transcriber.boot_model()

        self.state = AppStateType.HOME
        self.current_classification: ClassificationResult | None = None
        self.transcription_text = ""

        self._capture_phase: Optional[str] = None
        self._pending_top_path: Optional[str] = None
        self._pending_side_path: Optional[str] = None
        self._last_top_path: Optional[str] = None
        self._last_side_path: Optional[str] = None
        self._pending_volume_result: Optional[dict] = None
        self._volume_failed = False
        self._volume_pending = False
        self._volume_start_time: Optional[float] = None
        self._volume_ready_timer: Optional[QTimer] = None
        self._classification_saved_rock_id: Optional[str] = None
        self._classify_payload: Optional[dict] = None
        self._classification_applied = False

        self.classification_timeout_ms = 20_000
        self._classify_timeout = QTimer(self)
        self._classify_timeout.setSingleShot(True)
        self._classify_timeout.timeout.connect(self._on_classify_timeout)

        self.camera.capture_finished.connect(self._on_photo_captured)
        self.camera.capture_failed.connect(self._fail)
        self.classifier.finished.connect(self._on_classified)
        self.classifier.failed.connect(self._fail)
        self.volume_service.finished.connect(self._on_volume_finished)
        self.volume_service.failed.connect(self._on_volume_failed)
        self.transcriber.token.connect(self._on_transcription_token)
        self.transcriber.completed.connect(self._on_transcription_completed)
        self.transcriber.failed.connect(self._fail)

        self.transcriber.ready.connect(self._on_transcriber_ready)

        self.vtt_active = False
        self._was_session_finalized = False

    def assign_note_to_rock(self, note_ts: float, rock_id: str) -> None:
        self.store.update_voice_note_rock_id(note_ts, rock_id)
        self._publish_trip() # Refresh the UI to show it moved!
    
    def clear_all_trip_data(self) -> None:
        """Wipes all trip data and refreshes the UI."""
        self.store.delete_all_data()
        self.active_rock_id = None
        self._publish_trip() # Refreshes the UI to show an empty list!

    def _set_state(self, new_state: AppStateType) -> None:
        self.state = new_state
        self.state_changed.emit(new_state)

    def start_background_recording(self) -> None:
        if self.transcriber.is_recording:
            print("[VIEWMODEL] start_background_recording: already recording, ignoring", file=sys.stderr)
            return
        print("[VIEWMODEL] start_background_recording: starting new session", file=sys.stderr)
        self.transcription_text = ""
        self.transcription_changed.emit("")
        self.transcriber.start_recording()
        self.recording_status_changed.emit(True)

    def open_voice_page(self) -> None:
        # Push current transcript so the text box is up to date on arrival.
        self.transcription_changed.emit(self.transcription_text)
        self._set_state(AppStateType.VOICE_TO_TEXT)
    def reset_voice_context(self) -> None:
        """Clears the AI visual context and forces the next recording to be an orphan."""
        # 1. Force the database to save this as a standalone note
        self.active_rock_id = "ORPHAN"
        
        # 2. Wipe the visual_context.txt file so the transcriber stops forcing rock words
        project_root = Path(__file__).resolve().parent.parent.parent
        context_file = project_root / "ML-classifications" / "visual_context.txt"
        try:
            with open(context_file, "w") as f:
                f.write("") 
            print("[VIEWMODEL] Context explicitly reset by user. Note will be an ORPHAN.")
        except Exception as e:
            pass

    def go_home(self) -> None:
        if self.state == AppStateType.CAMERA_PREVIEW:
            self.camera.kill()
            self._capture_phase = None
            self._pending_top_path = None
            self._pending_side_path = None
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
        self._capture_phase = "top"
        self._pending_top_path = None
        self._pending_side_path = None
        if self._volume_ready_timer:
            self._volume_ready_timer.stop()
            self._volume_ready_timer = None
        self.camera.set_stop_after_capture(False)
        self._set_state(AppStateType.CAMERA_PREVIEW)

    def start_camera_stream(self, x: int, y: int, w: int, h: int) -> None:
        self.camera.start_preview(x, y, w, h)

    def trigger_capture(self) -> None:
        if self._capture_phase == "side":
            self.camera.set_stop_after_capture(True)
        self.camera.trigger_capture()

    def cancel_camera(self) -> None:
        self.camera.stop_preview()
        self.go_home()

    def get_review_image_paths(self) -> tuple[Optional[str], Optional[str]]:
        return (self._pending_top_path, self._pending_side_path)

    def confirm_captures_and_classify(self) -> None:
        self._pending_volume_result = None
        self._volume_failed = False
        self._volume_pending = False
        classify_path = self._pending_top_path or self._pending_side_path
        self._last_captured_path = classify_path
        self._set_state(AppStateType.CLASSIFYING)
        self._classify_timeout.start(self.classification_timeout_ms)
        self.classifier.classify(classify_path)

    def retake_captures(self) -> None:
        for p in (self._pending_top_path, self._pending_side_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        self._pending_top_path = None
        self._pending_side_path = None
        self._last_top_path = None
        self._last_side_path = None
        self._capture_phase = "top"
        self.camera.set_stop_after_capture(False)
        self._set_state(AppStateType.CAMERA_PREVIEW)

    def start_classification(self) -> None:
        self.current_classification = None
        self._set_state(AppStateType.CLASSIFYING)
        self.camera.start_preview()


    def capture_photo(self) -> None:
        """Call when user clicks Capture on the camera preview screen."""
        self.camera.capture()

    def reclassify(self) -> None:
        for p in (self._last_top_path, self._last_side_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        self.current_classification = None
        self._last_top_path = None
        self._last_side_path = None
        self.open_camera_preview()

    def _on_photo_captured(self, image_path: str) -> None:
        if self._capture_phase == "top":
            self._pending_top_path = image_path
            self._capture_phase = "side"
            self.camera.set_stop_after_capture(True)
            self.two_step_capture_message.emit("Rotate Rock and Capture 2nd View")
            return
        if self._capture_phase == "side":
            self._pending_side_path = image_path
            self._last_top_path = self._pending_top_path
            self._last_side_path = self._pending_side_path
            self._set_state(AppStateType.CONFIRM_CAPTURES)
            return
        self._last_captured_path = image_path
        self._set_state(AppStateType.CLASSIFYING)
        self._classify_timeout.start(self.classification_timeout_ms)
        self.classifier.classify(image_path)

    def _on_classified(self, payload: dict) -> None:
        self._classify_timeout.stop()
        self._classify_payload = payload
        self._classification_applied = False
        # Start volume estimation after classification so GPU is not shared (avoids Jetson OOM).
        if self._pending_top_path and self._pending_side_path:
            print(f"[VOLUME] Sending photos to volume estimation: top={self._pending_top_path!r}, side={self._pending_side_path!r}", file=sys.stderr)
            self._volume_pending = True
            self._volume_start_time = time.time()
            self.volume_service.estimate(self._pending_top_path, self._pending_side_path)
        else:
            print(f"[VOLUME] Skipping volume estimation: missing top or side image (top={self._pending_top_path!r}, side={self._pending_side_path!r})", file=sys.stderr)
        if self._pending_volume_result is not None:
            self._apply_classification_result()
            return
        if self._pending_volume_result is None:
            self._volume_ready_timer = QTimer(self)
            self._volume_ready_timer.setSingleShot(True)
            self._volume_ready_timer.timeout.connect(self._on_volume_ready_timeout)
            self._volume_ready_timer.start(10_000)
            return
        self._apply_classification_result()

    def _on_volume_ready_timeout(self) -> None:
        if self._volume_ready_timer:
            self._volume_ready_timer.stop()
            self._volume_ready_timer = None
        if self._classify_payload is not None and not self._classification_applied:
            self._apply_classification_result()

    def _apply_classification_result(self) -> None:
        if self._classify_payload is None or self._classification_applied:
            return
        self._classification_applied = True
        if self._volume_ready_timer:
            self._volume_ready_timer.stop()
            self._volume_ready_timer = None
        payload = self._classify_payload
        self._classify_payload = None
        vol = self._pending_volume_result
        if vol is not None and "volume_cm3" in vol:
            try:
                v = float(vol["volume_cm3"])
                estimated_volume = round(v, 1)
            except (TypeError, ValueError):
                estimated_volume = None
            mr = vol
            if mr.get("min_kg") is not None and mr.get("max_kg") is not None:
                try:
                    estimated_weight = f"{float(mr['min_kg']):.2f}–{float(mr['max_kg']):.2f} kg"
                except (TypeError, ValueError):
                    estimated_weight = None
            else:
                estimated_weight = None
        else:
            estimated_volume = None
            estimated_weight = None
        result = ClassificationResult(
            label=str(payload.get("label", "UNKNOWN")),
            confidence=float(payload.get("confidence", 0.0)),
            image_path=self._last_top_path or self._last_captured_path,
            side_image_path=self._last_side_path,
            estimated_volume=estimated_volume,
            estimated_weight=estimated_weight,
            raw=payload,
        )
        self.current_classification = result
        self.classification_changed.emit(result)
        if self._volume_pending:
            volume_str = "Volume = Calculating..."
        elif self._volume_failed:
            volume_str = "Volume = N/A"
        elif estimated_volume is not None:
            volume_str = f"Volume = {estimated_volume} cm³"
        else:
            volume_str = "Volume = N/A"
        self.volume_display_changed.emit(volume_str)
        self._set_state(AppStateType.CLASSIFIED)

    def save_classification(self) -> None:
        if self.current_classification:
            entry = self.store.save_rock(self.current_classification)
            self.active_rock_id = entry.rock_id
            self._classification_saved_rock_id = entry.rock_id
            label = self.current_classification.label
            
            # --- SHARED MAC PATH ---
            project_root = Path(__file__).resolve().parent.parent.parent
            context_file = project_root / "ML-classifications" / "visual_context.txt"
            
            try:
                with open(context_file, "w") as f:
                    f.write(label)
                print(f"[VIEWMODEL] Context file updated: {label}")
            except Exception as e:
                print(f"Warning: Could not save context bridge: {e}")
        self.go_home()

    def delete_classification(self) -> None:
        if self.current_classification:
            self._delete_classification_files(self.current_classification)
        self.current_classification = None
        self.go_home()

    def _delete_classification_files(self, result: ClassificationResult) -> None:
        """Delete the two images and _prediction.json for this classification from camera-pipeline/images/{date}/."""
        # Only delete files under ML-classifications/camera-pipeline/images/
        project_root = Path(__file__).resolve().parent.parent.parent
        images_root = project_root / "ML-classifications" / "camera-pipeline" / "images"
        try:
            images_root = images_root.resolve()
        except OSError:
            return
        to_delete: List[str] = []
        for path_str in (result.image_path, result.side_image_path):
            if not path_str:
                continue
            try:
                p = Path(path_str).resolve()
                if p.exists() and str(p).startswith(str(images_root)):
                    to_delete.append(str(p))
            except OSError:
                pass
        # _prediction.json is next to the top image (the one used for classification)
        if result.image_path:
            base, _ = os.path.splitext(result.image_path)
            pred_path = base + "_prediction.json"
            try:
                p = Path(pred_path).resolve()
                if p.exists() and str(p).startswith(str(images_root)):
                    to_delete.append(str(p))
            except OSError:
                pass
        for path_str in to_delete:
            try:
                os.remove(path_str)
                print(f"[VIEWMODEL] Deleted: {path_str}", file=sys.stderr)
            except OSError as e:
                print(f"[VIEWMODEL] Failed to delete {path_str}: {e}", file=sys.stderr)

    def _delete_pending_classification_files(self) -> None:
        """Delete pending session images, _prediction.json, and _volume.json under camera-pipeline/images/."""
        project_root = Path(__file__).resolve().parent.parent.parent
        images_root = project_root / "ML-classifications" / "camera-pipeline" / "images"
        try:
            images_root = images_root.resolve()
        except OSError:
            return
        to_delete: set = set()
        classify_path = self._pending_top_path or self._pending_side_path or self._last_captured_path
        for path_str in (self._pending_top_path, self._pending_side_path, self._last_captured_path):
            if not path_str:
                continue
            try:
                p = Path(path_str).resolve()
                if p.exists() and str(p).startswith(str(images_root)):
                    to_delete.add(str(p))
            except OSError:
                pass
        if classify_path:
            base, _ = os.path.splitext(classify_path)
            pred_path = base + "_prediction.json"
            try:
                p = Path(pred_path).resolve()
                if p.exists() and str(p).startswith(str(images_root)):
                    to_delete.add(str(p))
            except OSError:
                pass
        if self._pending_top_path:
            base, _ = os.path.splitext(self._pending_top_path)
            vol_path = base + "_volume.json"
            try:
                p = Path(vol_path).resolve()
                if p.exists() and str(p).startswith(str(images_root)):
                    to_delete.add(str(p))
            except OSError:
                pass
        for path_str in to_delete:
            try:
                os.remove(path_str)
                print(f"[VIEWMODEL] Deleted: {path_str}", file=sys.stderr)
            except OSError as e:
                print(f"[VIEWMODEL] Failed to delete {path_str}: {e}", file=sys.stderr)

    def _on_volume_finished(self, payload: dict) -> None:
        if self._volume_start_time is not None and (time.time() - self._volume_start_time) > 120:
            return
        self._volume_pending = False
        self._pending_volume_result = payload
        if self._classify_payload is not None and not self._classification_applied:
            self._apply_classification_result()
            return
        if self._classification_applied and self.current_classification:
            vol = payload.get("volume_cm3")
            mr = payload
            try:
                v = float(vol) if vol is not None else None
            except (TypeError, ValueError):
                v = None
            w = None
            if mr.get("min_kg") is not None and mr.get("max_kg") is not None:
                try:
                    w = f"{float(mr['min_kg']):.2f}–{float(mr['max_kg']):.2f} kg"
                except (TypeError, ValueError):
                    pass
            new_result = ClassificationResult(
                label=self.current_classification.label,
                confidence=self.current_classification.confidence,
                image_path=self.current_classification.image_path,
                side_image_path=self.current_classification.side_image_path,
                estimated_volume=round(v, 1) if v is not None else None,
                estimated_weight=w or self.current_classification.estimated_weight,
                raw=self.current_classification.raw,
            )
            self.current_classification = new_result
            self.classification_changed.emit(new_result)
            volume_str = f"Volume = {round(v, 1)} cm³" if v is not None else "Volume = N/A"
            self.volume_display_changed.emit(volume_str)
            if self._classification_saved_rock_id:
                self.store.update_rock_volume(self._classification_saved_rock_id, v, w)

    def _on_volume_failed(self, message: str) -> None:
        self._volume_failed = True
        self._volume_pending = False
        self.volume_display_changed.emit("Volume = N/A")
        print(f"[VOLUME] {message}", file=sys.stderr)

    def _on_classify_timeout(self) -> None:
        self.classifier.kill()
        self._fail("Classification timeout (20s)")

    def _abort_classification(self, error_message: Optional[str]) -> None:
        """Stop timers, kill classification and volume processes, delete images + .json files created, go home"""
        if self.state != AppStateType.CLASSIFYING:
            self.go_home()
            return
        self._classify_timeout.stop()
        if self._volume_ready_timer:
            self._volume_ready_timer.stop()
            self._volume_ready_timer = None
        self.classifier.kill()
        self.volume_service.kill()
        self._delete_pending_classification_files()
        self._classify_payload = None
        self._pending_volume_result = None
        self._classification_applied = False
        self._volume_pending = False
        self._volume_failed = False
        self._volume_start_time = None
        self.current_classification = None
        self._pending_top_path = None
        self._pending_side_path = None
        self._last_captured_path = None
        self._last_top_path = None
        self._last_side_path = None
        self.go_home()

    def cancel_classification(self) -> None:
        """if user cancels in analyzing page, clean up and return to home."""
        if self.state != AppStateType.CLASSIFYING:
            return
        self._abort_classification(None)

    def start_voice_to_text(self, silent: bool = False) -> None:
        import sys
        print("[VIEWMODEL] User-initiated VTT Start requested", file=sys.stderr)
        
        self.vtt_active = True
        self._was_session_finalized = False
        self.transcription_text = "" 
        self.transcription_changed.emit("")
        
        self.transcriber.start_recording()
        self.recording_status_changed.emit(True) 
        
        if not silent:
            self._set_state(AppStateType.VOICE_TO_TEXT)


    def stop_voice_to_text(self) -> None:
        import sys
        print("[VIEWMODEL] stop_voice_to_text() called", file=sys.stderr)
        self.vtt_active = False #!!!!!
        self.transcriber.stop_recording()
        self.recording_status_changed.emit(False)

    def redo_voice_to_text(self) -> None:
        import sys
        print("[VIEWMODEL] Redo requested", file=sys.stderr)
        
        self.transcriber.stop_recording()
        self.transcription_text = ""
        self.transcription_changed.emit("")
        self.recording_status_changed.emit(False)
        was_on_vtt = (self.state == AppStateType.VOICE_TO_TEXT)
        QTimer.singleShot(300, lambda: self.start_voice_to_text(silent=not was_on_vtt))

    def save_transcription(self) -> None:
        self._was_session_finalized = True
        text = self.transcription_text.strip()
        if text:
            # Pass the explicit rock_id so it links permanently!
            self.store.save_voice_note(text, cleaned=text, rock_id=getattr(self, "active_rock_id", None))

        self.stop_voice_to_text()
        self.vtt_active = False
        self.transcription_text = ""

        self.transcription_changed.emit("")
        self.recording_status_changed.emit(False)
        self.go_home()
    
    def make_rock_current(self, entry: RockEntry) -> None:
        """Sets an old rock as the target for new voice notes and updates the AI transcriber context."""
        self.active_rock_id = entry.rock_id
        label = entry.result.label
        project_root = Path(__file__).resolve().parent.parent.parent
        context_file = project_root / "ML-classifications" / "visual_context.txt"
        try:
            with open(context_file, "w") as f:
                f.write(label)
            print(f"[VIEWMODEL] Context explicitly overridden to: {label}")
        except Exception as e:
            pass

    def delete_rock_by_id(self, rock_id: str) -> None:
        self.store.delete_rock(rock_id)
        if getattr(self, "active_rock_id", None) == rock_id:
            self.active_rock_id = None
        self._publish_trip()

    def delete_voice_note_by_ts(self, ts: float) -> None:
        self.store.delete_voice_note(ts)
        self._publish_trip()

    def delete_transcription(self) -> None:
        import sys
        self._was_session_finalized = True
        self.vtt_active = False
        self.transcriber.stop_recording()
        self.transcription_text = ""
        self.transcription_changed.emit("")
        self.recording_status_changed.emit(False)
        self.go_home()

    def _on_transcription_token(self, chunk: str) -> None:
        import sys
        print(f"[VIEWMODEL] Received transcription token: '{chunk}'", file=sys.stderr)
        
        # --- DEBUG FIX: Force every new chunk onto its own line ---
        self.transcription_text += chunk.strip() + "\n"
        
        print(f"[VIEWMODEL] Updated transcription_text (length: {len(self.transcription_text)}): '{self.transcription_text[:200]}'", file=sys.stderr)
        self.transcription_changed.emit(self.transcription_text)

    # def _on_transcription_token(self, chunk: str) -> None:
    #     import sys
    #     print(f"[VIEWMODEL] Received transcription token: '{chunk}'", file=sys.stderr)
    #     # if self.state == AppStateType.VOICE_TO_TEXT_LOADING:
    #     #     self._set_state(AppStateType.VOICE_TO_TEXT)
    #     self.transcription_text += chunk
    #     print(f"[VIEWMODEL] Updated transcription_text (length: {len(self.transcription_text)}): '{self.transcription_text[:200]}'", file=sys.stderr)
    #     self.transcription_changed.emit(self.transcription_text)

    def _on_transcription_completed(self, final_text: str) -> None:
        import sys
        if self._was_session_finalized:
            return
        if not final_text or not final_text.strip() or "No audio recorded" in final_text:
            return
        
        print(f"[VIEWMODEL] CLEANUP: Replacing live text with final version", file=sys.stderr)
        self.transcription_text = final_text
        self.transcription_changed.emit(self.transcription_text)


    def _on_transcription_failed(self, message: str) -> None:
        print(f"[VIEWMODEL] Transcription error (non-fatal): {message}", file=sys.stderr)

    def _publish_trip(self) -> None:
        rocks = self.store.list_rocks()
        voice_notes = self.store.list_voice_notes()
        total_volume = 0.0
        total_weight = 0.0
        for r in rocks:
            if r.result.estimated_volume is not None:
                try:
                    total_volume += float(r.result.estimated_volume)
                except (TypeError, ValueError):
                    pass
            w = r.result.estimated_weight
            if w is not None:
                if isinstance(w, (int, float)):
                    total_weight += float(w)
                elif isinstance(w, str):
                    try:
                        a, b = w.replace(" kg", "").split("–")
                        total_weight += (float(a) + float(b)) / 2 # using average of min and max weight
                    except (ValueError, TypeError):
                        pass
        summary = TripSummary(rocks=rocks, total_volume=total_volume, total_weight=total_weight, voice_notes=voice_notes)
        self.trip_changed.emit(summary)

    def _fail(self, message: str) -> None:
        import sys
        print(f"\n[CRITICAL DEBUG] Failure detected: {message}", file=sys.stderr)
        # self._abort_classification(message)  <-- Comment this out!

