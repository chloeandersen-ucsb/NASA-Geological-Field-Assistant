# led-display

The main UI running on a 4 inch LED display. A PySide6 app that shows real-time rock classification results, voice note transcripts, and mission data.

## key files

| File | Description |
|---|---|
| `core/viewmodel.py` | app state + pipeline organization: triggers camera → classification → volume estimation. Manages `Store` (read/write to `sage_store/`), drives the UI |
| `ui/app_window.py` | all screens (`HomePage`, `CaptureReviewPage`, `ClassifiedPage`, `RockDetailPage`, etc.). Routes between them based on ViewModel state |
| `ui/joystick_navigator.py` | reads hardware joystick via `pygame` on a background thread, emits navigation signals |
| `services/process_service.py` | `QThread` subprocess wrappers for `CameraService`, `ClassificationService`, `VolumeService`, and `TranscriptionService` |
| `services/rock_summary.py` | runs `Phi-3-mini` (llama-cpp) locally to generate a plain-English summary from classification + voice notes |
| `sage_data/` | local data store for dev (`rocks.jsonl`, `voice_notes.jsonl`, `missions.json`), maps to `sage_store/` on the Jetson |
