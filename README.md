# **S**urface **A**ssistant for **G**eological **E**valuation 
A field system for real-time lunar rock classification. Runs on a NVIDIA Jetson Orin NX with a microphone, camera, and LED display. A ground UI on a separate machine lets Earth-based mission control review collected data.

## system overview

Camera captures an image of a rock, RockNet classifies it, and the volume estimator measures its dimensions. All information is shown on LED display in real time. Meanwhile the MEMS mic streams to the voice transcriber, which appends timestamped field notes. 

Everything gets written to `sage_store/` on the Jetson. After a session, `scripts/transfer_to_ground.ps1` pulls that folder to the ground machine, where `ground-UI` lets mission control browse the full mission log.

### where data lives

| Data | File(s) | Notes |
|---|---|---|
| Captured images | `ML-classifications/camera-pipeline/images/YYYYMMDD/*.jpg` | one folder per day |
| Rock classification | `*_prediction.json` + `sage_store/rocks.jsonl` | prediction.json stays with image; jsonl is the running log |
| Volume + mass | `*_volume.json` | merged into the rock's entry in `rocks.jsonl` |
| Voice notes | `sage_store/voice_notes.jsonl` | timestamped transcripts, keyed to rock sample |
| Mission index | `sage_store/missions.json` | groups rocks into named missions |

## quick start

```bash
make setup      # install dependencies + download models
make check      # verify setup
make run        # production mode (full pipeline)
make run-mock   # mock data (no hardware needed)
make run-mock-ml  # real voice transcription, mock ML
```

## display controls

| Key | Action |
|---|---|
| Joystick / arrow keys | navigate rocks |
| ESC | exit fullscreen |
| CTRL+C | quit |
| Quit button | quit |

## folders

| Folder | What it does |
|---|---|
| [`ML-classifications/`](ML-classifications/) | rock classifier |
| [`led-display/`](led-display/) | PySide6 UI |
| [`voiceNotes/`](voiceNotes/) | live speech-to-text for field annotations |
| [`rock-volume/`](rock-volume/) | volume + mass calculations |
| [`ground-UI/`](ground-UI/) | browser-based mission review application |
| [`pcb/`](pcb/) | MEMS microphone breakout board (KiCad) |
| [`microphone-testing/`](microphone-testing/) | audio test samples |
| [`scripts/`](scripts/) | data transfer utilities |

