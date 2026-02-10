# SAGE Project

## Setup

```bash
make setup      # Install dependencies (everything)
make clean      # Clean build artifacts
```

## Run

```bash
make run            # Production: real camera, real ML, real voice
make run-mock-ml    # Real voice; mock ML and mock camera
make run-mock-cam   # Real voice, real ML; sample image (no camera)
make run-mock       # Mock everything
```

## Display

- ESC: exit full screen
- F11: control full/not full screen
- Ctrl+C: quit application
- Or press "Quit" to quit
