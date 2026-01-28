# SAGE Project Setup

## Quick Start

```bash
make check      # Verify setup
make run        # Production mode
make run-mock   # Test with mock data
```

## Setup

### Jetson Orin NX

```bash
make setup
```

This installs:
- Python3 and pip
- PySide6
- Creates `/data/sage` data directory

### Mac (Development)

Using virtual environment (recommended):
```bash
cd led-display
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Or install directly:
```bash
pip3 install PySide6
```

Or use conda:
```bash
conda create -n sage-ui python=3.10
conda activate sage-ui
conda install pyside6
```

## Running

### Production Mode (Real Data)

```bash
make run
```

Uses real ML models and voiceNotes. Requires:
- `ML-classifications/rocknet_infer.py` and `best_rocknet.pt`
- `voiceNotes/improved2.py`

### Mock Mode (Testing)

```bash
make run-mock
```

Uses mock scripts for testing without real models.

### With Conda (Mac)

```bash
make run-conda
```

## Environment Variables

- `SAGE_STORE_DIR` - Data directory (default: `/data/sage` on Jetson, `./led-display/sage_data` on Mac)
- `SAGE_USE_MOCKS=1` - Use mock services
- `SAGE_ML_CLASSIFICATIONS_DIR` - Override ML-classifications path
- `SAGE_VOICE_TO_TEXT_DIR` - Override voiceNotes path
- `SAGE_ROCKNET_WEIGHTS` - Override model weights path
- `JETSON_PLATFORM=1` - Force Jetson mode detection

## Project Structure

```
capstone/
├── connector.py          # Centralized path/config management
├── Makefile             # Root build/run commands
├── led-display/         # Main UI application
│   ├── .gitignore       # Git ignore rules (excludes venv/, __pycache__, etc.)
│   ├── requirements.txt # Python dependencies (PySide6)
│   └── venv/            # Virtual environment (not in git)
├── ML-classifications/  # Rock classification model
└── voiceNotes/      # Voice transcription
```

### Dependencies

The `led-display/` folder includes:
- **`requirements.txt`**: Python package dependencies (PySide6)
- **`.gitignore`**: Excludes virtual environment, cache files, and data directories from version control

## Troubleshooting

- **Display issues on Jetson**: Set `QT_QPA_PLATFORM=xcb` if needed
- **Path errors**: Run `make check` to verify all paths
- **Permission errors**: Ensure data directory is writable (`sudo chown $USER:$USER /data/sage` on Jetson)
