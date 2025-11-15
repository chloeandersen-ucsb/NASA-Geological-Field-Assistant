# Connector

BLE bridge connecting Apple Watch to ML models on NVIDIA Jetson.

Exposes two BLE services that the Watch app uses:
- **Classification Service** - Triggers `rocknet_infer.py` to classify rocks
- **Transcription Service** - Runs `improved2.py` for voice-to-text

**ML models are automatically invoked** when the Watch sends commands

---

### Dependencies

```bash
cd /opt/capstone/connector

# Install connector dependencies
pip3 install -r requirements.txt

# Install test dependencies (optional)
pip3 install -r requirements-test.txt
```

**other file dependencies**
```bash
# For rocknet_infer.py
pip3 install torch torchvision timm pillow

# For improved2.py
pip3 install sounddevice nemo_toolkit numpy
```

### Setup Bluetooth

```bash
cd /opt/capstone/connector
sudo bash scripts/setup_bluetooth.sh
```

This configures the USB Bluetooth adapter as the default (once plugged into jetson).

### Configure Paths

Edit `config.yaml` and set your paths:

```yaml
classifier:
  mode: subprocess
  subprocess_cmd:
    - python3
    - /opt/capstone/ML-classifications/rocknet_infer.py
    - --weights
    - /opt/capstone/ML-classifications/best_rocknet.pt
    - --image
    - /opt/capstone/ML-classifications/latest.jpg

transcriber:
  mode: subprocess
  subprocess_cmd:
    - python3
    - /opt/capstone/voice-to-text/improved2.py
```

Or use environment variable:
```bash
export CAPSTONE_ROOT=/opt/capstone
# Scripts will auto-detect from this
```

### Run Connector

```bash
# Development mode
cd /opt/capstone/connector
bash scripts/run_dev.sh
```

should see:
```
[connector] Advertising 'GeoFieldAssistant-01' on /org/bluez/hci0
[connector] Services ready: Classification + Transcription
```

### Connect from Apple Watch

1. Open SAGE app on Watch
2. Watch auto-scans and connects to "GeoFieldAssistant-01"
3. Double-pinch to classify or record voice notes

**The connector automatically:**
- Runs `rocknet_infer.py` when Watch requests classification
- Runs `improved2.py` when Watch starts voice recording
- Sends results back to Watch via BLE

---

## Production Run (implementation still tbd...)

```bash
# Install as systemd service
sudo cp systemd/capstone-connector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable capstone-connector
sudo systemctl start capstone-connector

# Check status
sudo systemctl status capstone-connector
sudo journalctl -u capstone-connector -f
```

---

## BLE Protocol

### Services & Characteristics

**Classification Service:** `9FBF0B7E-5A89-4E6A-91F6-12C1B0B1A001`
- Command `...A101` (Write): `0x01` = classify now
- Result `...A102` (Notify): JSON `{"label":"...","confidence":0-100}`
- Status `...A103` (Notify): State updates

**Transcription Service:** `9FBF0B7E-5A89-4E6A-91F6-12C1B0B1B001`
- Command `...B101` (Write): `0x01` = start, `0x02` = stop
- Stream `...B102` (Notify): UTF-8 text tokens + control frames

**Control Frames:**
- `__START__` - Transcription started
- `__END__` - Transcription ended
- `__ERROR:CODE:message__` - Error occurred

---

## Directory Structure

```
connector/
├── capstone_connector/           # Main package
│   ├── server.py                 # BLE GATT server
│   ├── bridge.py                 # Command routing
│   ├── adapters.py               # ML model adapters
│   ├── constants.py              # Protocol constants
│   └── wrappers/                 # ML output parsers
│       ├── classify_from_rocknet.py
│       └── v2t_from_improved2.py
├── scripts/
│   ├── setup_bluetooth.sh        # BT setup
│   └── run_dev.sh                # Dev launcher
├── tests/                        # Comprehensive test suite
│   ├── test_protocol_compliance.py
│   ├── test_watch_app_simulation.py
│   └── test_timing_and_timeouts.py
├── config.yaml                   # Configuration
└── requirements.txt              # Python dependencies
```
---
