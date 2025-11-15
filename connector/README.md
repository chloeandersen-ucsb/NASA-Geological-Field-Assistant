# Connector - Jetson BLE Bridge

BLE bridge connecting Apple Watch to ML models on NVIDIA Jetson.

## What It Does

Exposes two BLE services that the Watch app uses:
- **Classification Service** - Triggers `rocknet_infer.py` to classify rocks
- **Transcription Service** - Runs `improved2.py` for voice-to-text

**ML models are automatically invoked** when the Watch sends commands - you don't run them manually.

---

## Quick Setup (From Scratch)

### Prerequisites
- NVIDIA Jetson (Nano/Xavier/Orin) with JetPack installed
- USB Bluetooth adapter (e.g., USB-BT500) - built-in BT often unreliable
- Python 3.8+

### 1. Clone Repository to Jetson

```bash
# SSH into Jetson
ssh jetson@<jetson-ip>

# Clone to /opt/capstone (recommended)
sudo mkdir -p /opt/capstone
sudo chown $USER:$USER /opt/capstone
cd /opt/capstone
git clone <your-repo-url> .
```

**Expected structure:**
```
/opt/capstone/
├── ML-classifications/
│   ├── rocknet_infer.py        # Your ML classification script
│   ├── best_rocknet.pt         # Model weights
│   └── latest.jpg              # Image to classify (camera updates this)
├── voice-to-text/
│   └── improved2.py            # Your voice ASR script
└── connector/                  # This folder
    ├── capstone_connector/
    ├── scripts/
    └── config.yaml
```

### 2. Install Dependencies

```bash
cd /opt/capstone/connector

# Install connector dependencies
pip3 install -r requirements.txt

# Install test dependencies (optional)
pip3 install -r requirements-test.txt
```

**ML dependencies** (install separately in your ML environment):
```bash
# For rocknet_infer.py
pip3 install torch torchvision timm pillow

# For improved2.py
pip3 install sounddevice nemo_toolkit numpy
```

### 3. Setup Bluetooth

```bash
cd /opt/capstone/connector
sudo bash scripts/setup_bluetooth.sh
```

This configures the USB Bluetooth adapter as the default.

### 4. Configure Paths

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

### 5. Run Connector

```bash
# Development mode
cd /opt/capstone/connector
bash scripts/run_dev.sh
```

You should see:
```
[connector] Advertising 'GeoFieldAssistant-01' on /org/bluez/hci0
[connector] Services ready: Classification + Transcription
```

### 6. Connect from Apple Watch

1. Open SAGE app on Watch
2. Watch auto-scans and connects to "GeoFieldAssistant-01"
3. Double-pinch to classify or record voice notes

**The connector automatically:**
- Runs `rocknet_infer.py` when Watch requests classification
- Runs `improved2.py` when Watch starts voice recording
- Sends results back to Watch via BLE

---

## Run on Boot (Production)

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

## Testing

```bash
cd /opt/capstone/connector

# Run all tests
pytest

# Run specific test suites
pytest tests/test_protocol_compliance.py -v   # BLE protocol
pytest tests/test_watch_app_simulation.py -v  # Watch integration
pytest tests/test_timing_and_timeouts.py -v   # Timing scenarios

# With coverage
pytest --cov=capstone_connector --cov-report=html
```

See `tests/README.md` for detailed test documentation.

---

## How It Works

### Classification Flow
```
Watch sends 0x01 → Connector runs rocknet_infer.py
→ Parses {"label":"Basalt","confidence":0.95}
→ Normalizes to {"label":"Basalt","confidence":95}
→ Sends to Watch via BLE notification
```

### Transcription Flow
```
Watch sends 0x01 → Connector runs improved2.py
→ Streams NDJSON tokens: {"type":"token","t":"the "}
→ Converts to plain UTF-8: "the "
→ Sends to Watch via BLE notifications
→ Watch sends 0x02 → Connector stops improved2.py
```

**ML models are subprocess-based** - connector spawns them on-demand and parses their output.

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

## Troubleshooting

**"No Bluetooth adapter found"**
```bash
# Check adapter
lsusb | grep Bluetooth
hciconfig

# Re-plug USB adapter and reboot
sudo reboot
```

**"Classification returns error"**
```bash
# Verify ML files exist
ls -la /opt/capstone/ML-classifications/best_rocknet.pt
ls -la /opt/capstone/ML-classifications/latest.jpg

# Test manually
cd /opt/capstone/ML-classifications
python3 rocknet_infer.py --weights best_rocknet.pt --image latest.jpg
```

**"Transcription not working"**
```bash
# Check audio device
python3 -c "import sounddevice; print(sounddevice.query_devices())"

# Test manually
cd /opt/capstone/voice-to-text
python3 improved2.py
# Should see "RECORDING NOW..." and transcription output
```

**"Watch can't connect"**
```bash
# Check connector is advertising
sudo systemctl status capstone-connector
sudo bluetoothctl
# > scan on
# Should see "GeoFieldAssistant-01"
```

**Tests failing**
```bash
# Run with verbose output
pytest -vv --tb=long

# Check test README
cat tests/README.md
```

---

## Configuration

Edit `config.yaml` to customize:

```yaml
device:
  adapter: hci0                    # Bluetooth adapter
  advertised_name: GeoFieldAssistant-01

runtime:
  asr_idle_timeout_s: 60          # Stop if no speech for 60s
  asr_hard_cap_min: 15            # Max 15min transcription
  notify_rate_hz: 15              # BLE notification rate
```

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

## Further Reading

- **BLE Protocol Details:** See `/watch/JETSON_CONFIGURATION.md`
- **Test Documentation:** See `tests/README.md`
- **Watch App Setup:** See `/watch/README.md`
