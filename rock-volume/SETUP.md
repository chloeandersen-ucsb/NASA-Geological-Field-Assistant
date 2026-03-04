# Rock Volume Measurement – Setup & Usage

## Prerequisites

Install required Python packages:

```bash
pip install -r requirements.txt
```

## Required: Download SAM Model

```bash
python download_sam.py
```

Downloads the Segment Anything model (~375 MB) to `~/.sam_models/`. This is required for automatic object segmentation.

## Optional: Camera Calibration

For best accuracy, calibrate your camera using a ChArUco board:

```bash
# Generate printable board
python charuco_calibration.py --board_only

# Capture 20+ calibration images, then run:
python charuco_calibration.py --images <images_directory>
```

Creates `camera_intrinsics.json` for lens distortion correction. Without this, the tool uses AprilTag markers for scaling (still accurate).

---

## Usage: Measure an Object

Measure an object using two orthogonal images (top and side view):

```bash
python measure_rock_volume.py --top top_view.jpg --side side_view.jpg
```

**Required Arguments:**
- `--top`: Path to top-down image
- `--side`: Path to side-view image

**Optional Arguments:**
- `--ref_cm`: AprilTag size in cm (default: 10.0 for tag36h11)
- `--dens_min`: Min density in g/cm³ (default: 2.5)
- `--dens_max`: Max density in g/cm³ (default: 3.0)
- `--out`: Output JSON path (default: `outputs/TIMESTAMP-measurement.json`)

**Output**: JSON file containing:
- Dimensions: L × W × H (cm)
- Volume (cm³)
- Mass range (g, kg)
- Shape analysis & segmentation details

---

## Test Run

```bash
python measure_rock_volume.py --top test_images/rock_centered_1.jpg --side test_images/rock_centered_2.jpg
```

Expected output: Volume ~8.7 cm³, mass range 0.022–0.026 kg.

