# Setup Guide

Get started with Rock Estimator in 3 steps.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Download SAM Model (Required)

```bash
python download_sam.py
```

This downloads the Segment Anything model (~375 MB) to `~/.sam_models/`.

**Note**: This is required for automatic object segmentation. If not installed, `measure_rock.py` will remind you.

## Step 3: (Optional) Calibrate Camera

For best accuracy, calibrate your camera using a printed ChArUco board:

```bash
# Generate board (printed calibration target)
python charuco_calibration.py --board_only

# Capture 20+ images of the board at different angles/distances
# Print board a 100% scale, ensure board is correct size
# Then run calibration
python charuco_calibration.py --images <path_to_calib_images>
```

This creates `camera_intrinsics.json` for lens correction.

If you skip this, the pipeline uses AprilTag markers for scaling only (still works fine).

---

## Usage: Measure an Object

Once setup is complete:

```bash
# Automatic measurement (top + side view)
python measure_rock.py --mode auto --top top_image.jpg --side side_image.jpg
```

**Output**: `outputs/measurement.json` with:
- Dimensions (L × W × H in cm)
- Volume (cm³)
- Mass range (g)
- All measurement details

---

## Quick Test

Test the pipeline on default test images:

```bash
python measure_rock.py --mode auto --top rock_centered_1.jpg --side rock_centered_2.jpg
```

