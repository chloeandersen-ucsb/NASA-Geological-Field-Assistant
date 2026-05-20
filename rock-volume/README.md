# rock-volume

Estimates the volume and mass of a rock sample from two photos (top-down + side view). Uses AprilTag markers for real-world scaling and SAM (Segment Anything) for segmentation.

## setup

```bash
make setup   # installs folder specific deps + downloads sam (~375 MB)
```

Optional — calibrate your camera for better accuracy:

```bash
python charuco_calibration.py --board_only   # print the ChArUco board
python charuco_calibration.py --images <dir> # run after capturing 20+ images
```

This helps for lens distortion correction. Without it, the tool falls back to AprilTag scaling.

**output JSON**
```json
{
  "dimensions_cm": { "L": ..., "W": ..., "H": ... },
  "volume_cm3": ...,
  "mass_range_g": [min, max],
  "mass_range_kg": [min, max]
}
```

## test run

```bash
python measure_rock_volume.py \
  --top test_images/rock_centered_1.jpg \
  --side test_images/rock_centered_2.jpg
```

Expected: ~8.7 cm³, mass ~0.022–0.026 kg.

## files

| File | Description |
|---|---|
| `measure_rock_volume.py` | main entry point |
| `volume_estimation_dual_view.py` | dual-view estimation core |
| `rock_volume_estimator_v0.py` | earlier single-view prototype |
| `charuco_calibration.py` | camera calibration tool |
| `download_sam.py` | downloads SAM model weights |
| `camera_intrinsics.json` | saved calibration (gitignored if large) |
| `test_images/` | sample images for testing |
