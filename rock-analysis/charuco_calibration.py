"""
ChArUco board generation + camera calibration (OpenCV 4.8).
Reference: https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/sandbox/ludovic/aruco_calibration_rotation.html
"""

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from cv2 import aruco
from reportlab.pdfgen import canvas

# Hardcoded ONE board (calib.io): 8x11, DICT_4X4_50, 20mm squares, 15mm markers
BOARD_ROWS = 11
BOARD_COLS = 8
BOARD_SQUARE_MM = 20.0
BOARD_MARKER_MM = 15.0
BOARD_DICT = "DICT_4X4_50"


def build_board():
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, BOARD_DICT))
    board = aruco.CharucoBoard(
        size=(BOARD_COLS, BOARD_ROWS),
        squareLength=BOARD_SQUARE_MM / 1000.0,
        markerLength=BOARD_MARKER_MM / 1000.0,
        dictionary=dictionary,
    )
    return board, dictionary


def generate_board_image(board, out_path: Path, dpi: int = 300, margin_mm: float = 10.0):
    """Generate board at exact physical dimensions for printing (PNG + PDF)."""
    MM_PER_INCH = 25.4
    square_px = int(BOARD_SQUARE_MM / MM_PER_INCH * dpi)
    width_px = square_px * BOARD_COLS
    height_px = square_px * BOARD_ROWS
    
    board_img = aruco.CharucoBoard.generateImage(board, (width_px, height_px), marginSize=0)
    
    # Add white margins to prevent printer cutoff
    margin_px = int(margin_mm / MM_PER_INCH * dpi)
    board_with_margin = np.ones((height_px + 2*margin_px, width_px + 2*margin_px), dtype=np.uint8) * 255
    board_with_margin[margin_px:margin_px+height_px, margin_px:margin_px+width_px] = board_img
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(out_path), board_with_margin)
    
    # Also generate PDF with exact physical dimensions (including margins)
    width_mm = BOARD_COLS * BOARD_SQUARE_MM + 2 * margin_mm
    height_mm = BOARD_ROWS * BOARD_SQUARE_MM + 2 * margin_mm
    width_in = width_mm / MM_PER_INCH
    height_in = height_mm / MM_PER_INCH
    width_pt = width_in * 72.0
    height_pt = height_in * 72.0

    pdf_path = out_path.with_suffix('.pdf')
    c = canvas.Canvas(str(pdf_path), pagesize=(width_pt, height_pt))
    c.drawImage(str(out_path), 0, 0, width=width_pt, height=height_pt, preserveAspectRatio=False, mask='auto')
    c.showPage()
    c.save()
    
    print(f"Generated board PNG: {board_with_margin.shape[1]}×{board_with_margin.shape[0]} px at {dpi} DPI")
    print(f"Generated board PDF: {width_mm:.0f}×{height_mm:.0f} mm ({BOARD_COLS}×{BOARD_ROWS} squares + {margin_mm}mm margins)")
    print(f"  PNG: {out_path}")
    print(f"  PDF: {pdf_path} (print at 100% scale, no page scaling)")


def read_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".tiff", ".bmp")
    paths = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
    if not paths:
        raise RuntimeError(f"No images found in {folder}")
    return paths


def maybe_downscale(gray: np.ndarray, max_dim: int | None):
    if max_dim is None:
        return gray
    h, w = gray.shape
    if max(h, w) <= max_dim:
        return gray
    scale = max_dim / float(max(h, w))
    return cv.resize(gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)


def calibrate(
    images_dir: Path,
    min_corners: int,
    max_dim: int | None,
    board_img_out: Path,
    out_json: Path,
):
    board, dictionary = build_board()
    generate_board_image(board, board_img_out)

    params = aruco.DetectorParameters()
    image_paths = read_images(images_dir)

    all_corners = []
    all_ids = []
    used = []
    rejected = []
    image_size = None

    for p in image_paths:
        img_bgr = cv.imread(str(p))
        if img_bgr is None:
            rejected.append((p.name, "read_fail"))
            continue
        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        gray = maybe_downscale(gray, max_dim)
        image_size = gray.shape[::-1]

        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
        markers_count = 0 if ids is None else len(ids)

        if ids is None or markers_count == 0:
            print(f"{p.name}: charuco_corners=0, aruco_markers=0")
            rejected.append((p.name, "no_markers"))
            continue

        num, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
        num_corners = 0 if ch_corners is None else int(num)
        print(f"{p.name}: charuco_corners={num_corners}, aruco_markers={markers_count}")

        if ch_corners is None or ch_ids is None or num_corners < min_corners:
            rejected.append((p.name, f"insufficient_corners_{num_corners}"))
            continue

        all_corners.append(ch_corners)
        all_ids.append(ch_ids)
        used.append(p.name)

    if len(all_corners) == 0:
        raise RuntimeError(f"No frames with sufficient ChArUco corners found (need >= {min_corners}).")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "rms_reprojection_error": float(ret),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "board": {
            "rows": BOARD_ROWS,
            "cols": BOARD_COLS,
            "square_mm": BOARD_SQUARE_MM,
            "marker_mm": BOARD_MARKER_MM,
            "dictionary": BOARD_DICT,
        },
        "used_images": used,
        "num_views": len(used),
        "rejected_images": rejected,
        "method": "charuco_standard",
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("\nCalibration complete")
    print(f"Saved intrinsics: {out_json}")
    print(f"Board image: {board_img_out}")
    print(f"Frames used: {len(used)} / {len(image_paths)}")
    print(f"RMS reprojection error: {ret:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="ChArUco calibration (OpenCV 4.8)")
    parser.add_argument("--images", default="src/images/camera_calibration", help="Folder with ChArUco images")
    parser.add_argument("--min_corners", type=int, default=6, help="Minimum ChArUco corners per frame")
    parser.add_argument("--max_dim", type=int, default=1800, help="Downscale max dimension (set 0 to disable)")
    parser.add_argument("--board_img", default="outputs/charuco_board.png", help="Output board image path")
    parser.add_argument("--out", default="outputs/camera_intrinsics.json", help="Output JSON path")
    parser.add_argument("--board_only", action="store_true", help="Only generate the board PNG+PDF and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    # Allow generating only the board (PNG+PDF) without calibration
    if args.board_only:
        board, _ = build_board()
        generate_board_image(board, Path(args.board_img))
        return

    max_dim = None if args.max_dim == 0 else args.max_dim
    calibrate(
        images_dir=Path(args.images),
        min_corners=args.min_corners,
        max_dim=max_dim,
        board_img_out=Path(args.board_img),
        out_json=Path(args.out),
    )


if __name__ == "__main__":
    main()