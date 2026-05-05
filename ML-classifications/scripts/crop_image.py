import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MANIFEST_VERSION = "1.0"
KNOWN_FAMILIES = {"basalt", "anorthosite", "breccia"}

SOURCE_WEIGHTS = {
    "lpi":            1.0,
    "astromaterials": 0.9,
    "manual":         1.0,
    "smithsonian":    0.7,
    "flickr":         1.0,
    "james_st_john":  1.0,
    "kaggle":         0.5,
    "comfyui":        0.6,
    "jsc":            1.0  
}
SOURCE_ORIGIN = {
    "lpi":            "lunar",
    "astromaterials": "lunar",
    "manual":         "lunar_earth",
    "smithsonian":    "earth",
    "flickr":         "earth",
    "james_st_john":  "earth",
    "kaggle":         "earth",
    "comfyui":        "synthetic",
    "jsc":            "lunar"
}


# ---------------------------------------------------------------------------
# ManifestManager
# ---------------------------------------------------------------------------

class ManifestManager:
    """Thread-safe manifest reader/writer with atomic saves and resume support."""

    def __init__(self, manifest_path: Path):
        self.path = manifest_path.resolve()
        self._lock = threading.Lock()
        self._data = self._load()
        self._index: dict[str, int] = {
            e["id"]: i for i, e in enumerate(self._data["images"])
        }

    def _load(self) -> dict:
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                data.setdefault("images", [])
                return data
            except json.JSONDecodeError as e:
                print(f"Warning: manifest file is corrupt ({e}), starting fresh.")
        return {
            "version":        MANIFEST_VERSION,
            "created":        datetime.now(timezone.utc).isoformat(),
            "updated":        datetime.now(timezone.utc).isoformat(),
            "source_weights": SOURCE_WEIGHTS,
            "source_origin":  SOURCE_ORIGIN,
            "images":         [],
        }

    def has(self, image_id: str) -> bool:
        with self._lock:
            return image_id in self._index

    def count(self) -> int:
        with self._lock:
            return len(self._index)

    def upsert(self, entry: dict) -> None:
        with self._lock:
            iid = entry["id"]
            if iid in self._index:
                self._data["images"][self._index[iid]] = entry
            else:
                self._index[iid] = len(self._data["images"])
                self._data["images"].append(entry)
            self._data["updated"] = datetime.now(timezone.utc).isoformat()
            self._atomic_save()

    def _atomic_save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        os.replace(str(tmp), str(self.path))


# ---------------------------------------------------------------------------
# Path inference helpers
# ---------------------------------------------------------------------------

def infer_family_source(file_path: Path) -> tuple[str | None, str | None]:
    """Walk path looking for images_raw/{family}/{source}."""
    parts = [p.lower() for p in file_path.parts]
    for i, part in enumerate(parts):
        if part == "images_raw" and i + 2 < len(parts):
            family = parts[i + 1]
            source = parts[i + 2]
            if family in KNOWN_FAMILIES:
                return family, source
    return None, None


def make_image_id(family: str | None, source: str | None, stem: str) -> str:
    return f"{family or 'unknown'}__{source or 'unknown'}__{stem}"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def letterbox_pad_to_square(image_bgr, size: int = 448):
    """Resize image to fit within size×size, pad remainder with black. Nothing is cropped."""
    h, w = image_bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def save_image(save_path: Path, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    if save_path.suffix.lower() in {".jpg", ".jpeg"}:
        pil_img.convert("RGB").save(save_path, quality=95)
    else:
        pil_img.save(save_path)


def build_output_name(name: str, suffix: str | None, ext: str) -> str:
    return f"{name}_{suffix}_resized{ext}" if suffix else f"{name}_resized{ext}"


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_one_image(
    file_path_str,
    output_folder_str,
    suffix=None,
    size=768,
    manifest_manager: ManifestManager | None = None,
):
    file_path   = Path(file_path_str)
    output_path = Path(output_folder_str)

    name      = file_path.stem
    ext       = file_path.suffix.lower()
    save_path = output_path / build_output_name(name, suffix, ext)

    family, source = infer_family_source(file_path)
    image_id = make_image_id(family, source, name)

    if manifest_manager is not None and manifest_manager.has(image_id):
        return f"[resume] Already in manifest: {file_path.name}"
    if manifest_manager is None and save_path.exists():
        return f"Skipping (already exists): {save_path.name}"

    image_bgr = cv2.imread(str(file_path))
    if image_bgr is None:
        return f"Could not read: {file_path.name}"

    final_bgr = letterbox_pad_to_square(image_bgr, size=size)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(save_path, final_bgr)

    if manifest_manager is not None:
        manifest_manager.upsert({
            "id":             image_id,
            "family":         family,
            "source":         source,
            "origin":         SOURCE_ORIGIN.get(source or "", "unknown"),
            "source_weight":  SOURCE_WEIGHTS.get(source or "", 1.0),
            "raw_path":       str(file_path.resolve()),
            "processed_path": str(save_path.resolve()),
            "features":       {},
            "label_status":   "unlabeled",
            "reviewed":       False,
            "reviewed_at":    None,
            "quality":        "ok",
            "processed_at":   datetime.now(timezone.utc).isoformat(),
        })

    return f"Saved: {save_path.name} | out=({size},{size})"


def process_images(
    input_folder,
    output_folder,
    suffix=None,
    size=768,
    workers=1,
    recursive=False,
    manifest_path=None,
):
    input_path  = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_path.resolve()}")

    output_path.mkdir(parents=True, exist_ok=True)

    glob = input_path.rglob("*") if recursive else input_path.iterdir()
    files = sorted(p for p in glob if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)

    if not files:
        print(f"No supported image files found in: {input_path.resolve()}")
        return

    manifest_manager: ManifestManager | None = None

    if manifest_path:
        mpath = Path(manifest_path)
        manifest_manager = ManifestManager(mpath)
        print(f"Manifest : {mpath}")
        print(f"Already in manifest (resume): {manifest_manager.count()}")

    print(f"Images found : {len(files)}")

    def get_output_folder(fp: Path) -> Path:
        if recursive:
            try:
                return output_path / fp.parent.relative_to(input_path)
            except ValueError:
                pass
        return output_path

    shared_kwargs = dict(suffix=suffix, size=size, manifest_manager=manifest_manager)

    if workers <= 1:
        for fp in files:
            out_dir = get_output_folder(fp)
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                print(process_one_image(str(fp), str(out_dir), **shared_kwargs))
            except Exception as e:
                print(f"Error processing {fp.name}: {e}")
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one_image, str(fp), str(get_output_folder(fp)), **shared_kwargs): fp
            for fp in files
        }
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print(f"Error in worker ({futures[future].name}): {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Letterbox-pad rock images to square and write dataset_manifest.json."
    )
    parser.add_argument("input_folder",  help="Folder containing source images")
    parser.add_argument("output_folder", help="Folder for processed images")
    parser.add_argument("--suffix",    default=None,  help="Optional text appended before '_resized'")
    parser.add_argument("--size",      type=int, default=768, help="Output size in pixels (default 768)")
    parser.add_argument("--workers",   type=int, default=1,   help="Parallel workers")
    parser.add_argument("--recursive", action="store_true",   help="Recurse into subfolders")
    parser.add_argument("--manifest",  default=None, help="Path to dataset_manifest.json (enables resume)")

    args = parser.parse_args()
    process_images(
        args.input_folder,
        args.output_folder,
        suffix=args.suffix,
        size=args.size,
        workers=args.workers,
        recursive=args.recursive,
        manifest_path=args.manifest,
    )