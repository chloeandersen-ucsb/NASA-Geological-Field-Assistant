"""
jsc.py
──────────────────────────────────────────────────────────────────────────────
Downloads ALL handheld rock photographs for basalt, anorthosite, and breccia
from the JSC Lunar Sample Curator catalog.

Pipeline per rock family:
  1. REST API /samplesbyclassification/{type}  →  list of GENERIC sample IDs
  2. HTML catalog page sampleinfo.cfm?sample={generic}  →  (photo_id, desc) pairs
  3. Filter: keep only handheld rock photos (skip thin sections, vacuum-vault shots)
  4. Download JPEGs to:
       ~/Desktop/capstone/ML-classifications/images_raw/{family}/jsc/

Resumable: skips files that already exist.
Writes _manifest.csv and _failures.csv to each output folder.

Usage:
    python scripts/jsc.py [--dry-run] [--workers N] [--list-classifications]

    --dry-run               Show what would be downloaded without saving anything
    --workers N             Parallel download threads (default: 4)
    --list-classifications  Print all valid API classification strings and exit
"""

import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.exceptions import ConnectionError, ReadTimeout

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE    = "https://curator.jsc.nasa.gov/rest/lunarapi/samples"
CATALOG_URL = "https://curator.jsc.nasa.gov/lunar/samplecatalog/sampleinfo.cfm"
JPEG_BASE   = "https://curator.jsc.nasa.gov/lunar/samplecatalog/photos/jpeg"

OUTPUT_ROOT = Path.home() / "Desktop/capstone/ML-classifications/images_raw"

# (API classification string, subtype allowlist or None for all, output folder)
#
# Crustal subtypes breakdown:
#   → anorthosite : Anorthosite, Pristine (unbrecciated highland crust)
#   → breccia     : Cataclasite, Cataclastic, Fragmental (impact-fragmented)
#   → skip        : Norite, Troctolite (visually distinct, would be noise)
#
# Breccia Anorthosite samples fall under "Breccia" classification — correctly
# kept in breccia since the dominant visual signal is breccia texture.
ROCK_CLASSES = [
    ("Basalt",  None,                                                    "basalt"),
    ("Crustal", frozenset({"Anorthosite", "Pristine"}),                  "anorthosite"),
    ("Crustal", frozenset({"Cataclasite", "Cataclastic", "Fragmental"}), "breccia"),
    ("Breccia", None,                                                    "breccia"),
]

TIMEOUT_S   = 45
RETRIES     = 3
API_SLEEP_S = 0.15   # between catalog page fetches (be polite)
IMG_SLEEP_S = 0.10   # between image downloads

# ── Description-based photo filter ───────────────────────────────────────────
#
# Goal: field-classification of handheld rocks → keep only photos that
# resemble what a geologist sees when holding a rock.
#
# KEEP:
#   "Color photograph ... ortho type"        — standardized 6-view mug shots of
#                                              the intact whole rock. Best data.
#   "Color photograph ... orientation of"    — additional orthogonal angles, same rock.
#   "Color photograph" (no special qualifier) — general rock photos, usually whole rock.
#
# SKIP — actively hurts the model:
#   "Black and white"      — domain shift vs all other color sources; grayscale
#                            augmentation (p=0.05 in training) handles this better
#                            than polluting the training set with B&W images
#   "Thin Section"         — microscope slides, completely different visual domain
#   "vacuum vault"         — rock sealed in a container, surface barely visible
#   "sawed surface"        — interior cut face, not exterior field appearance
#   "post cut"             — fragments after subdivision, same problem
#   "chips"                — same
#   "Display Case"         — glass glare, mounting hardware, bad background
#   "10017,15,74,280"      — multi-sample shots (colon or comma-separated sample
#                            numbers in description = multiple rocks in one frame)
#
# The filter is case-insensitive.

SKIP_IF_DESC_CONTAINS = [
    "black and white",
    "thin section",
    "vacuum vault",
    "sawed surface",
    "post cut",
    "chips",
    "display case",
]

def is_rock_photo(photo_id: str, description: str) -> bool:
    """
    Return True only for color photographs of the intact whole rock.
    Rejects B&W, thin sections, cut surfaces, multi-sample shots, and display cases.
    """
    desc_lower = description.lower()

    # Must be a color photograph
    if not desc_lower.startswith("color"):
        return False

    # Skip any of the known bad categories
    if any(bad in desc_lower for bad in SKIP_IF_DESC_CONTAINS):
        return False

    # Skip multi-sample photos: description references more than one sample number
    # (detected by colon separator in the Sample(s) list, e.g. "10017:10018:10019")
    if ":" in description:
        return False

    return True

# ── HTTP helpers ──────────────────────────────────────────────────────────────

session = requests.Session()
# Mimic a browser so the catalog pages render fully
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; lunar-sample-scraper/2.0)"
})

def get_json(url: str, label: str = "") -> list | dict | None:
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT_S)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 404:
                return None
            else:
                print(f"  [WARN] {label} HTTP {r.status_code} (attempt {attempt})")
        except (ReadTimeout, ConnectionError) as e:
            print(f"  [WARN] {label} {type(e).__name__} (attempt {attempt})")
            if attempt < RETRIES:
                time.sleep(1.0 * attempt)
    return None

def get_html(url: str, label: str = "") -> str | None:
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT_S)
            if r.status_code == 200:
                return r.text
            elif r.status_code == 404:
                return None
            else:
                print(f"  [WARN] {label} HTTP {r.status_code} (attempt {attempt})")
        except (ReadTimeout, ConnectionError) as e:
            print(f"  [WARN] {label} {type(e).__name__} (attempt {attempt})")
            if attempt < RETRIES:
                time.sleep(1.0 * attempt)
    return None

# ── Discovery ─────────────────────────────────────────────────────────────────

# Matches S##-##### and S##-###### (5 or 6 digit suffixes)
_PHOTO_ID_RE = re.compile(r'photo=([^"]+)"')
_DESC_RE     = re.compile(r'<p class="desc">([^<]+)</p>')

def get_all_generics(classification: str,
                     subtypes: frozenset | None = None) -> list[str]:
    """
    Return all GENERIC sample IDs for the given classification string.
    If subtypes is provided, only return samples whose SAMPLESUBTYPE is in that set.
    """
    url  = f"{API_BASE}/samplesbyclassification/{classification}"
    data = get_json(url, classification)
    if not data or not isinstance(data, list):
        return []
    generics = []
    for item in data:
        if "GENERIC" not in item:
            continue
        if subtypes is not None:
            subtype = (item.get("SAMPLESUBTYPE") or "").strip()
            if subtype not in subtypes:
                continue
        generics.append(str(item["GENERIC"]).strip())
    return generics


def get_photos_for_sample(generic: str) -> list[tuple[str, str]]:
    """
    Scrape the HTML catalog page for a sample and return a list of
    (photo_id, description) pairs for handheld rock photos only.
    """
    url  = f"{CATALOG_URL}?sample={generic}"
    html = get_html(url, f"catalog/{generic}")
    if not html:
        return []

    # Extract all photo= hrefs and their matching descriptions.
    # The HTML structure is:
    #   <a href="photoinfo.cfm?photo=S69-45214" title="...">
    #   ...
    #   <p class="title">Photo Number: S69-45214</p>
    #   <p class="desc">Black and white Processing photograph ...</p>
    #
    # We zip photo IDs with descriptions by extracting them in document order.
    photo_ids    = _PHOTO_ID_RE.findall(html)
    descriptions = _DESC_RE.findall(html)

    # photo_ids may have duplicates (the ID appears in href AND title attr).
    # The page repeats each ID exactly 4 times; descriptions appear once each.
    # Deduplicate photo_ids while preserving order, then zip with descriptions.
    seen = set()
    unique_ids = []
    for pid in photo_ids:
        if pid not in seen:
            seen.add(pid)
            unique_ids.append(pid)

    results = []
    for pid, desc in zip(unique_ids, descriptions):
        # Skip thin-section IDs (contain comma) regardless of description
        if "," in pid:
            continue
        if is_rock_photo(pid, desc):
            results.append((pid, desc))

    return results

# ── Download worker ───────────────────────────────────────────────────────────

def download_image(url: str, out_path: Path) -> tuple[bool, str]:
    if out_path.exists():
        return True, "already_exists"
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT_S)
            if r.status_code == 200:
                if len(r.content) < 2048:
                    return False, f"too_small_{len(r.content)}b"
                out_path.write_bytes(r.content)
                return True, ""
            elif r.status_code == 404:
                return False, "404"
            else:
                if attempt == RETRIES:
                    return False, f"HTTP_{r.status_code}"
        except (ReadTimeout, ConnectionError) as e:
            if attempt == RETRIES:
                return False, type(e).__name__
            time.sleep(1.0 * attempt)
    return False, "max_retries"

def download_one(args):
    pid, generic, out_dir = args
    url      = f"{JPEG_BASE}/{pid}.jpg"
    out_path = out_dir / f"{pid}_{generic}.jpg"
    time.sleep(IMG_SLEEP_S)
    ok, err = download_image(url, out_path)
    return pid, generic, ok, err

# ── Helpers ───────────────────────────────────────────────────────────────────

def list_classifications():
    """Print all valid classification strings from the API."""
    url  = f"{API_BASE}/listsampleclassification"
    data = get_json(url, "listsampleclassification")
    if not data:
        print("Could not fetch classification list.")
        return
    print("\nValid API classification strings:")
    seen = set()
    for item in data:
        t = item.get("SAMPLETYPE", "")
        if t and t not in seen:
            print(f"  {t}")
            seen.add(t)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover photos but do not download")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download threads (default: 4)")
    parser.add_argument("--list-classifications", action="store_true",
                        help="Print all valid API classification strings and exit")
    args = parser.parse_args()

    if args.list_classifications:
        list_classifications()
        return

    print("=" * 68)
    print("JSC Lunar Sample Image Scraper — basalt / anorthosite / breccia")
    print("=" * 68)
    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    for api_class, subtypes, folder_name in ROCK_CLASSES:
        out_dir = OUTPUT_ROOT / folder_name / "jsc"

        print(f"\n{'─'*60}")
        subtype_label = f" ({', '.join(sorted(subtypes))})" if subtypes else ""
        print(f"  Class : {api_class}{subtype_label}")
        print(f"  Output: {out_dir}")
        print(f"{'─'*60}")

        # Step 1: get all GENERIC IDs via REST API
        print("  Fetching sample list ...", end=" ", flush=True)
        generics = get_all_generics(api_class, subtypes)
        print(f"{len(generics)} samples")

        if not generics:
            print("  [SKIP] 0 samples — try --list-classifications to check the API string")
            continue

        # Step 2: scrape catalog HTML per sample to get photo IDs + descriptions
        print("  Scraping photo IDs from catalog pages ...")
        all_photos: list[tuple[str, str]] = []  # [(photo_id, generic), ...]
        samples_with_photos = 0

        for idx, generic in enumerate(generics, 1):
            photos = get_photos_for_sample(generic)
            if photos:
                samples_with_photos += 1
                for pid, _desc in photos:
                    all_photos.append((pid, generic))
            time.sleep(API_SLEEP_S)

            if idx % 50 == 0:
                print(f"    {idx}/{len(generics)} samples scanned — "
                      f"{len(all_photos)} photos so far "
                      f"({samples_with_photos} samples have photos)")

        # Deduplicate across samples (rare, but a photo can be listed under multiple generics)
        seen = set()
        unique_photos = []
        for pid, generic in all_photos:
            if pid not in seen:
                seen.add(pid)
                unique_photos.append((pid, generic))
        dupes = len(all_photos) - len(unique_photos)

        print(f"\n  ✓ {len(unique_photos)} unique rock photos "
              f"across {samples_with_photos}/{len(generics)} samples with photos"
              + (f" ({dupes} cross-sample dupes removed)" if dupes else ""))

        if args.dry_run:
            print(f"\n  Preview (first 10):")
            for pid, g in unique_photos[:10]:
                print(f"    {pid}  (sample {g})")
            if len(unique_photos) > 10:
                print(f"    ... and {len(unique_photos) - 10} more")
            continue

        if not unique_photos:
            print("  [SKIP] No photos found")
            continue

        # Step 3: create output directory
        out_dir.mkdir(parents=True, exist_ok=True)

        # Step 4: download in parallel
        print(f"\n  Downloading {len(unique_photos)} images ({args.workers} workers) ...")
        failures  = []
        downloaded = 0
        skipped   = 0
        work = [(pid, g, out_dir) for pid, g in unique_photos]

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download_one, w): w for w in work}
            for i, future in enumerate(as_completed(futures), 1):
                pid, generic, ok, err = future.result()
                if ok:
                    if err == "already_exists":
                        skipped += 1
                    else:
                        downloaded += 1
                else:
                    failures.append({"Photo_ID": pid, "Generic": generic, "Error": err})

                if i % 100 == 0 or i == len(unique_photos):
                    print(f"    {i}/{len(unique_photos)} — "
                          f"new: {downloaded}, skipped: {skipped}, failed: {len(failures)}")

        # Step 5: write logs
        if failures:
            fail_path = out_dir / "_failures.csv"
            with fail_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["Photo_ID", "Generic", "Error"])
                w.writeheader()
                w.writerows(failures)
            print(f"\n  ⚠  {len(failures)} failures → {fail_path}")

        manifest_path = out_dir / "_manifest.csv"
        with manifest_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Photo_ID", "Generic", "Family",
                                               "URL", "Local_Path"])
            w.writeheader()
            for pid, g in unique_photos:
                w.writerow({
                    "Photo_ID"  : pid,
                    "Generic"   : g,
                    "Family"    : folder_name,
                    "URL"       : f"{JPEG_BASE}/{pid}.jpg",
                    "Local_Path": str(out_dir / f"{pid}_{g}.jpg"),
                })
        print(f"  ✓  Manifest → {manifest_path}")
        print(f"  ✓  {folder_name}: {downloaded} downloaded, {skipped} already existed")

    print("\n" + "=" * 68)
    print("Done.")
    print("=" * 68)


if __name__ == "__main__":
    main()
    