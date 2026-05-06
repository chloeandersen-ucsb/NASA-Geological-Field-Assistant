"""
label_tool.py — manifest-aware interactive labeling tool.

Modes:
  default          Review images that have any feature = "n/a" (must be resolved)
  --review-all     Review every image, confirming or overriding auto-hints

Within each session features are pre-filled from auto-hints.
Only n/a features require a keypress in default mode.
All features require a keypress in --review-all mode (enter = keep current value).

Controls (shown per image):
  <number>  select choice
  enter     accept pre-filled / current value
  s         skip this image (n/a features stay n/a)
  r         reject image (marks quality=rejected, skips labeling)
  q         quit and save progress
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Features where "n/a" is a valid label class (not a missing-label sentinel)
VALID_NA_FEATURES: frozenset[str] = frozenset({"phenocryst_hint"})

# Labeler confidence levels → training loss weights (mirrors rocknet_v2.CONF_LEVELS)
CONF_LEVELS: dict[int, float] = {1: 0.5, 2: 0.75, 3: 1.0}
CONF_LABELS: dict[int, str]   = {1: "uncertain", 2: "likely", 3: "certain"}


def _feat_value(entry) -> str:
    """Extract feature value from either old string format or new dict format."""
    return entry["value"] if isinstance(entry, dict) else entry


def _feat_conf(entry) -> int:
    """Extract confidence (1-3) from a feature entry; defaults to 3 for old string format."""
    return entry.get("conf", 3) if isinstance(entry, dict) else 3


# ---------------------------------------------------------------------------
# Schemas (mirrored from crop_image.py — keep in sync)
# ---------------------------------------------------------------------------

BASALT_FEATURE_SCHEMA = {
    "groundmass_texture": ["aphanitic", "medium_grained", "phaneritic", "n/a"],
    "vesicularity":       ["low", "moderate", "high", "n/a"],
    "luster":             ["dull", "moderate", "metallic", "n/a"],
    "phenocryst_hint":    ["n/a", "olivine_like", "pyroxene_like"],
}
ANORTHOSITE_FEATURE_SCHEMA = {
    "crystal_fabric":     ["equigranular", "polygonal", "n/a"],
    "brightness":         ["moderate", "high", "n/a"],
    "surface_character":  ["weathered", "vitreous", "n/a"],
    "mafic_content_hint": ["low", "moderate", "n/a"],
}
BRECCIA_FEATURE_SCHEMA = {
    "clast_angularity": ["rounded", "angular", "n/a"],
    "sorting":          ["poor", "well", "n/a"],
    "support_fabric":   ["matrix_supported", "clast_supported", "n/a"],
}
FAMILY_SCHEMAS = {
    "basalt":      BASALT_FEATURE_SCHEMA,
    "anorthosite": ANORTHOSITE_FEATURE_SCHEMA,
    "breccia":     BRECCIA_FEATURE_SCHEMA,
}


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(data: dict, path: Path) -> None:
    """Atomic write: .tmp → rename."""
    tmp = path.with_suffix(".json.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(str(tmp), str(path))


def find_image_path(entry: dict) -> Path | None:
    """Return the best available image path: processed first, raw as fallback."""
    for key in ("processed_path", "raw_path"):
        p = entry.get(key)
        if p and Path(p).exists():
            return Path(p)
    return None


# ---------------------------------------------------------------------------
# Queue building
# ---------------------------------------------------------------------------

def has_incomplete_features(entry: dict) -> bool:
    features = entry.get("features", {})
    if not features:
        return True  # unlabeled entry — no features assigned yet
    return any(
        _feat_value(v) == "n/a"
        for k, v in features.items()
        if k not in VALID_NA_FEATURES
    )


def build_queue(
    images: list[dict],
    review_all: bool,
    family_filter: str | None,
    source_filter: str | None,
) -> list[dict]:
    queue = []
    for entry in images:
        if entry.get("quality") == "rejected":
            continue
        if family_filter and entry.get("family") != family_filter:
            continue
        if source_filter and entry.get("source") != source_filter:
            continue

        if review_all:
            queue.append(entry)
        else:
            # Default: only images with at least one n/a feature that aren't reviewed
            if not entry.get("reviewed", False) and has_incomplete_features(entry):
                queue.append(entry)

    return queue


# ---------------------------------------------------------------------------
# stdin helpers
# ---------------------------------------------------------------------------

def flush_stdin() -> None:
    """Discard any characters buffered in stdin (e.g. newlines left by plt.pause)."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except (ImportError, Exception):
        # Windows or non-tty: drain with msvcrt if available, else ignore
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_image(image_path: Path, title: str, figsize: tuple) -> None:
    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
        plt.close("all")
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)
    except Exception as e:
        print(f"  [display error: {e}]")


def format_scan_hints(scan: dict) -> str:
    if not scan:
        return "no scan data"
    parts = [
        f"brightness={scan.get('brightness_mean', '?')}",
        f"texture_var={scan.get('texture_variance', '?')}",
        f"saturation={scan.get('saturation_mean', '?')}",
        f"bucket={scan.get('dominant_value_bucket', '?')}",
    ]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def prompt_feature(
    feature_name: str,
    allowed_values: list[str],
    current_value: str,
    require_input: bool,
) -> str | None:
    """
    Prompt user to pick a feature value.

    require_input=True  → must choose (used for n/a features in default mode)
    require_input=False → enter keeps current_value (used in --review-all)

    Returns chosen value, "__SKIP__", or None (quit).
    """
    current_marker = f"  (current: {current_value})" if current_value else ""
    print(f"\n  {feature_name}{current_marker}")

    for i, v in enumerate(allowed_values, 1):
        star = " *" if v == current_value else ""
        print(f"    {i}. {v}{star}")

    if require_input:
        prompt_text = f"  Choose 1–{len(allowed_values)}: "
    else:
        prompt_text = f"  Choose 1–{len(allowed_values)} or enter to keep [{current_value}]: "

    while True:
        try:
            raw = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw == "" and not require_input:
            return current_value

        if raw.lower() in {"q", "quit"}:
            return None

        if raw.lower() in {"s", "skip"}:
            return "__SKIP__"

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(allowed_values):
                return allowed_values[idx]

        if raw in allowed_values:
            return raw

        print("  Invalid — try again, 's' to skip image, 'q' to quit.")


def prompt_confidence(current_conf: int) -> int | None:
    """
    Ask labeler for a confidence rating after they've chosen a feature value.

    Returns 1-3, or None if the user wants to quit.
    Pressing enter keeps the current confidence (defaults to 3=certain for new entries).
    """
    current_label = CONF_LABELS.get(current_conf, "certain")
    prompt_text = f"  Confidence? 1=uncertain  2=likely  3=certain  [enter={current_label}]: "
    while True:
        try:
            raw = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw == "":
            return current_conf
        if raw.lower() in {"q", "quit"}:
            return None
        if raw in {"1", "2", "3"}:
            return int(raw)
        print("  Invalid — enter 1, 2, 3, or press enter to keep.")


def label_entry(
    entry: dict,
    review_all: bool,
) -> tuple[dict | None, str]:
    """
    Interactively label one manifest entry.

    Returns (updated_features, action) where action is:
      "labeled"   — all features resolved
      "skip"      — user skipped
      "reject"    — user rejected image quality
      "quit"      — user wants to stop session
    """
    family = entry.get("family")
    schema = FAMILY_SCHEMAS.get(family or "", {})
    if not schema:
        return entry.get("features", {}), "skip"

    current_features = dict(entry.get("features", {}))
    updated = dict(current_features)

    for feature_name, allowed_values in schema.items():
        raw_current   = updated.get(feature_name, "n/a")
        current_value = _feat_value(raw_current)
        current_conf  = _feat_conf(raw_current)
        is_na = current_value == "n/a" and feature_name not in VALID_NA_FEATURES

        # In default mode: only prompt for n/a features; print and accept others
        if not review_all and not is_na:
            conf_str = f"  [conf={CONF_LABELS.get(current_conf, '?')}]" if isinstance(raw_current, dict) else ""
            print(f"  {feature_name:<22} → {current_value}  [accepted]{conf_str}")
            continue

        # In review-all mode OR for n/a features: prompt for value then confidence
        require_input = is_na
        result_value = prompt_feature(feature_name, allowed_values, current_value, require_input)

        if result_value is None:
            return None, "quit"
        if result_value == "__SKIP__":
            return None, "skip"

        result_conf = prompt_confidence(current_conf)
        if result_conf is None:
            return None, "quit"

        updated[feature_name] = {"value": result_value, "conf": result_conf}

    return updated, "labeled"


# ---------------------------------------------------------------------------
# Auto-review session (--review-auto mode)
# ---------------------------------------------------------------------------

def build_auto_review_queue(
    images: list[dict],
    family_filter: str | None,
    source_filter: str | None,
    min_conf: int,
) -> list[dict]:
    """
    Return auto_labeled entries that need human review, sorted by minimum
    feature confidence ascending (most uncertain images first).

    min_conf: only include entries where at least one feature conf <= min_conf.
    Default min_conf=3 includes everything; min_conf=1 shows only conf-1 images.
    """
    rows = []
    for entry in images:
        if entry.get("quality") == "rejected":
            continue
        if entry.get("label_status") != "auto_labeled":
            continue
        if family_filter and entry.get("family") != family_filter:
            continue
        if source_filter and entry.get("source") != source_filter:
            continue
        features = entry.get("features", {})
        if not features:
            continue
        min_feat_conf = min(
            (v["conf"] if isinstance(v, dict) else 3) for v in features.values()
        )
        if min_feat_conf > min_conf:
            continue
        rows.append((min_feat_conf, entry))
    rows.sort(key=lambda x: x[0])
    return [e for _, e in rows]


def _format_vlm_predictions(entry: dict, schema: dict) -> None:
    """Print VLM predictions with confidence markers and per-feature reasoning."""
    features  = entry.get("features", {})
    analysis  = entry.get("auto_label_analysis", {})
    meta      = entry.get("auto_label_meta", {})
    vlm_family = meta.get("vlm_family", entry.get("family", "?"))
    vlm_fconf  = meta.get("vlm_family_conf", "?")
    fconf_str  = CONF_LABELS.get(vlm_fconf, str(vlm_fconf)) if isinstance(vlm_fconf, int) else str(vlm_fconf)

    min_feat_conf = min(
        (v["conf"] if isinstance(v, dict) else 3) for v in features.values()
    ) if features else 1

    print(f"  VLM: {vlm_family} (conf={vlm_fconf}/{fconf_str}) | min_feature_conf: {min_feat_conf}")
    if meta.get("family_mismatch"):
        print("  *** FAMILY MISMATCH: VLM disagrees with manifest family ***")

    print()
    print("  VLM PREDICTIONS:")
    for feat_name in schema:
        raw = features.get(feat_name, "n/a")
        val  = raw["value"] if isinstance(raw, dict) else raw
        conf = raw["conf"]  if isinstance(raw, dict) else 3
        conf_label = CONF_LABELS.get(conf, "?")
        warn = " ⚠" if conf == 1 else ""
        print(f"  {feat_name:<22} → {val:<20} [conf={conf}: {conf_label}]{warn}")
        reasoning = analysis.get(feat_name, "")
        if reasoning:
            print(f"    ↳ {reasoning}")


def label_entry_auto_review(entry: dict) -> tuple[dict | None, str]:
    """
    Like label_entry() but pre-filled from VLM predictions.

    - Enter on a feature: keep VLM value and confidence unchanged
    - Number: override value, then prompt confidence (defaults to 3=certain)

    Returns (updated_features, action) — same contract as label_entry().
    """
    family = entry.get("family")
    schema = FAMILY_SCHEMAS.get(family or "", {})
    if not schema:
        return entry.get("features", {}), "skip"

    current_features = dict(entry.get("features", {}))
    updated          = dict(current_features)

    for feature_name, allowed_values in schema.items():
        raw_current   = updated.get(feature_name, {"value": "n/a", "conf": 1})
        current_value = _feat_value(raw_current)
        current_conf  = _feat_conf(raw_current)
        conf_label    = CONF_LABELS.get(current_conf, "?")
        warn          = " ⚠" if current_conf == 1 else ""

        print(f"\n  {feature_name:<22} [VLM: {current_value}, conf={current_conf}/{conf_label}{warn}]")
        for i, v in enumerate(allowed_values, 1):
            star = " *" if v == current_value else ""
            print(f"    {i}. {v}{star}")

        prompt_text = f"  Enter to accept VLM value, or choose 1–{len(allowed_values)}: "

        while True:
            try:
                raw = input(prompt_text).strip()
            except (EOFError, KeyboardInterrupt):
                return None, "quit"

            if raw == "":
                # Accept VLM prediction as-is (keep value and confidence)
                break

            if raw.lower() in {"q", "quit"}:
                return None, "quit"

            if raw.lower() in {"s", "skip"}:
                return None, "skip"

            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(allowed_values):
                    chosen_value = allowed_values[idx]
                    # Human override → prompt for confidence (default 3=certain)
                    new_conf = prompt_confidence(3)
                    if new_conf is None:
                        return None, "quit"
                    updated[feature_name] = {"value": chosen_value, "conf": new_conf}
                    break

            if raw in allowed_values:
                idx_in_list = allowed_values.index(raw)
                new_conf = prompt_confidence(3)
                if new_conf is None:
                    return None, "quit"
                updated[feature_name] = {"value": allowed_values[idx_in_list], "conf": new_conf}
                break

            print("  Invalid — try again, 's' to skip, 'q' to quit.")

    return updated, "labeled"


def run_review_auto_session(
    manifest_path: Path,
    family_filter: str | None,
    source_filter: str | None,
    min_conf: int,
    n_limit: int | None,
    figsize: tuple,
) -> None:
    """
    Review session for auto_labeled images.

    Controls per image:
      enter  → review features one by one (enter on each = accept VLM prediction)
      a      → accept ALL VLM predictions immediately (fastest path)
      r      → reject image quality
      q      → quit session
    """
    data   = load_manifest(manifest_path)
    images = data["images"]
    id_to_idx = {e["id"]: i for i, e in enumerate(images)}

    queue = build_auto_review_queue(images, family_filter, source_filter, min_conf)
    if n_limit:
        queue = queue[:n_limit]
    total = len(queue)

    if total == 0:
        print("No auto_labeled images found matching filters.")
        print("Run auto_label.py first, then use --review-auto.")
        return

    print(f"\nManifest : {manifest_path}")
    print(f"Mode     : review-auto (VLM predictions, lowest confidence first)")
    if family_filter:
        print(f"Family   : {family_filter}")
    if source_filter:
        print(f"Source   : {source_filter}")
    print(f"Min-conf : showing images with at least one feature conf <= {min_conf}")
    print(f"Queue    : {total} images")
    print("\nControls: enter=review features  a=accept all  r=reject  q=quit\n")

    labeled_count  = 0
    accepted_count = 0
    skipped_count  = 0
    rejected_count = 0

    for pos, entry in enumerate(queue, 1):
        image_path  = find_image_path(entry)
        family      = entry.get("family", "?")
        source      = entry.get("source", "?")
        origin      = entry.get("origin", "?")
        crop_conf   = entry.get("crop_meta", {}).get("confidence", "?")
        schema      = FAMILY_SCHEMAS.get(family, {})
        prev_status = entry.get("label_status", "?")

        print("=" * 72)
        print(f"[{pos}/{total}]  {Path(entry.get('processed_path') or entry.get('raw_path', '')).name}")
        print(f"  {family} / {source} / {origin} / crop_conf={crop_conf}  [{prev_status}]")
        _format_vlm_predictions(entry, schema)

        if image_path:
            title = f"{family}/{source}  [{pos}/{total}]  auto_labeled"
            display_image(image_path, title, figsize)
        else:
            print(f"  [image not found — raw: {entry.get('raw_path', '')}]")

        print()
        flush_stdin()
        try:
            pre = input("  Press enter to review features, 'a' to accept ALL, 'r' to reject, 'q' to quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if pre == "q":
            print("\nQuitting.")
            break

        if pre == "r":
            idx = id_to_idx[entry["id"]]
            data["images"][idx]["quality"]      = "rejected"
            data["images"][idx]["reviewed"]     = True
            data["images"][idx]["reviewed_at"]  = datetime.now(timezone.utc).isoformat()
            data["images"][idx]["label_status"] = "rejected"
            save_manifest(data, manifest_path)
            print(f"  Rejected: {entry['id']}")
            rejected_count += 1
            continue

        if pre == "a":
            # Accept all VLM predictions as-is
            idx = id_to_idx[entry["id"]]
            data["images"][idx]["reviewed"]     = True
            data["images"][idx]["reviewed_at"]  = datetime.now(timezone.utc).isoformat()
            data["images"][idx]["label_status"] = "reviewed"
            data["updated"] = datetime.now(timezone.utc).isoformat()
            save_manifest(data, manifest_path)
            print(f"  Accepted all VLM predictions.")
            accepted_count += 1
            continue

        # Enter → review features one by one
        updated_features, action = label_entry_auto_review(entry)

        if action == "quit":
            print("\nQuitting.")
            break

        if action == "skip":
            print("  Skipped.")
            skipped_count += 1
            continue

        idx = id_to_idx[entry["id"]]
        data["images"][idx]["features"]     = updated_features
        data["images"][idx]["reviewed"]     = True
        data["images"][idx]["reviewed_at"]  = datetime.now(timezone.utc).isoformat()
        data["images"][idx]["label_status"] = "reviewed"
        data["updated"] = datetime.now(timezone.utc).isoformat()
        save_manifest(data, manifest_path)
        labeled_count += 1
        print("  Saved — features confirmed.")

    plt.close("all")
    print(f"\n── Session complete ──────────────────────────────────────")
    print(f"  Accepted (all) : {accepted_count}")
    print(f"  Reviewed       : {labeled_count}")
    print(f"  Skipped        : {skipped_count}")
    print(f"  Rejected       : {rejected_count}")
    remaining = total - accepted_count - labeled_count - skipped_count - rejected_count
    print(f"  Remaining      : {remaining}")


# ---------------------------------------------------------------------------
# Main session
# ---------------------------------------------------------------------------

def run_session(
    manifest_path: Path,
    review_all: bool,
    family_filter: str | None,
    source_filter: str | None,
    figsize: tuple,
) -> None:
    data   = load_manifest(manifest_path)
    images = data["images"]

    # Build id → index map for fast updates
    id_to_idx = {e["id"]: i for i, e in enumerate(images)}

    queue = build_queue(images, review_all, family_filter, source_filter)
    total = len(queue)

    if total == 0:
        if review_all:
            print("No images found matching filters.")
        else:
            print("No incomplete images found. All n/a features resolved.")
            print("Use --review-all to review confirmed labels.")
        return

    mode_label = "review-all" if review_all else "incomplete (n/a features only)"
    print(f"\nManifest : {manifest_path}")
    print(f"Mode     : {mode_label}")
    if family_filter:
        print(f"Family   : {family_filter}")
    if source_filter:
        print(f"Source   : {source_filter}")
    print(f"Queue    : {total} images")
    print("\nControls: <number>=select  enter=keep  s=skip  r=reject  q=quit\n")

    labeled_count  = 0
    skipped_count  = 0
    rejected_count = 0

    for pos, entry in enumerate(queue, 1):
        image_path = find_image_path(entry)
        family     = entry.get("family", "?")
        source     = entry.get("source", "?")
        origin     = entry.get("origin", "?")
        crop_conf  = entry.get("crop_meta", {}).get("confidence", "?")
        scan       = entry.get("scan_hints", {})
        prev_status = entry.get("label_status", "?")

        print("=" * 72)
        print(
            f"[{pos}/{total}]  {Path(entry.get('processed_path') or entry.get('raw_path', '')).name}"
        )
        print(f"  {family} / {source} / {origin} / crop_conf={crop_conf}  [{prev_status}]")
        print(f"  scan: {format_scan_hints(scan)}")

        if image_path:
            title = f"{family}/{source}  [{pos}/{total}]"
            display_image(image_path, title, figsize)
        else:
            raw = entry.get("raw_path", "")
            print(f"  [image not found — raw: {raw}]")

        # Check for reject
        print()
        flush_stdin()
        try:
            pre = input("  Press enter to label, 'r' to reject, 'q' to quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if pre == "q":
            print("\nQuitting.")
            break

        if pre == "r":
            idx = id_to_idx[entry["id"]]
            data["images"][idx]["quality"]      = "rejected"
            data["images"][idx]["reviewed"]     = True
            data["images"][idx]["reviewed_at"]  = datetime.now(timezone.utc).isoformat()
            data["images"][idx]["label_status"] = "rejected"
            save_manifest(data, manifest_path)
            print(f"  Rejected: {entry['id']}")
            rejected_count += 1
            continue

        updated_features, action = label_entry(entry, review_all)

        if action == "quit":
            print("\nQuitting.")
            break

        if action == "skip":
            print(f"  Skipped.")
            skipped_count += 1
            continue

        # action == "labeled"
        idx = id_to_idx[entry["id"]]
        data["images"][idx]["features"]     = updated_features
        data["images"][idx]["reviewed"]     = True
        data["images"][idx]["reviewed_at"]  = datetime.now(timezone.utc).isoformat()
        data["images"][idx]["label_status"] = "reviewed"
        data["updated"] = datetime.now(timezone.utc).isoformat()

        save_manifest(data, manifest_path)
        labeled_count += 1

        remaining_na = [k for k, v in updated_features.items() if _feat_value(v) == "n/a"]
        if remaining_na:
            print(f"  Saved (still n/a: {', '.join(remaining_na)})")
        else:
            print(f"  Saved — all features resolved.")

    plt.close("all")
    print(f"\n── Session complete ──────────────────────────────────────")
    print(f"  Labeled  : {labeled_count}")
    print(f"  Skipped  : {skipped_count}")
    print(f"  Rejected : {rejected_count}")
    print(f"  Remaining in queue: {total - labeled_count - skipped_count - rejected_count}")


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def print_manifest_summary(data: dict, family_filter: str | None = None, source_filter: str | None = None) -> None:
    """
    Comprehensive dataset distribution summary — mirrors auto_label.py --summary.
    Covers pipeline status, per-family per-feature value distributions,
    class-imbalance warnings, low-confidence flags, and ML readiness.
    """
    from collections import Counter

    images = data.get("images", [])
    if family_filter:
        images = [e for e in images if e.get("family") == family_filter]
    if source_filter:
        images = [e for e in images if e.get("source") == source_filter]

    total = len(images)
    if total == 0:
        print("  No images found matching filters.")
        return

    n_reviewed = sum(1 for e in images if e.get("reviewed", False))
    n_rejected = sum(1 for e in images if e.get("quality") == "rejected")
    incomplete = sum(1 for e in images if has_incomplete_features(e) and not e.get("reviewed"))
    by_status  = Counter(e.get("label_status", "unlabeled") for e in images)

    print(f"\n{'─'*70}")
    print(f"  DATASET SUMMARY")
    print(f"{'─'*70}")
    print(f"  Total images : {total:,}")
    print(f"  Reviewed     : {n_reviewed:,}  ({n_reviewed/total*100:.0f}%)")
    print(f"  Incomplete   : {incomplete:,}  (n/a features unresolved)")
    print(f"  Rejected     : {n_rejected:,}  ({n_rejected/total*100:.0f}%)")
    print()
    print(f"  Pipeline breakdown:")
    for status, count in sorted(by_status.items(), key=lambda x: -x[1]):
        bar = "█" * max(1, round(count / total * 30))
        print(f"    {status:<24} {count:>5}  ({count/total*100:4.0f}%)  {bar}")

    # Per-family breakdowns using each family's schema (including n/a options)
    for family, schema in FAMILY_SCHEMAS.items():
        fimages = [e for e in images if e.get("family") == family]
        if not fimages:
            continue

        n = len(fimages)
        labeled = [
            e for e in fimages
            if e.get("features") and e.get("label_status") in ("auto_labeled", "reviewed")
        ]
        n_labeled = len(labeled)

        sources = Counter(e.get("source", "?") for e in fimages)
        origins = Counter(e.get("origin", "?") for e in fimages)
        fstatus = Counter(e.get("label_status", "unlabeled") for e in fimages)

        print(f"\n  {'─'*66}")
        print(f"  {family.upper()}  ({n:,} total  |  {n_labeled:,} labeled)")

        src_str  = "  ".join(f"{s}={c}" for s, c in sources.most_common())
        orig_str = "  ".join(f"{o}={c}" for o, c in origins.most_common())
        stat_str = "  ".join(f"{s}={c}" for s, c in fstatus.most_common())
        print(f"  Sources  : {src_str or '—'}")
        print(f"  Origins  : {orig_str or '—'}")
        print(f"  Status   : {stat_str}")

        if n_labeled == 0:
            print(f"  (no labeled images — run auto_label.py)")
            continue

        print(f"\n  Feature value distributions  (n={n_labeled} labeled images):")

        imbalance_notes: list[str] = []
        conf1_notes: list[tuple[str, int]] = []

        for feat_name, allowed_values in schema.items():
            val_counts  = Counter()
            conf_counts = Counter()
            for e in labeled:
                raw = e.get("features", {}).get(feat_name)
                if raw is None:
                    continue
                val  = _feat_value(raw)
                conf = _feat_conf(raw)
                val_counts[val]   += 1
                conf_counts[conf] += 1

            feat_total = sum(val_counts.values())
            if feat_total == 0:
                continue

            dist_parts = [
                f"{v}={val_counts.get(v, 0)}({val_counts.get(v, 0)/feat_total*100:.0f}%)"
                for v in allowed_values
            ]
            dist_str = "  ".join(dist_parts)

            # Imbalance: skip n/a-only features (phenocryst_hint) from imbalance check
            # because n/a is a real valid class there
            real_vals = {v: c for v, c in val_counts.items() if v != "n/a" or feat_name in VALID_NA_FEATURES}
            real_total = sum(real_vals.values())
            max_pct  = max(real_vals.values()) / real_total * 100 if real_total > 0 else 0
            n_conf1  = conf_counts.get(1, 0)
            flags    = ""
            if max_pct > 80:
                flags += "  ⚠ imbalanced"
            if n_conf1 > 0:
                flags += f"  [conf1={n_conf1}]"

            print(f"    {feat_name:<22} {dist_str}{flags}")

            if max_pct > 80 and real_total > 0:
                dom = max(real_vals, key=real_vals.get)
                imbalance_notes.append(f"{feat_name}: '{dom}' dominates at {max_pct:.0f}%")
            if n_conf1 > 0:
                conf1_notes.append((feat_name, n_conf1))

        if imbalance_notes:
            print()
            print(f"  ⚠  Class imbalance (one value >80%  →  model may not learn minority classes):")
            for note in imbalance_notes:
                print(f"       {note}")

        if conf1_notes:
            total_conf1 = sum(c for _, c in conf1_notes)
            print(f"  ⚑  Low-confidence labels (conf=1) — {total_conf1} uncertain labels need review:")
            for feat, cnt in sorted(conf1_notes, key=lambda x: -x[1]):
                print(f"       {feat:<22} {cnt} images")

    # ML readiness
    MIN_SAMPLES = 50
    print(f"\n  {'─'*66}")
    print(f"  ML TRAINING READINESS CHECK")
    print(f"  {'─'*66}")

    for family, schema in FAMILY_SCHEMAS.items():
        fimages  = [e for e in images if e.get("family") == family]
        labeled  = [e for e in fimages if e.get("features") and e.get("label_status") in ("auto_labeled", "reviewed")]
        rev_imgs = [e for e in fimages if e.get("label_status") == "reviewed"]
        n_labeled, n_rev = len(labeled), len(rev_imgs)

        issues: list[str] = []
        if n_labeled < MIN_SAMPLES:
            issues.append(f"only {n_labeled} labeled  (recommend {MIN_SAMPLES}+)")

        for feat_name, allowed_values in schema.items():
            val_counts = Counter()
            for e in labeled:
                raw = e.get("features", {}).get(feat_name)
                if raw is None:
                    continue
                val = _feat_value(raw)
                val_counts[val] += 1
            real_vals  = {v: c for v, c in val_counts.items() if v != "n/a" or feat_name in VALID_NA_FEATURES}
            real_total = sum(real_vals.values())
            if real_total > 0 and max(real_vals.values()) / real_total > 0.80:
                dom = max(real_vals, key=real_vals.get)
                issues.append(f"{feat_name} skewed ('{dom}' {real_vals[dom]/real_total*100:.0f}%)")

        icon = "⚠" if issues else "✓"
        issue_str = "; ".join(issues) if issues else "looks good"
        print(f"  {icon} {family:<14} {n_labeled:>4} labeled  ({n_rev} human-reviewed)  →  {issue_str}")

    n_auto  = sum(1 for e in images if e.get("label_status") == "auto_labeled")
    n_unlbl = sum(1 for e in images if e.get("label_status", "unlabeled") in ("unlabeled", "auto_hint", "auto_label_failed"))
    print()
    print(f"  Needs human review : {n_auto:,}   →  python label_tool.py --review-auto")
    print(f"  Not yet labeled    : {n_unlbl:,}   →  python auto_label.py")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive labeling tool — reads/writes dataset_manifest.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to dataset_manifest.json",
    )
    parser.add_argument(
        "--review-all", action="store_true",
        help="Review ALL images (not just incomplete). Shows current values; enter=keep.",
    )
    parser.add_argument(
        "--family", choices=["basalt", "anorthosite", "breccia"], default=None,
        help="Filter queue to one rock family.",
    )
    parser.add_argument(
        "--source", default=None,
        help="Filter queue to one source (e.g. lpi, smithsonian, manual).",
    )
    parser.add_argument(
        "--figsize", default="9,7",
        help="Matplotlib figure size as width,height (default: 9,7).",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print manifest summary and exit (no labeling).",
    )
    parser.add_argument(
        "--review-auto", action="store_true",
        help=(
            "Review auto_labeled images (from auto_label.py). "
            "Shows VLM predictions; 'a' accepts all, enter reviews per-feature. "
            "Images sorted by lowest confidence first."
        ),
    )
    parser.add_argument(
        "--min-conf", type=int, default=3, metavar="N", choices=[1, 2, 3],
        help=(
            "With --review-auto: only show images with at least one feature conf <= N. "
            "Default 3 shows all. Use 1 to show only very uncertain images."
        ),
    )
    parser.add_argument(
        "--n", type=int, default=None, metavar="N",
        help="With --review-auto: limit session to N images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        print("Run crop_image.py --manifest first to generate it.")
        sys.exit(1)

    data = load_manifest(manifest_path)

    if args.summary:
        print_manifest_summary(data, family_filter=args.family, source_filter=args.source)
        return

    try:
        figsize = tuple(int(x.strip()) for x in args.figsize.split(","))
    except ValueError:
        figsize = (9, 7)

    if args.review_auto:
        if args.review_all:
            print("Error: --review-auto and --review-all are mutually exclusive.")
            sys.exit(1)
        run_review_auto_session(
            manifest_path=manifest_path,
            family_filter=args.family,
            source_filter=args.source,
            min_conf=args.min_conf,
            n_limit=args.n,
            figsize=figsize,
        )
        return

    run_session(
        manifest_path=manifest_path,
        review_all=args.review_all,
        family_filter=args.family,
        source_filter=args.source,
        figsize=figsize,
    )


if __name__ == "__main__":
    main()
