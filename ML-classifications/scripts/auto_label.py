"""
auto_label.py — VLM-based auto-labeling for dataset_manifest.json

Uses Claude Vision to classify geological features for each rock image,
writing predictions back to the manifest with confidence scores (1–3).

Label status flow:
  unlabeled / auto_hint  →  auto_labeled  →  (human review via label_tool --review-auto)
                                          →  reviewed  (if --auto-accept-threshold met)
  (parse failure)        →  auto_label_failed

Usage:
  python auto_label.py --manifest path/to/dataset_manifest.json [options]
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from PIL import Image


# ── .env loader ───────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    """
    Load KEY=VALUE pairs from a .env file into os.environ (without overwriting
    existing env vars). Searches the script's directory, then its parent.
    """
    script_dir = Path(__file__).parent
    candidates = [script_dir / ".env", script_dir.parent / ".env", script_dir.parent.parent / ".env"]
    for env_path in candidates:
        if env_path.exists():
            with env_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key   = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            return


_load_dotenv()


# ── Model choices ─────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "claude-haiku-4-5-20251001"
DEFAULT_WORKERS = 2          # conservative default; raise with --workers if quota allows
MAX_RETRIES     = 5          # max attempts per image on rate-limit / transient errors
RETRY_BASE_SEC  = 4.0        # first retry delay in seconds (doubles each attempt)

# ── System prompt (cached — stays constant across every request) ──────────────
# Must exceed 1024 tokens to be cache-eligible on the Anthropic API.
# The schema definitions + geology notes provide visual grounding that helps
# the model identify features from photographs.

SYSTEM_PROMPT = """\
You are a geological feature classifier for rock sample photographs. You analyze
images of rock specimens and classify their visual features according to a strict
schema for three rock families: basalt, anorthosite, and breccia.

CRITICAL INSTRUCTIONS — READ BEFORE CLASSIFYING:
1. The rock family (basalt, anorthosite, or breccia) is ALREADY KNOWN and confirmed
   from the image source. Do NOT attempt to reclassify the family. Your only job is
   to classify the visual features WITHIN the given family's schema.
2. You must look carefully at the actual photograph and describe what you observe
   before assigning any label. Do not default to the first option in any list. Each
   label must be justified by something you can visually confirm in the image.
   For every feature ask: "What specific visual evidence supports this choice?"

══════════════════════════════════════════════════════════════
BASALT — DETAILED VISUAL CRITERIA
══════════════════════════════════════════════════════════════

── groundmass_texture ───────────────────────────────────────
This is the most important feature to get right. Look closely at the rock surface.

  aphanitic:
    WHAT YOU SEE: The rock surface looks uniformly dark, smooth, or dusty with no
    visible individual mineral grains. It resembles a fine-grained ceramic or dense
    black stone. No sparkle, no flecks, no individual crystal faces distinguishable.
    The texture is like dark velvet or fine sandpaper — uniform throughout.
    ASSIGN aphanitic ONLY if you genuinely cannot resolve individual mineral grains.

  medium_grained:
    WHAT YOU SEE: The rock surface has a speckled or salt-and-pepper appearance.
    You can see individual mineral grains or flecks — small but discrete spots of
    varying shade (light pyroxene or feldspar flecks in a darker matrix, or vice versa).
    Crystals are roughly 0.5–2 mm. The surface does not look uniformly smooth —
    there is a granular texture visible. Some crystal faces may catch light slightly.
    ASSIGN medium_grained if you can see discrete grains/flecks even if small.

  phaneritic:
    WHAT YOU SEE: Clearly distinguishable interlocking crystals larger than ~2 mm.
    You can count and identify individual crystal grains. The rock looks coarsely
    crystalline — like a dark granite or gabbro. Individual mineral species may be
    distinguishable by colour and shape. Crystal faces are obvious.
    ASSIGN phaneritic only if the crystalline texture is unmistakably coarse.

  DEFAULT TRAP TO AVOID: Do not assign aphanitic just because the image is dark,
  small, or the rock surface is partially obscured. If you cannot determine the
  texture (blurry image, rock in a bag, only one face visible), assign your best
  guess with conf=1. Genuinely medium_grained and phaneritic samples EXIST in
  this dataset — you will encounter them.

── vesicularity ─────────────────────────────────────────────
Look for circular or irregular holes/pits/voids in the rock surface.

  low:
    WHAT YOU SEE: The rock surface is mostly solid with few or no visible holes.
    Less than ~10% of the visible surface shows voids. The rock looks dense.

  moderate:
    WHAT YOU SEE: Scattered holes/pits visible but the rock is still mostly solid.
    Roughly 10–40% of the surface area shows voids. You notice the bubbles but
    they do not dominate the appearance.

  high:
    WHAT YOU SEE: The rock is visibly bubbly, spongy, or porous. Voids are abundant
    and obvious — they cover more than ~40% of the visible surface, or the rock
    has a clearly frothy/pumiceous appearance. Vesicles dominate the texture.

── luster ───────────────────────────────────────────────────
Look at how light reflects off the rock surface.

  dull:
    WHAT YOU SEE: The surface absorbs light and appears matte with no sheen or
    reflective highlights. Like unglazed pottery or a dusty road.

  moderate:
    WHAT YOU SEE: Some areas catch light or show a subdued sheen. Not mirror-like
    but noticeably more reflective than dull. Small bright specks from crystal faces.

  metallic:
    WHAT YOU SEE: The surface has a clearly reflective, glossy, or glassy sheen —
    like dark glass, polished metal, or vitreous obsidian. Bright specular highlights
    visible. This is uncommon but real — assign it when clearly present.

── phenocryst_hint ──────────────────────────────────────────
Look for larger, isolated crystals embedded in a finer groundmass.

  n/a:
    WHAT YOU SEE: No isolated crystals stand out from the groundmass. The texture
    is homogeneous — no grains appear noticeably larger than the surrounding matrix.

  olivine_like:
    WHAT YOU SEE: Rounded or sub-rounded grains with a greenish, yellowish-green,
    or pale green tint, clearly larger than the surrounding matrix. Often glassy.

  pyroxene_like:
    WHAT YOU SEE: Dark (black or very dark brown), blocky or tabular crystals that
    stand out against the groundmass. Tend to be angular with well-defined faces.

══════════════════════════════════════════════════════════════
ANORTHOSITE — DETAILED VISUAL CRITERIA
══════════════════════════════════════════════════════════════

── crystal_fabric ───────────────────────────────────────────

  equigranular:
    WHAT YOU SEE: All mineral grains appear roughly the same size throughout.
    The texture is uniform — no grains are noticeably larger or smaller than others.
    Like a mosaic of similar-sized white/grey tiles.

  polygonal:
    WHAT YOU SEE: Crystal boundaries form angular, straight-edged, or geometric
    interlocking shapes — like a tiled floor with angular grout lines. The grain
    boundaries look like fitted polygons rather than rounded or irregular shapes.

── brightness ───────────────────────────────────────────────

  moderate:
    WHAT YOU SEE: The rock is light-coloured (pale grey or off-white) but has
    noticeable dark patches, grey areas, or darker mineral grains mixed throughout.
    It does not look strikingly white.

  high:
    WHAT YOU SEE: The rock is strikingly white or very pale. Minimal dark minerals.
    It stands out as bright even compared to typical light-coloured rocks.

── surface_character ────────────────────────────────────────

  weathered:
    WHAT YOU SEE: The surface looks rough, corroded, pitted, or chalky. There may
    be brown or orange staining, crumbly edges, or a powdery coating. Fresh crystal
    faces are mostly absent.

  vitreous:
    WHAT YOU SEE: The surface shows fresh, glassy, or clean crystal faces with a
    slight sheen. Grain boundaries are crisp. The rock looks unaltered and clean.

── mafic_content_hint ───────────────────────────────────────

  low:
    WHAT YOU SEE: The rock is overwhelmingly white/light with very few or no dark
    mineral grains. If any dark spots exist they are sparse and small.

  moderate:
    WHAT YOU SEE: Noticeable dark (black, dark grey, or dark green) mineral grains
    are scattered throughout. They are obvious enough that you'd describe the rock
    as "white with dark spots" rather than simply "white."

══════════════════════════════════════════════════════════════
BRECCIA — DETAILED VISUAL CRITERIA
══════════════════════════════════════════════════════════════

── clast_angularity ─────────────────────────────────────────

  rounded:
    WHAT YOU SEE: Rock fragments have smooth, curved, or sub-rounded edges and
    corners. They look worn or abraded — like river pebbles.

  angular:
    WHAT YOU SEE: Rock fragments have sharp, jagged edges and corners. They look
    freshly broken — like crushed rock or shattered glass.

── sorting ──────────────────────────────────────────────────

  poor:
    WHAT YOU SEE: A chaotic mix of clast sizes — tiny fragments next to large ones,
    with no size preference. The rock looks jumbled and heterogeneous.

  well:
    WHAT YOU SEE: The clasts are mostly similar in size. The rock looks more
    uniform — like a gravel of consistent grade.

── support_fabric ───────────────────────────────────────────

  matrix_supported:
    WHAT YOU SEE: Large clasts appear to float in a fine-grained background
    material (matrix). The matrix fills the space between clasts; clasts do not
    directly touch each other.

  clast_supported:
    WHAT YOU SEE: Large clasts visibly touch or press against each other. There
    is little or no matrix between them — the clasts form the structural framework.

══════════════════════════════════════════════════════════════
CONFIDENCE SCALE
══════════════════════════════════════════════════════════════

3 = certain   — You can clearly see the defining visual feature described above.
                No ambiguity. You would bet on this classification.
2 = likely    — The visual evidence suggests this class but is not fully definitive.
                There is some ambiguity (e.g., borderline texture, partial occlusion).
1 = uncertain — The image is too dark, blurry, small, or occluded to be confident.
                You are guessing more than observing. Use this honestly.

IMPORTANT: conf=1 is not a failure — it correctly flags images for human review.
Do not inflate confidence to avoid uncertainty. Honest conf=1 labels are valuable.

══════════════════════════════════════════════════════════════
LUNAR vs. TERRESTRIAL CONTEXT
══════════════════════════════════════════════════════════════

When the user message identifies the sample origin, use it to calibrate your
expectations before classifying features.

Lunar samples (JSC, LPI, Astromaterials, Manual — lunar):
- No water or biological weathering — absent orange/brown iron staining
- Space weathering creates a thin amorphous glassy rind that may make the surface
  appear slightly darker or more matte than the fresh interior rock
- Anorthosite: expect vitreous surface_character and high brightness (pristine
  plagioclase, no hydration alteration)
- Basalt: commonly aphanitic to fine-grained; ilmenite-rich varieties are very dark
  with a metallic sheen; vesicularity is common in mare basalt flows; olivine and
  pyroxene phenocrysts occur but are not universal

Terrestrial samples (James St. John / Flickr, Smithsonian, Kaggle):
- Weathering rinds, iron-oxide staining, clay coatings, or patina may be present
- surface_character tends toward "weathered" more often than for lunar samples
- Luster may appear dull due to a surface coating even on otherwise fresh specimens
- Amygdaloidal basalt (vesicles filled with secondary minerals) can look different
  from vesicular basalt — vesicle outlines still indicate moderate to high vesicularity

══════════════════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════════════════

Return ONLY valid JSON — no markdown fences, no prose outside the JSON:

{
  "family": "<basalt|anorthosite|breccia>",
  "family_conf": <1|2|3>,
  "analysis": {
    "<feature_name>": "<one sentence describing what you observe in the image that leads to your classification>",
    ...
  },
  "features": {
    "<feature_name>": {"value": "<label_from_schema>", "conf": <1|2|3>},
    ...
  }
}

Rules:
- The "analysis" field MUST be completed before "features". Write what you observe,
  then derive the label. This prevents defaulting.
- Include analysis and features ONLY for the identified family's schema.
- Use ONLY the exact label strings listed above — no variations.
- The "family" field should match the family hint given by the user unless you are
  certain the specimen is a different rock type.
- Do not wrap in markdown code blocks.
"""


# ── Feature schemas (mirrors rocknet_v2.py / label_tool.py) ──────────────────

FEATURE_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "basalt": {
        "groundmass_texture": ["aphanitic", "medium_grained", "phaneritic"],
        "vesicularity":       ["low", "moderate", "high"],
        "luster":             ["dull", "moderate", "metallic"],
        "phenocryst_hint":    ["n/a", "olivine_like", "pyroxene_like"],
    },
    "anorthosite": {
        "crystal_fabric":     ["equigranular", "polygonal"],
        "brightness":         ["moderate", "high"],
        "surface_character":  ["weathered", "vitreous"],
        "mafic_content_hint": ["low", "moderate"],
    },
    "breccia": {
        "clast_angularity": ["rounded", "angular"],
        "sorting":          ["poor", "well"],
        "support_fabric":   ["matrix_supported", "clast_supported"],
    },
}


# ── Geological keyword hints (extracted from sample name / file name) ─────────
# Maps lowercase keyword → plain-English hint for the VLM user message.
# Hints prime the model toward likely feature values without overriding its
# visual observation — they are labelled as hints, not instructions.

_GEO_KEYWORD_HINTS: list[tuple[str, str]] = [
    # vesicularity / texture terms
    ("amygdaloidal",  "Amygdaloidal texture: vesicles filled with secondary minerals — expect moderate to high vesicularity"),
    ("vesicular",     "Described as vesicular — expect moderate to high vesicularity"),
    ("scoriaceous",   "Scoriaceous / frothy texture — expect high vesicularity"),
    ("pumiceous",     "Pumice-like texture — expect high vesicularity"),
    # luster / mineralogy
    ("ilmenite",      "Ilmenite is a dark metallic iron-titanium oxide — expect dark groundmass, possibly metallic luster"),
    ("high ti",       "High-Ti basalt — characteristically dark, dense, aphanitic to fine-grained groundmass"),
    ("high-ti",       "High-Ti basalt — characteristically dark, dense, aphanitic to fine-grained groundmass"),
    ("low ti",        "Low-Ti basalt — slightly lighter colour than high-Ti; can be medium-grained"),
    ("low-ti",        "Low-Ti basalt — slightly lighter colour than high-Ti; can be medium-grained"),
    ("glassy",        "Glassy/vitreous texture noted — expect metallic or moderate luster"),
    ("vitrophyre",    "Vitrophyre (glassy groundmass with phenocrysts) — expect metallic luster"),
    ("kreep",         "KREEP basalt (K, REE, P-rich lunar type) — typically aphanitic, dark"),
    # phenocrysts
    ("olivine",       "Olivine noted — look for rounded greenish phenocrysts (olivine_like)"),
    ("pyroxene",      "Pyroxene noted — look for dark blocky crystals (pyroxene_like phenocrysts)"),
    ("pigeonite",     "Pigeonite (low-Ca pyroxene) noted — look for dark blocky crystals (pyroxene_like phenocrysts)"),
    ("augite",        "Augite (clinopyroxene) noted — look for dark blocky crystals (pyroxene_like)"),
    ("porphyritic",   "Porphyritic texture: phenocrysts likely present — look carefully for larger isolated crystals"),
    # anorthosite / feldspar
    ("plagioclase",   "Plagioclase-rich — expect high brightness for anorthosite"),
    ("anorthite",     "Anorthite (Ca-rich plagioclase) noted — expect high brightness"),
    ("ferroan",       "Ferroan anorthosite — pristine lunar highland type; expect high brightness and vitreous surface"),
    # weathering
    ("weathered",     "Described as weathered — surface_character likely 'weathered'"),
    ("altered",       "Hydrothermally or chemically altered — surface_character likely 'weathered'"),
    # breccia
    ("regolith",      "Regolith breccia — fine-grained matrix supporting variable clast sizes (likely matrix_supported, poor sorting)"),
    ("impact",        "Impact breccia — angular clasts expected from shock fragmentation"),
    ("polymict",      "Polymict breccia — multiple clast lithologies, typically poor sorting"),
]


def _extract_geo_hints(sample_name: str) -> list[str]:
    """Return geological hint strings from known keywords in the sample name."""
    lower = sample_name.lower()
    seen: set[str] = set()
    hints: list[str] = []
    for keyword, hint in _GEO_KEYWORD_HINTS:
        if keyword in lower and hint not in seen:
            seen.add(hint)
            hints.append(hint)
    return hints


def _build_origin_context(origin: str | None, source: str | None) -> str | None:
    """Return a one-sentence geological framing based on origin."""
    if origin == "lunar":
        return (
            "Lunar sample: no water weathering or biological alteration; "
            "may have a thin space-weathering rind (slightly darker surface); "
            "expect pristine crystal faces."
        )
    if origin in ("earth", "terrestrial") or source in ("james_st_john", "smithsonian", "kaggle"):
        return (
            "Terrestrial sample: surface weathering, iron staining, "
            "or hydrothermal alteration possible."
        )
    return None


def _build_user_message(entry: dict) -> str:
    """
    Build an enriched per-image user prompt that includes sample metadata and
    geological hints extracted from the filename / sample name.
    """
    family  = entry.get("family", "unknown")
    origin  = entry.get("origin")
    source  = entry.get("source")

    # Extract sample name from the id: everything after the second "__"
    entry_id    = entry.get("id", "")
    parts       = entry_id.split("__", 2)
    sample_name = parts[2] if len(parts) >= 3 else ""

    lines: list[str] = [
        f"Family: {family} (confirmed — do not change this)",
    ]

    if origin:
        lines.append(f"Origin: {origin}")
    if source:
        lines.append(f"Source: {source}")
    if sample_name:
        lines.append(f'Sample name: "{sample_name}"')

    origin_ctx = _build_origin_context(origin, source)
    if origin_ctx:
        lines.append(f"Origin context: {origin_ctx}")

    geo_hints = _extract_geo_hints(sample_name)
    if geo_hints:
        lines.append("Geological hints (from sample name — use to calibrate, not override visual observation):")
        for hint in geo_hints:
            lines.append(f"  • {hint}")

    scan = entry.get("scan_hints")
    if scan:
        lines.append(
            f"Image metrics: brightness={scan.get('brightness_mean', '?')}, "
            f"texture_variance={scan.get('texture_variance', '?')}, "
            f"saturation={scan.get('saturation_mean', '?')}"
        )

    lines.append("")
    lines.append(
        f"Classify only the {family} features visible in this image and return JSON."
    )

    return "\n".join(lines)


# ── Manifest I/O ─────────────────────────────────────────────────────────────

def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(data: dict, path: Path) -> None:
    """Atomic write via .tmp → rename."""
    tmp = path.with_suffix(".json.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(str(tmp), str(path))


# ── Image encoding ────────────────────────────────────────────────────────────

def encode_image_base64(image_path: Path) -> str:
    """Open image, convert to RGB JPEG at quality=85, return base64 string."""
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG", quality=85)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_vlm_response(raw_text: str, expected_family: str) -> dict | None:
    """
    Parse and validate the VLM JSON response.

    Returns dict with: family, family_conf, family_mismatch, features
    Returns None on parse failure or schema violation.
    """
    text = raw_text.strip()
    # Strip markdown fences if the model added them despite instructions
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:] if not lines[-1].strip().startswith("```") else lines[1:-1]
        text  = "\n".join(inner).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    # Family is always the manifest value — ignore whatever the model returned
    family = expected_family
    if family not in FEATURE_SCHEMAS:
        return None

    family_conf = data.get("family_conf")
    if family_conf not in (1, 2, 3):
        family_conf = 2  # default to "likely" if model omitted or returned invalid

    # Features must match the confirmed family's schema
    schema       = FEATURE_SCHEMAS[family]
    raw_features = data.get("features", {})
    if not isinstance(raw_features, dict):
        return None

    features: dict[str, dict] = {}
    for feat_name, allowed_values in schema.items():
        feat_entry = raw_features.get(feat_name)
        if feat_entry is None:
            return None  # missing required feature
        if not isinstance(feat_entry, dict):
            return None
        value = feat_entry.get("value")
        conf  = feat_entry.get("conf")
        if value not in allowed_values:
            return None
        if conf not in (1, 2, 3):
            return None
        features[feat_name] = {"value": value, "conf": conf}

    # Extract optional analysis field (reasoning per feature)
    analysis = data.get("analysis", {})
    if not isinstance(analysis, dict):
        analysis = {}

    return {
        "family":      family,
        "family_conf": family_conf,
        "features":    features,
        "analysis":    analysis,
    }


# ── Single-image VLM call ─────────────────────────────────────────────────────

def vlm_label_one(
    client: anthropic.Anthropic,
    entry: dict,
    image_path: Path,
    model_id: str,
) -> dict:
    """
    Call Claude Vision to label one image.

    Returns a result dict:
      success=True  → features, family_vlm, family_conf, family_mismatch, usage
      success=False → error, usage
    """
    family      = entry.get("family", "unknown")
    b64         = encode_image_base64(image_path)
    user_text   = _build_user_message(entry)

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": user_text,
                            },
                        ],
                    }
                ],
            )
            break  # success — exit retry loop

        except anthropic.RateLimitError as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                return {"success": False, "error": f"rate_limit_exhausted: {exc}", "usage": {}}
            delay = RETRY_BASE_SEC * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)

        except anthropic.APIError as exc:
            return {"success": False, "error": str(exc), "usage": {}}
    else:
        # Should not reach here, but guard anyway
        return {"success": False, "error": str(last_exc), "usage": {}}

    raw_text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens":          response.usage.input_tokens,
        "output_tokens":         response.usage.output_tokens,
        "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        "cache_read_tokens":     getattr(response.usage, "cache_read_input_tokens", 0),
    }

    parsed = parse_vlm_response(raw_text, family)
    if parsed is None:
        return {
            "success": False,
            "error":   f"parse_error: {raw_text[:200]}",
            "usage":   usage,
        }

    return {
        "success":     True,
        "features":    parsed["features"],
        "analysis":    parsed["analysis"],
        "family":      parsed["family"],
        "family_conf": parsed["family_conf"],
        "usage":       usage,
    }


# ── Queue builders ────────────────────────────────────────────────────────────

def build_labeling_queue(
    images: list[dict],
    family_filter: str | None,
    source_filter: str | None,
    force_relabel: bool,
) -> list[dict]:
    """Return entries that still need VLM labeling."""
    always_skip = {"reviewed", "rejected"}
    skip_statuses = always_skip | (set() if force_relabel else {"auto_labeled"})

    queue = []
    for entry in images:
        if entry.get("quality") == "rejected":
            continue
        if entry.get("label_status", "unlabeled") in skip_statuses:
            continue
        if family_filter and entry.get("family") != family_filter:
            continue
        if source_filter and entry.get("source") != source_filter:
            continue
        queue.append(entry)
    return queue


def build_review_queue(
    images: list[dict],
    family_filter: str | None,
    source_filter: str | None = None,
) -> list[dict]:
    """Return auto_labeled entries sorted by minimum feature confidence (ascending)."""
    rows = []
    for entry in images:
        if entry.get("label_status") != "auto_labeled":
            continue
        if family_filter and entry.get("family") != family_filter:
            continue
        if source_filter and entry.get("source") != source_filter:
            continue
        features = entry.get("features", {})
        min_conf = (
            min((v["conf"] if isinstance(v, dict) else 3) for v in features.values())
            if features else 1
        )
        rows.append((min_conf, entry))
    rows.sort(key=lambda x: x[0])
    return [e for _, e in rows]


# ── Batch labeling ────────────────────────────────────────────────────────────

def run_batch_labeling(
    manifest_path: Path,
    model_id: str,
    workers: int,
    checkpoint_every: int,
    dry_run: bool,
    family_filter: str | None,
    source_filter: str | None,
    auto_accept_threshold: int | None,
    force_relabel: bool,
) -> None:
    data      = load_manifest(manifest_path)
    images    = data["images"]
    id_to_idx = {e["id"]: i for i, e in enumerate(images)}

    queue = build_labeling_queue(images, family_filter, source_filter, force_relabel)
    total = len(queue)

    if total == 0:
        print("No images to label. All entries are already auto_labeled, reviewed, or rejected.")
        print("Use --force-relabel to re-label already auto_labeled entries.")
        return

    mode_str = "[DRY RUN] " if dry_run else ""
    print(f"\n{mode_str}Auto-labeling {total} images with {model_id}")
    if family_filter:
        print(f"Family filter       : {family_filter}")
    if source_filter:
        print(f"Source filter       : {source_filter}")
    if auto_accept_threshold:
        print(f"Auto-accept threshold: all confs >= {auto_accept_threshold}")
    print(f"Workers             : {workers}  |  checkpoint every {checkpoint_every}")
    print()

    client = anthropic.Anthropic(
        default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    lock             = threading.Lock()
    completed        = 0
    success_count    = 0
    fail_count       = 0
    auto_accepted_ct = 0
    total_cache_read = 0
    total_input      = 0

    def process_entry(entry: dict) -> tuple[str, dict]:
        """Worker: returns (entry_id, result_dict)."""
        image_path = None
        for key in ("processed_path", "raw_path"):
            p = entry.get(key)
            if p and Path(p).exists():
                image_path = Path(p)
                break
        if image_path is None:
            return entry["id"], {"success": False, "error": "image_not_found", "usage": {}}
        return entry["id"], vlm_label_one(client, entry, image_path, model_id)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_entry, e): e for e in queue}

        for future in as_completed(futures):
            entry_id, result = future.result()
            idx = id_to_idx[entry_id]

            with lock:
                completed += 1
                now_str = datetime.now(timezone.utc).isoformat()

                if result["success"]:
                    success_count += 1
                    usage = result.get("usage", {})
                    total_cache_read += usage.get("cache_read_tokens", 0)
                    total_input      += usage.get("input_tokens", 0)

                    features = result["features"]
                    analysis = result.get("analysis", {})

                    auto_accepted = bool(
                        auto_accept_threshold
                        and all(f["conf"] >= auto_accept_threshold for f in features.values())
                    )
                    if auto_accepted:
                        auto_accepted_ct += 1

                    final_status = "reviewed" if auto_accepted else "auto_labeled"
                    auto_label_meta = {
                        "model":             model_id,
                        "labeled_at":        now_str,
                        "family_conf":       result["family_conf"],
                        "cache_read_tokens": usage.get("cache_read_tokens", 0),
                        "auto_accepted":     auto_accepted,
                    }

                    if not dry_run:
                        data["images"][idx]["features"]        = features
                        data["images"][idx]["label_status"]    = final_status
                        data["images"][idx]["auto_label_meta"] = auto_label_meta
                        if analysis:
                            data["images"][idx]["auto_label_analysis"] = analysis
                        if auto_accepted:
                            data["images"][idx]["reviewed"]    = True
                            data["images"][idx]["reviewed_at"] = now_str

                    status_str = "auto_accepted→reviewed" if auto_accepted else "auto_labeled"
                    cr  = usage.get("cache_read_tokens", 0)
                    cc  = usage.get("cache_creation_tokens", 0)
                    inp = usage.get("input_tokens", 0)
                    cache_str = f"cache_create={cc}" if cc else f"cache_read={cr}"

                    print(
                        f"  [{completed:>{len(str(total))}}/{total}] ok  "
                        f"{entry_id[:45]}  → {status_str}  "
                        f"(in={inp} {cache_str})"
                    )
                    for feat_name, feat_val in features.items():
                        val        = feat_val["value"]
                        conf       = feat_val["conf"]
                        reasoning  = analysis.get(feat_name, "")
                        reason_str = f"  ← {reasoning}" if reasoning else ""
                        print(f"    {feat_name:<22} {val:<18} ({conf}){reason_str}")

                else:
                    fail_count += 1
                    if not dry_run:
                        data["images"][idx]["label_status"]    = "auto_label_failed"
                        data["images"][idx]["auto_label_meta"] = {
                            "model":      model_id,
                            "labeled_at": now_str,
                            "error":      result.get("error", "unknown"),
                        }
                    err = result.get("error", "")[:80]
                    print(f"  [{completed:>{len(str(total))}}/{total}] FAIL {entry_id[:48]}  {err}")

                if not dry_run and completed % checkpoint_every == 0:
                    data["updated"] = now_str
                    save_manifest(data, manifest_path)
                    print(f"  [checkpoint saved at {completed}/{total}]")

    if not dry_run:
        data["updated"] = datetime.now(timezone.utc).isoformat()
        save_manifest(data, manifest_path)

    cache_pct = (total_cache_read / max(total_input, 1)) * 100
    print(f"\n── Batch complete ──────────────────────────────────────────")
    print(f"  Total         : {total}")
    print(f"  Succeeded     : {success_count}")
    print(f"  Failed        : {fail_count}")
    if auto_accept_threshold:
        print(f"  Auto-accepted : {auto_accepted_ct}  (marked reviewed, no human review needed)")
    print(f"  Cache hit rate: {total_cache_read:,} / {total_input:,} tokens ({cache_pct:.0f}%)")
    if dry_run:
        print("  [dry-run — manifest not written]")
    else:
        print(f"  Manifest      : {manifest_path}")


# ── Dataset summary ───────────────────────────────────────────────────────────

def print_dataset_summary(manifest_path: Path, family_filter: str | None, source_filter: str | None) -> None:
    """
    Print a comprehensive data-distribution summary to help assess dataset quality
    before training RockNet v2.  Covers:
      • Overall pipeline status (unlabeled → auto_labeled → reviewed)
      • Per-family: source & origin breakdown, feature-value distributions,
        class-imbalance warnings, and low-confidence (conf=1) flags
      • ML readiness check (sample size, imbalance, pending human review)
    """
    from collections import Counter

    data   = load_manifest(manifest_path)
    images = data.get("images", [])

    if family_filter:
        images = [e for e in images if e.get("family") == family_filter]
    if source_filter:
        images = [e for e in images if e.get("source") == source_filter]

    total = len(images)
    if total == 0:
        print("  No images found matching filters.")
        return

    # ── Overall pipeline ──────────────────────────────────────────────────────
    by_status = Counter(e.get("label_status", "unlabeled") for e in images)
    n_reviewed = sum(1 for e in images if e.get("reviewed", False))
    n_rejected = sum(1 for e in images if e.get("quality") == "rejected")

    print(f"\n{'─'*70}")
    print(f"  DATASET SUMMARY  —  {manifest_path.name}")
    print(f"{'─'*70}")
    print(f"  Total images : {total:,}")
    print(f"  Reviewed     : {n_reviewed:,}  ({n_reviewed/total*100:.0f}%)")
    print(f"  Rejected     : {n_rejected:,}  ({n_rejected/total*100:.0f}%)")
    print()
    print(f"  Pipeline breakdown:")
    for status, count in sorted(by_status.items(), key=lambda x: -x[1]):
        bar = "█" * max(1, round(count / total * 30))
        print(f"    {status:<24} {count:>5}  ({count/total*100:4.0f}%)  {bar}")

    # ── Per-family breakdowns ─────────────────────────────────────────────────
    for family, schema in FEATURE_SCHEMAS.items():
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

        for feat_name, valid_values in schema.items():
            val_counts  = Counter()
            conf_counts = Counter()
            for e in labeled:
                raw = e.get("features", {}).get(feat_name)
                if raw is None:
                    continue
                val  = raw["value"] if isinstance(raw, dict) else raw
                conf = raw.get("conf", 3) if isinstance(raw, dict) else 3
                val_counts[val]   += 1
                conf_counts[conf] += 1

            feat_total = sum(val_counts.values())
            if feat_total == 0:
                continue

            dist_parts = [
                f"{v}={val_counts.get(v, 0)}({val_counts.get(v, 0)/feat_total*100:.0f}%)"
                for v in valid_values
            ]
            dist_str = "  ".join(dist_parts)

            max_pct  = max(val_counts.values()) / feat_total * 100
            n_conf1  = conf_counts.get(1, 0)
            flags    = ""
            if max_pct > 80:
                flags += "  ⚠ imbalanced"
            if n_conf1 > 0:
                flags += f"  [conf1={n_conf1}]"

            print(f"    {feat_name:<22} {dist_str}{flags}")

            if max_pct > 80:
                dom = val_counts.most_common(1)[0][0]
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
            print(f"  ⚑  Low-confidence labels (conf=1) — {total_conf1} uncertain labels need human review:")
            for feat, cnt in sorted(conf1_notes, key=lambda x: -x[1]):
                print(f"       {feat:<22} {cnt} images")

    # ── ML readiness check ────────────────────────────────────────────────────
    print(f"\n  {'─'*66}")
    print(f"  ML TRAINING READINESS CHECK")
    print(f"  {'─'*66}")

    MIN_SAMPLES = 50

    for family, schema in FEATURE_SCHEMAS.items():
        fimages  = [e for e in images if e.get("family") == family]
        labeled  = [e for e in fimages if e.get("features") and e.get("label_status") in ("auto_labeled", "reviewed")]
        rev_imgs = [e for e in fimages if e.get("label_status") == "reviewed"]
        n_labeled, n_rev = len(labeled), len(rev_imgs)

        issues: list[str] = []
        if n_labeled < MIN_SAMPLES:
            issues.append(f"only {n_labeled} labeled  (recommend {MIN_SAMPLES}+)")

        for feat_name, valid_values in schema.items():
            val_counts = Counter()
            for e in labeled:
                raw = e.get("features", {}).get(feat_name)
                if raw is None:
                    continue
                val = raw["value"] if isinstance(raw, dict) else raw
                val_counts[val] += 1
            feat_total = sum(val_counts.values())
            if feat_total > 0 and max(val_counts.values()) / feat_total > 0.80:
                dom = val_counts.most_common(1)[0][0]
                issues.append(f"{feat_name} skewed ('{dom}' {max(val_counts.values())/feat_total*100:.0f}%)")

        icon = "⚠" if issues else "✓"
        issue_str = "; ".join(issues) if issues else "looks good"
        print(f"  {icon} {family:<14} {n_labeled:>4} labeled  ({n_rev} human-reviewed)  →  {issue_str}")

    n_auto   = sum(1 for e in images if e.get("label_status") == "auto_labeled")
    n_unlbl  = sum(1 for e in images if e.get("label_status", "unlabeled") in ("unlabeled", "auto_hint", "auto_label_failed"))
    print()
    print(f"  Needs human review : {n_auto:,}   →  python label_tool.py --review-auto")
    print(f"  Not yet labeled    : {n_unlbl:,}   →  python auto_label.py")
    print()


# ── Review queue printer ──────────────────────────────────────────────────────

def print_review_queue(manifest_path: Path, family_filter: str | None, source_filter: str | None) -> None:
    data  = load_manifest(manifest_path)
    queue = build_review_queue(data["images"], family_filter, source_filter)

    if not queue:
        print("No auto_labeled images found. Run auto_label.py first.")
        return

    conf_labels = {1: "uncertain", 2: "likely", 3: "certain"}
    print(f"\n── Review queue: {len(queue)} images, sorted by min feature confidence ──")
    print(f"  {'Rank':<5} {'MinConf':<12} {'Family':<12} {'Source':<12} ID")
    print("  " + "─" * 72)
    for rank, entry in enumerate(queue, 1):
        features = entry.get("features", {})
        min_conf = (
            min((v["conf"] if isinstance(v, dict) else 3) for v in features.values())
            if features else 1
        )
        cl = conf_labels.get(min_conf, "?")
        print(
            f"  {rank:<5} {min_conf} ({cl:<8})  "
            f"{entry.get('family', '?'):<12} "
            f"{entry.get('source', '?'):<12} "
            f"{entry.get('id', '?')[:50]}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM-based auto-labeling for rock images using Claude Vision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", required=True, help="Path to dataset_manifest.json")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        choices=["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, metavar="N",
        help=f"Concurrent API threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10, metavar="N",
        help="Save manifest every N completions (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Make API calls but do not write manifest changes",
    )
    parser.add_argument(
        "--family", choices=["basalt", "anorthosite", "breccia"], default=None,
        help="Limit labeling to one rock family",
    )
    parser.add_argument(
        "--source", default=None, metavar="SOURCE",
        help="Limit labeling to one source (e.g. kaggle, lpi, smithsonian, astromaterials)",
    )
    parser.add_argument(
        "--auto-accept-threshold", type=int, default=None, metavar="N",
        choices=[1, 2, 3],
        help="Auto-mark as reviewed if all feature confs >= N, skipping human review",
    )
    parser.add_argument(
        "--force-relabel", action="store_true",
        help="Re-label entries already marked auto_labeled",
    )
    parser.add_argument(
        "--review-queue", action="store_true",
        help="Print the sorted human-review queue and exit (no API calls)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help=(
            "Print a comprehensive data-distribution summary (per-family per-feature "
            "counts, class balance, confidence quality, ML readiness) and exit."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    if args.summary:
        print_dataset_summary(manifest_path, args.family, args.source)
        return

    if args.review_queue:
        print_review_queue(manifest_path, args.family, args.source)
        return

    run_batch_labeling(
        manifest_path=manifest_path,
        model_id=args.model,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        dry_run=args.dry_run,
        family_filter=args.family,
        source_filter=args.source,
        auto_accept_threshold=args.auto_accept_threshold,
        force_relabel=args.force_relabel,
    )


if __name__ == "__main__":
    main()
