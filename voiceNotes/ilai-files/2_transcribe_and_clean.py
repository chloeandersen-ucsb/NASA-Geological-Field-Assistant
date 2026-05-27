#!/usr/bin/env python3
"""
STEP 2: Transcribe segments with Whisper large-v3, then apply geology
word correction and quality filtering.

Why Whisper large-v3 instead of your NeMo model?
  - It has much better coverage of technical/domain vocabulary
  - We use it ONLY for generating training labels (not for live inference)
  - The geology word list forces correct spellings for known hard words

Outputs a NeMo-format manifest.json ready for fine-tuning.

Usage:
    # Install first (run once):
    pip install openai-whisper

    python 2_transcribe_and_clean.py \
        --segments_dir ./segments \
        --output_manifest ./data/train_manifest.json \
        --geology_wordlist geology_words.txt
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


# ── Geology word list ─────────────────────────────────────────────────────────
# These are injected into Whisper as initial prompt tokens so it biases
# toward correct spellings. Add any domain words your data contains.
GEOLOGY_WORDS = [
    # Minerals
    "feldspar", "orthoclase", "plagioclase", "quartz", "olivine", "pyroxene",
    "amphibole", "hornblende", "augite", "biotite", "muscovite", "chlorite",
    "calcite", "dolomite", "magnetite", "hematite", "pyrite", "galena",
    "sphalerite", "chalcopyrite", "malachite", "azurite", "gypsum", "halite",
    "fluorite", "apatite", "zircon", "tourmaline", "garnet", "kyanite",
    "sillimanite", "andalusite", "staurolite", "epidote", "serpentine",
    "talc", "barite", "anhydrite", "corundum", "spinel", "rutile",

    # Rock types
    "granite", "basalt", "rhyolite", "andesite", "diorite", "gabbro",
    "peridotite", "dunite", "obsidian", "pumice", "tuff", "ignimbrite",
    "sandstone", "shale", "mudstone", "siltstone", "limestone", "dolostone",
    "chert", "conglomerate", "breccia", "arkose", "greywacke", "turbidite",
    "schist", "gneiss", "quartzite", "marble", "slate", "phyllite",
    "amphibolite", "granulite", "eclogite", "blueschist", "hornfels",

    # Igneous processes
    "magma", "lava", "intrusive", "extrusive", "plutonic", "volcanic",
    "subvolcanic", "hypabyssal", "batholith", "pluton", "laccolith",
    "sill", "dike", "xenolith", "phenocryst", "porphyritic", "aphanitic",
    "phaneritic", "vesicular", "amygdaloidal", "fractionation",
    "differentiation", "crystallization", "solidus", "liquidus",
    "partial melting", "mantle", "lithosphere", "asthenosphere",

    # Sedimentary processes
    "deposition", "erosion", "weathering", "diagenesis", "lithification",
    "compaction", "cementation", "porosity", "permeability", "facies",
    "stratigraphy", "unconformity", "transgression", "regression",
    "alluvial", "fluvial", "aeolian", "lacustrine", "deltaic", "marine",
    "evaporite", "biochemical", "clastic", "siliciclastic",

    # Metamorphic processes
    "metamorphism", "recrystallization", "foliation", "lineation",
    "schistosity", "cleavage", "banding", "porphyroblast", "retrograde",
    "prograde", "contact metamorphism", "regional metamorphism",
    "subduction", "accretion", "terrane", "suture zone",

    # Structural geology
    "anticline", "syncline", "monocline", "fold", "thrust fault",
    "normal fault", "reverse fault", "strike-slip fault", "horst", "graben",
    "rift", "orogeny", "tectonics", "plate tectonics", "subduction zone",
    "mid-ocean ridge", "transform boundary", "accretionary prism",

    # Geochemistry / petrology
    "silica", "SiO2", "mafic", "felsic", "ultramafic", "intermediate",
    "alkaline", "peraluminous", "calc-alkaline", "tholeiitic",
    "trace element", "REE", "rare earth", "isotope", "radiometric",
    "geochronology", "U-Pb", "Ar-Ar", "geothermal gradient",

    # Field terms
    "outcrop", "exposure", "bedding", "joint", "fracture", "vein",
    "mineralization", "alteration", "hydrothermal", "ore deposit",
    "geomorphology", "topography", "drainage basin", "watershed",
]

# Common Whisper mis-transcriptions of geology words → correct form
# Format: (regex_pattern, replacement)
CORRECTION_RULES = [
    (r"\bfelt\s*spar\b",        "feldspar"),
    (r"\bfeld\s*spar\b",        "feldspar"),
    (r"\bfelspar\b",            "feldspar"),
    (r"\bplagio\s*clase\b",     "plagioclase"),
    (r"\bortho\s*clase\b",      "orthoclase"),
    (r"\bpyro\s*xene\b",        "pyroxene"),
    (r"\bamphi\s*bole\b",       "amphibole"),
    (r"\born\s*blend\b",        "hornblende"),
    (r"\bhorn\s*blend\b",       "hornblende"),
    (r"\bbio\s*tite\b",         "biotite"),
    (r"\bmusco\s*vite\b",       "muscovite"),
    (r"\boliv\s*ine\b",         "olivine"),
    (r"\bgar\s*net\b",          "garnet"),
    (r"\bmagnet\s*ite\b",       "magnetite"),
    (r"\bhemat\s*ite\b",        "hematite"),
    (r"\bpyr\s*ite\b",          "pyrite"),
    (r"\bgran\s*ite\b",         "granite"),
    (r"\bba\s*salt\b",          "basalt"),
    (r"\brhyo\s*lite\b",        "rhyolite"),
    (r"\bandes\s*ite\b",        "andesite"),
    (r"\bper\s*id\s*otite\b",   "peridotite"),
    (r"\bgab\s*bro\b",          "gabbro"),
    (r"\bquartz\s*ite\b",       "quartzite"),
    (r"\bschist\b",             "schist"),
    (r"\bgneiss\b",             "gneiss"),
    (r"\bphyl\s*lite\b",        "phyllite"),
    (r"\bstratigraphy\b",       "stratigraphy"),
    (r"\borog\s*eny\b",         "orogeny"),
    (r"\btec\s*tonics\b",       "tectonics"),
    (r"\bsubduc\s*tion\b",      "subduction"),
    (r"\bdiagen\s*esis\b",      "diagenesis"),
    (r"\blithif\s*ication\b",   "lithification"),
    (r"\bhydro\s*thermal\b",    "hydrothermal"),
    (r"\bgeo\s*chronology\b",   "geochronology"),
    (r"\bgeomorphology\b",      "geomorphology"),
    (r"\bma\s*fic\b",           "mafic"),
    (r"\bfel\s*sic\b",          "felsic"),
    (r"\bultra\s*mafic\b",      "ultramafic"),
]


def apply_corrections(text: str) -> str:
    """Apply geology-specific spelling corrections to a transcript."""
    text = text.strip()
    for pattern, replacement in CORRECTION_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Normalise whitespace
    text = " ".join(text.split())
    return text


def is_quality_transcript(text: str, duration_s: float) -> tuple[bool, str]:
    """
    Return (keep, reason) — filter out transcripts that will hurt training.
    """
    if not text or len(text.strip()) < 5:
        return False, "too short"

    words = text.split()
    word_count = len(words)

    # Words-per-second sanity check (normal speech: 1.5–4.5 wps)
    wps = word_count / max(duration_s, 0.1)
    if wps < 0.8:
        return False, f"too few words for duration ({wps:.1f} wps)"
    if wps > 6.0:
        return False, f"too many words for duration ({wps:.1f} wps) — likely wrong"

    # Reject if >20% of characters are non-ASCII (music, noise, foreign)
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.2:
        return False, "too many non-ASCII characters"

    # Reject very repetitive output (Whisper hallucination sign)
    if word_count > 4:
        unique_ratio = len(set(w.lower() for w in words)) / word_count
        if unique_ratio < 0.4:
            return False, f"too repetitive (unique ratio {unique_ratio:.2f})"

    return True, "ok"


def transcribe_segments(segments_dir: str, output_manifest: str,
                        geology_wordlist_path: str | None = None,
                        whisper_model: str = "large-v3") -> None:
    try:
        import whisper
    except ImportError:
        print("ERROR: whisper not installed. Run:  pip install openai-whisper",
              file=sys.stderr)
        sys.exit(1)

    # Build the geology prompt — Whisper uses this to bias token probabilities
    prompt_words = GEOLOGY_WORDS.copy()
    if geology_wordlist_path and os.path.exists(geology_wordlist_path):
        with open(geology_wordlist_path) as f:
            extra = [line.strip() for line in f if line.strip()]
        prompt_words.extend(extra)
        print(f"Loaded {len(extra)} extra words from {geology_wordlist_path}")

    # Keep prompt under Whisper's 224-token limit
    geology_prompt = ", ".join(prompt_words[:80])

    print(f"Loading Whisper {whisper_model}... (this may take a minute)")
    model = whisper.load_model(whisper_model)
    print("Model loaded.")

    wav_files = sorted(Path(segments_dir).rglob("*.wav"))
    if not wav_files:
        print(f"ERROR: No .wav files found under {segments_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(wav_files)} segments to transcribe")

    os.makedirs(Path(output_manifest).parent, exist_ok=True)

    kept = 0
    skipped = 0
    t0 = time.time()

    with open(output_manifest, "w", encoding="utf-8") as out_f:
        for i, wav_path in enumerate(wav_files):
            # Progress update every 50 files
            if i % 50 == 0 and i > 0:
                elapsed  = time.time() - t0
                rate     = i / elapsed
                eta      = (len(wav_files) - i) / rate
                print(f"  [{i}/{len(wav_files)}]  "
                      f"kept={kept}  skipped={skipped}  "
                      f"ETA {eta/60:.0f}min")

            # Get duration from filename or via ffprobe
            try:
                result = __import__("subprocess").run(
                    ["ffprobe", "-v", "quiet", "-show_entries",
                     "format=duration", "-of", "csv=p=0", str(wav_path)],
                    capture_output=True, text=True
                )
                duration_s = float(result.stdout.strip())
            except Exception:
                duration_s = 10.0  # fallback

            # Transcribe
            try:
                result = model.transcribe(
                    str(wav_path),
                    language="en",
                    initial_prompt=geology_prompt,
                    temperature=0.0,        # greedy — more deterministic labels
                    condition_on_previous_text=False,  # avoid hallucination drift
                )
                raw_text = result["text"]
            except Exception as e:
                print(f"  WARNING: Failed to transcribe {wav_path.name}: {e}")
                skipped += 1
                continue

            # Clean and validate
            corrected_text = apply_corrections(raw_text)
            keep, reason   = is_quality_transcript(corrected_text, duration_s)

            if not keep:
                skipped += 1
                continue

            # Write NeMo manifest line
            entry = {
                "audio_filepath": str(wav_path.resolve()),
                "duration":       round(duration_s, 3),
                "text":           corrected_text.lower(),  # NeMo CTC expects lowercase
            }
            out_f.write(json.dumps(entry) + "\n")
            kept += 1

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Transcription complete in {elapsed/60:.1f} min")
    print(f"  Kept:    {kept} segments")
    print(f"  Skipped: {skipped} segments (quality filtered)")
    print(f"  Manifest written to: {output_manifest}")
    print(f"\nNext step: run  3_finetune_nemo.py")


def split_train_val(manifest_path: str, val_ratio: float = 0.05) -> None:
    """Split a manifest into train/val sets."""
    import random
    lines = open(manifest_path).readlines()
    random.shuffle(lines)
    n_val   = max(1, int(len(lines) * val_ratio))
    n_train = len(lines) - n_val

    base  = manifest_path.replace(".json", "")
    train = base + "_train.json"
    val   = base + "_val.json"

    with open(train, "w") as f:
        f.writelines(lines[:n_train])
    with open(val, "w") as f:
        f.writelines(lines[n_train:])

    print(f"Split: {n_train} train  |  {n_val} val")
    print(f"  → {train}")
    print(f"  → {val}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe segments with Whisper + geology correction")
    parser.add_argument("--segments_dir",     required=True)
    parser.add_argument("--output_manifest",  required=True)
    parser.add_argument("--geology_wordlist", default=None,
                        help="Optional .txt file with extra geology words (one per line)")
    parser.add_argument("--whisper_model",    default="large-v3",
                        choices=["medium", "large", "large-v2", "large-v3"],
                        help="Whisper model size (large-v3 recommended)")
    parser.add_argument("--split_val",        action="store_true",
                        help="Also split into train/val manifests")
    args = parser.parse_args()

    transcribe_segments(
        args.segments_dir,
        args.output_manifest,
        args.geology_wordlist,
        args.whisper_model,
    )

    if args.split_val:
        split_train_val(args.output_manifest)


if __name__ == "__main__":
    main()
