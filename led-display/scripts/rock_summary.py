from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "led-display" / "models" / "Phi-3-mini-4k-instruct-q4.gguf"

# Maps each RockNet feature name to its corresponding AI summary bucket.
FEATURE_TO_BUCKET: dict[str, str] = {
    # Appearance
    "luster":             "Color & Appearance",
    "brightness":         "Color & Appearance",
    # Mineralogy
    "phenocryst_hint":    "Mineralogy & Composition",
    "mafic_content_hint": "Mineralogy & Composition",
    # Texture / structure
    "groundmass_texture": "Texture & Structure",
    "crystal_fabric":     "Texture & Structure",
    "clast_angularity":   "Texture & Structure",
    "sorting":            "Texture & Structure",
    "support_fabric":     "Texture & Structure",
    # Weathering
    "vesicularity":       "Weathering & Alteration",
    "surface_character":  "Weathering & Alteration",
}


def extract_displayable_features(raw: Optional[dict]) -> dict[str, str]:
    """Return {feature_name: humanized_value} for all displayable RockNet features."""
    if not raw:
        return {}
    features_data = raw.get("features") or {}
    result: dict[str, str] = {}
    for name, feat in features_data.items():
        if not isinstance(feat, dict) or not feat.get("display", False):
            continue
        value = feat.get("value", "")
        if not value or value in ("n/a", "uncertain"):
            continue
        result[name] = value.replace("_", " ")
    return result


def features_to_summary_string(
    label: str,
    features: dict[str, str],
    volume: Optional[float],
    weight,
) -> str:
    """Build a 6-line summary string from classification features alone (no LLM)."""
    buckets: dict[str, list[str]] = {
        "Color & Appearance": [],
        "Mineralogy & Composition": [],
        "Texture & Structure": [],
        "Weathering & Alteration": [],
        "Dimensions & Weight": [],
        "Field Context & Sampling Notes": [],
    }

    for feat_name, feat_value in features.items():
        bucket = FEATURE_TO_BUCKET.get(feat_name)
        if bucket:
            readable = feat_name.replace("_", " ")
            buckets[bucket].append(f"{readable}: {feat_value}")

    dim_parts: list[str] = []
    if volume is not None:
        dim_parts.append(f"{volume:.1f} cm³")
    weight_str = _normalize_text(str(weight)) if weight else ""
    if weight_str and weight_str.lower() != "none":
        dim_parts.append(weight_str)
    if dim_parts:
        buckets["Dimensions & Weight"].append("; ".join(dim_parts))

    if label and label.lower() not in ("other", "unknown rock", ""):
        buckets["Field Context & Sampling Notes"].append(
            f"Classified as {label} by RockNet"
        )

    lines: list[str] = []
    for bucket_name, parts in buckets.items():
        value = "; ".join(parts) if parts else "Not specified."
        lines.append(f"- {bucket_name}: {value}")
    return "\n".join(lines)


def _build_features_prompt_block(features: dict[str, str]) -> str:
    """Build the classification features section to inject into the LLM prompt."""
    if not features:
        return ""
    lines = [f"  {name.replace('_', ' ')}: {value}" for name, value in features.items()]
    return "=== CLASSIFICATION FEATURES (RockNet image analysis) ===\n" + "\n".join(lines)


FILLER_PATTERNS = (
    r"\bi(?: am|'m)\s+talking\s+about\s+a\s+rock\b",
    r"\bthis\s+is\s+a\s+rock\b",
    r"\bhere\s+is\s+a\s+rock\b",
    r"\bthis\s+rock\s+(?:is|has|as)\b",
    r"\bit\s+is\b",
)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").split()).strip()


def _collapse_duplicate_halves(text: str) -> str:
    words = text.split()
    if len(words) >= 6 and len(words) % 2 == 0:
        midpoint = len(words) // 2
        if words[:midpoint] == words[midpoint:]:
            return " ".join(words[:midpoint])
    return text


def _strip_repeated_ngrams(text: str) -> str:
    words = text.split()
    if len(words) < 4:
        return text

    compact: list[str] = []
    i = 0
    while i < len(words):
        repeated_size = 0
        max_n = min(8, (len(words) - i) // 2)
        for n in range(max_n, 1, -1):
            if words[i:i + n] == words[i + n:i + (2 * n)]:
                repeated_size = n
                break
        if repeated_size:
            compact.extend(words[i:i + repeated_size])
            i += repeated_size * 2
            continue
        compact.append(words[i])
        i += 1
    return " ".join(compact)


def _clean_note(text: str) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return ""

    cleaned = re.sub(r"\bthis\s+rock\s+as\b", "this rock has", cleaned, flags=re.IGNORECASE)
    cleaned = _collapse_duplicate_halves(cleaned)
    cleaned = _strip_repeated_ngrams(cleaned)

    for pattern in FILLER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    return cleaned


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = _normalize_text(item).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(_normalize_text(item))
    return ordered


def _resolve_model_path() -> Path:
    env_model_path = os.environ.get("SAGE_SUMMARIZER_MODEL_PATH")
    candidate = Path(env_model_path).expanduser() if env_model_path else DEFAULT_MODEL_PATH
    if not candidate.exists():
        raise FileNotFoundError(f"Summarizer model not found: {candidate}")
    return candidate


def _build_report_prompt(payload: dict, notes: list[str]) -> str:
    label = _normalize_text(payload.get("label", "Unknown rock"))
    volume = payload.get("estimated_volume")
    weight = _normalize_text(payload.get("estimated_weight"))
    features: dict[str, str] = payload.get("features") or {}

    metadata: list[str] = [f"Rock type: {label}"]
    if volume is not None:
        metadata.append(f"Estimated volume: {volume} cm3")
    if weight:
        metadata.append(f"Estimated weight: {weight}")

    joined_notes = "\n".join(f"- {note}" for note in notes)
    features_block = _build_features_prompt_block(features)
    retry_instruction = " Please provide a more detailed, varied, and comprehensive synthesis than a standard report." if payload.get("is_retry") else ""

    sections: list[str] = [f"=== METADATA ===\n{chr(10).join(metadata)}"]
    if features_block:
        sections.append(features_block)
    sections.append(f"=== FIELD AUDIO TRANSCRIPTS ===\n{joined_notes}")
    synthesis_note = "classification features, " if features_block else ""
    sections.append(
        "=== INSTRUCTIONS ===\n"
        f"Synthesize the metadata, {synthesis_note}and transcripts above into a concise field report. "
        "You MUST copy the exact 6-line template below word-for-word and fill in the brackets. Do not add, remove, or rename any lines. If data is missing, replace the bracket with 'Not specified'.\n"
        f"{retry_instruction}\n\n"
        "TEMPLATE:\n"
        "- Color & Appearance: [fill in]\n"
        "- Mineralogy & Composition: [fill in]\n"
        "- Texture & Structure: [fill in]\n"
        "- Weathering & Alteration: [fill in]\n"
        "- Dimensions & Weight: [fill in]\n"
        "- Field Context & Sampling Notes: [fill in]"
    )
    return "\n\n".join(sections)


def _load_llama() -> Llama:
    if Llama is None:
        raise RuntimeError("llama-cpp-python is not installed in the project environment.")

    model_path = _resolve_model_path()
    n_ctx = int(os.environ.get("SAGE_SUMMARIZER_CTX", "2048"))
    n_threads = int(os.environ.get("SAGE_SUMMARIZER_THREADS", "4"))
    n_gpu_layers = int(os.environ.get("SAGE_SUMMARIZER_GPU_LAYERS", "0"))

    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


def _extract_response_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""

    first_choice = choices[0]
    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(part for part in text_parts if part)

    text = first_choice.get("text")
    return text if isinstance(text, str) else ""


def _clean_generated_summary(text: str) -> str:
    # DEBUG: See what the AI actually said before we process it
    print(f"\n--- RAW AI OUTPUT ---\n{text}\n---------------------\n")

    # 1. The Absolute Source of Truth: Our unbreakable 6 rows
    final_fields = {
        "Color & Appearance": "Not specified.",
        "Mineralogy & Composition": "Not specified.",
        "Texture & Structure": "Not specified.",
        "Weathering & Alteration": "Not specified.",
        "Dimensions & Weight": "Not specified.",
        "Field Context & Sampling Notes": "Not specified."
    }

    # 2. Clean up any weird AI markdown formatting
    summary = re.sub(r"^```[a-zA-Z]*\n", "", text.strip())
    summary = re.sub(r"\n```$", "", summary)

    # 3. Aggressively hunt through the AI's output line-by-line
    for line in summary.split('\n'):
        for key in final_fields.keys():
            # This regex looks for the category name, ignoring dashes, spaces, or bolding (**)
            pattern = re.compile(rf"^\s*[-*]*\s*(?:\*\*)?{re.escape(key)}(?:\*\*)?\s*:\s*(.*)", re.IGNORECASE)
            match = pattern.search(line)
            
            if match:
                value = match.group(1).strip()
                # If the AI actually provided an answer, save it!
                if value and "[fill in]" not in value.lower() and "not specified" not in value.lower():
                    final_fields[key] = value
                break # Move to the next line

    # 4. Reconstruct the perfect 6-line string for the UI
    perfect_summary = []
    for key, value in final_fields.items():
        perfect_summary.append(f"- {key}: {value}")

    return "\n".join(perfect_summary)


def _generate_summary(payload: dict, notes: list[str]) -> str:
    llm = _load_llama()
    prompt = _build_report_prompt(payload, notes)

    is_retry = payload.get("is_retry", False)
    temperature = 0.6 if is_retry else 0.1

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "Summarize these geological field notes into a brief, professional report.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
        top_p=0.9,
        max_tokens=400,
    )
    return _clean_generated_summary(_extract_response_text(response))


def summarize_payload(payload: dict) -> dict:
    rock_id = payload.get("rock_id")
    features: dict[str, str] = payload.get("features") or {}

    cleaned_notes: list[str] = []
    for note in payload.get("notes", []):
        cleaned_note = _clean_note(note)
        if cleaned_note:
            cleaned_notes.append(cleaned_note)
    notes = _dedupe(cleaned_notes)

    if not notes:
        if features:
            summary = features_to_summary_string(
                label=payload.get("label", ""),
                features=features,
                volume=payload.get("estimated_volume"),
                weight=payload.get("estimated_weight"),
            )
            return {"rock_id": rock_id, "summary": summary}
        return {
            "rock_id": rock_id,
            "summary": "No associated recordings to summarize yet.",
        }

    summary = _generate_summary(payload, notes).strip() or "AI summary unavailable."
    return {"rock_id": rock_id, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    result = summarize_payload(payload)
    with open(args.output_json, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
