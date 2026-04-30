from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "led-display" / "models" / "Phi-3-mini-4k-instruct-q4.gguf"

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

    metadata: list[str] = [f"Rock type: {label}"]
    if volume is not None:
        metadata.append(f"Estimated volume: {volume} cm3")
    if weight:
        metadata.append(f"Estimated weight: {weight}")

    joined_notes = "\n".join(f"- {note}" for note in notes)
    
    return (
        f"=== METADATA ===\n{chr(10).join(metadata)}\n\n"
        f"=== FIELD AUDIO TRANSCRIPTS ===\n{joined_notes}\n\n"
        "=== INSTRUCTIONS ===\n"
        "Synthesize the metadata and transcripts above into a concise field report. "
        "You must output ONLY the following template. If information for a specific category is missing from the data, write 'Not specified'. Keep descriptions brief and objective.\n\n"
        "- Color & Appearance: \n"
        "- Mineralogy & Composition: \n"
        "- Texture & Structure: \n"
        "- Weathering & Alteration: \n"
        "- Dimensions & Weight: \n"
        "- Field Context & Sampling Notes: "
    )


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
    # DEBUG: See what the AI actually said before we chop it
    print(f"\n--- RAW AI OUTPUT ---\n{text}\n---------------------\n")

    summary = re.sub(r"^```[a-zA-Z]*\n", "", text.strip())
    summary = re.sub(r"\n```$", "", summary)
    
    # Updated Regex to match the new, expanded prompt categories
    match = re.search(
        r"(?i)(Color & Appearance|Mineralogy & Composition|Texture & Structure|Weathering & Alteration|Dimensions & Weight|Field Context & Sampling Notes):", 
        summary
    )
    
    if match:
        # Start from the beginning of the line where the first category appears
        start_pos = summary.rfind('\n', 0, match.start()) + 1
        summary = summary[start_pos:]
        
    return summary.strip()


def _generate_summary(payload: dict, notes: list[str]) -> str:
    llm = _load_llama()
    prompt = _build_report_prompt(payload, notes)
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
        temperature=0.1,
        top_p=0.9,
        max_tokens=400,
    )
    return _clean_generated_summary(_extract_response_text(response))


def summarize_payload(payload: dict) -> dict:
    rock_id = payload.get("rock_id")
    cleaned_notes: list[str] = []
    for note in payload.get("notes", []):
        cleaned_note = _clean_note(note)
        if cleaned_note:
            cleaned_notes.append(cleaned_note)
    notes = _dedupe(cleaned_notes)

    if not notes:
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
