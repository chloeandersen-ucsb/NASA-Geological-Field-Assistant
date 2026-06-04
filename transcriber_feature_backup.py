import time
app_start_time = time.time()
 
import sounddevice as sd
import nemo.collections.asr as nemo_asr
import numpy as np
import queue
import sys
import torch
torch.cuda.set_per_process_memory_fraction(0.4, 0)
import os
import difflib
import jellyfish 
import argparse 
from datetime import datetime
from spellchecker import SpellChecker
from textblob import TextBlob
 
# ------------------------------------------------------------------
# PART 1: CONFIGURATION
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Live ASR Transcriber")
parser.add_argument("--use-base", action="store_true", help="Use base fastconformer model instead of local fine-tuned model")
parser.add_argument("--raw-asr", action="store_true", help="Disable dictionary correction (preserve LLM formatting)")
args = parser.parse_args()
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
target_model_name = os.environ.get("SAGE_MODEL_FILENAME", "newest_model.nemo")
MODEL_FILE = os.path.join(SCRIPT_DIR, target_model_name)
 
# This is the path to the bridge file
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
CONTEXT_FILE = os.path.join(PROJECT_ROOT, "ML-classifications", "visual_context.txt")
 
def get_current_visual_context():
    try:
        if os.path.exists(CONTEXT_FILE):
            with open(CONTEXT_FILE, "r") as f:
                label = f.read().strip()
                if label: return [label]
    except:
        pass
    return [] # Default fallback
 
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5        
WINDOW_DURATION = 4.0       
ENERGY_THRESHOLD = 0.001    
PRINT_SILENCE = True
 
spell = SpellChecker()
 
# ------------------------------------------------------------------
# GEOLOGY WHITELIST
# Words the spellchecker must NEVER alter. Populated from training data
# frequency analysis — only words seen 1+ times in the corpus.
# Add new words here as your model learns them.
# ------------------------------------------------------------------
GEOLOGY_WHITELIST = {
    # High frequency (>10x in training data)
    "feldspar", "granite", "metamorphic", "sedimentary", "igneous",
    "landsat", "imagery", "magma", "marble", "gneiss", "limestone",
    "sandstone", "quartz", "plagioclase", "porphyritic", "quartzite",
    "metamorphism", "lava", "reflectance", "amphibolite", "basalt",
    "tourmaline", "shale",
    # Medium frequency (1-10x)
    "biotite", "garnet", "schist", "andesite", "gabbro", "amphibole",
    "hornblende", "conglomerate", "rhyolite", "phenocrysts", "phenocryst",
    "foliation", "granulite", "calcite", "dolostone", "zircon", "olivine",
    "phyllite", "slate", "outcrop", "mafic", "muscovite", "silica",
    "schorl", "lithification", "unconformity", "tectonics", "gypsum",
    "orthoclase", "pyroxene", "porphyroblast", "porphyroblasts",
    # Common geology terms that spellchecker mangles
    "hydrothermal", "alteration", "mineralization", "diagenesis",
    "stratigraphy", "orogeny", "subduction", "anticline", "syncline",
    "felsic", "ultramafic", "peridotite", "hematite", "magnetite",
    "recrystallized", "recrystallization", "geomorphology", "anorthosite"
}
 
# Pre-load the geology whitelist into the spellchecker so it never
# flags these as misspellings.
spell.word_frequency.load_words(GEOLOGY_WHITELIST)
 
# ------------------------------------------------------------------
# EXPLICIT CORRECTION RULES
# For words where Jaro-Winkler matching isn't reliable enough.
# These are applied as a first pass, before any other correction.
# Pattern: (regex, correct_form) — add new entries as you find them.
# ------------------------------------------------------------------
import re as _re
EXPLICIT_CORRECTIONS = [
    # plagioclase — covers plagiaclase, plagiiaclase, plagiiaclays, plagiiaclayse, etc.
    (_re.compile(r"\bplagi[ai]+[oa]?cl[ae][ys]+e?\b", _re.I), "plagioclase"),
    (_re.compile(r"\bplagi[ai]+[oa]?cl[ae]+s[e]?\b", _re.I), "plagioclase"),
    # feldspar — covers "felts sp", "felds spar", "felt spar"
    (_re.compile(r"\bfel[dt]s?\s*sp[ae]r\b", _re.I), "feldspar"),
    (_re.compile(r"\bfelts\s+sp\b", _re.I), "feldspar"),
    # gneiss — covers gneis, niceis, gnais
    (_re.compile(r"\bgn?[ae][iy]+s{1,2}\b", _re.I), "gneiss"),
    (_re.compile(r"\bni[cs]e[iy]+s\b", _re.I), "gneiss"),
    # phenocrysts — covers phenachrysts, phenochrysts, phenocrysps
    (_re.compile(r"\bphen[ao]c[rh]+[iy]*[sc]?[tp]+s?\b", _re.I), "phenocrysts"),
    # schist — covers shist, shchist, shifists
    (_re.compile(r"\bsh[ci]+[hy]?[iy]*sts?\b", _re.I), "schist"),
    (_re.compile(r"\bshist\b", _re.I), "schist"),
    # metamorphism — covers metamorphhoism, metamorphoism, metamorphhoism
    (_re.compile(r"\bmet[ae]m[ao]rph+[oi]+[sz][im]*\b", _re.I), "metamorphism"),
    # recrystallized variants
    (_re.compile(r"\brecrystal+[il]+[sz]ed?\b", _re.I), "recrystallized"),
    # quartzite variants
    (_re.compile(r"\bqu[ao]r[dt]z?[sz]?[iy]+te?\b", _re.I), "quartzite"),
    # amphibolite variants
    (_re.compile(r"\bamphi[bv]o[lh][iy]+te?\b", _re.I), "amphibolite"),
    # anorthosite — ASR splits it across 2-4 words.
    # Observed: "on north side site", "on north side sideite", "a north site", "anortho site"
    (_re.compile(r"\ban\s*[ao]rth[ao]\s*s[iy]+te?\b", _re.I), "anorthosite"),
    (_re.compile(r"\b(?:on|an?)\s+north\s+(?:side?\s+)?s(?:ite?|ideite?|ight)\b", _re.I), "anorthosite"),
]
 
def apply_explicit_corrections(text: str) -> str:
    """Apply regex-based corrections before any other processing."""
    for pattern, replacement in EXPLICIT_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text


def _repair_visual_context(text: str, visual_keywords: list) -> str:
    """
    General-purpose repair for visual context keywords mangled by ASR.
    The ASR splits geology terms across 2-6 words in unpredictable ways
    (e.g. "anorthosite" → "in the north a yes not"). Regex can't cover all
    variants, so we use character-level SequenceMatcher on sliding windows.

    Only fires when the keyword is absent from the transcript — never replaces
    an already-correct word. Tries large windows first so the entire mangled
    span is replaced, not just part of it.
    """
    if not text or not visual_keywords:
        return text

    words = text.split()

    for phrase in visual_keywords:
        target = phrase.lower().strip()
        target_sq = target.replace(" ", "")
        if len(target_sq) < 5:
            continue  # too short — false-positive risk not worth it

        # Already present? Nothing to do.
        if target_sq in "".join(w.lower().strip(".,!?") for w in words):
            continue

        best_score = 0.0
        best_start = -1
        best_size = 0

        # Scan ALL window sizes and keep the globally best-scoring match.
        # Early-break-on-large-window was wrong: a 5-word window that includes
        # an extra real word (e.g. "rock") scores lower than the correct 4-word
        # window that covers only the mangled span.
        # Length-ratio guard: chunk must be 0.6–2.2× the target length to
        # avoid matching a completely unrelated long sentence.
        # Scan ALL window sizes, down to 0 so we actually check 1-word chunks!
        for wsize in range(min(7, len(words)), 0, -1):
            for start in range(len(words) - wsize + 1):
                chunk = "".join(w.lower().strip(".,!?") for w in words[start:start + wsize])
                ratio = len(chunk) / len(target_sq)
                
                # STRICTER LENGTH RATIO: The chunk cannot be wildly longer or shorter than the target
                if not (0.75 <= ratio <= 1.4):
                    continue
                    
                score = difflib.SequenceMatcher(None, target_sq, chunk).ratio()
                if score > best_score:
                    best_score = score
                    best_start = start
                    best_size = wsize

        # 1-word window: needs high confidence (0.78) but not 0.82 — that was
        # too tight and caused "anothersite" (score ≈ 0.82) to fail on floating-point.
        # Multi-word window: 0.60 — these squish differently so a lower bar is fine.
        required_score = 0.78 if best_size == 1 else 0.60

        if best_score >= required_score and best_start >= 0:
            print(
                f"[CONTEXT REPAIR] '{' '.join(words[best_start:best_start+best_size])}'"
                f" → '{target}' (score={best_score:.2f})",
                file=sys.stderr
            )
            words = (words[:best_start] + [target] + words[best_start + best_size:])

    return " ".join(words)


def multimodal_correction(transcript, visual_keywords):
    if not transcript: return ""

    # Always apply explicit regex corrections first — even in raw_asr mode,
    # since these fix known systematic model errors, not English spelling.
    transcript = apply_explicit_corrections(transcript)

    # Visual context repair also runs before the raw_asr gate — it's not a
    # spellchecker, it's a targeted fix for the specific rock type being observed.
    transcript = _repair_visual_context(transcript, visual_keywords)

    if args.raw_asr: return transcript
 
    # 1. MASTER TARGET LIST — always includes geology whitelist so the
    # n-gram matcher can snap common geology words even without visual context
    targets = list(GEOLOGY_WHITELIST)
    for phrase in visual_keywords:
        targets.append(phrase.lower())
        targets.extend(phrase.lower().split())
    
    if targets:
        spell.word_frequency.load_words(targets)
        
    targets = list(set(targets))
 
    # --- THE SAFE STUTTER & GARBAGE FILTER ---
    raw_words = transcript.split()
 
    # Self-correction collapser: the model often outputs sequences like
    # "plagiiaclase and plagioclase" or "gneis gneiss" where it corrects
    # itself mid-stream. Keep the LAST occurrence of near-duplicate runs.
    # Only fires when the words are directly adjacent (no skipping over
    # real content words between them).
    collapsed = []
    i = 0
    while i < len(raw_words):
        current = raw_words[i].lower().strip(".,?!")
        # Only look at the immediately next word — no skipping
        if (i + 1 < len(raw_words)):
            ahead = raw_words[i + 1].lower().strip(".,?!")
            if (len(current) >= 4 and len(ahead) >= 4 and
                    (current.startswith(ahead[:4]) or ahead.startswith(current[:4]))):
                # Adjacent self-correction — keep the longer/later one
                collapsed.append(raw_words[i + 1] if len(ahead) >= len(current) else raw_words[i])
                i += 2
                continue
        collapsed.append(raw_words[i])
        i += 1
 
    words = []
    for w in collapsed:
        clean = w.lower().strip(".,?!")
        # Kill exact consecutive duplicates that slipped through
        if words and clean == words[-1].lower().strip(".,?!"):
            continue
        # Kill stray consonants (allow only a, i, o)
        if len(clean) == 1 and clean not in ["a", "i", "o"]:
            continue
        words.append(w)
 
    corrected_words = []
    skip_next = 0
 
    for i in range(len(words)):
        if skip_next > 0:
            skip_next -= 1
            continue
 
        best_match = None
        max_skip = 0
 
        # 2. SLIDING N-GRAM WINDOW
        # Use window=4 for visual context keywords only — ASR can split a single
        # technical word (e.g. "anorthosite") into up to 4 tokens. The wider window
        # is gated on visual_keywords to avoid false positives on common words.
        visual_targets = [kw.replace(" ", "").lower() for phrase in visual_keywords
                          for kw in ([phrase] + phrase.split())]
        window_sizes = [4, 3, 2, 1] if visual_targets else [3, 2, 1]
        for window_size in window_sizes:
            if i + window_size <= len(words):
                chunk = "".join(words[i:i+window_size]).lower().strip(".,?!")

                # For window=4, only check visual context targets to keep it tight
                candidate_targets = visual_targets if window_size == 4 else targets

                for target in candidate_targets:
                    target_squished = target.replace(" ", "")
                    spelling_score = jellyfish.jaro_winkler_similarity(chunk, target_squished)
                    sounds_alike = (jellyfish.metaphone(chunk) == jellyfish.metaphone(target_squished))

                    threshold = 0.88 if len(target_squished) <= 4 else 0.85

                    length_ratio = min(len(chunk), len(target_squished)) / max(len(chunk), len(target_squished))

                    if length_ratio > 0.5 and (sounds_alike or spelling_score > threshold):
                        best_match = target
                        max_skip = window_size - 1
                        break

                if best_match:
                    break
 
        if best_match:
            corrected_words.append(best_match)
            skip_next = max_skip
        else:
            # 3. SPELLCHECKER FALLBACK
            # Only run spellcheck if the word is NOT in our geology whitelist.
            # Geology terms are intentionally not in the English dictionary —
            # running spellcheck on them guarantees wrong "corrections".
            clean_w = words[i].lower().strip(".,?!")
            if clean_w and clean_w not in spell and clean_w not in GEOLOGY_WHITELIST:
                corrected = spell.correction(clean_w)
                corrected_words.append(corrected if corrected else words[i])
            else:
                corrected_words.append(words[i])
            
    return " ".join(corrected_words)
 
# ------------------------------------------------------------------
# PART 2: DEVICE & MODEL SETUP (MAC-SPECIFIC RESTORE)
# ------------------------------------------------------------------
# We MUST use map_location here or the Jetson weights will crash the Mac
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
 
# Mock distributed to prevent the "Redirects" crash on MacOS
import types
if not hasattr(torch, "distributed"):
    torch.distributed = types.SimpleNamespace()
torch.distributed.is_initialized = lambda: False
 
if args.use_base:
    print("Loading base model: stt_en_fastconformer_ctc_large...")
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name="stt_en_fastconformer_ctc_large",
        map_location=device
    )
    print("Base model loaded successfully!")
else:
    print(f"Loading fine-tuned model: {MODEL_FILE}...")
    # map_location=device is the ONLY way this works on your MacBook
    asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        restore_path=MODEL_FILE, 
        map_location=device
    )
    print("Fine-tuned model loaded successfully!")
 
asr_model.freeze()
asr_model = asr_model.to(device)
asr_model.eval()
sys.stdout.flush()

# Warmup: force cuBLAS to allocate its workspace now, while GPU memory is free,
# before the LLM loads. Without this, the first recording triggers cublasCreate()
# with less headroom and hits CUBLAS_STATUS_ALLOC_FAILED on the Jetson.
try:
    _w_signal = torch.zeros(1, SAMPLE_RATE, dtype=torch.float32, device=device)
    _w_len = torch.tensor([SAMPLE_RATE], dtype=torch.int64, device=device)
    with torch.no_grad():
        asr_model.forward(input_signal=_w_signal, input_signal_length=_w_len)
    del _w_signal, _w_len
    print("NeMo warmup complete.")
    sys.stdout.flush()
except Exception as _warmup_err:
    print(f"NeMo warmup failed (non-fatal): {_warmup_err}", file=sys.stderr)

# Load the LLM BEFORE signalling ready to the UI.
# The LLM maps ~2.2 GB into memory; if it loads concurrently with the camera
# pipeline the two compete for NVMM and the camera gets ENOMEM.
# Loading here (during the loading screen) ensures all memory is settled before
# the user can open the camera preview.
from llama_cpp import Llama

LLM_MODEL_PATH = os.path.join(PROJECT_ROOT, "led-display", "models", "Phi-3-mini-4k-instruct-q4.gguf")

try:
    llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=4096, n_gpu_layers=0, verbose=False)
    print("LLM loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load LLM. Falling back to standard correction. Error: {e}")
    llm = None
sys.stdout.flush()

# Signal the UI that everything is ready. The home screen appears here, AFTER
# both NeMo and the LLM are fully in memory.
print("READY")
sys.stdout.flush()
 
# ------------------------------------------------------------------
# PART 3: LIVE INFERENCE LOGIC
# ------------------------------------------------------------------
def rms(x):
    return np.sqrt(np.mean(np.square(x.astype(np.float64))))
 
# def smart_merge(final_text, new_chunk):
#     """Now returns a TUPLE: (full_merged_text, entirely_new_suffix)"""
#     final_text = final_text.strip()
#     new_chunk = new_chunk.strip()
    
#     if not final_text: return new_chunk, new_chunk
#     if not new_chunk: return final_text, ""
 
#     old_words = final_text.split()
#     new_words = new_chunk.split()
 
#     # --- 1. NEW: AGGRESSIVE STUTTER KILLER ---
#     # Strip consecutive duplicate words created by ASR model boundary artifacts
#     deduped_new = [new_words[0]]
#     for w in new_words[1:]:
#         if w.lower() != deduped_new[-1].lower():
#             deduped_new.append(w)
#     new_words = deduped_new
 
#     # --- 2. LIST-BASED SEQUENCE MATCHER ---
#     search_window = old_words[-15:]
#     s = difflib.SequenceMatcher(None, [w.lower() for w in search_window], [w.lower() for w in new_words])
#     match = s.find_longest_match(0, len(search_window), 0, len(new_words))
    
#     # We trust the match if it's 2+ words, OR if it's 1 word but aligns near the beginning of the new audio
#     if match.size >= 2 or (match.size == 1 and match.b <= 2):
#         absolute_chop_idx = len(old_words) - len(search_window) + match.a
#         merged_words = old_words[:absolute_chop_idx] + new_words[match.b:]
#         merged_text = " ".join(merged_words)
        
#         new_suffix = " ".join(new_words[match.b + match.size:])
#         return merged_text, new_suffix
#     else:
#         # PANIC PROTOCOL: If no overlap is found, safely append the ENTIRE new chunk!
#         # (Your old code only appended the last word, causing massive drops)
#         safe_append = " ".join(new_words)
#         return final_text + " " + safe_append, safe_append
 
def smart_merge(final_text, new_chunk):
    final_text = final_text.strip()
    new_chunk = new_chunk.strip()
    
    if not final_text: return new_chunk, new_chunk
    if not new_chunk: return final_text, ""
 
    old_words = final_text.split()
    new_words = new_chunk.split()
 
    # 1. INTERNAL STUTTER KILLER
    deduped_new = [new_words[0]]
    for w in new_words[1:]:
        if w.lower() != deduped_new[-1].lower():
            deduped_new.append(w)
    new_words = deduped_new
 
    # 2. EXACT PREFIX-SUFFIX MATCHER (Prevents double cheatgrass)
    max_overlap = 0
    for i in range(len(new_words), 0, -1):
        if len(old_words) >= i:
            if [w.lower() for w in old_words[-i:]] == [w.lower() for w in new_words[:i]]:
                max_overlap = i
                break
                
    if max_overlap > 0:
        new_suffix_words = new_words[max_overlap:]
        merged_text = " ".join(old_words + new_suffix_words)
        new_suffix = " ".join(new_suffix_words)
        return merged_text, new_suffix
        
    # 3. FUZZY MATCHER (Fallback)
    search_window = old_words[-15:]
    s = difflib.SequenceMatcher(None, [w.lower() for w in search_window], [w.lower() for w in new_words])
    match = s.find_longest_match(0, len(search_window), 0, len(new_words))
    
    if match.size >= 2 or (match.size == 1 and match.b <= 2):
        absolute_chop_idx = len(old_words) - len(search_window) + match.a
        merged_words = old_words[:absolute_chop_idx] + new_words[match.b:]
        merged_text = " ".join(merged_words)
        new_suffix = " ".join(new_words[match.b + match.size:])
        return merged_text, new_suffix
    else:
        # PANIC PROTOCOL: Only append the very last word to prevent screen flooding
        safe_append = new_words[-1] if new_words else ""
        return final_text + " " + safe_append, safe_append
        
# ------------------------------------------------------------------
# PART 4: MAIN LOOP
# ------------------------------------------------------------------
audio_q = queue.Queue()
full_audio_buffer = []
 
def audio_callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())
 
# Check for Sennheiser → then explicit "pulse" PA device → then system default.
# The "default" ALSA device (via pulse plugin) can hang on Jetson; the "pulse"
# PortAudio device routes through PulseAudio correctly and supports multiplexing.
DEVICE_INDEX = None
for i, dev in enumerate(sd.query_devices()):
    if 'Sennheiser' in dev['name'] and dev['max_input_channels'] > 0:
        DEVICE_INDEX = i
        break
if DEVICE_INDEX is None:
    for i, dev in enumerate(sd.query_devices()):
        if dev['name'] == 'pulse' and dev['max_input_channels'] > 0:
            DEVICE_INDEX = i
            break
if DEVICE_INDEX is None:
    DEVICE_INDEX = sd.default.device[0]
 
# Start the stream once and leave it running permanently.
# Stopping/restarting causes PulseAudio to buffer audio during the gap and
# deliver it as "stale" frames at the start of the next recording, which
# contaminates the full-buffer NeMo pass with the previous session's audio.
# Instead, we keep the stream alive and drain-and-discard the audio queue
# whenever recording_active is False.
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, device=DEVICE_INDEX, callback=audio_callback)
stream.start()
 
chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
window_size = int(SAMPLE_RATE * WINDOW_DURATION)
rolling_buffer = np.zeros((1, 0), dtype=np.float32)
final_transcript = ""
recording_active = False

total_frames_recorded = 0
fresh_frames = 0          # frames added to rolling_buffer AFTER the stale-discard phase
last_inference_frames = 0
# PulseAudio has ~200 ms of internal latency even with a persistent stream.
# Discard the first 200 ms from the rolling buffer (live inference only) as a
# lightweight guard. The audio buffers themselves are fully reset on every start,
# so this is purely defensive against the OS audio stack, not saved recordings.
_STALE_DISCARD_FRAMES = int(SAMPLE_RATE * 0.2)
_frames_discarded = 0
 
last_raw_text = ""
stable_chunks = 0
last_llm_output = ""   # track previous recording's LLM output for cache-replay detection
 
try:
    while True:
        import select
        if select.select([sys.stdin], [], [], 0.01)[0]:
            raw_line = sys.stdin.readline()
            if not raw_line: break
            line = raw_line.strip().lower()
            
            if line == "start":
                recording_active = True

                while not audio_q.empty():
                    try:
                        audio_q.get_nowait()
                    except queue.Empty:
                        break

                full_audio_buffer = []
                rolling_buffer = np.zeros((1,0), dtype=np.float32)
                total_frames_recorded = 0
                fresh_frames = 0
                last_inference_frames = 0
                silence_chunks = 0
                _frames_discarded = 0

                stable_chunks = 0
                last_raw_text = ""
                final_transcript = ""

                if llm:
                    try: llm.reset()
                    except Exception: pass
                    try: llm._ctx.kv_cache_clear()
                    except Exception: pass

                # Stream is always running — nothing to start.
                print("\nRECORDING NOW... (Fine-tuned Model)\n")
                sys.stdout.flush()
            elif line == "stop":
                if not recording_active:
                    continue
                # Stream stays running; audio collected while not recording is
                # continuously discarded by the idle drain below.
                recording_active = False
 
                if last_raw_text:
                    live_context = get_current_visual_context()
                    
                    # NOTE: No head drop here. The head drop is a live-streaming
                    # artifact fix only — applying it at stop time corrupts the
                    # last window's first word. The full buffer re-transcription
                    # below supersedes this anyway.
                    corrected_text = multimodal_correction(last_raw_text, live_context).replace("⁇", "").strip()
                    if corrected_text:
                        new_full_transcript, new_part = smart_merge(final_transcript, corrected_text)
                        if new_part:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            print(f"[{timestamp}] {new_full_transcript}")
                        final_transcript = new_full_transcript

                import select
                import time
                # Wait up to 0.55s to ensure we catch the delayed 50ms 'start' from the UI
                if select.select([sys.stdin], [], [], 0.55)[0]:
                    # Vacuum up ALL queued commands in the pipe
                    pending = []
                    while select.select([sys.stdin], [], [], 0.0)[0]:
                        cmd = sys.stdin.readline().strip().lower()
                        if cmd in ["start", "stop"]: 
                            pending.append(cmd)
                        
                    # Only start fresh if the VERY LAST command was a start
                    # Only start fresh if the VERY LAST command was a start
                    if pending and pending[-1] == "start":
                        print("\n[REDO DETECTED] Discarding audio. Starting fresh...\n")
                        recording_active = True
                        
                        while not audio_q.empty():
                            try: audio_q.get_nowait()
                            except queue.Empty: break
                                
                        full_audio_buffer = []
                        rolling_buffer = np.zeros((1,0), dtype=np.float32)
                        total_frames_recorded = 0
                        fresh_frames = 0
                        last_inference_frames = 0
                        silence_chunks = 0
                        _frames_discarded = 0

                        stable_chunks = 0
                        last_raw_text = ""
                        final_transcript = ""

                        if llm:
                            try: llm.reset()
                            except Exception: pass
                            try: llm._ctx.kv_cache_clear()
                            except Exception: pass
                        
                        time.sleep(0.2)
                        print("\nRECORDING NOW... (Fine-tuned Model)\n")
                            
                        sys.stdout.flush()
                        continue
                # ------------------------------------
 
                print("\n\nStopping stream... Preparing final transcript...")
                
                # ------------------------------------------------------------------
                # PART 5: FINAL CLEANUP (DICTIONARY FIRST, LLM LAST)
                # ------------------------------------------------------------------
                if full_audio_buffer:
                    print("Processing full audio buffer for maximum accuracy...")
                    full_audio = np.concatenate(full_audio_buffer, axis=0).astype(np.float32)
                    max_val = np.max(np.abs(full_audio))
                    if max_val > 0: full_audio = full_audio / (max_val + 1e-9)
                    full_signal = torch.tensor(full_audio, dtype=torch.float32, device=device).unsqueeze(0)
                    full_len = torch.tensor([full_signal.shape[1]], dtype=torch.int64, device=device)
 
                    with torch.no_grad():
                        logits = asr_model.forward(input_signal=full_signal, input_signal_length=full_len)
                        if isinstance(logits, tuple): logits = logits[0]
                        pred_tokens = logits.argmax(dim=-1)
                        transcripts = asr_model.decoding.ctc_decoder_predictions_tensor(pred_tokens)
                        final_raw = transcripts[0].text if hasattr(transcripts[0], 'text') else str(transcripts[0])
                        
                        if not final_raw.strip() and final_transcript.strip():
                            print("\n[WARNING] Mic pop squashed buffer. Sending live text to LLM...")
                            final_raw = final_transcript

                        # Sanity-check the full-buffer pass against the live rolling-window
                        # transcript. If word overlap is < 30%, NeMo hallucinated (this happens
                        # on very short recordings or after the GPU processed a different session).
                        # Fall back to the live transcript which we know was tracking the audio.
                        if final_transcript.strip() and final_raw.strip():
                            import re as _re2
                            def _words(s): return set(_re2.sub(r'[^\w\s]', '', s.lower()).split())
                            fb_words = _words(final_raw)
                            live_words = _words(final_transcript)
                            if fb_words and live_words:
                                overlap = len(fb_words & live_words) / max(len(fb_words), len(live_words))
                                if overlap < 0.3:
                                    print(f"[WARN] Full-buffer NeMo output shares only {overlap:.0%} words with live transcript — likely hallucination. Using live transcript.", file=sys.stderr)
                                    sys.stderr.flush()
                                    final_raw = final_transcript

                        probs = torch.softmax(logits, dim=-1)
                        top_probs, top_indices = torch.topk(probs, k=4, dim=-1)
                        vocab = asr_model.decoder.vocabulary
                        
                        ambiguous_notes = []
                        for t in range(logits.shape[1]):
                            top_prob = top_probs[0, t, 0].item()
                            top_idx = top_indices[0, t, 0].item()
                            
                            if top_idx < len(vocab) and top_prob < 0.85:
                                alts = []
                                for k in range(4):
                                    idx = top_indices[0, t, k].item()
                                    prob = top_probs[0, t, k].item()
                                    if idx < len(vocab) and prob > 0.01:
                                        token = vocab[idx].replace(' ', '')
                                        if token:
                                            alts.append(token)
                                if len(alts) > 1:
                                    ambiguous_notes.append(f"[{'/'.join(alts)}]")
                        
                        confidence_hint = ""
                        if ambiguous_notes:
                            confidence_hint = "The ASR model was unsure about some sounds. Here are the top alternatives it considered for the ambiguous parts (in chronological order): " + " ".join(ambiguous_notes)
 
                    visual_context = get_current_visual_context()
                    dictionary_cleaned_text = multimodal_correction(final_raw, visual_context).replace("⁇", "")

                    # NeMo sometimes returns only ⁇ tokens for non-geology speech — stripping
                    # them leaves an empty string, which bypasses the LLM and produces a blank
                    # final transcript. Fall back to the live streaming text in that case.
                    if not dictionary_cleaned_text.strip() and final_transcript.strip():
                        print("[WARN] Full-buffer transcription was empty after cleaning — using live transcript.", file=sys.stderr)
                        sys.stderr.flush()
                        dictionary_cleaned_text = final_transcript

                    sys.stdout.flush() # TEST
                    
                    final_clean_text = dictionary_cleaned_text
                    
                    if llm and dictionary_cleaned_text.strip():
                        print(f"NeMo full-buffer: {dictionary_cleaned_text}")
                        print("Running LLM formatting...")
                        sys.stdout.flush()

                        unique_id = time.time() # <--- NEW: Cache buster
                        
                        if confidence_hint:
                            print("Passing confidence notes to LLM...")
                            context_str = ", ".join(visual_context)
                            context_hint = ""
                            if context_str:
                                context_hint = (
                                    f"\nContext: the speaker is looking at a {context_str} rock."
                                    f" Silently correct any word that sounds like '{context_str}' to '{context_str}'."
                                    f" Do NOT add notes, asterisks, or explanations about the correction."
                                )
                            system_prompt = f"""<|system|>
System Time: {unique_id}
You are a robotic geology transcript formatter. Add correct capitalization and punctuation. Correct obvious ASR errors using the context below. Output ONLY the corrected sentence — no notes, no markup, no explanations.{context_hint}
{confidence_hint}<|end|>"""
                        else:
                            system_prompt = f"""<|system|>
System Time: {unique_id}
You are a robotic text formatter. Your ONLY job is to add proper capitalization and punctuation to the text.
Output ONLY the formatted text. Do not add conversational filler. Do not change the words.<|end|>"""

                        # Few-shot example for context-correction path: shows silent word fix
                        if confidence_hint and context_str:
                            prompt = f"""{system_prompt}
<|user|>
Raw ASR: i found a granit rock here<|end|>
<|assistant|>
I found a granite rock here.<|end|>
<|user|>
Transcript ID: {unique_id}
Raw ASR: {dictionary_cleaned_text}<|end|>
<|assistant|>"""
                        else:
                            prompt = f"""{system_prompt}
<|user|>
Raw ASR: i found this dark piece magnesium underground crater need record much<|end|>
<|assistant|>
I found this dark piece magnesium underground crater. Need record much.<|end|>
<|user|>
Transcript ID: {unique_id}
Raw ASR: {dictionary_cleaned_text}<|end|>
<|assistant|>"""
 
                        # Scale max_tokens to input size — a safe upper bound is
                        # 1.5x the input word count (punctuation + capitalisation
                        # add tokens but never more than ~50% overhead).
                        # Floor at 256, cap at 1024 to stay within Phi-3's n_ctx.
                        input_word_count = len(dictionary_cleaned_text.split())
                        llm_max_tokens = max(256, min(1024, int(input_word_count * 1.5)))
 
                        try:
                            response = llm(
                                prompt,
                                max_tokens=llm_max_tokens,
                                stop=["<|end|>"],
                                temperature=0.1 if not confidence_hint else 0.2
                            )

                            llm_output = response["choices"][0]["text"].strip()
                            print(f"[LLM] raw output: {repr(llm_output)}", file=sys.stderr)
                            sys.stderr.flush()

                            if "Here is the formatted text:" in llm_output:
                                llm_output = llm_output.split("Here is the formatted text:")[-1].strip()
                            elif "Here is the corrected text:" in llm_output:
                                llm_output = llm_output.split("Here is the corrected text:")[-1].strip()
                            # Strip any **Note:** / *Note:* / Note: meta-commentary Phi-3 sometimes adds
                            import re as _re_llm
                            llm_output = _re_llm.split(r'\s*\*{0,2}Note\b', llm_output, flags=_re_llm.I)[0].strip()

                            # --- CACHE CORRUPTION GUARD ---
                            import re
                            def get_words(s): return re.sub(r'[^\w\s]', '', s.lower()).split()

                            dict_words = get_words(dictionary_cleaned_text)
                            llm_words = get_words(llm_output)
                            prev_llm_words = get_words(last_llm_output)

                            asr_vs_llm = difflib.SequenceMatcher(None, dict_words, llm_words).ratio() if (dict_words and llm_words) else 1.0
                            llm_vs_prev = difflib.SequenceMatcher(None, llm_words, prev_llm_words).ratio() if (llm_words and prev_llm_words) else 0.0

                            # Threshold lowered to 0.65: a partial replay (e.g. same opening
                            # phrase with slight word differences) now triggers a reload.
                            cache_replay = (llm_vs_prev > 0.65)
                            cache_diverged = (dict_words and llm_words and asr_vs_llm < 0.4)
                            # Catch the case where LLM output matches last session more than it
                            # matches the current NeMo transcript — the hallmark of a partial
                            # cache replay that slips under the absolute threshold.
                            cache_biased = (prev_llm_words and dict_words and llm_words and
                                            llm_vs_prev > asr_vs_llm + 0.2 and llm_vs_prev > 0.45)

                            if cache_diverged or cache_replay or cache_biased:
                                reason = (f"replay of previous output (llm_vs_prev={llm_vs_prev:.2f})" if cache_replay
                                          else f"output biased toward previous session (llm_vs_prev={llm_vs_prev:.2f} > asr_vs_llm={asr_vs_llm:.2f})" if cache_biased
                                          else f"diverged from ASR (sim={asr_vs_llm:.2f})")
                                print(f"[LLM] Cache corruption detected ({reason}). Reloading LLM...", file=sys.stderr)
                                sys.stderr.flush()

                                print("[STATUS] Model rebooting due to cache interruption. Please hold on.")
                                sys.stdout.flush()

                                # Nuke and rebuild the LLM
                                try:
                                    if hasattr(llm, 'close'): llm.close()
                                except Exception: pass
                                del llm
                                import gc
                                gc.collect()

                                from llama_cpp import Llama
                                llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=4096, n_gpu_layers=0, verbose=False)

                                # Retry inference on fresh model
                                response = llm(
                                    prompt,
                                    max_tokens=llm_max_tokens,
                                    stop=["<|end|>"],
                                    temperature=0.1 if not confidence_hint else 0.2
                                )

                                llm_output = response["choices"][0]["text"].strip()
                                print(f"[LLM] raw output after reload: {repr(llm_output)}", file=sys.stderr)

                                if "Here is the formatted text:" in llm_output:
                                    llm_output = llm_output.split("Here is the formatted text:")[-1].strip()
                                elif "Here is the corrected text:" in llm_output:
                                    llm_output = llm_output.split("Here is the corrected text:")[-1].strip()

                                # After reload the fresh LLM STILL produces the same output
                                # as last time → NeMo's full-buffer pass is biased/hallucinating.
                                # Re-run the LLM with the live-streaming transcript as input instead.
                                # Live inference uses independent 4-second windows so it doesn't
                                # suffer the same long-range bias as the full-buffer NeMo pass.
                                post_reload_words = get_words(llm_output)
                                still_replay = (difflib.SequenceMatcher(None, post_reload_words, prev_llm_words).ratio() > 0.85
                                                if (post_reload_words and prev_llm_words) else False)
                                if still_replay:
                                    live_input = final_transcript.strip() or dictionary_cleaned_text
                                    print(f"[LLM] Still replaying after reload — re-running with live transcript: {repr(live_input)}", file=sys.stderr)
                                    sys.stderr.flush()
                                    live_prompt = prompt.replace(
                                        f"Raw ASR: {dictionary_cleaned_text}",
                                        f"Raw ASR: {live_input}"
                                    )
                                    live_response = llm(
                                        live_prompt,
                                        max_tokens=max(256, min(1024, int(len(live_input.split()) * 1.5))),
                                        stop=["<|end|>"],
                                        temperature=0.1
                                    )
                                    llm_output = live_response["choices"][0]["text"].strip()
                                    if "Here is the formatted text:" in llm_output:
                                        llm_output = llm_output.split("Here is the formatted text:")[-1].strip()
                                    elif "Here is the corrected text:" in llm_output:
                                        llm_output = llm_output.split("Here is the corrected text:")[-1].strip()
                                    print(f"[LLM] live-input output: {repr(llm_output)}", file=sys.stderr)
                            # ------------------------------

                            if llm_output:
                                final_clean_text = llm_output
                                last_llm_output = llm_output
                            else:
                                print("[LLM] Empty output — keeping raw ASR as final text", file=sys.stderr)
                                sys.stderr.flush()
                        except Exception as e:
                            print(f"\n[LLM ERROR] Formatting failed: {e}", file=sys.stderr)
                            print("Falling back to dictionary cleaned text.", file=sys.stderr)
                            sys.stderr.flush()
    
                    print("\n" + "="*40)
                    print("FINAL TRANSCRIPT:")
                    print("="*40)
                    print(final_clean_text)
                    print("="*40)
                else:
                    print("No audio recorded.")
                    
                print("STREAMING COMPLETE")
                sys.stdout.flush()
 
        if recording_active:
            # 1. Drain the audio queue and count total frames
            while not audio_q.empty():
                block = audio_q.get()
                block_mono = block[:, 0].reshape(1, -1)
                n_frames = block_mono.shape[1]

                total_frames_recorded += n_frames

                # Discard the first _STALE_DISCARD_FRAMES from BOTH buffers.
                # Stale frames are PulseAudio latency from the previous recording —
                # they must not reach the rolling buffer (live inference) or the
                # full_audio_buffer (final NeMo pass), or NeMo hallucinates the
                # previous session's words over the current transcript.
                if _frames_discarded < _STALE_DISCARD_FRAMES:
                    discard_n = min(n_frames, _STALE_DISCARD_FRAMES - _frames_discarded)
                    _frames_discarded += discard_n
                    block_mono = block_mono[:, discard_n:]
                    if block_mono.shape[1] == 0:
                        continue

                full_audio_buffer.append(block_mono.flatten())
                fresh_frames += block_mono.shape[1]
                rolling_buffer = np.concatenate((rolling_buffer, block_mono), axis=1)
 
            # 2. Start inference ONLY after 2.0 seconds of audio is collected
            # Gate on fresh_frames (post-stale-discard) so the first NeMo inference
            # only fires once the rolling buffer is filled with genuinely new audio.
            if fresh_frames >= int(SAMPLE_RATE * 2.0):

                # Prevent bursts: only trigger if 0.5s of NEW audio has arrived since the last inference
                if fresh_frames - last_inference_frames >= chunk_size:

                    last_inference_frames = fresh_frames
 
                    # Cap the buffer to the maximum window size (4.0s)
                    if rolling_buffer.shape[1] > window_size:
                        rolling_buffer = rolling_buffer[:, -window_size:]
 
                    current_visual_context = get_current_visual_context()
                    window_audio = rolling_buffer 
                    
                    if rms(window_audio.flatten()) < ENERGY_THRESHOLD:
                        continue
 
                    max_val = np.max(np.abs(window_audio))
                    if max_val > 0: window_audio = window_audio / (max_val + 1e-9)
                    tensor_in = torch.from_numpy(window_audio).float().to(device)
                    len_in = torch.tensor([tensor_in.shape[1]], dtype=torch.int64, device=device)
 
                    with torch.no_grad():
                        logits = asr_model.forward(input_signal=tensor_in, input_signal_length=len_in)
                        if isinstance(logits, tuple): logits = logits[0]
                        pred_tokens = logits.argmax(dim=-1)
                        transcripts = asr_model.decoding.ctc_decoder_predictions_tensor(pred_tokens)
                        raw_text = transcripts[0].text if hasattr(transcripts[0], 'text') else str(transcripts[0])
 
                    # --- THE TEXT-STABILITY FLUSHER ---
                    if raw_text == last_raw_text:
                        stable_chunks += 1
                    else:
                        stable_chunks = 0
                    last_raw_text = raw_text
 
                    raw_words = raw_text.split()
                    
                    # If the transcript is actively changing, the last word is likely half-spoken. Drop it!
                    # If the transcript has stayed EXACTLY the same for 2 loops (1.0s), the sentence is done. Keep it!
                    if len(raw_words) > 1 and stable_chunks < 2:
                        raw_words = raw_words[:-1]
                        
                    # Head Drop: If the buffer is sliding (4.0s+), the FIRST word is an artifact of the old audio cut. Drop it!
                    if fresh_frames > window_size and len(raw_words) > 1:
                        raw_words = raw_words[1:]
                        
                    raw_text = " ".join(raw_words)
 
                    live_context = get_current_visual_context()
                    corrected_text = multimodal_correction(raw_text, live_context).replace("⁇", "").strip()
                    
                    if corrected_text:
                        new_full_transcript, new_part = smart_merge(final_transcript, corrected_text)

                        if new_part:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            # Emit full transcript so the UI can replace (not append) its state.
                            # Emitting only new_part caused stale words to persist when smart_merge
                            # self-corrected an earlier error (e.g. "fourth" → "fifth").
                            print(f"[{timestamp}] {new_full_transcript}")
                            sys.stdout.flush()

                        final_transcript = new_full_transcript
                        
        else:
            # Not recording: drain and discard so the queue never accumulates
            # stale audio that would bleed into the next recording's full_audio_buffer.
            while not audio_q.empty():
                try:
                    audio_q.get_nowait()
                except queue.Empty:
                    break

        time.sleep(0.01)

except KeyboardInterrupt:
    stream.stop()
except Exception as _crash:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    sys.exit(1)