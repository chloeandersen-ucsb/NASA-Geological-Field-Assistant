import time
app_start_time = time.time()

import sounddevice as sd
import nemo.collections.asr as nemo_asr
import numpy as np
import queue
import sys
import torch
import os
import difflib
import jellyfish 
from datetime import datetime
from spellchecker import SpellChecker
from textblob import TextBlob

# ------------------------------------------------------------------
# PART 1: CONFIGURATION
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# PART 1: CONFIGURATION
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, "newest_model.nemo")

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
    return ["Gneiss"] # Default fallback

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5        
WINDOW_DURATION = 2.5       
ENERGY_THRESHOLD = 0.001    
PRINT_SILENCE = True

spell = SpellChecker()

GEOLOGY_TRIGGERS = { 
    "rock", "sample", "specimen", "mineral", "crystal", "crystals",
    "sediment", "lava", "ash", "crater",
    "granite", "basalt", "gneiss", "schist", "olivine", "magnesium", "underground",
    "luster", "fine-grained", "coarse", "foliated",
    "vesicular", "porous", "crystalline", "volcanic", "space", "piece"
}

spell.word_frequency.load_words(list(GEOLOGY_TRIGGERS))

def has_geology_context(words, index, window=3):
    start = max(0, index - window)
    end = min(len(words), index + window + 1)
    nearby_words = words[start:index] + words[index+1:end]
    for w in nearby_words:
        clean_w = w.lower().strip(".,?!")
        if clean_w in GEOLOGY_TRIGGERS:
            return True
    return False

def multimodal_correction(transcript, visual_keywords):
    if not transcript: return ""

    # 1. MASTER TARGET LIST
    targets = []
    for phrase in visual_keywords:
        targets.append(phrase.lower())
        targets.extend(phrase.lower().split())
    targets.extend(list(GEOLOGY_TRIGGERS))
    targets = list(set(targets))

    # --- THE SAFE STUTTER & GARBAGE FILTER ---
    raw_words = transcript.split()
    words = []
    for w in raw_words:
        clean = w.lower().strip(".,?!")
        # Kill consecutive model stutters (e.g., "magnesium magnesium")
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
        for window_size in [3, 2, 1]:
            if i + window_size <= len(words):
                chunk = "".join(words[i:i+window_size]).lower().strip(".,?!")
                
                for target in targets:
                    target_squished = target.replace(" ", "")
                    spelling_score = jellyfish.jaro_winkler_similarity(chunk, target_squished)
                    sounds_alike = (jellyfish.metaphone(chunk) == jellyfish.metaphone(target_squished))
                    
                    # Safe threshold! No more "outcrop" hallucinations.
                    threshold = 0.88 if len(target_squished) <= 4 else 0.78
                    
                    if sounds_alike or spelling_score > threshold:
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
            clean_w = words[i].lower().strip(".,?!")
            if clean_w and clean_w not in spell:
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

print(f"Loading fine-tuned model: {MODEL_FILE}...")
# map_location=device is the ONLY way this works on your MacBook
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    restore_path=MODEL_FILE, 
    map_location=device
)
asr_model.freeze()
asr_model = asr_model.to(device)
asr_model.eval()
print("Fine-tuned model loaded successfully!")
sys.stdout.flush()

# --- NEW: BOOT THE LOCAL LLM ---
from llama_cpp import Llama

print("Loading Phi-3 LLM for post-processing...")
# Adjust the model_path to wherever you keep your .gguf files
LLM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Phi-3-mini-4k-instruct-q4.gguf")

try:
    # n_gpu_layers=-1 offloads it to the Jetson's GPU for maximum speed
    llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=512, n_gpu_layers=-1, verbose=False)
    print("LLM loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load LLM. Falling back to standard correction. Error: {e}")
    llm = None
sys.stdout.flush()

# ------------------------------------------------------------------
# PART 3: LIVE INFERENCE LOGIC
# ------------------------------------------------------------------
def rms(x):
    return np.sqrt(np.mean(np.square(x.astype(np.float64))))

def smart_merge(final_text, new_chunk):
    """Now returns a TUPLE: (full_merged_text, entirely_new_suffix)"""
    final_text = final_text.strip()
    new_chunk = new_chunk.strip()
    
    if not final_text: return new_chunk, new_chunk
    if not new_chunk: return final_text, ""

    old_words = final_text.split()
    new_words = new_chunk.split()

    # --- 1. THE STUTTER KILLER ---
    # Instantly drop exact duplicate words passing between chunks
    if old_words and new_words and old_words[-1].lower() == new_words[0].lower():
        new_words = new_words[1:]
        if not new_words: return final_text, ""

    # --- 2. LIST-BASED SEQUENCE MATCHER ---
    # Grab the last 15 words to search for the overlap
    search_window = old_words[-15:]
    
    # difflib works on lists of strings! It will find the longest sequence of identical words.
    s = difflib.SequenceMatcher(None, [w.lower() for w in search_window], [w.lower() for w in new_words])
    match = s.find_longest_match(0, len(search_window), 0, len(new_words))
    
    # If we find an overlap of at least 2 words (or 1 word if the new audio chunk is tiny)
    if match.size >= 2 or (match.size == 1 and len(new_words) <= 3):
        absolute_chop_idx = len(old_words) - len(search_window) + match.a
        
        # Splicing: Old text up to the match + the ENTIRE new chunk starting from the match
        merged_words = old_words[:absolute_chop_idx] + new_words[match.b:]
        merged_text = " ".join(merged_words)
        
        # Return only the genuinely new text to print to the screen
        new_suffix = " ".join(new_words[match.b + match.size:])
        return merged_text, new_suffix
    else:
        # PANIC PROTOCOL: Stream moves safely forward without repeating massive blocks
        safe_append = " ".join(new_words[-1:])
        return final_text + " " + safe_append, safe_append
        
# ------------------------------------------------------------------
# PART 4: MAIN LOOP
# ------------------------------------------------------------------
audio_q = queue.Queue()
full_audio_buffer = []

def audio_callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())

# Check for Sennheiser or default
DEVICE_INDEX = None
for i, dev in enumerate(sd.query_devices()):
    if 'Sennheiser' in dev['name']:
        DEVICE_INDEX = i
        break
if DEVICE_INDEX is None:
    DEVICE_INDEX = sd.default.device[0]

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, device=DEVICE_INDEX, callback=audio_callback)

chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
window_size = int(SAMPLE_RATE * WINDOW_DURATION)
rolling_buffer = np.zeros((1, 0), dtype=np.float32)
final_transcript = ""
recording_active = False

try:
    while True:
        import select
        if select.select([sys.stdin], [], [], 0.01)[0]:
            raw_line = sys.stdin.readline()
            if not raw_line: break
            line = raw_line.strip().lower()
            
            if line == "start":
                recording_active = True
                full_audio_buffer = []
                rolling_buffer = np.zeros((1,0), dtype=np.float32)
                final_transcript = ""
                stream.start()
                print("\nRECORDING NOW... (Fine-tuned Model)\n")
                sys.stdout.flush()
            elif line == "stop":
                stream.stop()
                recording_active = False
                print("\n\nStopping stream... Preparing final transcript...")
                
                # ------------------------------------------------------------------
                # PART 5: FINAL CLEANUP (LLM-POWERED)
                # ------------------------------------------------------------------
                if full_audio_buffer:
                    print("Processing full audio buffer for maximum accuracy...")
                    full_audio = np.concatenate(full_audio_buffer, axis=0).astype(np.float32)
                    max_val = np.max(np.abs(full_audio))
                    if max_val > 0: full_audio = full_audio / (max_val + 1e-9)
                    full_signal = torch.tensor(full_audio, dtype=torch.float32, device=device).unsqueeze(0)
                    full_len = torch.tensor([full_signal.shape[1]], dtype=torch.int64, device=device)

                    with torch.no_grad():
                        try:
                            # 1. Get the massive single-pass transcription from NeMo
                            preds = asr_model.transcribe([full_audio])
                            final_raw = preds[0].text if hasattr(preds[0], 'text') else str(preds[0])
                        except:
                            logits = asr_model.forward(input_signal=full_signal, input_signal_length=full_len)
                            if isinstance(logits, tuple): logits = logits[0]
                            pred_tokens = logits.argmax(dim=-1)
                            transcripts = asr_model.decoding.ctc_decoder_predictions_tensor(pred_tokens)
                            final_raw = transcripts[0].text

                    # 2. LLM Post-Processing Layer
                    llm_corrected_text = final_raw
                    visual_context = get_current_visual_context()[0]
                    
                    if llm and final_raw.strip():
                        print("Running LLM spellcheck...")
                        
                        # Phi-3 strict User-Block Prompt
                        prompt = f"""<|user|>
You are a robotic text formatter. Your ONLY job is to add capitalization and punctuation.
You must output the EXACT same words in the EXACT same order. 
DO NOT delete words. DO NOT add words. DO NOT fix grammar. 
If a word is a clear phonetic typo of a geology term related to "{visual_context}", you may correct its spelling.

Example Raw: i found this dark piece magnesium underground crater need record much
Example Fixed: I found this dark piece of magnesium underground a crater. I need to record more.

Raw ASR: {final_raw}
Fixed ASR:<|end|>
<|assistant|>"""

                        response = llm(
                            prompt, 
                            max_tokens=150, 
                            stop=["<|end|>", "\n"], 
                            temperature=0.1 # Low temperature so it doesn't get "creative"
                        )
                        
                        llm_output = response["choices"][0]["text"].strip()
                        if llm_output:
                            llm_corrected_text = llm_output

                    # 3. Final Jaro-Winkler Dictionary Sweep
                    final_clean_text = multimodal_correction(llm_corrected_text, get_current_visual_context()).replace("⁇", "")
    
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
            while not audio_q.empty():
                block = audio_q.get()
                block_mono = block[:, 0].reshape(1, -1)
                rolling_buffer = np.concatenate((rolling_buffer, block_mono), axis=1)
                full_audio_buffer.append(block_mono.flatten())

            if rolling_buffer.shape[1] >= window_size:
                current_visual_context = get_current_visual_context()

                window_audio = rolling_buffer[:, :window_size]
                if rms(window_audio.flatten()) < ENERGY_THRESHOLD:
                    rolling_buffer = rolling_buffer[:, chunk_size:]
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

                # --- NEW: THE UNSTABLE TAIL DROP ---
                # The last word in a rolling window is almost always physically chopped in half.
                # We drop it here. In the next 0.5s loop, it will be fully inside the window and transcribe perfectly.
                raw_words = raw_text.split()
                if len(raw_words) > 2:
                    raw_text = " ".join(raw_words[:-2])
                # -----------------------------------

                # 1. Grab the live context from the file
                live_context = get_current_visual_context()
                
                # 2. Feed it into your correction brain
                corrected_text = multimodal_correction(raw_text, live_context)
                
                # --- RULE 1: Destroy the weird question marks ---
                corrected_text = corrected_text.replace("⁇", "").strip()
                
                if corrected_text:
                    # Unpack the tuple!
                    new_full_transcript, new_part = smart_merge(final_transcript, corrected_text)
                    
                    if new_part:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        print(f"[{timestamp}] {new_part}")
                        sys.stdout.flush()
                            
                    final_transcript = new_full_transcript
                rolling_buffer = rolling_buffer[:, chunk_size:]
        time.sleep(0.01)

except KeyboardInterrupt:
    stream.stop()