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
    "rock", "stone", "sample", "specimen", "mineral", "crystal",
    "sediment", "magma", "lava", "ash", "outcrop", "formation",
    "granite", "basalt", "gneiss", "schist", 
    "luster", "grain", "fine-grained", "coarse", "foliated",
    "vesicular", "porous", "crystalline", "volcanic",
}

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

    # --- THE FIX: Break multi-word labels into individual words ---
    expanded_context = []
    for phrase in visual_keywords:
        # Splits "basalt pigeonite" into ["basalt", "pigeonite"]
        expanded_context.extend(phrase.split())
    
    # Remove duplicates (e.g., if context was ["basalt", "basalt pigeonite"])
    expanded_context = list(set(expanded_context))

    blob = TextBlob(transcript)
    blob_tags = blob.tags
    words = transcript.split()
    corrected_words = []
    
    for i, word in enumerate(words):
        best_candidate = word
        clean_word = word.lower().strip(".,?!")
        current_tag = blob_tags[i][1] if i < len(blob_tags) else "XX"
        prev_tag = blob_tags[i-1][1] if i > 0 and i-1 < len(blob_tags) else "XX"
        
        # We now loop over the split words ("basalt", "pigeonite") instead of the full phrase
        for visual_context in expanded_context:
            context_clean = visual_context.lower()
            sounds_alike = (jellyfish.metaphone(clean_word) == jellyfish.metaphone(context_clean))
            spelling_score = jellyfish.jaro_winkler_similarity(clean_word, context_clean)
            is_match = (sounds_alike or spelling_score > 0.85)
            
            if is_match:
                if i+1 < len(words):
                    next_word_clean = words[i+1].lower().strip(".,?!")
                    next_sounds_alike = (jellyfish.metaphone(next_word_clean) == jellyfish.metaphone(context_clean))
                    if current_tag.startswith("JJ") and next_sounds_alike:
                        best_candidate = word 
                        break
                if not current_tag.startswith("JJ") or has_geology_context(words, i):
                    # Uses the single word (e.g., "basalt") instead of the whole phrase
                    best_candidate = visual_context 
                    break
                if current_tag.startswith("JJ") and prev_tag.startswith("JJ"):
                    best_candidate = visual_context
                    break
                    
        corrected_words.append(best_candidate)
        
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

# ------------------------------------------------------------------
# PART 3: LIVE INFERENCE LOGIC
# ------------------------------------------------------------------
def rms(x):
    return np.sqrt(np.mean(np.square(x.astype(np.float64))))

def smart_merge(final_text, new_chunk):
    # Clean up lingering spaces and handle empty strings
    final_text = final_text.strip()
    new_chunk = new_chunk.strip()
    
    if not final_text: return new_chunk
    if not new_chunk: return final_text
    
    # Restrict the search zone to the last 80 characters
    search_zone = final_text[-80:] if len(final_text) > 80 else final_text
    
    # --- RULE 3: The Ghost Tail Filter ---
    if new_chunk in search_zone:
        return final_text
    
    # Find the Longest Common Substring (The Nucleus)
    s = difflib.SequenceMatcher(None, search_zone, new_chunk)
    match = s.find_longest_match(0, len(search_zone), 0, len(new_chunk))
    
    # --- RULE 2: Apply the 4-Character Nucleus Merge ---
    if match.size >= 4:
        # 1. Grab the clean nucleus
        overlap_str = search_zone[match.a : match.a + match.size]
        
        # 2. Grab everything in the OLD text BEFORE the nucleus
        absolute_chop = len(final_text) - len(search_zone) + match.a
        prefix = final_text[:absolute_chop]
        
        # 3. Grab everything in the NEW text AFTER the nucleus
        # This automatically deletes garbage like the "g" or "t" at the start!
        suffix = new_chunk[match.b + match.size:]
        
        return prefix + overlap_str + suffix
        
    else:
        # No valid nucleus found, just append it normally
        return final_text + " " + new_chunk
        
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
                # PART 5: FINAL CLEANUP (RESTORED)
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
                            # The massive single-pass transcription
                            preds = asr_model.transcribe([full_audio])
                            final_raw = preds[0].text if hasattr(preds[0], 'text') else str(preds[0])
                        except:
                            logits = asr_model.forward(input_signal=full_signal, input_signal_length=full_len)
                            if isinstance(logits, tuple): logits = logits[0]
                            pred_tokens = logits.argmax(dim=-1)
                            transcripts = asr_model.decoding.ctc_decoder_predictions_tensor(pred_tokens)
                            final_raw = transcripts[0].text

                    # Fetch the live context one last time for the final sweep
                    final_clean_text = multimodal_correction(final_raw, get_current_visual_context()).replace("⁇", "")
    
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

                # 1. Grab the live context from the file
                live_context = get_current_visual_context()
                
                # 2. Feed it into your correction brain
                corrected_text = multimodal_correction(raw_text, live_context)
                
                # --- RULE 1: Destroy the weird question marks ---
                corrected_text = corrected_text.replace("⁇", "").strip()
                
                if corrected_text:
                    # --- RULE 2: Mash overlapping text together ---
                    new_full_transcript = smart_merge(final_transcript, corrected_text)
                    
                    if len(new_full_transcript) > len(final_transcript):
                        # Only grab the completely new words to send to the UI
                        new_part = new_full_transcript[len(final_transcript):].strip()
                        if new_part:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            print(f"[{timestamp}] {new_part}")
                            sys.stdout.flush()
                            
                    final_transcript = new_full_transcript
                rolling_buffer = rolling_buffer[:, chunk_size:]
        time.sleep(0.01)

except KeyboardInterrupt:
    stream.stop()