import time
app_start_time = time.time() # Capture the exact moment the script starts

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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, "newest_model.nemo")
CURRENT_VISUAL_CONTEXT = ["Gneiss"] 

# Audio Tunables
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5        
WINDOW_DURATION = 2.5       # Keeps context long enough to hear "nice rock"
ENERGY_THRESHOLD = 0.001    
PRINT_SILENCE = True

# Context Helpers
spell = SpellChecker()

GEOLOGY_TRIGGERS = { 
    "rock", "stone", "sample", "specimen", "mineral", "crystal",
    "sediment", "magma", "lava", "ash", "outcrop", "formation",
    "granite", "basalt", "gneiss", "schist", 
    "luster", "grain", "fine-grained", "coarse", "foliated",
    "vesicular", "porous", "crystalline", "volcanic",
}

def has_geology_context(words, index, window=3):
    """Checks for geology trigger words nearby."""
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
    
    blob = TextBlob(transcript)
    # Get tags: [('nice', 'JJ'), ('Gneiss', 'NN'), ('rock', 'NN')]
    blob_tags = blob.tags
    words = transcript.split()
    corrected_words = []
    
    for i, word in enumerate(words):
        best_candidate = word
        clean_word = word.lower().strip(".,?!")
        
        # Get tags for current and surrounding words
        current_tag = blob_tags[i][1] if i < len(blob_tags) else "XX"
        prev_tag = blob_tags[i-1][1] if i > 0 and i-1 < len(blob_tags) else "XX"
        next_tag = blob_tags[i+1][1] if i+1 < len(blob_tags) else "XX"

        for visual_context in visual_keywords:
            context_clean = visual_context.lower()
            
            # Phonetic match
            sounds_alike = (jellyfish.metaphone(clean_word) == jellyfish.metaphone(context_clean))
            spelling_score = jellyfish.jaro_winkler_similarity(clean_word, context_clean)
            is_match = (sounds_alike or spelling_score > 0.85)

            if is_match:
                # --- NEW REFINED LOGIC ---
                
                # 1. If it's a "Nice Gneiss" situation:
                # If THIS word is an adjective (JJ) AND the NEXT word is a noun (NN) 
                # that also sounds like Gneiss, keep this one as "nice".
                if i+1 < len(words):
                    next_word_clean = words[i+1].lower().strip(".,?!")
                    next_sounds_alike = (jellyfish.metaphone(next_word_clean) == jellyfish.metaphone(context_clean))
                    if current_tag.startswith("JJ") and next_sounds_alike:
                        best_candidate = word # Keep as "nice"
                        break

                # 2. If it is clearly a Noun or has "rock/sample" context, swap to Gneiss
                if not current_tag.startswith("JJ") or has_geology_context(words, i):
                    best_candidate = visual_context
                    break
                
                # 3. Double Adjective catch: "nice nice rock" -> "nice Gneiss rock"
                if current_tag.startswith("JJ") and prev_tag.startswith("JJ"):
                    best_candidate = visual_context
                    break
        
        corrected_words.append(best_candidate)
        
    return " ".join(corrected_words)

# ------------------------------------------------------------------
# PART 2: DEVICE & MODEL SETUP
# ------------------------------------------------------------------
if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")

if not hasattr(torch, "distributed"):
    import types
    torch.distributed = types.SimpleNamespace()

if not hasattr(torch.distributed, "is_initialized"):
    torch.distributed.is_initialized = lambda: False

DEVICE_INDEX = None
print("Searching for Sennheiser XS LAV...")
for i, dev in enumerate(sd.query_devices()):
    if 'Sennheiser XS LAV USB-C' in dev['name']:
        DEVICE_INDEX = i
        break
if DEVICE_INDEX is None:
    print("Warning: Sennheiser LAV not found. Using default input.")
    DEVICE_INDEX = sd.default.device[0]

if not os.path.exists(MODEL_FILE):
    print(f"ERROR: Could not find {MODEL_FILE}")
    exit(1)

print(f"Loading local model: {MODEL_FILE}...")
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=MODEL_FILE)
asr_model.freeze()
asr_model = asr_model.to(device)
asr_model.eval()
print("Model loaded successfully!")

# ------------------------------------------------------------------
# PART 3: SMART MERGE HELPER
# ------------------------------------------------------------------
def rms(x):
    return np.sqrt(np.mean(np.square(x.astype(np.float64))))

def smart_merge(final_text, new_chunk):
    """
    Intelligently merges a new live chunk into the final text.
    Handles 'unstable' tails (e.g., 'night' -> 'nice').
    """
    final_text = final_text.strip()
    new_chunk = new_chunk.strip()
    
    if not final_text: return new_chunk
    if not new_chunk: return final_text
    
    final_words = final_text.split()
    new_words = new_chunk.split()
    
    # 1. Try to find the start of 'new_chunk' inside 'final_text'
    # We look at the last 6 words of final_text to find overlap
    lookback = min(len(final_words), 10)
    search_zone = final_words[-lookback:]
    
    best_overlap_index = -1
    
    # Check if the start of new_words exists in search_zone
    # We require at least 2 words to match to be confident, 
    # OR 1 word if it's a very short chunk.
    min_match = 2 if len(new_words) > 1 else 1
    
    for i in range(len(search_zone)):
        # Construct a potential overlap from search_zone[i:]
        potential_overlap = search_zone[i:]
        len_overlap = len(potential_overlap)
        
        # Does new_words START with this overlap?
        if len(new_words) >= len_overlap:
            if new_words[:len_overlap] == potential_overlap:
                best_overlap_index = i
                break
    
    # 2. If overlap found, stitch them
    if best_overlap_index != -1:
        # Calculate where in the absolute final_words list the match happened
        absolute_index = (len(final_words) - lookback) + best_overlap_index
        
        # We assume the new_chunk is the "truth" and overwrite the tail
        # Take everything BEFORE the overlap from old text
        prefix = final_words[:absolute_index]
        return " ".join(prefix + new_words)
    
    # 3. If no overlap, just append (fallback)
    return final_text + " " + new_chunk


# ------------------------------------------------------------------
# PART 4: MAIN LOOP
# ------------------------------------------------------------------
audio_q = queue.Queue()
full_audio_buffer = []

def audio_callback(indata, frames, time_info, status):
    if status: print(status, file=sys.stderr)
    audio_q.put(indata.copy())

chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
window_size = int(SAMPLE_RATE * WINDOW_DURATION)
rolling_buffer = np.zeros((1, 0), dtype=np.float32)

startup_duration = time.time() - app_start_time

print(f"\n{'='*40}")
print(f"STARTUP TIME: {startup_duration:.2f} seconds")
print(f"{'='*40}\n")

print("\nRECORDING... (Ctrl+C to stop)\n")
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, device=DEVICE_INDEX, callback=audio_callback)

last_printed_was_silence = False
final_transcript = ""

try:
    stream.start()
    while True:
        while not audio_q.empty():
            block = audio_q.get()
            block_mono = block[:, 0].reshape(1, -1)
            rolling_buffer = np.concatenate((rolling_buffer, block_mono), axis=1)
            full_audio_buffer.append(block_mono.flatten())

        if rolling_buffer.shape[1] >= window_size:
            window_audio = rolling_buffer[:, :window_size]
            energy = rms(window_audio.flatten())
            timestamp = datetime.now().strftime("%H:%M:%S")

            if energy < ENERGY_THRESHOLD:
                if PRINT_SILENCE and not last_printed_was_silence:
                    print(f"[{timestamp}] (silence)")
                    sys.stdout.flush()
                    last_printed_was_silence = True
                rolling_buffer = rolling_buffer[:, chunk_size:]
                continue

            # Decode
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

            # Correct
            corrected_text = multimodal_correction(raw_text, CURRENT_VISUAL_CONTEXT)
            corrected_text = corrected_text.replace("⁇", "")

            # Merge & Print
            if corrected_text:
                last_printed_was_silence = False
                
                # A. Smart Merge (Handles 'night' -> 'nice' replacement)
                new_full_transcript = smart_merge(final_transcript, corrected_text)
                
                # B. Determine what is NEW
                # If the text got shorter (rewrite), print the whole new chunk? 
                # Or just print the difference.
                if len(new_full_transcript) > len(final_transcript):
                    # Simple append case
                    new_part = new_full_transcript[len(final_transcript):].strip()
                    if new_part:
                        print(f"[{timestamp}] {new_part}")
                        sys.stdout.flush()
                elif new_full_transcript != final_transcript:
                    # Replacement case (Text changed but didn't necessarily grow)
                    # We can't "unprint" in the console easily, so we just print the fix
                    # For the GUI, this might result in "night nice", but it's better than duplication.
                    pass 
                
                final_transcript = new_full_transcript

            rolling_buffer = rolling_buffer[:, chunk_size:]

        time.sleep(0.01)

except KeyboardInterrupt:
    stream.stop()
    print("\n\nStopping stream... Preparing final transcript...")

# ------------------------------------------------------------------
# PART 5: FINAL CLEANUP
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
            preds = asr_model.transcribe([full_audio])
            final_raw = preds[0].text if hasattr(preds[0], 'text') else str(preds[0])
        except:
            logits = asr_model.forward(input_signal=full_signal, input_signal_length=full_len)
            if isinstance(logits, tuple): logits = logits[0]
            pred_tokens = logits.argmax(dim=-1)
            transcripts = asr_model.decoding.ctc_decoder_predictions_tensor(pred_tokens)
            final_raw = transcripts[0].text

    final_clean_text = multimodal_correction(final_raw, CURRENT_VISUAL_CONTEXT).replace("⁇", "")
    
    print("\n" + "="*40)
    print("FINAL TRANSCRIPT:")
    print("="*40)
    print(final_clean_text)
    print("="*40)
else:
    print("No audio recorded.")
