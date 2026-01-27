import sounddevice as sd
import nemo.collections.asr as nemo_asr
import numpy as np
import queue
import time
from datetime import datetime
import torch
import os
import difflib
import jellyfish 
from spellchecker import SpellChecker # NEW

spell = SpellChecker() # NEW
GEOLOGY_TRIGGERS = { # NEW
    # --- KEEP THESE (Strong Nouns) ---
    "rock", "stone", "sample", "specimen", "mineral", "crystal",
    "sediment", "magma", "lava", "ash", "outcrop", "formation",
    "granite", "basalt", "gneiss", "schist", # Add common rock types as triggers too
    
    # --- KEEP THESE (Specific Adjectives) ---
    "luster", "grain", "fine-grained", "coarse", "foliated",
    "vesicular", "porous", "crystalline", "volcanic",
    
    # --- REMOVE THESE (Too Generic - They cause false positives) ---
    # "hard", "soft", "shiny", "color", "texture", 
    # "look", "observing", "holding", "beautiful", "covered", "surface"
}

def has_geology_context(words, index, window=3): # NEW
    """
    Looks at words before and after the target index.
    Returns True if a 'Geology Trigger' word is found nearby.
    """
    start = max(0, index - window)
    end = min(len(words), index + window + 1)
    
    nearby_words = words[start:index] + words[index+1:end]
    
    for w in nearby_words:
        # Check if the root of the word is in our trigger list
        # (Simple check: is 'rock' in 'rocks'?)
        clean_w = w.lower().strip(".,?!")
        if clean_w in GEOLOGY_TRIGGERS:
            return True
            
    return False

# ------------------------------
# Configuration
# ------------------------------
MODEL_FILE = "geology_model2-2-2.nemo"

CURRENT_VISUAL_CONTEXT = ["Gneiss"] 

# ------------------------------
# Device Setup
# ------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: mps (Mac GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: cuda")
else:
    device = torch.device("cpu")
    print("Using device: cpu")

if not hasattr(torch, "distributed"):
    import types
    torch.distributed = types.SimpleNamespace(is_initialized=lambda: False)
elif not hasattr(torch.distributed, "is_initialized"):
    torch.distributed.is_initialized = lambda: False

# ------------------------------
# Load Model
# ------------------------------
if not os.path.exists(MODEL_FILE):
    print(f"ERROR: Could not find {MODEL_FILE}")
    exit(1)

print(f"Loading local model: {MODEL_FILE}...")
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=MODEL_FILE)
asr_model.freeze()
asr_model = asr_model.to(device)
asr_model.eval()
print("Model loaded successfully!")

# ------------------------------
# TUNED PARAMETERS
# ------------------------------
sr = 16000
chunk_duration = 0.5        # Fast updates (0.5s)
window_duration = 2.5       # 2.5s window (Good balance of context vs speed)
chunk_size = int(sr * chunk_duration)
window_size = int(sr * window_duration)

audio_queue = queue.Queue()
rolling_buffer = np.zeros((1, 0), dtype=np.float32)

# ------------------------------
# Helper Functions
# ------------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def decode_tensor(tensor):
    with torch.no_grad():
        signal_lengths = torch.tensor([tensor.shape[0]]).to(device)
        logits = asr_model.forward(input_signal=tensor.unsqueeze(0), input_signal_length=signal_lengths)
        if isinstance(logits, tuple): logits = logits[0]
        pred_tokens = logits.argmax(dim=-1)
        transcripts = asr_model.decoding.ctc_decoder_predictions_tensor(pred_tokens)
        return transcripts[0].text

def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# def multimodal_correction(transcript, context_words):
#     if not transcript: return ""
    
#     words = transcript.split()
#     corrected_words = []
    
#     for word in words:
#         best_candidate = word.lower().strip(".,?!")
        
#         # Check against our visual keywords (e.g., "Quartz")
#         for context in context_words:
#             # Check 1: Do they sound alike? (Metaphone)
#             # "Quartz" -> KRTS, "Quart" -> KRT
#             if jellyfish.metaphone(word) == jellyfish.metaphone(context):
#                 best_candidate = context
#                 break
            
#             # Check 2: Are they spelled almost identically? (Levenshtein)
#             # Fixes "Quarz" -> "Quartz" (1 char diff)
#             if jellyfish.levenshtein_distance(word.lower(), context.lower()) <= 2:
#                 # Extra check: Don't replace short words like "at" with "Bat"
#                 if len(word) > 3: 
#                     best_candidate = context
#                     break
        
#         corrected_words.append(best_candidate)
        
#     return " ".join(corrected_words)

def multimodal_correction(transcript, visual_keywords):
    if not transcript: return ""
    
    words = transcript.split()
    corrected_words = []
    
    for i, word in enumerate(words):
        best_candidate = word
        clean_word = word.lower().strip(".,?!")
        
        # Check this word against our visual cheat sheet (e.g., "Quartz")
        for visual_context in visual_keywords:
            context_clean = visual_context.lower()
            
            # 1. PHONETIC CHECK: Does it sound like the rock?
            sounds_alike = (jellyfish.metaphone(clean_word) == jellyfish.metaphone(context_clean))
            
            # 2. SPELLING CHECK: Is it spelled similarly?
            spelling_score = jellyfish.jaro_winkler_similarity(clean_word, context_clean)
            
            # --- THE NEW "CONTEXT GATE" ---
            
            # Condition A: It's a "Bad" word (not in dictionary)
            # If it's gibberish (e.g. "quar"), we are aggressive.
            # We don't even need context; just fix it.
            if clean_word not in spell and (sounds_alike or spelling_score > 0.8):
                best_candidate = visual_context
                break

            # Condition B: It's a "Good" word (e.g. "Courts" or "Quart")
            # This is where we apply the new Context Logic.
            elif clean_word in spell:
                
                # Only swap if it sounds/looks very close...
                if (sounds_alike or spelling_score > 0.9):
                    
                    # ...AND we see a "Geology Trigger" nearby!
                    if has_geology_context(words, i):
                        best_candidate = visual_context
                        break
                    else:
                        # "Judicial Courts" -> No geology context nearby.
                        # Do NOT swap.
                        pass

        corrected_words.append(best_candidate)
        
    return " ".join(corrected_words)

# ------------------------------
# Main Loop
# ------------------------------
print("\nPreparing microphone...")
try:
    stream = sd.InputStream(channels=1, samplerate=sr, callback=audio_callback)
    stream.start()
    print("RECORDING... (Ctrl+C to stop)\n")
    
    last_text = ""
    
    while True:
        while not audio_queue.empty():
            data = audio_queue.get()
            rolling_buffer = np.concatenate((rolling_buffer, data.T), axis=1)

        if rolling_buffer.shape[1] >= window_size:
            # 1. Grab window & Decode
            window_audio = rolling_buffer[:, :window_size]
            waveform_tensor = torch.from_numpy(window_audio).float().squeeze(0).to(device)
            current_text = decode_tensor(waveform_tensor).strip()

            current_text = multimodal_correction(current_text, CURRENT_VISUAL_CONTEXT) 
            
            timestamp = datetime.now().strftime("%H:%M:%S")

            # --- THE "GROW ONLY" FILTER ---
            
            # Case 1: Same text? Do nothing.
            if current_text == last_text:
                pass
            
            # Case 2: Growing? (e.g., "We are" -> "We are currently")
            # We check if the new text contains the old text OR is longer and very similar
            elif len(current_text) > len(last_text) and (last_text in current_text or similarity(last_text, current_text) > 0.6):
                print(f"[{timestamp}] {current_text}")
                last_text = current_text
                
            # Case 3: Totally new context? (Window shifted completely)
            # If the text is different AND there is very little overlap, assume it's a new line
            elif similarity(last_text, current_text) < 0.4 and len(current_text) > 5:
                 print(f"[{timestamp}] {current_text}")
                 last_text = current_text
                 
            # Case 4: Getting Shorter? (The Glitch)
            # If it gets shorter but is still similar (e.g. "Gneiss" -> "Nice"), IGNORE IT.
            # We keep 'last_text' as the memory of the "good" version.
            else:
                pass 

            # Slide window
            rolling_buffer = rolling_buffer[:, chunk_size:]

        time.sleep(0.01)

except KeyboardInterrupt:
    stream.stop()
    print("\nDone.")
