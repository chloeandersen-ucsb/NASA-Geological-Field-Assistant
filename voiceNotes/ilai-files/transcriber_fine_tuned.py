'''
- Last Updated: 02/03/2026
- Tried adding punctuation for final transcript. Didn't work well (see # FORMAT)
- Smart grammar ("nice Gneiss rock")
- Merged cleanup & newest model

- Tasks:
- TEST ON ORIN
'''
import sounddevice as sd
import nemo.collections.asr as nemo_asr
import numpy as np
import queue
import time
import sys
import torch
import os
import difflib
import jellyfish 
from datetime import datetime
from spellchecker import SpellChecker
from textblob import TextBlob
# from deepmultilingualpunctuation import PunctuationModel # FORMAT

# ------------------------------------------------------------------
# PART 1: CONFIGURATION & GEOLOGY CONTEXT
# ------------------------------------------------------------------
MODEL_FILE = "newest_model.nemo"
CURRENT_VISUAL_CONTEXT = ["Gneiss"] 

# Audio Tunables
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5        # Fast updates
WINDOW_DURATION = 2.0       # 2.0s window (Good balance for context)
ENERGY_THRESHOLD = 0.001    # RMS threshold for silence
PRINT_SILENCE = True

# Context Helpers
spell = SpellChecker()

GEOLOGY_TRIGGERS = { 
    # Strong Nouns
    "rock", "stone", "sample", "specimen", "mineral", "crystal",
    "sediment", "magma", "lava", "ash", "outcrop", "formation",
    "granite", "basalt", "gneiss", "schist", 
    # Specific Adjectives
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
    
    # 1. Get Grammar Tags for the whole sentence
    # TextBlob will label each word: ('very', 'RB'), ('nice', 'JJ'), ('rock', 'NN')
    blob = TextBlob(transcript)
    blob_tags = [tag for word, tag in blob.tags]
    
    words = transcript.split()
    corrected_words = []
    
    for i, word in enumerate(words):
        best_candidate = word
        clean_word = word.lower().strip(".,?!")
        
        # --- SMART LOGIC: Part-of-Speech Check ---
        # If the word is "nice" and it's being used as an ADJECTIVE (JJ), 
        # trust the grammar and don't turn it into a rock.
        # current_tag = blob_tags.get(clean_word, "")
        current_tag = blob_tags[i]
        prev_tag = blob_tags[i-1] if i > 0 and i-1 < len(blob_tags) else "XX"
        
        if current_tag.startswith("JJ"):
             corrected_words.append(word)
             continue
        # -----------------------------------------

        for visual_context in visual_keywords:
            context_clean = visual_context.lower()
            
            # Phonetic & Spelling checks (Same as before)
            sounds_alike = (jellyfish.metaphone(clean_word) == jellyfish.metaphone(context_clean))
            spelling_score = jellyfish.jaro_winkler_similarity(clean_word, context_clean)
            is_match = (sounds_alike or spelling_score > 0.9)

            if is_match:
                if prev_tag.startswith("RB"):
                    break
                if prev_tag.startswith("JJ"):
                    best_candidate = visual_context
                    break
                if has_geology_context(words, i):
                    best_candidate = visual_context
                    break
        
        corrected_words.append(best_candidate)
        
    return " ".join(corrected_words)

# ------------------------------------------------------------------
# PART 2: DEVICE & MODEL SETUP (Merged)
# ------------------------------------------------------------------

# 1. Setup Torch Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if not hasattr(torch, "distributed"):
    import types
    torch.distributed = types.SimpleNamespace(is_initialized=lambda: False)
elif not hasattr(torch.distributed, "is_initialized"):
    torch.distributed.is_initialized = lambda: False

# 2. Find Mic
DEVICE_INDEX = None
print("Searching for Sennheiser XS LAV...")
for i, dev in enumerate(sd.query_devices()):
    if 'Sennheiser XS LAV USB-C' in dev['name']:
        DEVICE_INDEX = i
        break

if DEVICE_INDEX is not None:
    print(f"Using LAV MIC on device index {DEVICE_INDEX}")
else:
    print("Warning: Sennheiser LAV not found. Using default input.")
    DEVICE_INDEX = sd.default.device[0]

sys.stdout.flush()

# 3. Load Your Local Model
if not os.path.exists(MODEL_FILE):
    print(f"ERROR: Could not find {MODEL_FILE}")
    print("Please ensure your .nemo file is in the same folder.")
    exit(1)

print(f"Loading local model: {MODEL_FILE}...")
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=MODEL_FILE)
asr_model.freeze()
asr_model = asr_model.to(device)
asr_model.eval()
print("Model loaded successfully!")

# Load punctuation model - FORMAT
print("Loading Punctuation Model...")
# punct_model = PunctuationModel(model="unikei/distilbert-base-re-punctuate")
# print("Punctuation Model loaded!")

# ------------------------------------------------------------------
# PART 3: AUDIO PROCESSING HELPERS
# ------------------------------------------------------------------

def rms(x):
    return np.sqrt(np.mean(np.square(x.astype(np.float64))))

def incremental_merge(prev, new):
    """Merges overlapping text streams cleanly."""
    prev = prev.strip()
    new = new.strip()
    if not prev: return new
    prev_words = prev.split()
    new_words = new.split()
    max_overlap = 0
    # Simple overlap check
    for i in range(len(prev_words)):
        suffix = prev_words[i:]
        if new_words[:len(suffix)] == suffix:
            max_overlap = len(suffix)
    fresh = new_words[max_overlap:]
    if not fresh: return prev
    return prev + " " + " ".join(fresh)

# Audio Queue Setup
audio_q = queue.Queue()
full_audio_buffer = [] # Stores all audio for the final cleanup

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(indata.copy())

# Buffer Math
chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
window_size = int(SAMPLE_RATE * WINDOW_DURATION)
rolling_buffer = np.zeros((1, 0), dtype=np.float32)

# ------------------------------------------------------------------
# PART 4: MAIN LOOP
# ------------------------------------------------------------------
print("\nRECORDING... (Ctrl+C to stop)\n")

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    device=DEVICE_INDEX,
    callback=audio_callback,
)

last_printed_was_silence = False
final_transcript = ""

try:
    stream.start()
    while True:
        # 1. Drain Queue
        while not audio_q.empty():
            block = audio_q.get()
            block_mono = block[:, 0].reshape(1, -1)
            # Live buffer
            rolling_buffer = np.concatenate((rolling_buffer, block_mono), axis=1)
            # Final buffer (keep everything)
            full_audio_buffer.append(block_mono.flatten())

        # 2. Process Window if ready
        if rolling_buffer.shape[1] >= window_size:
            window_audio = rolling_buffer[:, :window_size]
            energy = rms(window_audio.flatten())
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Silence Check
            if energy < ENERGY_THRESHOLD:
                if PRINT_SILENCE and not last_printed_was_silence:
                    print(f"[{timestamp}] (silence)")
                    sys.stdout.flush()
                    last_printed_was_silence = True
                # Slide window
                rolling_buffer = rolling_buffer[:, chunk_size:]
                continue

            # 3. Decode Window (Forward Pass)
            max_val = np.max(np.abs(window_audio))
            if max_val > 0:
                window_audio = window_audio / (max_val + 1e-9)

            tensor_in = torch.from_numpy(window_audio).float().to(device)
            len_in = torch.tensor([tensor_in.shape[1]], dtype=torch.int64, device=device)

            with torch.no_grad():
                logits = asr_model.forward(input_signal=tensor_in, input_signal_length=len_in)
                if isinstance(logits, tuple): logits = logits[0]
                pred_tokens = logits.argmax(dim=-1)
                transcripts = asr_model.decoding.ctc_decoder_predictions_tensor(pred_tokens)
                raw_text = transcripts[0].text if hasattr(transcripts[0], 'text') else str(transcripts[0])

            # 4. Apply Multimodal Correction (THE KEY MERGE STEP)
            corrected_text = multimodal_correction(raw_text, CURRENT_VISUAL_CONTEXT)
            
            # 5. Print & Merge
            if corrected_text:
                last_printed_was_silence = False
                print(f"[{timestamp}] {corrected_text}")
                sys.stdout.flush()
                
                # Update rolling transcript
                final_transcript = incremental_merge(final_transcript, corrected_text)
            
            # Slide window
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
    
    # Concatenate all recorded audio
    full_audio = np.concatenate(full_audio_buffer, axis=0).astype(np.float32)
    
    # Normalize
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / (max_val + 1e-9)

    # Convert to Tensor for Final Pass
    full_signal = torch.tensor(full_audio, dtype=torch.float32, device=device).unsqueeze(0)
    full_len = torch.tensor([full_signal.shape[1]], dtype=torch.int64, device=device)

    # Decode entire sequence at once
    with torch.no_grad():
        # Try .transcribe first (easier), fall back to forward pass if model type differs
        try:
            # Note: .transcribe usually expects a list of paths or arrays
            preds = asr_model.transcribe([full_audio])
            final_raw = preds[0].text if hasattr(preds[0], 'text') else str(preds[0])
        except:
            # Fallback to manual forward pass
            logits = asr_model.forward(input_signal=full_signal, input_signal_length=full_len)
            if isinstance(logits, tuple): logits = logits[0]
            pred_tokens = logits.argmax(dim=-1)
            transcripts = asr_model.decoding.ctc_decoder_predictions_tensor(pred_tokens)
            final_raw = transcripts[0].text

    # Apply Multimodal Correction one last time to the Clean Transcript
    final_clean_text = multimodal_correction(final_raw, CURRENT_VISUAL_CONTEXT).replace("⁇", "")

    # try: # FORMAT
    #     # Restore punctuation
    #     punctuated_text = punct_model.restore_punctuation(corrected_text)
        
    #     # --- SAFETY RAIL ---
    #     # If input had spaces but output doesn't, the model corrupted the text.
    #     if " " in corrected_text.strip() and " " not in punctuated_text.strip():
    #         raise ValueError("Space deletion detected")
            
    # except Exception as e:
    #     print(f"Warning: Punctuation model failed ({e}). Reverting to basic formatting.")
        
    #     # Fallback: Manual Capitalization & Period
    #     clean = corrected_text.strip()
    #     if clean:
    #         punctuated_text = clean[0].upper() + clean[1:]
    #         if punctuated_text[-1] not in ".?!":
    #             punctuated_text += "."
    #     else:
    #         punctuated_text = ""


    print("\n" + "="*40)
    print("FINAL TRANSCRIPT:")
    print("="*40)
    print(final_clean_text)
    print("="*40)
else:
    print("No audio recorded.")