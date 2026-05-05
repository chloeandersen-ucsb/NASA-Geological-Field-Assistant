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
    return [] # Default fallback

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5        
WINDOW_DURATION = 4.0       
ENERGY_THRESHOLD = 0.001    
PRINT_SILENCE = True

spell = SpellChecker()

def multimodal_correction(transcript, visual_keywords):
    if args.raw_asr: return transcript
    if not transcript: return ""

    # 1. MASTER TARGET LIST
    targets = []
    for phrase in visual_keywords:
        targets.append(phrase.lower())
        targets.extend(phrase.lower().split())
    
    if targets:
        spell.word_frequency.load_words(targets)
        
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
                    
                    # --- NEW: Stricter Thresholds & Length Checks ---
                    # Make the long-word threshold stricter (0.85 instead of 0.78)
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

# --- NEW: BOOT THE HEAVIER LLM ---
from llama_cpp import Llama

LLM_MODEL_PATH = os.path.join(PROJECT_ROOT, "led-display", "models", "Phi-3-mini-4k-instruct-q4.gguf")

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

total_frames_recorded = 0
last_inference_frames = 0

last_raw_text = ""
stable_chunks = 0

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
                total_frames_recorded = 0
                last_inference_frames = 0
                silence_chunks = 0
                last_raw_text = ""
                final_transcript = ""
                stream.start()
                print("\nRECORDING NOW... (Fine-tuned Model)\n")
                sys.stdout.flush()
            elif line == "stop":
                stream.stop()
                recording_active = False

                if last_raw_text:
                    live_context = get_current_visual_context()
                    
                    # We must run the Head Drop here too so it merges cleanly!
                    raw_words = last_raw_text.split()
                    if total_frames_recorded > window_size and len(raw_words) > 1:
                        raw_words = raw_words[1:]
                    flushed_raw = " ".join(raw_words)
                    
                    corrected_text = multimodal_correction(flushed_raw, live_context).replace("⁇", "").strip()
                    if corrected_text:
                        new_full_transcript, new_part = smart_merge(final_transcript, corrected_text)
                        if new_part:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            print(f"[{timestamp}] {new_part}")
                            sys.stdout.flush()

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
                    
                    sys.stdout.flush() # TEST
                    
                    final_clean_text = dictionary_cleaned_text
                    
                    if llm and dictionary_cleaned_text.strip():
                        print("Running LLM formatting...")
                        if confidence_hint:
                            print("Passing confidence notes to LLM...")
                            context_str = ", ".join(visual_context)
                            system_prompt = f"""<|system|>
You are an expert geology assistant and robotic text formatter. Your job is to format the text with proper capitalization and punctuation.
Visual Context: {context_str}
{confidence_hint}

If the ASR text contains errors, use the visual context and the phonetic alternatives to choose the best geological words. Keep the text as close to the original as possible. Output ONLY the final formatted transcript. Do not include conversational filler.<|end|>"""
                        else:
                            system_prompt = f"""<|system|>
You are a robotic text formatter. Your ONLY job is to add proper capitalization and punctuation to the text.
Output ONLY the formatted text. Do not add conversational filler. Do not change the words.<|end|>"""

                        prompt = f"""{system_prompt}
<|user|>
Raw ASR: i found this dark piece magnesium underground crater need record much<|end|>
<|assistant|>
I found this dark piece magnesium underground crater. Need record much.<|end|>
<|user|>
Raw ASR: {dictionary_cleaned_text}<|end|>
<|assistant|>"""

                        response = llm(
                            prompt, 
                            max_tokens=200, 
                            stop=["<|end|>"],
                            temperature=0.1 if not confidence_hint else 0.2
                        )
                        
                        llm_output = response["choices"][0]["text"].strip()
                        
                        if "Here is the formatted text:" in llm_output:
                            llm_output = llm_output.split("Here is the formatted text:")[-1].strip()
                        elif "Here is the corrected text:" in llm_output:
                            llm_output = llm_output.split("Here is the corrected text:")[-1].strip()
                        
                        if llm_output:
                            final_clean_text = llm_output
    
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
                rolling_buffer = np.concatenate((rolling_buffer, block_mono), axis=1)
                full_audio_buffer.append(block_mono.flatten())
                
                total_frames_recorded += block_mono.shape[1]

            # 2. Start inference ONLY after 2.0 seconds of audio is collected
            if total_frames_recorded >= int(SAMPLE_RATE * 2.0):
                
                # 3. Prevent bursts: only trigger if 0.5s of NEW audio has arrived since the last inference
                if total_frames_recorded - last_inference_frames >= chunk_size:
                    
                    # Sync the tracker directly to prevent the "inference bomb" lag
                    last_inference_frames = total_frames_recorded

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
                    if total_frames_recorded > window_size and len(raw_words) > 1:
                        raw_words = raw_words[1:]
                        
                    raw_text = " ".join(raw_words)

                    live_context = get_current_visual_context()
                    corrected_text = multimodal_correction(raw_text, live_context).replace("⁇", "").strip()
                    
                    if corrected_text:
                        new_full_transcript, new_part = smart_merge(final_transcript, corrected_text)
                        
                        if new_part:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            # Standard print statement with a newline, safe for IDEs and PySide6
                            print(f"[{timestamp}] {new_part}")
                            sys.stdout.flush()
                                
                        final_transcript = new_full_transcript
                        
        time.sleep(0.01)

except KeyboardInterrupt:
    stream.stop()