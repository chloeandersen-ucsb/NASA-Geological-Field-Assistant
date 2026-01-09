'''
LOCALLY TESTED ON MACBOOK NOT YET JETSON ORIN COMPATIBLE (last update 01/09/26)
- better accuracy on rolling window (best accuracy so far)
- clean readable final transcript implemented (best accuracy and cleanliness so far)
'''

import sounddevice as sd
import torch
import nemo.collections.asr as nemo_asr
import queue
import numpy as np
import time
from datetime import datetime

DEVICE_INDEX = 0             # setting sound device
SAMPLE_RATE = 16000        
CHUNK_DURATION = 0.5         # seconds (small blocks gathered from callback)
WINDOW_DURATION = 1.45       # seconds (what we send to ASR)
ENERGY_THRESHOLD = 0.0015    # RMS energy threshold (tune up/down if it still prints silence)
PRINT_SILENCE = True         # whether to print (silence) lines
full_audio_buffer = []


# Load model
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_fastconformer_ctc_large")
asr_model = asr_model.eval()

# Audio setup
BLOCK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
CHUNK_SIZE = BLOCK_SIZE
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)

audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())
    return None

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    device=DEVICE_INDEX,
    callback=audio_callback,
    blocksize=BLOCK_SIZE
)

# Normalizing Outputs to String
def normalize_prediction(pred):
    if pred is None:
        return ""
    if isinstance(pred, tuple):
        pred = pred[0]
    while isinstance(pred, list):
        pred = pred[0] if pred else ""
    if not isinstance(pred, str):
        return ""
    return pred.strip()

# Merging new token with previous token if overlap
def incremental_merge(prev, new):
    prev = prev.strip()
    new = new.strip()
    if not prev:
        return new
    prev_words = prev.split()
    new_words = new.split()
    max_overlap = 0
    for i in range(len(prev_words)):
        suffix = prev_words[i:]
        if new_words[:len(suffix)] == suffix:
            max_overlap = len(suffix)
    fresh = new_words[max_overlap:]
    if not fresh:
        return prev
    return prev + " " + " ".join(fresh)

# Root Square Mean Math
def rms(x):
    return np.sqrt(np.mean(np.square(x.astype(np.float64))))

# Main
print("Devices (selecting DEVICE_INDEX={}):".format(DEVICE_INDEX))
print(sd.query_devices())
print("Default device tuple:", sd.default.device)
print("\nRECORDING NOW... (Ctrl+C to stop)\n")

rolling_buffer = np.zeros((1, 0), dtype=np.float32)
final_transcript = ""
segments = []
segment_start = None
last_printed_was_silence = False

stream.start()
try:
    while True:
        while not audio_q.empty():
            block = audio_q.get()
            block_mono = block[:, 0].reshape(1, -1)

            # keep existing rolling buffer
            rolling_buffer = np.concatenate((rolling_buffer, block_mono), axis=1)

            # also append to full buffer for final transcript
            full_audio_buffer.append(block_mono.flatten())


        # Run ASR on the first window if we have enough size
        if rolling_buffer.shape[1] >= WINDOW_SIZE:
            window_audio = rolling_buffer[:, :WINDOW_SIZE]  
            energy = rms(window_audio.flatten())
            timestamp = datetime.now().strftime("%H:%M:%S")

            if energy < ENERGY_THRESHOLD:
                # Silence
                if PRINT_SILENCE and not last_printed_was_silence:
                    print(f"[{timestamp}] (silence)")
                    last_printed_was_silence = True
                # Slide window forward by one chunk to keep overlap
                rolling_buffer = rolling_buffer[:, CHUNK_SIZE:]
                continue

            # Non-silent
            max_val = np.max(np.abs(window_audio))
            if max_val > 0:
                window_audio = window_audio / (max_val + 1e-9)

            # Torch Tensor
            signal = torch.tensor(window_audio, dtype=torch.float32).to(next(asr_model.parameters()).device)
            length = torch.tensor([signal.shape[1]], dtype=torch.int64)

            
            with torch.no_grad():
                out = asr_model.forward(input_signal=signal, input_signal_length=length)

            logits = out[0] if isinstance(out, tuple) else out

            # Decoding
            pred = asr_model.decoding.ctc_decoder_predictions_tensor(logits)

            # Normalizing 
            normalized = ""
            if isinstance(pred, tuple):
                normalized = normalize_prediction(pred[0])
            else:
                if isinstance(pred, list) and len(pred) > 0:
                    normalized = normalize_prediction(pred[0])
                else:
                    normalized = normalize_prediction(pred)

            if normalized:
                last_printed_was_silence = False
                print(f"[{timestamp}] {normalized}")

                # Implementing Merge into Final Transcript
                updated = incremental_merge(final_transcript, normalized)
                if updated != final_transcript:
                    now = datetime.now().strftime("%H:%M:%S")
                    if segment_start is None:
                        segment_start = now
                        segments.append({"start": now, "end": now, "text": updated})
                    else:
                        segments[-1]["text"] = updated
                        segments[-1]["end"] = now
                    final_transcript = updated
            else:
                if PRINT_SILENCE and not last_printed_was_silence:
                    print(f"[{timestamp}] (silence)")
                    last_printed_was_silence = True

            # Keeps overlap
            rolling_buffer = rolling_buffer[:, CHUNK_SIZE:]

        time.sleep(0.03)

except KeyboardInterrupt:
    stream.stop()
    print("\nSTREAMING COMPLETE.\n")

# Final transcript
if full_audio_buffer:
    # print("FINAL TRANSCRIPT:\n")

    full_audio = np.concatenate(full_audio_buffer, axis=0)

    # normalize full audio
    full_audio = full_audio.astype(np.float32)
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / (max_val + 1e-9)

    signal = torch.tensor(full_audio).unsqueeze(0).to(next(asr_model.parameters()).device)
    length = torch.tensor([signal.shape[1]], dtype=torch.int64)

    with torch.no_grad():
        out = asr_model.forward(input_signal=signal, input_signal_length=length)

    logits = out[0] if isinstance(out, tuple) else out

    pred = asr_model.decoding.ctc_decoder_predictions_tensor(logits)
    final_text = normalize_prediction(pred)

    print("FINAL TRANSCRIPT:\n")
    print(final_text)

else:
    print("No speech detected.")
