import sounddevice as sd
import nemo.collections.asr as nemo_asr
import numpy as np
import queue
import time
from datetime import datetime
import torch

# ------------------------------
# Setup torch device (GPU if available)
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Handle torch.distributed (required by NeMo)
if not hasattr(torch, "distributed"):
    import types
    torch.distributed = types.SimpleNamespace(is_initialized=lambda: False)
elif not hasattr(torch.distributed, "is_initialized"):
    torch.distributed.is_initialized = lambda: False

# ------------------------------
# Load ASR model
# ------------------------------
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    model_name="stt_en_fastconformer_ctc_large"
)
asr_model = asr_model.to(device)
asr_model.eval()

# ------------------------------
# Streaming parameters
# ------------------------------
sr = 16000
chunk_duration = 0.5        # seconds per audio capture
window_duration = 1.0       # seconds per ASR window
chunk_size = int(sr * chunk_duration)
window_size = int(sr * window_duration)
batch_size = 3               # number of windows to batch for decoding

audio_queue = queue.Queue()
rolling_buffer = np.zeros((1, 0), dtype=np.float32)
final_transcript = []
last_text = ""  # avoid printing duplicates

# ------------------------------
# Audio callback
# ------------------------------
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

# ------------------------------
# Waveform decoding
# ------------------------------
def decode_batch_waveforms(batch_waveforms):
    """
    batch_waveforms: torch tensor of shape [batch, time]
    """
    with torch.no_grad():
        signal_lengths = torch.tensor([w.shape[0] for w in batch_waveforms]).to(device)
        logits = asr_model.forward(
            input_signal=batch_waveforms, input_signal_length=signal_lengths
        )
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_tokens = logits.argmax(dim=-1)
        # greedy decoding
        transcripts = asr_model.decoding.ctc_decoder_predictions_tensor(pred_tokens)
        return [t.text for t in transcripts]

# ------------------------------
# Start streaming
# ------------------------------
stream = sd.InputStream(channels=1, samplerate=sr, callback=audio_callback)
stream.start()
print("RECORDING NOW... (Ctrl+C to stop)\n")

try:
    while True:
        # Add new audio to rolling buffer
        while not audio_queue.empty():
            data = audio_queue.get()
            rolling_buffer = np.concatenate((rolling_buffer, data.T), axis=1)

        # Collect multiple windows for batch decoding
        batch_waveforms = []
        while rolling_buffer.shape[1] >= window_size and len(batch_waveforms) < batch_size:
            window_audio = rolling_buffer[:, :window_size]
            waveform_tensor = torch.from_numpy(window_audio).float().squeeze(0).to(device)  # [window_size]
            batch_waveforms.append(waveform_tensor)
            rolling_buffer = rolling_buffer[:, chunk_size:]  # slide by chunk_size

        if batch_waveforms:
            batch_tensor = torch.stack(batch_waveforms, dim=0)  # [batch, window_size]
            phrases = decode_batch_waveforms(batch_tensor)
            timestamp = datetime.now().strftime("%H:%M:%S")
            for phrase in phrases:
                if phrase != last_text and phrase.strip() != "":
                    print(f"[{timestamp}] ['{phrase}']")
                    final_transcript.append((timestamp, phrase))
                    last_text = phrase

        time.sleep(0.01)  # short sleep for low latency

except KeyboardInterrupt:
    stream.stop()
    print("\nSTREAMING COMPLETE.\n")
    print("FINAL TRANSCRIPT:\n")
    for ts, phrase in final_transcript:
        print(f"[{ts}] ['{phrase}']")
