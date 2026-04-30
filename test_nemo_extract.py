import torch
import nemo.collections.asr as nemo_asr
import numpy as np

device = torch.device("cpu")
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_fastconformer_ctc_large", map_location=device)
model.eval()

# Let's run it on an actual test wav file to get realistic logits
# We can use microphone-testing/test.wav
import librosa
audio, sr = librosa.load("microphone-testing/test.wav", sr=16000)
tensor_in = torch.from_numpy(audio).float().unsqueeze(0)
len_in = torch.tensor([tensor_in.shape[1]], dtype=torch.int64)

with torch.no_grad():
    logits = model.forward(input_signal=tensor_in, input_signal_length=len_in)
    if isinstance(logits, tuple): logits = logits[0]
    
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
    vocab = model.decoder.vocabulary
    
    notes = []
    for t in range(logits.shape[1]):
        top_prob = top_probs[0, t, 0].item()
        top_idx = top_indices[0, t, 0].item()
        
        if top_idx < len(vocab) and top_prob < 0.90:
            alts = []
            for k in range(5):
                idx = top_indices[0, t, k].item()
                prob = top_probs[0, t, k].item()
                if idx < len(vocab) and prob > 0.001:
                    token = vocab[idx].replace(' ', '')
                    if token:
                        alts.append(f"{token}")
            if len(alts) > 1:
                notes.append(" or ".join(alts))
    
    pred_tokens = logits.argmax(dim=-1)
    transcripts = model.decoding.ctc_decoder_predictions_tensor(pred_tokens)
    
    print("Transcript:", transcripts[0].text)
    print("Ambiguous token alternatives:")
    for n in notes:
        print("  -", n)
