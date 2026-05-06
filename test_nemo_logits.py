import torch
import nemo.collections.asr as nemo_asr

device = torch.device("cpu")
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_fastconformer_ctc_large", map_location=device)
model.eval()

# dummy input
tensor_in = torch.randn(1, 16000).float()
len_in = torch.tensor([16000], dtype=torch.int64)

with torch.no_grad():
    logits = model.forward(input_signal=tensor_in, input_signal_length=len_in)
    if isinstance(logits, tuple): logits = logits[0]
    
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=3, dim=-1)
    
    print("Vocab size:", len(model.decoder.vocabulary))
    
    for t in range(min(5, logits.shape[1])):
        print(f"Time {t}:")
        for k in range(3):
            idx = top_indices[0, t, k].item()
            prob = top_probs[0, t, k].item()
            if idx < len(model.decoder.vocabulary):
                token = model.decoder.vocabulary[idx]
            else:
                token = "<BLANK>"
            print(f"  {token} ({prob:.4f})")
