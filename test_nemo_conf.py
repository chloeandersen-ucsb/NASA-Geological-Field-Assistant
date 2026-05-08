import torch
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

device = torch.device("cpu")
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_fastconformer_ctc_large", map_location=device)
model.eval()

# To get word confidence, we need to modify the decoding config
decoding_cfg = model.cfg.decoding
decoding_cfg.preserve_alignments = True
decoding_cfg.compute_timestamps = True
# Try setting confidence config
if 'confidence_cfg' not in decoding_cfg:
    decoding_cfg.confidence_cfg = OmegaConf.create({})
decoding_cfg.confidence_cfg.preserve_word_confidence = True

model.change_decoding_strategy(decoding_cfg)

# dummy input
tensor_in = torch.randn(1, 16000).float()
len_in = torch.tensor([16000], dtype=torch.int64)

with torch.no_grad():
    logits = model.forward(input_signal=tensor_in, input_signal_length=len_in)
    if isinstance(logits, tuple): logits = logits[0]
    
    pred_tokens = logits.argmax(dim=-1)
    transcripts = model.decoding.ctc_decoder_predictions_tensor(logits)
    
    res = transcripts[0]
    print(res.text)
    if hasattr(res, 'word_confidence'):
        print(res.word_confidence)
