# voiceNotes

Live speech-to-text for field annotations. Streams audio from the MEMS mic and transcribes in real time using NVIDIA NeMo's FastConformer model, manually fine-tuned & trained on geology/lunar terminology.

## models

| Model | File | Notes |
|---|---|---|
| Fine-tuned FastConformer | `ilai-files/newest_model.nemo` | trained on geology vocab, ~442 MB |
| Base FastConformer | (downloaded at runtime) | `stt_en_fastconformer_ctc_large` from NGC |

## main transcriber

`ilai-files/transcriber_fine_tuned.py` is the production transcriber. It:
1. Streams audio in 0.5s chunks
2. Runs ASR over a 4s sliding window
3. Cleans up / post-processes output with spell-check + TextBlob (with punctuation)
4. Reads `visual_context.txt` from ML-classifications to bias transcription toward the current rock type
5. Writes transcripts to `led-display/sage_data/voice_notes.jsonl`

```bash
python ilai-files/transcriber_fine_tuned.py
python ilai-files/transcriber_fine_tuned.py --use-base   # skip fine-tuned model
python ilai-files/transcriber_fine_tuned.py --raw-asr    # disable spell correction
```

## files

| File | Description |
|---|---|
| `improved2.py` | standalone transcriber (no context integration) |
| `ilai-files/original_model.py` | original training script |
| `ilai-files/score.py` | WER scoring against reference transcripts |
| `shruthi-files/rtt.py` | earlier real-time transcription prototype |
| `shruthi-files/rtt_lav.py` | lavalier mic variant |
| `shruthi-files/rtt_finalclean.py` | cleaned-up version of the RTT prototype |

## visual context integration

The transcriber reads `ML-classifications/visual_context.txt` to get the currently-displayed rock classification. This feeds into the ASR context so geology terms for the right rock type get weighted more heavily (e.g. if the display shows "basalt", the model knows to favor "vesicularity" over "visibility").
