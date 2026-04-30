import re
import os

filepath = "voiceNotes/ilai-files/transcriber_fine_tuned.py"
with open(filepath, "r") as f:
    content = f.read()

# 1. Add argparse at the top
if 'import argparse' not in content:
    content = content.replace('import jellyfish', 'import jellyfish \nimport argparse')

# 2. Add config block
config_block = """# ------------------------------------------------------------------
# PART 1: CONFIGURATION
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Live ASR Transcriber")
parser.add_argument("--use-base", action="store_true", help="Use base fastconformer model instead of local fine-tuned model")
parser.add_argument("--raw-asr", action="store_true", help="Disable dictionary correction (preserve LLM formatting)")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))"""
content = re.sub(r'# ------------------------------------------------------------------\n# PART 1: CONFIGURATION\n# ------------------------------------------------------------------\nSCRIPT_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)', config_block, content)

# 3. Replace multimodal_correction and GEOLOGY_TRIGGERS
old_mm_pattern = r'GEOLOGY_TRIGGERS = \{.*?return False\n\ndef multimodal_correction\(transcript, visual_keywords\):.*?return " "\.join\(corrected_words\)'
new_mm = '''def multimodal_correction(transcript, visual_keywords):
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
                    
                    # Safe threshold! No more "outcrop" hallucinations.
                    threshold = 0.88 if len(target_squished) <= 4 else 0.78
                    
                    if sounds_alike or spelling_score > threshold:
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
            
    return " ".join(corrected_words)'''
content = re.sub(old_mm_pattern, new_mm, content, flags=re.DOTALL)

# 4. Modify Model Loading
old_model_load = r'print\(f"Loading fine-tuned model: \{MODEL_FILE\}..."\)\n# map_location=device is the ONLY way this works on your MacBook\nasr_model = nemo_asr\.models\.EncDecCTCModelBPE\.restore_from\(\n    restore_path=MODEL_FILE, \n    map_location=device\n\)\nasr_model\.freeze\(\)\nasr_model = asr_model\.to\(device\)\nasr_model\.eval\(\)\nprint\("Fine-tuned model loaded successfully!"\)\nsys\.stdout\.flush\(\)'
new_model_load = '''if args.use_base:
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
sys.stdout.flush()'''
content = re.sub(old_model_load, new_model_load, content)

# 5. Update LLM Model to Llama-3 8B
content = content.replace('print("Loading Phi-3 LLM for post-processing...")', 'print("Loading Llama-3 8B LLM for post-processing...")')
content = content.replace('LLM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Phi-3-mini-4k-instruct-q4.gguf")', 'LLM_MODEL_PATH = os.path.join(PROJECT_ROOT, "led-display", "models", "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")')

# 6. Replace final cleanup block with new alternative extraction and LLM formatting
old_final_cleanup = r'                    with torch\.no_grad\(\):.*?print\("FINAL TRANSCRIPT:"\)'
new_final_cleanup = '''                    with torch.no_grad():
                        # Always get logits to extract confidence alternatives
                        logits = asr_model.forward(input_signal=full_signal, input_signal_length=full_len)
                        if isinstance(logits, tuple): logits = logits[0]
                        pred_tokens = logits.argmax(dim=-1)
                        transcripts = asr_model.decoding.ctc_decoder_predictions_tensor(pred_tokens)
                        final_raw = transcripts[0].text if hasattr(transcripts[0], 'text') else str(transcripts[0])
                        
                        # Extract low confidence alternatives
                        probs = torch.softmax(logits, dim=-1)
                        top_probs, top_indices = torch.topk(probs, k=4, dim=-1)
                        vocab = asr_model.decoder.vocabulary
                        
                        ambiguous_notes = []
                        for t in range(logits.shape[1]):
                            top_prob = top_probs[0, t, 0].item()
                            top_idx = top_indices[0, t, 0].item()
                            
                            # If it's not a blank token and confidence is low
                            if top_idx < len(vocab) and top_prob < 0.85:
                                alts = []
                                for k in range(4):
                                    idx = top_indices[0, t, k].item()
                                    prob = top_probs[0, t, k].item()
                                    # Only include plausible alternative tokens
                                    if idx < len(vocab) and prob > 0.01:
                                        token = vocab[idx].replace(' ', '') # Clean subword marker
                                        if token:
                                            alts.append(token)
                                if len(alts) > 1:
                                    ambiguous_notes.append(f"[{'/'.join(alts)}]")
                        
                        confidence_hint = ""
                        if ambiguous_notes:
                            confidence_hint = "The ASR model was unsure about some sounds. Here are the top alternatives it considered for the ambiguous parts (in chronological order): " + " ".join(ambiguous_notes)

                    visual_context = get_current_visual_context()

                    # 2. RUN THE DICTIONARY SWEEP FIRST!
                    # Fix the geology terms while it's all still raw, lowercase ASR text
                    dictionary_cleaned_text = multimodal_correction(final_raw, visual_context).replace("⁇", "")
                    
                    # 3. LLM Post-Processing Layer (Formatting + Confidence Check)
                    final_clean_text = dictionary_cleaned_text
                    
                    if llm and dictionary_cleaned_text.strip():
                        print("Running LLM formatting...")
                        if confidence_hint:
                            print("Passing confidence notes to LLM...")
                            context_str = ", ".join(visual_context)
                            system_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert geology assistant and robotic text formatter. Your job is to format the text with capitalization and punctuation.
Visual Context: {context_str}
{confidence_hint}

If the ASR text contains errors, use the visual context and the phonetic alternatives to choose the best geological words. Do NOT completely rewrite the sentence. Keep it as close to the original as possible, just fixing errors and formatting.
CRITICAL: Do NOT output conversational text, explanations, or introductions like "Here is the formatted text". Output ONLY the final transcript."""
                        else:
                            system_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a robotic text formatter. Your ONLY job is to add capitalization and punctuation.
You must output the EXACT same words in the EXACT same order. 
DO NOT delete words. DO NOT add words. DO NOT fix grammar. 
CRITICAL: Do NOT output conversational text, explanations, or introductions like "Here is the formatted text". Output ONLY the formatted text."""

                        # Perfect Llama-3 Few-Shot Template
                        prompt = f"""{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
Raw ASR: i found this dark piece magnesium underground crater need record much<|eot_id|><|start_header_id|>assistant<|end_header_id|>
I found this dark piece magnesium underground crater. Need record much.<|eot_id|><|start_header_id|>user<|end_header_id|>
Raw ASR: {dictionary_cleaned_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

                        response = llm(
                            prompt, 
                            max_tokens=200, 
                            stop=["<|eot_id|>"], # REMOVED the \\n stop token!
                            temperature=0.1 if not confidence_hint else 0.2
                        )
                        
                        llm_output = response["choices"][0]["text"].strip()
                        
                        # Strip conversational filler if LLM ignores instructions
                        if "Here is the formatted text:" in llm_output:
                            llm_output = llm_output.split("Here is the formatted text:")[-1].strip()
                        elif "Here is the corrected text:" in llm_output:
                            llm_output = llm_output.split("Here is the corrected text:")[-1].strip()
                        elif llm_output.startswith("Here"):
                            lines = llm_output.split('\\n')
                            if len(lines) > 1:
                                llm_output = '\\n'.join(lines[1:]).strip()
                        
                        # Only accept the LLM text if it isn't completely empty
                        if llm_output:
                            final_clean_text = llm_output
    
                    print("\\n" + "="*40)
                    print("FINAL TRANSCRIPT:")'''

content = re.sub(old_final_cleanup, new_final_cleanup, content, flags=re.DOTALL)

with open(filepath, "w") as f:
    f.write(content)

print("Patching complete.")
