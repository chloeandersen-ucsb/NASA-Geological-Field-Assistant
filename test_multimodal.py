import argparse
from spellchecker import SpellChecker
import jellyfish

class Args:
    raw_asr = False
args = Args()

spell = SpellChecker()

def multimodal_correction(transcript, visual_keywords):
    if args.raw_asr: return transcript
    if not transcript: return ""

    targets = []
    for phrase in visual_keywords:
        targets.append(phrase.lower())
        targets.extend(phrase.lower().split())
    
    if targets:
        spell.word_frequency.load_words(targets)
        
    targets = list(set(targets))

    raw_words = transcript.split()
    words = []
    for w in raw_words:
        clean = w.lower().strip(".,?!")
        if words and clean == words[-1].lower().strip(".,?!"):
            continue
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

        for window_size in [3, 2, 1]:
            if i + window_size <= len(words):
                chunk = "".join(words[i:i+window_size]).lower().strip(".,?!")
                
                for target in targets:
                    target_squished = target.replace(" ", "")
                    spelling_score = jellyfish.jaro_winkler_similarity(chunk, target_squished)
                    sounds_alike = (jellyfish.metaphone(chunk) == jellyfish.metaphone(target_squished))
                    
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
            clean_w = words[i].lower().strip(".,?!")
            if clean_w and clean_w not in spell:
                corrected = spell.correction(clean_w)
                corrected_words.append(corrected if corrected else words[i])
            else:
                corrected_words.append(words[i])
            
    return " ".join(corrected_words)

print(multimodal_correction("i'm testing to see if it works", ["Gneiss"]))
print(multimodal_correction("i am testing to see if it works", ["Gneiss"]))
