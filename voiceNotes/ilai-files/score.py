import jiwer
import string
import re
from spellchecker import SpellChecker
from difflib import SequenceMatcher

spell = SpellChecker()

# ==========================================
#  1. PASTE GROUND TRUTH
# ==========================================
GROUND_TRUTH = """
Look at the banding on this gneiss formation
"""

# ==========================================
#  2. PASTE RAW LOG OUTPUT
# ==========================================
RAW_LOG_OUTPUT = """
[23:28:14] look at the banding ⁇
[23:28:14] look at the banding on this not ⁇
[23:28:14] look at the banding on this nice form ⁇
[23:28:15] look at the banding on this gneiss formation
"""

KEYWORDS = [
    "metamorphic", 
    "transition", 
    "foliation",
    "gneiss", 
    "quartz", 
    "feldspar", 
    "mica schist", 
    "fracture", 
    "cleavage", 
    "luster", 
    "vitreous",
    "pearly"
]


for k in KEYWORDS:
    clean_k = k.replace("-", " ")
    spell.word_frequency.load_words(clean_k.split())

# ==========================================
#  CORE FUNCTIONS
# ==========================================

def basic_clean(text):
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
    text = text.replace('??', '').replace('⁇', '')
    text = text.lower()
    text = text.replace("-", " ") 
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # --- FIX: REMOVE FILLER WORDS ---
    words = text.split()
    # List of fillers to remove (including "uh", "um", "ah")
    fillers = {"uh", "um", "ah", "hmm", "hm", "er"}
    
    clean_words = [w for w in words if w not in fillers]
    
    return clean_words

def zipper_stitch(raw_log):
    lines = raw_log.strip().split('\n')
    full_transcript_str = ""
    
    MIN_OVERLAP = 6
    
    for line in lines:
        new_words = basic_clean(line)
        # If the line was just "uh" or "um", it's now empty, so we skip it entirely
        if not new_words: continue
        
        new_line_str = " ".join(new_words)
        
        if not full_transcript_str:
            full_transcript_str = new_line_str
            continue
            
        tail_len = 50
        tail = full_transcript_str[-tail_len:]
        head = new_line_str[:tail_len]
        
        matcher = SequenceMatcher(None, tail, head)
        match = matcher.find_longest_match(0, len(tail), 0, len(head))
        
        if match.size >= MIN_OVERLAP:
            cut_from_end = len(tail) - match.a
            base_transcript = full_transcript_str[:-cut_from_end]
            new_part = new_line_str[match.b:]
            full_transcript_str = base_transcript + new_part
        else:
            last_word = full_transcript_str.split()[-1] if full_transcript_str else ""
            first_word = new_line_str.split()[0] if new_line_str else ""
            
            if last_word == first_word:
                new_line_str = " ".join(new_line_str.split()[1:])
                full_transcript_str += " " + new_line_str
            else:
                full_transcript_str += " " + new_line_str

    return full_transcript_str

def filter_gibberish(text):
    words = text.split()
    valid_words = []
    for w in words:
        if w in spell or w in spell.word_frequency.dictionary:
            valid_words.append(w)
    return " ".join(valid_words)

def deduplicate_stutters(text):
    words = text.split()
    if not words: return ""
    output = []
    for w in words:
        if not output or output[-1] != w:
            output.append(w)
    return " ".join(output)

# ==========================================
#  SCORING
# ==========================================

def calculate_metrics(truth, stitched_hyp, keywords):
    t_clean = " ".join(basic_clean(truth))
    h_deduped = deduplicate_stutters(stitched_hyp)
    h_filtered = filter_gibberish(h_deduped)
    
    print(f"\n--- Reconstructed Transcript ---")
    print(f"'{h_filtered}'")
    print(f"\n--- Ground Truth ---")
    print(f"'{t_clean}'")
    
    wer_score = jiwer.wer(t_clean, h_filtered)
    
    keyword_hits = 0
    active_keywords = []
    t_padded = f" {t_clean} "
    h_padded = f" {h_filtered} "
    
    print(f"\n--- Keyword Check ---")
    for k in keywords:
        k_clean = " ".join(basic_clean(k))
        if f" {k_clean} " in t_padded:
            active_keywords.append(k_clean)
            if f" {k_clean} " in h_padded:
                print(f"✅ Found: {k_clean}")
                keyword_hits += 1
            else:
                print(f"❌ MISSED: {k_clean}")

    recall_score = keyword_hits / len(active_keywords) if active_keywords else 0.0
    return wer_score, recall_score

# Run
stitched = zipper_stitch(RAW_LOG_OUTPUT)
wer, recall = calculate_metrics(GROUND_TRUTH, stitched, KEYWORDS)

print("\n" + "="*30)
print("   FINAL CAPSTONE SCORES")
print("="*30)
print(f"Word Error Rate (WER):      {wer:.2%}")
print(f"Geology Keyword Recall:     {recall:.2%}")
print("="*30 + "\n")
