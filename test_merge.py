import difflib

def smart_merge(final_text, new_chunk):
    final_text = final_text.strip()
    new_chunk = new_chunk.strip()
    
    if not final_text: return new_chunk, new_chunk
    if not new_chunk: return final_text, ""

    old_words = final_text.split()
    new_words = new_chunk.split()

    if old_words and new_words and old_words[-1].lower() == new_words[0].lower():
        new_words = new_words[1:]
        if not new_words: return final_text, ""

    search_window = old_words[-15:]
    
    s = difflib.SequenceMatcher(None, [w.lower() for w in search_window], [w.lower() for w in new_words])
    match = s.find_longest_match(0, len(search_window), 0, len(new_words))
    
    if match.size >= 2 or (match.size == 1 and len(new_words) <= 3):
        absolute_chop_idx = len(old_words) - len(search_window) + match.a
        
        merged_words = old_words[:absolute_chop_idx] + new_words[match.b:]
        merged_text = " ".join(merged_words)
        
        new_suffix = " ".join(new_words[match.b + match.size:])
        return merged_text, new_suffix
    else:
        safe_append = " ".join(new_words[-1:])
        return final_text + " " + safe_append, safe_append

# simulate rolling buffer streaming
stream = [
    "i'm testing",
    "testing to see",
    "to see if it",
    "see if it works"
]
final = ""
for c in stream:
    final, new = smart_merge(final, c)
    print(f"chunk: '{c}', new: '{new}', final: '{final}'")
