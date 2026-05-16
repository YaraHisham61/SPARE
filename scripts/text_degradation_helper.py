import random
import string

def inject_typos(text, level=1):
    """
    Injects random character substitutions into the text.
    level: Number of substitutions to perform.
    """
    if not text or level == 0:
        return text
    
    text_list = list(text)
    valid_indices = [i for i, char in enumerate(text_list) if char != ' ']
    n_changes = min(level, len(valid_indices))
    to_change = random.sample(valid_indices, n_changes)
    
    for idx in to_change:
        original_char = text_list[idx]
        replacement = random.choice(string.ascii_lowercase)
        while replacement == original_char.lower():
            replacement = random.choice(string.ascii_lowercase)
        text_list[idx] = replacement
        
    return "".join(text_list)