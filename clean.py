import re

def clean_srt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', content)
    
    # 2. Optional removing of empty lines
    clean_text = re.sub(r'\n\s*\n', '\n', clean_text).strip()
    
    return clean_text