import re
from datetime import datetime

# --- CONFIGURAZIONE ---
CAPTION_FILE = "oppenheimer_visual_corpus.txt"
SRT_FILE = "/home/lucap/projects/testQwen/data/text/Oppenheimer282023%29.srt"
OUTPUT_FILE = "oppenheimer_multimodal_corpus.txt"

def time_to_seconds(t_str):
    """Converte HH:MM:SS o HH:MM:SS,mmm in secondi totali."""
    t_str = t_str.replace(',', '.') # Gestisce virgola SRT
    parts = t_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

def parse_srt(srt_path):
    """Estrae i sottotitoli in una lista di dizionari {start, end, text}."""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regex per catturare tempo e testo (standard SRT)
    pattern = re.compile(r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\n\d+\n|$)', re.DOTALL)
    subs = []
    for match in pattern.finditer(content):
        subs.append({
            'start': time_to_seconds(match.group(1)),
            'end': time_to_seconds(match.group(2)),
            'text': match.group(3).replace('\n', ' ').strip()
        })
    return subs

def merge_corpora():
    print("--- Avvio fusione multimodale ---")
    subtitles = parse_srt(SRT_FILE)
    
    with open(CAPTION_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        current_caption_block = ""
        current_header = ""
        
        for line in f_in:
            # Rileva l'inizio di un nuovo segmento
            header_match = re.search(r'--- SEGMENT \d+ \[([\d:]+) - ([\d:]+)\] ---', line)
            
            if header_match:
                # Se avevamo un blocco precedente, processalo prima di iniziare il nuovo
                if current_header:
                    process_and_write(f_out, current_header, current_caption_block, subtitles)
                
                current_header = line.strip()
                current_caption_block = ""
                # Estrai i tempi del segmento per il filtro
                start_seg = time_to_seconds(header_match.group(1))
                end_seg = time_to_seconds(header_match.group(2))
                current_segment_times = (start_seg, end_seg)
            else:
                if line.strip(): # Accumula il testo della caption
                    current_caption_block += line
        
        # Scrivi l'ultimo blocco
        if current_header:
            process_and_write(f_out, current_header, current_caption_block, subtitles)

def process_and_write(f_out, header, caption_text, all_subs):
    # Estrai tempi dal titolo del segmento
    times = re.search(r'\[([\d:]+) - ([\d:]+)\]', header)
    s_start = time_to_seconds(times.group(1))
    s_end = time_to_seconds(times.group(2))
    
    # Filtra i sottotitoli che cadono in questo range (con un piccolo margine)
    relevant_subs = [s['text'] for s in all_subs if s['start'] >= s_start and s['start'] < s_end]
    dialogue_text = " ".join(relevant_subs)
    
    # Formattazione finale del blocco per il self-study
    f_out.write(f"{header}\n")
    f_out.write(f"VISUAL DESCRIPTION: {caption_text.strip()}\n")
    if dialogue_text:
        f_out.write(f"DIALOGUE CONTEXT: {dialogue_text}\n")
    else:
        f_out.write(f"DIALOGUE CONTEXT: [No dialogue in this sequence]\n")
    f_out.write("\n")

if __name__ == "__main__":
    merge_corpora()
    print(f"Fusione completata. Corpus pronto in: {OUTPUT_FILE}")