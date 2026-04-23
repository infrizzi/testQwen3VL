import re

# --- USED FOR CONVERTING THE TIMESTAMPS IN THE OPPENHEIMER VISUAL CORPUS ---
def convert_timestamps(input_file, output_file):
    # Regex per trovare la riga dell'intestazione e catturare l'indice del segmento
    pattern = re.compile(r"^--- SEGMENT (\d+) \(.*\) ---")
    
    chunk_duration = 30  # secondi
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            match = pattern.match(line)
            if match:
                segment_idx = int(match.group(1))
                
                # Calcolo del tempo: segment_idx * 30 secondi
                start_sec = segment_idx * chunk_duration
                end_sec = start_sec + chunk_duration
                
                # Formattazione in HH:MM:SS
                def format_time(s):
                    h = s // 3600
                    m = (s % 3600) // 60
                    sec = s % 60
                    return f"{h:02d}:{m:02d}:{sec:02d}"
                
                timestamp = f"{format_time(start_sec)} - {format_time(end_sec)}"
                
                # Scrivi la nuova riga con il timestamp
                f_out.write(f"--- SEGMENT {segment_idx} [{timestamp}] ---\n")
            else:
                # Scrivi le righe del testo invariate
                f_out.write(line)

    print(f"Conversione completata! File salvato in: {output_file}")

# Esecuzione
convert_timestamps("oppenheimer_visual_corpus.txt", "oppenheimer_visual_corpus_timestamped.txt")