import torch
import os
import time
import subprocess
from pathlib import Path
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import qwen_vl_utils

# ==========================================
# 1. CONFIGURAZIONE PERCORSI E PARAMETRI
# ==========================================
VIDEO_NAME = os.getenv("VIDEO_NAME", "2001_A_Space_Odyssey")
BASE_DIR = Path("/work/tesi_lpaladino/data/videos/")

VIDEO_INPUT = BASE_DIR / f"{VIDEO_NAME}/{VIDEO_NAME}.mp4"
CHUNKS_DIR = BASE_DIR / f"{VIDEO_NAME}/chunks/"
OUTPUT_CORPUS = BASE_DIR / f"{VIDEO_NAME}/{VIDEO_NAME}_visual_corpus.txt"
SEGMENT_TIME = 30  # secondi
OVERLAP_TIME = 2   # secondi di sovrapposizione
MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"

os.makedirs(CHUNKS_DIR, exist_ok=True)
os.environ["QWEN_VL_VIDEO_READER_BACKEND"] = "decord"

# ==========================================
# 2. FUNZIONE DI SEGMENTAZIONE (FFMPEG)
# ==========================================
def split_video(input_file, chunk_dir, seg_time, overlap):
    print(f"--- Inizio segmentazione video: {input_file} ---")
    # Ottieni la durata totale
    cmd = [
        "ffprobe", "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        input_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip().split('\n')[-1]
    total_duration = float(output)
    print(f"Durata totale rilevata: {total_duration} secondi")

    start = 0
    chunk_idx = 0
    chunk_paths = []

    while start < total_duration:
        output_path = os.path.join(chunk_dir, f"chunk_{chunk_idx:04d}.mp4")
        # Comando ffmpeg per estrarre il segmento senza ricodifica pesante
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(seg_time),
            "-i", input_file, "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "128k", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        chunk_paths.append(output_path)
        
        start += (seg_time - overlap)
        chunk_idx += 1
        if chunk_idx % 10 == 0:
            print(f"Creati {chunk_idx} segmenti...")
            
    return chunk_paths

# ==========================================
# 3. SETUP MODELLO QWEN3-VL
# ==========================================
print(f"--- Caricamento modello: {MODEL_PATH} ---")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.video_processor.max_frames = 1024
processor.video_processor.min_frames = 16

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# Ottimizzazioni per video lunghi/dettagliati
qwen_vl_utils.vision_process.VIDEO_MAX_TOKEN_NUM = 2048
qwen_vl_utils.vision_process.FPS_MAX_FRAMES = 1024
qwen_vl_utils.vision_process.MODEL_SEQ_LEN = 128000 

# ==========================================
# 4. LOOP DI INFERENZA E SCRITTURA CORPUS
# ==========================================
def run_visual_captioning(chunks):
    prompt_caption = (
        "Provide a highly concise, precise, and detailed description of the scene. "
        "Do not include any artistic interpretations, subjective commentary, or mention the camera, model, or the video itself. "
        "Focus strictly and directly on describing the physical actions, the setting, the characters, and the objects present."
    )
    
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        for i, chunk_path in enumerate(chunks):
            print(f"Elaborazione {i+1}/{len(chunks)}: {chunk_path}")
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": chunk_path},
                    {"type": "text", "text": prompt_caption}
                ]
            }]

            # Preparazione input
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = qwen_vl_utils.process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            # Generazione
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values_videos=inputs.get("pixel_values_videos", None),
                    video_grid_thw=inputs.get("video_grid_thw", None),
                    pixel_values=inputs.get("pixel_values", None),
                    image_grid_thw=inputs.get("image_grid_thw", None),
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.5,
                    repetition_penalty=1.1
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                caption = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            # Scrittura su file (formato compatibile con self-study)
            f.write(f"--- SEGMENT {i} ({chunk_path}) ---\n")
            f.write(caption.strip() + "\n\n")
            f.flush() # Salva costantemente
            
            # Pulizia VRAM per evitare frammentazione
            torch.cuda.empty_cache()

# Esecuzione pipeline
if __name__ == "__main__":
    # Se i chunk esistono già, puoi saltare questo passaggio
    video_chunks = sorted([os.path.join(CHUNKS_DIR, f) for f in os.listdir(CHUNKS_DIR) if f.endswith(".mp4")])
    if not video_chunks:
        video_chunks = split_video(VIDEO_INPUT, CHUNKS_DIR, SEGMENT_TIME, OVERLAP_TIME)
    
    run_visual_captioning(video_chunks)
    print(f"--- Corpus completato: {OUTPUT_CORPUS} ---")