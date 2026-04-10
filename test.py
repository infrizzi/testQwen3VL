import torch
import os
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. CONFIGURAZIONE MODELLO
# ==========================================
model_path = "Qwen/Qwen3-VL-4B-Instruct"
os.environ["QWEN_VL_VIDEO_READER_BACKEND"] = "decord"

print(f"--- Caricamento modello: {model_path} ---")

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# ==========================================
# 2. PREPARAZIONE INPUT (VIDEO)
# ==========================================
# Configurazione per il video: campioniamo 2 frame al secondo per non eccedere i token
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/homes/lpaladino/testQwen3VL/data/0IdYJGBmguM.mp4",
                "fps": 1.0, 
            },
            {"type": "text", "text": "Descrivi cosa succede in questo video in breve."}
        ]
    }
]

# Applichiamo il template
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Testo dopo template: '{text}'")

# Estrazione info visuali con qwen_vl_utils
# Nota: process_vision_info gestisce il caricamento del file video
image_inputs, video_inputs = process_vision_info(messages)

# Generazione dei tensori
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

if "video_grid_thw" in inputs:
    print(f"Video Grid Shape (THW): {inputs['video_grid_thw'].shape}")
    print(f"Video Grid THW values (T,H,W): {inputs['video_grid_thw']}")

# ==========================================
# 3. GENERAZIONE
# ==========================================
print("--- Generazione in corso (Inference Video)... ---")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        output_logits=True,
        return_dict_in_generate=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

# Logits
stacked_logits = torch.stack(outputs.logits)

print("\n" + "="*50)
print(f"SHAPE ANALISI:")
print(f"- Input IDs: {inputs.input_ids.shape}") # (batch, seq_len)
print(f"- Logits Stacked: {stacked_logits.shape}") 
print(f"- Vocab Size: {model.config.vocab_size}")
print("="*50)

# Decodifica
generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
response = processor.batch_decode(
    generated_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)[0]

print("\n" + "="*50)
print("RISPOSTA DEL MODELLO:")
print(f"'{response.strip()}'")
print("="*50)