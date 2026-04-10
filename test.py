import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. CONFIGURAZIONE MODELLO E AMBIENTE
# ==========================================
model_path = "Qwen/Qwen3-VL-4B-Instruct"

print(f"--- Caricamento modello: {model_path} ---")

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # Attivato come richiesto
    device_map="auto"
)

# Ottimizzazione per inferenza
model.config.use_cache = True

# ==========================================
# 2. PREPARAZIONE INPUT (MESSAGGI)
# ==========================================
# TEST SOLO TESTO (per verifica rapida)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Ciao! Sei attivo con Flash Attention? Presentati brevemente."}
        ]
    }
]

# ESEMPIO VIDEO (Decommenta questo blocco per usare il video)
"""
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/homes/lpaladino/data/0IdYJGBmguM.mp4"},
            {"type": "text", "text": "Descrivi cosa succede nel video."}
        ]
    }
]
"""

# Applichiamo il template e processiamo i dati visuali
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

# Generazione dei tensori per la GPU
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

# ==========================================
# 3. GENERAZIONE (INFERENZA)
# ==========================================
print("--- Generazione in corso... ---")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        output_logits=True,
        return_dict_in_generate=True,
        do_sample=True,      # True per usare temperature
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1 # Evita che il modello si incastri
    )

# ==========================================
# 4. DECODIFICA E LOGITS
# ==========================================
generated_ids = outputs.sequences

# Tagliamo l'input per isolare solo la risposta nuova
# Usiamo l'indice della prima dimensione per gestire il batch
input_len = inputs.input_ids.shape[1]
generated_ids_trimmed = generated_ids[:, input_len:]

# Decodifica robusta
response = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

# Se la risposta è vuota, proviamo a vedere se ci sono token speciali "nascosti"
if not response.strip():
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]

# Estrazione Logits (Stack dei tensori generati ad ogni step)
all_logits = torch.stack(outputs.logits)

# ==========================================
# 5. OUTPUT FINALE
# ==========================================
print("\n" + "="*50)
print("RISPOSTA DEL MODELLO:")
print(f"'{response.strip()}'")
print("="*50)
print(f"Dettagli Tecnici:")
print(f"- Input tokens: {input_len}")
print(f"- Output tokens generati: {generated_ids_trimmed.shape[1]}")
print(f"- Logits shape: {all_logits.shape}")
print("="*50)