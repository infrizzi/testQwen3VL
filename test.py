import torch
import os
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. SETUP AMBIENTE E MODELLO
# ==========================================
# Forza il backend decord per evitare errori di metadati
os.environ["QWEN_VL_VIDEO_READER_BACKEND"] = "decord"
model_path = "Qwen/Qwen3-VL-4B-Instruct"

print(f"--- Caricamento modello e processor: {model_path} ---")

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# ==========================================
# 2. PREPARAZIONE INPUT
# ==========================================
video_path = "/homes/lpaladino/testQwen3VL/data/0IdYJGBmguM.mp4"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "fps": 1.0, # 1 frame al secondo per bilanciare dettaglio e memoria
            },
            {"type": "text", "text": "Descrivi brevemente cosa succede nel video."}
        ]
    }
]

# Generiamo il testo base dal template (contiene il placeholder <|video_pad|>)
prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Testo dopo apply_chat_template:\n{prompt_text}\n")


# Estraiamo i tensori del video tramite qwen_vl_utils
image_inputs, video_inputs = process_vision_info(messages)
print(f"Video inputs (prima di espansione): {video_inputs}")

# Lui vedrà <|video_pad|> nel testo e lo espanderà in base alla shape di video_inputs.
inputs = processor(
    text=[prompt_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

# Debug delle dimensioni per verifica
print(f"Input IDs shape (dopo espansione pad): {inputs.input_ids.shape}")
if "video_grid_thw" in inputs:
    print(f"Video Grid THW: {inputs.video_grid_thw.tolist()}")

# ==========================================
# 3. GENERAZIONE ED ESTRAZIONE LOGITS
# ==========================================
print("--- Avvio generazione ---")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        output_logits=True,
        return_dict_in_generate=True,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

# Estrazione dei Logits (Shape: [Tokens_generati, Batch, Vocab_size])
stacked_logits = torch.stack(outputs.logits)

# Decodifica della sola risposta (tagliando l'input)
generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
response = processor.batch_decode(
    generated_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)[0]

# ==========================================
# 4. OUTPUT FINALE
# ==========================================
print("\n" + "="*50)
print("RISPOSTA DEL MODELLO:")
print(f"'{response.strip()}'")
print("="*50)
print(f"ANALISI LOGITS:")
print(f"- Numero token generati: {stacked_logits.shape[0]}")
print(f"- Shape del tensore logits: {stacked_logits.shape}")
print("="*50)