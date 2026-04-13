import torch
import os
import json
import time
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import  qwen_vl_utils

# ==========================================
# 1. MODEL AND PROCESSOR SETUP
# ==========================================
os.environ["QWEN_VL_VIDEO_READER_BACKEND"] = "decord"
model_path = "Qwen/Qwen3-VL-4B-Instruct"

print(f"--- Loading model and processor: {model_path} ---")

processor = AutoProcessor.from_pretrained(model_path)
processor.video_processor.max_frames = 1024
processor.video_processor.min_frames = 16 

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

qwen_vl_utils.vision_process.VIDEO_MAX_TOKEN_NUM = 2048
qwen_vl_utils.vision_process.FPS_MAX_FRAMES = 512
qwen_vl_utils.vision_process.MODEL_SEQ_LEN = 128000

# ==========================================
# 2. INPUT PROCESSING
# ==========================================
with open('messages.json', 'r') as f:
    data = json.load(f)

messages = data["messages"]

for i, entry in enumerate(messages):
    print(f"\n--- Message {i+1} ---")

    # Reset GPU memory tracks and start timer for visual processing
    torch.cuda.reset_peak_memory_stats()
    start_visual = time.perf_counter()

    current_message = [entry]

    # Template generation
    prompt_text = processor.apply_chat_template(current_message, tokenize=False, add_generation_prompt=True)
    print("\n" + "="*30 + " DEBUG 1: CHAT TEMPLATE " + "="*30)
    print(f"Formatted Prompt:\n{prompt_text}")
    print("="*84)

    # Take visual tensors
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(current_message)
    print("\n" + "="*30 + " DEBUG 2: VISION INFO " + "="*30)
    print(f"Video Inputs Type: {type(video_inputs)}")
    if video_inputs:
        print(f"Number of videos processed: {len(video_inputs)}")
        print(f"Raw Tensor Shape (before processing): {video_inputs[0].shape}") 
        # [Frames, Channels, Height, Width]
    print("="*84)

    # Final input processing -> visual placeholders are expanded
    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # End timer for visual processing
    end_visual = time.perf_counter()
    visual_duration = end_visual - start_visual

    # VRAM usage
    current_vram = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB

    print("\n" + "="*30 + " DEBUG 3: VRAM USAGE & TIME SPENT " + "="*30)
    print(f"Current VRAM Usage: {current_vram:.3f} GB")
    print(f"Peak VRAM Usage: {peak_vram:.3f} GB")
    print(f"Visual Processing Time: {visual_duration:.3f} seconds")
    print("="*84)

    print("\n" + "="*30 + " DEBUG 4: FRAME & GRID ANALYSIS " + "="*30)
    if "video_grid_thw" in inputs:
        # video_grid_thw contains [T, H, W]
        grid = inputs.video_grid_thw[0]
        t, h, w = grid[0].item(), grid[1].item(), grid[2].item()
        
        patches_per_frame = h * w
        tokens_per_frame = patches_per_frame // 4 # Spatial pooling 2x2
        
        print(f"Grid Dimensions -> T (Frames/Time): {t}, H (Height Patches): {h}, W (Width Patches): {w}")
        print(f"Each frame consists of {patches_per_frame} raw patches.")
        print(f"After 2x2 spatial pooling, each frame contributes {tokens_per_frame} latent tokens.")
        print(f"Total Video Tokens Calculation: ({t} frames) * ({tokens_per_frame} tokens/frame) = {t * tokens_per_frame}")
    else:
        print("No video grid found in inputs.")
    print("="*84)

    video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    actual_pad_count = (inputs.input_ids == video_pad_id).sum().item()

    # Total expected tokens using the formula: T * H * W / 4
    expected_tokens = (inputs.video_grid_thw[:, 0] * inputs.video_grid_thw[:, 1] * inputs.video_grid_thw[:, 2] // 4).sum().item()

    print("\n" + "="*30 + " DEBUG 5: TOKEN SYNC " + "="*30)
    print(f"Actual <|video_pad|> tokens found in text: {actual_pad_count}")
    print(f"Expected tokens calculated from grid:    {expected_tokens}")

    if actual_pad_count == expected_tokens:
        print("STATUS: SUCCESS - Dimensions are perfectly aligned.")
    else:
        print(f"STATUS: ERROR - Mismatch detected! Difference: {abs(actual_pad_count - expected_tokens)}")
    print("="*84)

    # ==========================================
    # 3. GENERATION AND LOGITS
    # ==========================================
    print("--- Starting generation ---")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values_videos=inputs.get("pixel_values_videos", None),
            video_grid_thw=inputs.get("video_grid_thw", None),
            pixel_values=inputs.get("pixel_values", None),
            image_grid_thw=inputs.get("image_grid_thw", None),
            max_new_tokens=512,
            output_logits=True,
            return_dict_in_generate=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    # Logits extraction (Shape: [Token_generated, Batch, Vocab_size])
    stacked_logits = torch.stack(outputs.logits)

    # Decoding only the generated answer
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
    print("FINAL RESPONSE:")
    print(f"'{response.strip()}'")
    print("="*50)
    print(f"ANALISI LOGITS:")
    print(f"- Generated tokens: {stacked_logits.shape[0]}")
    print(f"- Logits shape: {stacked_logits.shape}")
    print("="*50)