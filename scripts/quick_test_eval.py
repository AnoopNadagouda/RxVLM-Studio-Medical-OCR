import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
from pathlib import Path
from PIL import Image
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Patch ROPE
import transformers.modeling_rope_utils as rope_utils
def _init_default_rope(config, device=None, seq_len=None, **kwargs):
    base = getattr(config, "rope_theta", 10000)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = getattr(config, "hidden_size", 1024) // getattr(config, "num_attention_heads", 16)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim))
    attention_factor = 1.0
    return inv_freq, attention_factor

rope_utils.ROPE_INIT_FUNCTIONS['default'] = _init_default_rope

from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from scripts.dataset_loader import preprocess_image_aspect_ratio

print("Loading processor & model...", flush=True)
processor = AutoProcessor.from_pretrained("models/PaddleOCR-VL-1.6", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "models/PaddleOCR-VL-1.6",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(base_model, "checkpoints/prescription_ocr_8020_best")
model.to("cuda:0")
model.eval()
print("Model loaded successfully!", flush=True)

recs = []
with open("processed/test_20.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            recs.append(json.loads(line))

recs = recs[:3]
prompt_text = "Transcribe the prescription structured JSON accurately."

t0 = time.time()
for idx, rec in enumerate(recs, 1):
    img = Image.open(rec["image_path"]).convert("RGB")
    img_padded = preprocess_image_aspect_ratio(img, max_dim=768)
    
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    p_str = processor.apply_chat_template(msgs, add_generation_prompt=True)
    
    inputs = processor(text=[p_str], images=[img_padded], return_tensors="pt")
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None
        )

    
    in_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[0][in_len:]
    pred_text = processor.decode(new_ids, skip_special_tokens=True).strip()
    print(f"Sample {idx}: Prediction len = {len(pred_text)} chars | Preview: {pred_text[:60]}...", flush=True)

print(f"3 samples took: {time.time() - t0:.2f} seconds", flush=True)
