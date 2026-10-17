import os
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_PATH = "models/PaddleOCR-VL-1.6"
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Processor loaded!")
manifest_path = "ocr_manifests/vlm_stage1_train.jsonl"
with open(manifest_path, 'r', encoding='utf-8') as f:
    sample = json.loads(f.readline())
print("Sample item:", sample)
img_path = sample['image']
convs = sample['conversations']
human_text = ""
target_text = ""
for msg in convs:
    if msg['from'] == 'human':
        human_text = msg['value']
    elif msg['from'] == 'gpt':
        target_text = msg['value']
print("Human prompt:", repr(human_text))
print("Target text:", repr(target_text))
img = Image.open(img_path).convert("RGB")
print("Image size:", img.size)

# Format inputs for processor
if hasattr(processor, "apply_chat_template"):
    try:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": human_text.replace("<image>\n", "")}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
        ]
        prompt_str = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        print("Chat template prompt:", repr(prompt_str))
    except Exception as e:
        print("Chat template error:", e)

inputs = processor(text=[human_text], images=[img], return_tensors="pt")
print("Processor output keys:", inputs.keys())
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape {v.shape}, dtype {v.dtype}")
