import os
import sys
import torch
import transformers.modeling_rope_utils as rope_utils

# 1. Patch transformers ROPE_INIT_FUNCTIONS['default'] for modern transformers compatibility
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
from peft import LoraConfig, get_peft_model
from scripts.dataset_loader import MedicineWordDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_PATH = "models/PaddleOCR-VL-1.6"
print("1. Loading Processor and Model...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Using device: {device}, dtype: {dtype}", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
)

# Configure LoRA
target_mods = ['q_proj', 'v_proj', 'vision_projection']
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_mods,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.to(device)
model.print_trainable_parameters()

# Test dataset & collator on 2 samples
dataset = MedicineWordDataset('ocr_manifests/vlm_stage1_train.jsonl', is_train=True)
batch_items = [dataset[0], dataset[1]]

images = [item['image'] for item in batch_items]
prompts = [item['prompt'] for item in batch_items]
targets = [item['target_text'] for item in batch_items]
full_texts = [f"{p}\n{t}" for p, t in zip(prompts, targets)]

print("Processing batch through processor...", flush=True)
inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

labels = inputs["input_ids"].clone()
for i, prompt in enumerate(prompts):
    p_inputs = processor(text=[prompt], return_tensors="pt")
    p_len = p_inputs["input_ids"].shape[1]
    labels[i, :p_len] = -100

if "attention_mask" in inputs:
    labels[inputs["attention_mask"] == 0] = -100

inputs["labels"] = labels

print("Testing forward pass...", flush=True)
outputs = model(**inputs)
loss = outputs.loss
print(f"Forward step loss: {loss.item():.4f}", flush=True)

print("Testing backward pass...", flush=True)
loss.backward()
print("Backward step successful!", flush=True)
