import os
import torch
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

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_PATH = "models/PaddleOCR-VL-1.6"
print("Loading model config & structure...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

print("Model loaded successfully!")
module_names = [name for name, _ in model.named_modules()]

q_proj_modules = [m for m in module_names if 'q_proj' in m]
v_proj_modules = [m for m in module_names if 'v_proj' in m]
vision_proj_modules = [m for m in module_names if 'vision' in m or 'mlp' in m or 'proj' in m]

print(f"Found {len(q_proj_modules)} q_proj modules")
print(f"Found {len(v_proj_modules)} v_proj modules")
print("Vision/projection related modules sample:", vision_proj_modules[:10])

target_mods = ['q_proj', 'v_proj', 'vision_projection']
matching = [m for m in module_names if any(m.endswith(t) or f".{t}." in m or m.startswith(t) or t in m for t in target_mods)]
print(f"Matching modules for target_modules={target_mods}: {len(matching)}")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_mods,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
