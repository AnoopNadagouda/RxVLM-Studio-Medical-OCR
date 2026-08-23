import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Add workspace directory to python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Patch transformers ROPE_INIT_FUNCTIONS['default'] for modern transformers compatibility
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
from peft import LoraConfig, get_peft_model, PeftModel
from scripts.dataset_loader import MedicineWordDataset

MODEL_PATH = "models/PaddleOCR-VL-1.6"
STAGE2_CHECKPOINT = "./checkpoints/stage2_multiline_best"
TRAIN_MANIFEST = "ocr_manifests/vlm_master_train.jsonl"
VAL_MANIFEST = "ocr_manifests/vlm_master_val.jsonl"
CHECKPOINT_DIR = Path("./checkpoints/stage3_fullpage_best")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
EVAL_EVERY_STEPS = 20
LORA_R = 16
LORA_ALPHA = 32
TARGET_MODULES = ['q_proj', 'v_proj', 'vision_projection']

def custom_collate_fn(batch, processor):
    images = [item['image'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    targets = [item['target_text'] for item in batch]
    
    formatted_prompts = []
    for prompt in prompts:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        formatted_prompts.append(processor.apply_chat_template(msgs, add_generation_prompt=True))
        
    full_texts = [f"{p}{t}" for p, t in zip(formatted_prompts, targets)]
    inputs = processor(text=full_texts, images=images, padding=True, max_length=512, truncation=True, return_tensors="pt")
    
    labels = inputs["input_ids"].clone()
    tokenizer = getattr(processor, "tokenizer", None)
    for i, p_text in enumerate(formatted_prompts):
        p_len = len(tokenizer.encode(p_text)) if tokenizer is not None else 10
        labels[i, :p_len] = -100
        
    if "attention_mask" in inputs:
        labels[inputs["attention_mask"] == 0] = -100
        
    inputs["labels"] = labels
    return inputs

class Collator:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, batch):
        return custom_collate_fn(batch, self.processor)

def evaluate(model, val_loader, processor, device, use_amp, dtype):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device=device, dtype=dtype) if (v.dtype == torch.float32 and k in ["pixel_values", "images"]) else v.to(device=device)
                else:
                    inputs[k] = v
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
                outputs = model(**inputs)
                loss = outputs.loss
            if loss is not None and not math.isnan(loss.item()):
                total_loss += loss.item()
                total_steps += 1
    model.train()
    return total_loss / max(1, total_steps)

def main():
    print("=" * 75, flush=True)
    print("Stage 3 Fine-Tuning: PaddleOCR-VL-1.6 Full Prescription Layout Parsing", flush=True)
    print("=" * 75, flush=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()
    dtype = torch.float16 if use_amp else torch.float32
    print(f"Target Device: {device} | Precision AMP: {use_amp} (dtype={dtype})", flush=True)

    print("\n[Phase 1] Loading Datasets...", flush=True)
    train_dataset = MedicineWordDataset(TRAIN_MANIFEST, is_train=True, max_dim=448)
    val_dataset = MedicineWordDataset(VAL_MANIFEST, is_train=False, max_dim=448)
    
    print("\n[Phase 2] Loading PaddleOCR-VL-1.6 Base Model & Stage 2 Weights...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    stage3_path = Path(CHECKPOINT_DIR)
    stage2_path = Path(STAGE2_CHECKPOINT)
    start_step = 0
    best_val_loss = float("inf")
    
    if stage3_path.exists() and (stage3_path / "adapter_config.json").exists():
        print(f"Resuming Stage 3 LoRA Weights from {CHECKPOINT_DIR} (Step 35)...", flush=True)
        model = PeftModel.from_pretrained(model, str(stage3_path), is_trainable=True)
        start_step = 35
        best_val_loss = 11.3021
    elif stage2_path.exists() and (stage2_path / "adapter_config.json").exists():
        print(f"Loading Stage 2 LoRA Weights from {STAGE2_CHECKPOINT}...", flush=True)
        model = PeftModel.from_pretrained(model, STAGE2_CHECKPOINT, is_trainable=True)
    else:
        print("Configuring new LoRA weights...", flush=True)
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.to(device)

    collator = Collator(processor)
    num_workers = 2 if device.startswith("cuda") else 0
    pin_memory = True if device.startswith("cuda") else False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_subset = torch.utils.data.Subset(val_dataset, list(range(min(40, len(val_dataset)))))
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    total_training_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = max(1, int(total_training_steps * 0.05))
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_training_steps)
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\nTraining Plan:", flush=True)
    print(f"  - Total Epochs: {NUM_EPOCHS}", flush=True)
    print(f"  - Total Training Samples: {len(train_dataset)}", flush=True)
    print(f"  - Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS}", flush=True)
    print(f"  - Total Optimizer Steps: {total_training_steps}", flush=True)

    global_step = start_step
    if start_step > 0:
        for _ in range(start_step):
            scheduler.step()

    start_time = time.time()
    
    log_history = {"train_loss": [], "val_convergence": []}

    print("\n[Phase 3] Launching Stage 3 Fine-Tuning Loop...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("-" * 95, flush=True)
    print(f"{'Epoch':<6} | {'Step':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'LR':<10} | {'Status'}", flush=True)
    print("-" * 95, flush=True)

    model.train()
    optimizer.zero_grad()

    for epoch in range(1, NUM_EPOCHS + 1):
        running_loss = 0.0
        step_in_epoch = 0

        for step, batch in enumerate(train_loader):
            step_in_epoch += 1

            inputs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device=device, dtype=dtype) if (v.dtype == torch.float32 and k in ["pixel_values", "images"]) else v.to(device=device)
                else:
                    inputs[k] = v

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
                outputs = model(**inputs)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS

            if step_in_epoch % GRAD_ACCUM_STEPS == 0 or step == len(train_loader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                avg_step_loss = running_loss / GRAD_ACCUM_STEPS
                running_loss = 0.0
                current_lr = scheduler.get_last_lr()[0]

                if global_step % 5 == 0 or global_step == 1:
                    print(f"[Stage 3 | Epoch {epoch}] Step {global_step} | Train Loss: {avg_step_loss:.4f} | LR: {current_lr:.2e}", flush=True)

                log_history["train_loss"].append({
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(avg_step_loss, 4)
                })

                if global_step % EVAL_EVERY_STEPS == 0:
                    val_loss = evaluate(model, val_loader, processor, device, use_amp, dtype)
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                        model.save_pretrained(CHECKPOINT_DIR)
                        processor.save_pretrained(CHECKPOINT_DIR)
                        status_msg = f"[BEST SAVED @ step {global_step}]"
                    else:
                        status_msg = f"[Val Eval @ step {global_step}]"

                    print(f"{epoch:<6} | {global_step:<8} | {avg_step_loss:<12.4f} | {val_loss:<10.4f} | {current_lr:<10.2e} | {status_msg}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n[Success] Stage 3 Fine-Tuning Completed in {elapsed/60:.2f} minutes!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR.resolve()}")

    with open(CHECKPOINT_DIR / "training_history.json", 'w', encoding='utf-8') as f:
        json.dump(log_history, f, indent=2)

if __name__ == "__main__":
    main()
