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
from peft import LoraConfig, get_peft_model
from scripts.dataset_loader import MedicineWordDataset

# Hyperparameters per Stage 2 Multiline instructions
MODEL_PATH = "models/PaddleOCR-VL-1.6"
TRAIN_MANIFEST = "ocr_manifests/vlm_stage2_train.jsonl"
VAL_MANIFEST = "ocr_manifests/vlm_stage2_val.jsonl"
CHECKPOINT_DIR = Path("./checkpoints/stage2_multiline_best")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
EVAL_EVERY_STEPS = 25
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
    
    inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
    
    labels = inputs["input_ids"].clone()
    tokenizer = getattr(processor, "tokenizer", None)
    for i, p_text in enumerate(formatted_prompts):
        if tokenizer is not None:
            p_len = len(tokenizer.encode(p_text))
        else:
            p_len = 10
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
    print("=" * 70, flush=True)
    print("Stage 2 Fine-Tuning: PaddleOCR-VL-1.6 on Multi-Line Prescription Datasets", flush=True)
    print("=" * 70, flush=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()
    dtype = torch.float16 if use_amp else torch.float32
    print(f"Target Device: {device} | Precision AMP: {use_amp} (dtype={dtype})", flush=True)

    # 1. Load Datasets
    print("\n[Phase 1] Loading Datasets...", flush=True)
    train_dataset = MedicineWordDataset(TRAIN_MANIFEST, is_train=True, max_dim=768)
    val_dataset = MedicineWordDataset(VAL_MANIFEST, is_train=False, max_dim=768)
    
    # 2. Load Model & Processor
    print("\n[Phase 2] Loading PaddleOCR-VL-1.6 Model & Processor...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    
    # 3. Apply LoRA Configuration & Gradient Checkpointing
    print("\n[Phase 3] Configuring LoRA & Gradient Checkpointing...", flush=True)
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
    model.to(device)
    model.print_trainable_parameters()

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
    val_subset = torch.utils.data.Subset(val_dataset, list(range(min(50, len(val_dataset)))))

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    total_training_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_training_steps * 0.05)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\nTraining Plan:", flush=True)
    print(f"  - Device: {device}", flush=True)
    print(f"  - Total Epochs: {NUM_EPOCHS}", flush=True)
    print(f"  - Batch Size: {BATCH_SIZE} (Grad Accum: {GRAD_ACCUM_STEPS} -> Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS})", flush=True)
    print(f"  - Data Loader: num_workers={num_workers}, pin_memory={pin_memory}", flush=True)
    print(f"  - Total Optimizer Steps: {total_training_steps}", flush=True)
    print(f"  - Eval Frequency: Every {EVAL_EVERY_STEPS} steps", flush=True)

    best_val_loss = float("inf")
    global_step = 0
    start_time = time.time()
    
    log_history = {
        "train_loss": [],
        "val_convergence": []
    }

    print("\n[Phase 4] Launching Stage 2 Fine-Tuning Loop...", flush=True)
    print("-" * 95, flush=True)
    print(f"{'Epoch':<6} | {'Step':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'LR':<10} | {'Latency':<10} | {'VRAM (Alloc/Max)':<22} | {'Status'}", flush=True)
    print("-" * 95, flush=True)

    model.train()
    optimizer.zero_grad()

    for epoch in range(1, NUM_EPOCHS + 1):
        running_loss = 0.0
        step_in_epoch = 0

        for step, batch in enumerate(train_loader):
            step_start = time.time()
            step_in_epoch += 1
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            inputs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.float32 and k in ["pixel_values", "images"]:
                        inputs[k] = v.to(device=device, dtype=dtype)
                    else:
                        inputs[k] = v.to(device=device)
                else:
                    inputs[k] = v

            if use_amp:
                with torch.amp.autocast('cuda', dtype=dtype):
                    outputs = model(**inputs)
                    loss = outputs.loss / GRAD_ACCUM_STEPS
            else:
                outputs = model(**inputs)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS

            print(f"[Epoch {epoch}] Batch {step+1}/{len(train_loader)} | Micro-loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}", flush=True)
            sys.stdout.flush()

            if step_in_epoch % GRAD_ACCUM_STEPS == 0 or step == len(train_loader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                step_duration = time.time() - step_start
                avg_step_loss = running_loss / GRAD_ACCUM_STEPS
                running_loss = 0.0
                current_lr = scheduler.get_last_lr()[0]

                if torch.cuda.is_available():
                    vram_alloc_mib = torch.cuda.memory_allocated() / (1024 ** 2)
                    vram_max_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    vram_alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    vram_str = f"{vram_alloc_mib:.0f} MiB ({vram_alloc_gb:.2f}G) / {vram_max_mib:.0f} MiB"
                else:
                    vram_str = "CPU N/A"

                latency_str = f"{step_duration:.2f}s/step"
                
                if global_step % 10 == 0 or global_step == 1:
                    print(f"\n[ITERATION STEP {global_step}] Train Loss: {avg_step_loss:.4f} | LR: {current_lr:.2e} | Latency: {latency_str} | VRAM: {vram_str}", flush=True)

                log_history["train_loss"].append({
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(avg_step_loss, 4),
                    "latency_sec": round(step_duration, 4),
                    "vram": vram_str
                })

                status_msg = ""
                val_loss_str = "N/A"
                if global_step % EVAL_EVERY_STEPS == 0:
                    val_loss = evaluate(model, val_loader, processor, device, use_amp, dtype)
                    val_loss_str = f"{val_loss:.4f}"
                    
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                        model.save_pretrained(CHECKPOINT_DIR)
                        processor.save_pretrained(CHECKPOINT_DIR)
                        status_msg = f"[BEST SAVED @ step {global_step}]"
                    else:
                        status_msg = f"[Val Eval @ step {global_step}]"

                    log_history["val_convergence"].append({
                        "step": global_step,
                        "epoch": epoch,
                        "val_loss": round(val_loss, 4),
                        "is_best": is_best
                    })

                latency_str = f"{step_duration:.2f}s/step"
                print(f"{epoch:<6} | {global_step:<8} | {avg_step_loss:<12.4f} | {val_loss_str:<10} | {current_lr:<10.2e} | {latency_str:<10} | {vram_str:<18} | {status_msg}", flush=True)

    final_val_loss = evaluate(model, val_loader, processor, device, use_amp, dtype)
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        model.save_pretrained(CHECKPOINT_DIR)
        processor.save_pretrained(CHECKPOINT_DIR)
        final_msg = f"[FINAL BEST SAVED @ step {global_step}]"
    else:
        final_msg = f"[Final Val Eval]"

    log_history["val_convergence"].append({
        "step": global_step,
        "epoch": NUM_EPOCHS,
        "val_loss": round(final_val_loss, 4),
        "is_best": final_val_loss == best_val_loss
    })

    print("-" * 75, flush=True)
    print(f"{NUM_EPOCHS:<6} | {global_step:<8} | {'-':<12} | {final_val_loss:<12.4f} | {scheduler.get_last_lr()[0]:<10.2e} | {final_msg}", flush=True)
    print("-" * 75, flush=True)

    elapsed = time.time() - start_time
    print(f"\n[Success] Stage 2 Fine-Tuning Completed in {elapsed/60:.2f} minutes!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Checkpoint saved to: {CHECKPOINT_DIR.resolve()}")

    log_file = CHECKPOINT_DIR / "training_history.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_history, f, indent=2)
    print(f"Training log history saved to: {log_file}")

if __name__ == "__main__":
    main()
