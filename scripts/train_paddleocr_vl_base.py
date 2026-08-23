import os
import sys
import json
import time
import math
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image

# Set OpenMP Duplicate Lib Fix for Windows Anaconda environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add workspace base directory to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Apply Transformers RoPE compatibility patch
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

from transformers import AutoProcessor, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# -----------------------------------------------------------------------------
# Configuration Constants per Task Requirements
# -----------------------------------------------------------------------------
MODEL_PATH = "models/PaddleOCR-VL-1.6"
TRAIN_MANIFEST = BASE_DIR / "ocr_manifests" / "clean_composite_train.jsonl"
VAL_MANIFEST = BASE_DIR / "ocr_manifests" / "clean_composite_val.jsonl"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "paddleocr_vl_clean_base"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# 4GB VRAM Protections
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
EVAL_EVERY_STEPS = 10
LORA_R = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# -----------------------------------------------------------------------------
# Evaluation Helper Functions (CER & WER)
# -----------------------------------------------------------------------------
def compute_levenshtein(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

def calculate_cer(ref, pred):
    if not ref:
        return 0.0 if not pred else 1.0
    dist = compute_levenshtein(list(ref), list(pred))
    return dist / float(len(ref))

def calculate_wer(ref, pred):
    ref_words = ref.strip().split()
    pred_words = pred.strip().split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    dist = compute_levenshtein(ref_words, pred_words)
    return dist / float(len(ref_words))

# -----------------------------------------------------------------------------
# Dataset & Custom Collator with Loss Masking (-100)
# -----------------------------------------------------------------------------
class CleanCompositeMedOCRDataset(Dataset):
    def __init__(self, manifest_path):
        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line.strip()))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = BASE_DIR / item["image"]
        human_prompt = ""
        target_text = ""
        for msg in item["conversations"]:
            if msg["from"] == "human":
                human_prompt = msg["value"].replace("<image>\n", "").strip()
            elif msg["from"] == "gpt":
                target_text = msg["value"].strip()
                
        img = Image.open(img_path).convert("RGB")
        return {
            "image": img,
            "prompt": human_prompt if human_prompt else "Transcribe the prescription text in the image accurately.",
            "target_text": target_text
        }

class Collator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", None)
        
    def __call__(self, batch):
        images = [item["image"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        targets = [item["target_text"] for item in batch]
        
        formatted_prompts = []
        for p in prompts:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
            if hasattr(self.processor, "apply_chat_template"):
                p_str = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
            else:
                p_str = f"<image>\n{p}\nAssistant: "
            formatted_prompts.append(p_str)
            
        full_texts = [f"{p}{t}" for p, t in zip(formatted_prompts, targets)]
        
        inputs = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        
        labels = inputs["input_ids"].clone()
        
        # Mask prompt token IDs with -100 so cross-entropy loss is computed exclusively on transcription output
        for i, p_str in enumerate(formatted_prompts):
            if self.tokenizer is not None:
                p_ids = self.tokenizer.encode(p_str, add_special_tokens=False)
                p_len = len(p_ids)
            else:
                p_len = 15
            labels[i, :p_len] = -100
            
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
            
        inputs["labels"] = labels
        return inputs

# -----------------------------------------------------------------------------
# Evaluation Loop (Loss, CER, WER)
# -----------------------------------------------------------------------------
def evaluate(model, val_dataset, processor, device, dtype):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    total_cer = 0.0
    total_wer = 0.0
    num_eval_samples = min(50, len(val_dataset))
    
    collator = Collator(processor)
    eval_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, list(range(num_eval_samples))),
        batch_size=1,
        shuffle=False,
        collate_fn=collator
    )
    
    with torch.no_grad():
        for batch in eval_loader:
            inputs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.float32 and k in ["pixel_values", "images"]:
                        inputs[k] = v.to(device=device, dtype=dtype)
                    else:
                        inputs[k] = v.to(device=device)
                else:
                    inputs[k] = v
                    
            with torch.amp.autocast('cuda', dtype=dtype):
                outputs = model(**inputs)
                loss = outputs.loss
                if loss is not None and not math.isnan(loss.item()):
                    total_loss += loss.item()
                    total_steps += 1
                    
    # Compute CER and WER on decoded evaluation predictions
    for idx in range(num_eval_samples):
        item = val_dataset[idx]
        img = item["image"]
        ref_text = item["target_text"]
        prompt = item["prompt"]
        
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        p_str = processor.apply_chat_template(msgs, add_generation_prompt=True) if hasattr(processor, "apply_chat_template") else f"<image>\n{prompt}\nAssistant: "
        
        eval_inputs = processor(text=[p_str], images=[img], return_tensors="pt")
        tensor_inputs = {}
        for k, v in eval_inputs.items():
            if isinstance(v, torch.Tensor):
                tensor_inputs[k] = v.to(device=device, dtype=dtype) if (v.dtype == torch.float32 and k in ["pixel_values", "images"]) else v.to(device=device)
                
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                generated_ids = model.generate(**tensor_inputs, max_new_tokens=48)
                
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if pred_text.startswith(prompt):
            pred_text = pred_text[len(prompt):].strip()
            
        sample_cer = calculate_cer(ref_text, pred_text)
        sample_wer = calculate_wer(ref_text, pred_text)
        
        total_cer += sample_cer
        total_wer += sample_wer
        
    model.train()
    avg_loss = total_loss / max(1, total_steps)
    avg_cer = total_cer / float(num_eval_samples)
    avg_wer = total_wer / float(num_eval_samples)
    
    return avg_loss, avg_cer, avg_wer

# -----------------------------------------------------------------------------
# Main Fine-Tuning Execution Function
# -----------------------------------------------------------------------------
def main():
    print("=" * 85, flush=True)
    print("Native PaddleOCR-VL-1.6 Fine-Tuning on Leak-Free Composite Dataset", flush=True)
    print("=" * 85, flush=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()
    dtype = torch.float16 if use_amp else torch.float32
    print(f"Hardware Environment: Device = {device} | Precision = AMP FP16 ({dtype})", flush=True)
    
    # Step 1: Load Datasets
    print("\n[Phase 1] Loading Clean Composite Datasets...", flush=True)
    train_dataset = CleanCompositeMedOCRDataset(TRAIN_MANIFEST)
    val_dataset = CleanCompositeMedOCRDataset(VAL_MANIFEST)
    print(f"Dataset Loaded: Train = {len(train_dataset)} samples | Val = {len(val_dataset)} samples", flush=True)
    
    # Step 2: Load Processor & Model
    print(f"\n[Phase 2] Loading PaddleOCR-VL-1.6 Architecture from {MODEL_PATH}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    print("Base PaddleOCR-VL-1.6 model loaded into memory.", flush=True)
    
    # Step 3: LoRA & Gradient Checkpointing Configuration
    print("\n[Phase 3] Configuring PEFT LoRA and 4GB VRAM Protections...", flush=True)
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )
    
    total_training_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_training_steps * 0.05)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    print("\n[Phase 4] Launching Fine-Tuning Loop...", flush=True)
    print("-" * 105, flush=True)
    print(f"{'Epoch':<6} | {'Step':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val CER':<10} | {'Val WER':<10} | {'Status'}", flush=True)
    print("-" * 105, flush=True)
    
    best_val_cer = float("inf")
    global_step = 0
    start_time = time.time()
    
    model.train()
    optimizer.zero_grad()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        running_loss = 0.0
        step_in_epoch = 0
        
        for step, batch in enumerate(train_loader):
            step_in_epoch += 1
            
            # VRAM protection: periodic cache clearance
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
                    
            with torch.amp.autocast('cuda', dtype=dtype):
                outputs = model(**inputs)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                
            scaler.scale(loss).backward()
            running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            if (step + 1) % 4 == 0 or step == len(train_loader) - 1:
                print(f"Batch {step+1}/{len(train_loader)} | Micro-loss: {loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f}", flush=True)
                
            if step_in_epoch % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                avg_train_loss = running_loss / GRADIENT_ACCUMULATION_STEPS
                running_loss = 0.0
                
                # Validation evaluation step
                if global_step % EVAL_EVERY_STEPS == 0 or global_step == total_training_steps:
                    print(f"\n---> Running Periodic Validation at Step {global_step}...", flush=True)
                    val_loss, val_cer, val_wer = evaluate(model, val_dataset, processor, device, dtype)
                    
                    status = "Standard"
                    if val_cer < best_val_cer:
                        best_val_cer = val_cer
                        status = "Saved Best Checkpoint"
                        model.save_pretrained(CHECKPOINT_DIR)
                        processor.save_pretrained(CHECKPOINT_DIR)
                        
                    print(f"{epoch:<6} | {global_step:<8} | {avg_train_loss:<12.4f} | {val_loss:<10.4f} | {val_cer:<10.4f} | {val_wer:<10.4f} | {status}", flush=True)
                else:
                    if global_step % 5 == 0:
                        print(f"{epoch:<6} | {global_step:<8} | {avg_train_loss:<12.4f} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | Progress Step", flush=True)
                    
    total_time = time.time() - start_time
    print("-" * 105, flush=True)
    print(f"[OK] Training complete in {total_time/60.0:.2f} minutes.", flush=True)
    print(f"Best Validation CER: {best_val_cer:.4f}", flush=True)
    print(f"Best Checkpoint Location: {CHECKPOINT_DIR}", flush=True)

if __name__ == "__main__":
    main()
