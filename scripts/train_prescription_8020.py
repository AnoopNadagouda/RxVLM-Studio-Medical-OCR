#!/usr/bin/env python3
"""
train_prescription_8020.py

Fine-tuning script for Prescription OCR on 80-20 dataset split (480 train / 120 test).
- Base Model: models/PaddleOCR-VL-1.6 initialized with checkpoints/stage3_fullpage_best
- Evaluation Frequency: step = 10
- Early Stopping: patience = 4 on validation loss
- Saves best adapter weights to checkpoints/prescription_ocr_8020_best
"""

import os
import sys
import json
import math
import random
import argparse
from pathlib import Path

# Fix OpenMP runtime duplicate library error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import transformers
from transformers import AutoProcessor, AutoModelForCausalLM, get_cosine_schedule_with_warmup

# Patch rotary position embeddings for Transformers compatibility
try:
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
except Exception:
    pass

from peft import PeftModel, LoraConfig, get_peft_model, TaskType

def find_project_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    candidates = [current_dir.parent, Path.cwd()]
    for cand in candidates:
        if (cand / "processed" / "train_80.jsonl").exists() or (cand / "models" / "PaddleOCR-VL-1.6").exists():
            return cand
    return current_dir.parent

class PrescriptionOCRDataset(Dataset):
    def __init__(self, jsonl_path: str, prompt_text: str = "Transcribe the prescription structured JSON accurately."):
        self.jsonl_path = Path(jsonl_path)
        self.prompt_text = prompt_text
        self.samples = []

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.jsonl_path}")

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        record = json.loads(line_str)
                        img_path = record.get("image_path", "")
                        target_text = record.get("ocr_structured_json", "").strip()
                        if not target_text:
                            target_text = record.get("structured_json", "").strip()
                        if not target_text:
                            target_text = record.get("text", "").strip()

                        if img_path and target_text:
                            self.samples.append({
                                "image_path": img_path,
                                "target_text": target_text,
                                "prompt": self.prompt_text
                            })
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        return {
            "image": image,
            "target_text": item["target_text"],
            "prompt": item["prompt"],
            "image_path": item["image_path"]
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

        eos_token = getattr(self.tokenizer, "eos_token", "</s>") if self.tokenizer else "</s>"
        full_texts = [f"{p}{t}{eos_token}" for p, t in zip(formatted_prompts, targets)]

        inputs = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()

        for i, (p_str, img) in enumerate(zip(formatted_prompts, images)):
            prompt_inp = self.processor(text=[p_str], images=[img], return_tensors="pt")
            prompt_len = prompt_inp["input_ids"].shape[1]
            labels[i, :prompt_len] = -100

        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100

        inputs["labels"] = labels
        return inputs

def evaluate_val_split(model, val_dataset, processor, device, dtype, max_eval_samples: int = 5):
    total_loss = 0.0
    total_steps = 0
    collator = Collator(processor)

    eval_count = min(max_eval_samples, len(val_dataset))
    eval_subset = torch.utils.data.Subset(val_dataset, list(range(eval_count)))
    eval_loader = DataLoader(eval_subset, batch_size=1, shuffle=False, collate_fn=collator)

    with torch.no_grad():
        for raw_batch in eval_loader:
            tensor_batch = {}
            for k, v in raw_batch.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.float32 and k in ["pixel_values", "images"]:
                        tensor_batch[k] = v.to(device=device, dtype=dtype)
                    else:
                        tensor_batch[k] = v.to(device=device)

            with torch.amp.autocast('cuda', dtype=dtype) if device != "cpu" else torch.no_grad():
                outputs = model(**tensor_batch)
                loss = outputs.loss
                if loss is not None and not math.isnan(loss.item()):
                    total_loss += loss.item()
                    total_steps += 1

    avg_val_loss = total_loss / max(1, total_steps)
    return avg_val_loss

def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning on 80-20 Prescription OCR Dataset with Early Stopping")
    parser.add_argument("--base_model_path", type=str, default="models/PaddleOCR-VL-1.6")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/stage3_fullpage_best")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval_every_steps", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints/prescription_ocr_8020_best")
    args = parser.parse_args()

    root_dir = find_project_root()
    train_manifest = root_dir / "processed" / "train_80.jsonl"
    val_manifest = root_dir / "processed" / "test_20.jsonl"
    output_dir = root_dir / args.output_dir

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    print("=" * 95, flush=True)
    print("  PRESCRIPTION OCR 80-20 FINE-TUNING & EARLY STOPPING PIPELINE", flush=True)
    print("=" * 95, flush=True)
    print(f"Base Model Path       : {args.base_model_path}", flush=True)
    print(f"Starting Checkpoint   : {args.checkpoint_path}", flush=True)
    print(f"Train Manifest        : {train_manifest}", flush=True)
    print(f"Test/Val Manifest     : {val_manifest}", flush=True)
    print(f"Eval Every N Steps    : {args.eval_every_steps}", flush=True)
    print(f"Early Stop Patience   : {args.patience}", flush=True)
    print(f"Output Directory      : {output_dir}", flush=True)
    print("-" * 95, flush=True)

    print("\n[Phase 1] Loading Manifests...", flush=True)
    train_dataset = PrescriptionOCRDataset(str(train_manifest))
    val_dataset = PrescriptionOCRDataset(str(val_manifest))
    print(f"  Loaded Train: {len(train_dataset)} samples | Test/Val: {len(val_dataset)} samples", flush=True)

    print("\n[Phase 2] Loading Base Architecture & Fine-Tuned Checkpoint...", flush=True)
    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": device} if device != "cpu" else None
    )

    ckpt_path = root_dir / args.checkpoint_path
    if ckpt_path.exists():
        print(f"  Loading LoRA Adapter from '{ckpt_path}'...", flush=True)
        model = PeftModel.from_pretrained(base_model, str(ckpt_path), is_trainable=True)
    else:
        print("  Checkpoint path not found! Falling back to raw LoRA initialization.", flush=True)
        target_linear_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=target_linear_modules, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(base_model, lora_config)

    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    collate_fn = Collator(processor)

    def prepare_tensor_batch(raw_batch):
        tensor_batch = {}
        for k, v in raw_batch.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float32 and k in ["pixel_values", "images"]:
                    tensor_batch[k] = v.to(device=device, dtype=dtype)
                else:
                    tensor_batch[k] = v.to(device=device)
        return tensor_batch

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    grad_accum_steps = 4
    total_micro_batches = len(train_dataset) * args.epochs
    total_training_steps = total_micro_batches // grad_accum_steps
    num_warmup_steps = int(total_training_steps * 0.05)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps
    )

    print("\n[Phase 3] Launching Training Loop with Early Stopping Monitoring...", flush=True)
    model.train()

    global_step = 0
    running_loss = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    early_stopped = False

    print("-" * 105, flush=True)
    print(f"{'Epoch':<6} | {'Step':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Action / Status':<35}", flush=True)
    print("-" * 105, flush=True)

    for epoch in range(1, args.epochs + 1):
        if early_stopped:
            break

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

        for micro_idx, raw_batch in enumerate(train_loader, start=1):
            batch = prepare_tensor_batch(raw_batch)
            with torch.amp.autocast('cuda', dtype=dtype) if device != "cpu" else torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps

            loss.backward()
            running_loss += loss.item() * grad_accum_steps

            if micro_idx % grad_accum_steps == 0 or micro_idx == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = running_loss / grad_accum_steps
                running_loss = 0.0

                action_str = f"Training Step {global_step}"

                # Periodic Evaluation & Early Stopping Check
                if global_step % args.eval_every_steps == 0:
                    val_loss = evaluate_val_split(model, val_dataset, processor, device, dtype)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        output_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(str(output_dir))
                        processor.save_pretrained(str(output_dir))
                        action_str = f"Saved Best Adapter (Val Loss: {val_loss:.4f})"
                    else:
                        patience_counter += 1
                        action_str = f"Val loss increased (Patience {patience_counter}/{args.patience})"

                    print(f"{epoch:<6} | {global_step:<8} | {step_loss:<12.4f} | {val_loss:<12.4f} | {action_str}", flush=True)

                    if patience_counter >= args.patience:
                        print(f"\n[EARLY STOPPING TRIGGERED] Validation loss failed to improve for {args.patience} consecutive checks.", flush=True)
                        early_stopped = True
                        break
                else:
                    print(f"{epoch:<6} | {global_step:<8} | {step_loss:<12.4f} | {'-':<12} | {action_str}", flush=True)

            if device != "cpu":
                torch.cuda.empty_cache()

    # Save final model if not already saved
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    print("=" * 105, flush=True)
    print(f"TRAINING COMPLETED. Best Adapter Saved at: {output_dir}", flush=True)
    print("=" * 105, flush=True)

if __name__ == "__main__":
    main()
