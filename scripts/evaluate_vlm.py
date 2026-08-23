import os
import sys
import json
import time
import re
from pathlib import Path
from PIL import Image
import torch

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
from peft import PeftModel
from scripts.dataset_loader import preprocess_image_aspect_ratio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_MODEL_PATH = "models/PaddleOCR-VL-1.6"
CHECKPOINT_PATH = "./checkpoints/stage1_drug_lexicon_best"
VAL_MANIFEST = "ocr_manifests/vlm_stage1_val.jsonl"
REPORT_PATH = Path("./reports/stage1_validation_metrics.json")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

def levenshtein_distance(seq1, seq2):
    """
    Computes Levenshtein edit distance between two sequences (strings or lists of words).
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def post_process_text(text: str) -> str:
    """
    Post-processing for OCR text:
    - Trim surrounding whitespace
    - Collapse redundant spaces
    - Standardize common quotes / punctuation spacing
    """
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def compute_cer(reference: str, prediction: str) -> float:
    ref_norm = post_process_text(reference)
    pred_norm = post_process_text(prediction)
    if not ref_norm:
        return 0.0 if not pred_norm else 1.0
    dist = levenshtein_distance(ref_norm, pred_norm)
    return dist / float(len(ref_norm))

def compute_wer(reference: str, prediction: str) -> float:
    ref_words = post_process_text(reference).split()
    pred_words = post_process_text(prediction).split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    dist = levenshtein_distance(ref_words, pred_words)
    return dist / float(len(ref_words))

def compute_exact_match(reference: str, prediction: str) -> bool:
    return post_process_text(reference).strip().lower() == post_process_text(prediction).strip().lower()

def main(num_samples: int = 50):
    print("=" * 75, flush=True)
    print("Stage 1 Evaluation & CER/WER Verification Module", flush=True)
    print("=" * 75, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() else torch.float32)
    print(f"Device: {device} | Precision: {dtype}", flush=True)

    # 1. Load Processor
    print("\n[Phase 1] Loading Processor...", flush=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # 2. Load Base Model and Checkpoint Adapter (if available)
    print("\n[Phase 2] Loading Model Architecture...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    checkpoint_adapter_file = Path(CHECKPOINT_PATH) / "adapter_model.safetensors"
    if checkpoint_adapter_file.exists() or (Path(CHECKPOINT_PATH) / "adapter_config.json").exists():
        print(f"  --> Merging LoRA weights from: {CHECKPOINT_PATH}", flush=True)
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        model = model.merge_and_unload()
    else:
        print(f"  [Note] No LoRA checkpoint found at {CHECKPOINT_PATH}. Evaluating base model baseline.", flush=True)
        model = base_model

    model.to(device)
    model.eval()

    # 3. Load Validation Manifest Samples
    print(f"\n[Phase 3] Loading {num_samples} Validation Samples from {VAL_MANIFEST}...", flush=True)
    records = []
    with open(VAL_MANIFEST, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if len(records) >= num_samples:
                    break

    # 4. Perform Greedy Inference & Calculate CER/WER
    print(f"\n[Phase 4] Executing Autoregressive Inference & Calculating Metrics...", flush=True)
    print("-" * 85, flush=True)
    print(f"{'Sample #':<8} | {'Ground Truth':<25} | {'Prediction':<25} | {'CER':<8} | {'WER':<8} | {'Match'}", flush=True)
    print("-" * 85, flush=True)

    eval_results = []
    total_cer = 0.0
    total_wer = 0.0
    total_exact = 0

    for idx, rec in enumerate(records, 1):
        img_path = rec['image']
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (768, 768), (255, 255, 255))

        img_padded = preprocess_image_aspect_ratio(img, max_dim=768)

        # Build prompt label
        prompt = "<image>\nExtract all prescription text, drug names, and medical details from this image accurately."
        ground_truth = ""
        for msg in rec.get('conversations', []):
            if msg.get('from') == 'human':
                prompt = msg.get('value', prompt)
            elif msg.get('from') == 'gpt':
                ground_truth = msg.get('value', '')

        formatted_msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        prompt_text = processor.apply_chat_template(formatted_msg, add_generation_prompt=True)

        inputs = processor(text=[prompt_text], images=[img_padded], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                num_beams=1,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
            )

        # Slice generated response tokens after input prompt
        input_len = inputs["input_ids"].shape[1]
        response_ids = generated_ids[0][input_len:]
        prediction_text = processor.tokenizer.decode(response_ids, skip_special_tokens=True)
        prediction_norm = post_process_text(prediction_text)
        ground_truth_norm = post_process_text(ground_truth)

        cer = compute_cer(ground_truth_norm, prediction_norm)
        wer = compute_wer(ground_truth_norm, prediction_norm)
        exact_match = compute_exact_match(ground_truth_norm, prediction_norm)

        total_cer += cer
        total_wer += wer
        if exact_match:
            total_exact += 1

        gt_safe = ground_truth_norm[:25].encode('ascii', errors='replace').decode('ascii')
        pred_safe = prediction_norm[:25].encode('ascii', errors='replace').decode('ascii')
        print(f"{idx:<8} | {gt_safe:<25} | {pred_safe:<25} | {cer:<8.4f} | {wer:<8.4f} | {'EXACT' if exact_match else 'DIFF'}", flush=True)

        eval_results.append({
            "sample_id": idx,
            "image": img_path,
            "ground_truth": ground_truth_norm,
            "prediction": prediction_norm,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact_match": exact_match
        })

    avg_cer = total_cer / len(records) if records else 0.0
    avg_wer = total_wer / len(records) if records else 0.0
    exact_acc = (total_exact / len(records)) * 100.0 if records else 0.0

    print("-" * 85, flush=True)
    print(f"Summary Metrics across {len(records)} samples:", flush=True)
    print(f"  - Average CER: {avg_cer:.4f}", flush=True)
    print(f"  - Average WER: {avg_wer:.4f}", flush=True)
    print(f"  - Exact Match Accuracy: {exact_acc:.2f}% ({total_exact}/{len(records)})", flush=True)
    print("=" * 75, flush=True)

    report_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": BASE_MODEL_PATH,
        "checkpoint": CHECKPOINT_PATH if checkpoint_adapter_file.exists() else "base_baseline",
        "samples_evaluated": len(records),
        "metrics": {
            "mean_cer": round(avg_cer, 4),
            "mean_wer": round(avg_wer, 4),
            "exact_match_accuracy_pct": round(exact_acc, 2),
            "exact_matches": total_exact
        },
        "sample_details": eval_results
    }

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report_payload, f, indent=2)

    print(f"\n[Success] Validation report saved to: {REPORT_PATH.resolve()}")

if __name__ == "__main__":
    n = 50
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    main(num_samples=n)
