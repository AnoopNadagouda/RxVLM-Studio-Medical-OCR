import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

BASE_MODEL_PATH = "models/PaddleOCR-VL-1.6"
CHECKPOINT_PATH = "./checkpoints/stage2_multiline_best"
VAL_MANIFEST = "ocr_manifests/vlm_stage2_val.jsonl"
REPORT_PATH = Path("./reports/stage2_multiline_metrics.json")
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
    r"""
    Post-processing for OCR text:
    - Strip LaTeX formatting markup (\[, \], \text, \(, \))
    - Collapse redundant inline spaces and token repeats
    - Preserve line breaks (\n) for multiline evaluation
    """
    if not text:
        return ""
    text = text.replace(r"\[", "").replace(r"\]", "").replace(r"\(", "").replace(r"\)", "").replace(r"\text", "").replace("\\", "")
    text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    lines = [re.sub(r'[ \t]+', ' ', line.strip()) for line in text.split('\n') if line.strip()]
    return '\n'.join(lines).strip()

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

def extract_dosage_tokens(text: str):
    """
    Extract dosage patterns such as:
    - Frequencies: '1-0-1', '1-1-1', '1 - 0 - 1', '0-0-1', '1-0-0'
    - Strengths/quantities: '500mg', '500 mg', '10ml', '10 ml', '1 tab', '2 tabs', '1 cap'
    """
    if not text:
        return []
    text_lower = text.lower()
    dosage_freq_pattern = r'\b\d+\s*-\s*\d+\s*-\s*\d+\b'
    dosage_quant_pattern = r'\b\d+\s*(?:mg|ml|gm|g|mcg|tab|tabs|tablet|tablets|cap|caps|capsule|capsules)\b'
    
    freqs = re.findall(dosage_freq_pattern, text_lower)
    quants = re.findall(dosage_quant_pattern, text_lower)
    
    freqs_norm = [re.sub(r'\s+', '', f) for f in freqs]
    quants_norm = [re.sub(r'\s+', '', q) for q in quants]
    
    return freqs_norm + quants_norm

def compute_dosage_accuracy(reference: str, prediction: str):
    """
    Computes precision/recall for dosage patterns (e.g. '1-0-1', '500mg')
    """
    ref_dosages = extract_dosage_tokens(reference)
    pred_dosages = extract_dosage_tokens(prediction)
    
    if not ref_dosages:
        return 1.0 if not pred_dosages else 1.0
    
    matched = 0
    ref_copy = list(ref_dosages)
    for p in pred_dosages:
        if p in ref_copy:
            matched += 1
            ref_copy.remove(p)
            
    recall = matched / float(len(ref_dosages))
    return recall

def compute_line_structural_accuracy(reference: str, prediction: str):
    """
    Evaluates line-level structural metrics:
    - Line Count Exact Match: whether predicted number of lines equals reference lines.
    - Average Line-by-Line Edit Distance Ratio.
    """
    ref_lines = [l.strip() for l in reference.strip().split('\n') if l.strip()]
    pred_lines = [l.strip() for l in prediction.strip().split('\n') if l.strip()]
    
    line_count_match = len(ref_lines) == len(pred_lines)
    
    if not ref_lines:
        line_cer = 0.0 if not pred_lines else 1.0
    else:
        max_lines = max(len(ref_lines), len(pred_lines))
        total_cer = 0.0
        for i in range(max_lines):
            r = ref_lines[i] if i < len(ref_lines) else ""
            p = pred_lines[i] if i < len(pred_lines) else ""
            total_cer += compute_cer(r, p)
        line_cer = total_cer / float(max_lines)
        
    return {
        "line_count_match": line_count_match,
        "ref_line_count": len(ref_lines),
        "pred_line_count": len(pred_lines),
        "line_level_cer": line_cer
    }

def main(num_samples: int = 50):
    print("=" * 75, flush=True)
    print("Stage 2 Evaluation: Multi-Line CER, WER & Structural Accuracy Pipeline", flush=True)
    print("=" * 75, flush=True)

    # Memory efficient device handling
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Device: {device} | Precision: {dtype}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Load Processor
    print("\n[Phase 1] Loading Processor...", flush=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # 2. Load Model & Adapter
    print(f"\n[Phase 2] Loading Base Model from {BASE_MODEL_PATH}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    ckpt_path = Path(CHECKPOINT_PATH)
    if ckpt_path.exists() and (ckpt_path / "adapter_config.json").exists():
        print(f"Loading Stage 2 LoRA Adapter from {CHECKPOINT_PATH}...", flush=True)
        model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
    else:
        print(f"[Warning] LoRA Adapter not found at {CHECKPOINT_PATH}. Evaluator running on Base Model.", flush=True)

    try:
        model.to(device)
    except RuntimeError:
        print("[Memory Notice] Switching evaluator device to CPU to avoid VRAM conflict with training process...", flush=True)
        device = "cpu"
        dtype = torch.float32
        model.to(device)

    model.eval()

    # 3. Load Validation Manifest
    val_manifest_path = Path(VAL_MANIFEST)
    if not val_manifest_path.exists():
        val_manifest_path = Path("C:/Users/ANOOP/Downloads/ocr_manifests/vlm_stage2_val.jsonl")

    print(f"\n[Phase 3] Loading Validation Manifest from {val_manifest_path}...", flush=True)
    records = []
    with open(val_manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    eval_records = records[:num_samples] if num_samples else records
    print(f"Evaluating {len(eval_records)} validation samples...", flush=True)

    # 4. Evaluation Loop
    results = []
    total_cer = 0.0
    total_wer = 0.0
    exact_matches = 0
    line_count_matches = 0
    total_dosage_acc = 0.0

    print("\n[Phase 4] Executing Evaluation Loop...", flush=True)
    start_time = time.time()

    for idx, rec in enumerate(eval_records, 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        img_path = rec['image']
        
        ref_text = ""
        prompt_text = "<image>\nExtract all prescription text, drug names, and medical details from this image accurately."
        for msg in rec.get('conversations', []):
            if msg.get('from') == 'human':
                prompt_text = msg.get('value', prompt_text)
            elif msg.get('from') == 'gpt':
                ref_text = msg.get('value', '')

        try:
            raw_img = Image.open(img_path).convert("RGB")
        except Exception:
            raw_img = Image.new("RGB", (768, 768), (255, 255, 255))
            
        img_padded = preprocess_image_aspect_ratio(raw_img, max_dim=768)

        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        formatted_prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)

        inputs = processor(text=[formatted_prompt], images=[img_padded], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None
            )

        in_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[0][in_len:]
        pred_text = processor.decode(new_ids, skip_special_tokens=True).strip()

        cer = compute_cer(ref_text, pred_text)
        wer = compute_wer(ref_text, pred_text)
        exact_match = post_process_text(ref_text).strip().lower() == post_process_text(pred_text).strip().lower()
        
        dosage_acc = compute_dosage_accuracy(ref_text, pred_text)
        line_struct = compute_line_structural_accuracy(ref_text, pred_text)

        total_cer += cer
        total_wer += wer
        if exact_match:
            exact_matches += 1
        if line_struct["line_count_match"]:
            line_count_matches += 1
        total_dosage_acc += dosage_acc

        rec_res = {
            "id": idx,
            "image": img_path,
            "reference": ref_text,
            "prediction": pred_text,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact_match": exact_match,
            "line_structure": line_struct,
            "dosage_accuracy": round(dosage_acc, 4)
        }
        results.append(rec_res)

        if idx % 10 == 0 or idx == 1 or idx == len(eval_records):
            print(f"[{idx}/{len(eval_records)}] CER: {cer:.4f} | WER: {wer:.4f} | Line Match: {line_struct['line_count_match']} | Dosage Acc: {dosage_acc:.2f}", flush=True)

    elapsed = time.time() - start_time
    avg_cer = total_cer / max(1, len(eval_records))
    avg_wer = total_wer / max(1, len(eval_records))
    exact_match_acc = (exact_matches / float(max(1, len(eval_records)))) * 100.0
    line_match_acc = (line_count_matches / float(max(1, len(eval_records)))) * 100.0
    avg_dosage_acc = (total_dosage_acc / float(max(1, len(eval_records)))) * 100.0

    # Load validation loss from training_history.json if present
    val_loss = None
    history_file = Path(CHECKPOINT_PATH) / "training_history.json"
    if history_file.exists():
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                hist = json.load(f)
                val_conv = hist.get("val_convergence", [])
                if val_conv:
                    val_loss = val_conv[-1].get("val_loss")
        except Exception:
            pass

    summary_metrics = {
        "checkpoint": str(CHECKPOINT_PATH),
        "total_evaluated": len(eval_records),
        "evaluation_time_sec": round(elapsed, 2),
        "validation_loss": val_loss,
        "mean_cer": round(avg_cer, 4),
        "mean_wer": round(avg_wer, 4),
        "exact_match_accuracy_pct": round(exact_match_acc, 2),
        "line_count_structural_accuracy_pct": round(line_match_acc, 2),
        "dosage_syntax_accuracy_pct": round(avg_dosage_acc, 2)
    }

    report = {
        "summary": summary_metrics,
        "sample_details": results
    }

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 75, flush=True)
    print("STAGE 2 MULTI-LINE EVALUATION RESULTS SUMMARY", flush=True)
    print("=" * 75, flush=True)
    print(f" Checkpoint Path: {CHECKPOINT_PATH}", flush=True)
    print(f" Evaluated Samples: {len(eval_records)}", flush=True)
    print(f" Execution Latency: {elapsed:.2f} seconds ({elapsed/len(eval_records):.2f}s/sample)", flush=True)
    if val_loss is not None:
        print(f" Validation Loss: {val_loss:.4f}", flush=True)
    print(f" Character Error Rate (CER): {avg_cer:.4f}", flush=True)
    print(f" Word Error Rate (WER): {avg_wer:.4f}", flush=True)
    print(f" Exact Match Accuracy: {exact_match_acc:.2f}%", flush=True)
    print(f" Line Structural Accuracy: {line_match_acc:.2f}%", flush=True)
    print(f" Dosage Syntax Accuracy ('1-0-1', '500mg'): {avg_dosage_acc:.2f}%", flush=True)
    print(f"\nDetailed evaluation report saved to: {REPORT_PATH.resolve()}", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    num_eval = 50
    if len(sys.argv) > 1:
        try:
            num_eval = int(sys.argv[1])
        except ValueError:
            pass
    main(num_samples=num_eval)
