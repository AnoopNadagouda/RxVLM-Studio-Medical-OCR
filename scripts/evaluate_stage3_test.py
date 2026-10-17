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
CHECKPOINT_PATH = "./checkpoints/stage3_fullpage_best"

TEST_MANIFESTS = {
    "Prescription_Dataset_600_Test": "processed/test.jsonl",
    "Master_VLM_Test_Split": "C:/Users/ANOOP/Downloads/ocr_manifests/vlm_test.jsonl"
}

REPORT_PATH = Path("./reports/stage3_test_metrics_breakdown.json")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

def levenshtein_distance(seq1, seq2):
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
        "line_level_cer": round(line_cer, 4)
    }

def evaluate_dataset(manifest_key, manifest_path, model, processor, device, max_samples=None):
    print(f"\n" + "=" * 75)
    print(f"Evaluating Manifest: {manifest_key} ({manifest_path})")
    print("=" * 75)
    
    p = Path(manifest_path)
    if not p.exists():
        print(f"[ERROR] Manifest file not found: {p}")
        return None

    records = []
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    print(f"Evaluating {len(records)} test samples...")
    results = []
    total_cer = 0.0
    total_wer = 0.0
    exact_matches = 0
    line_count_matches = 0
    total_dosage_acc = 0.0

    start_time = time.time()

    for idx, rec in enumerate(records, 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        img_path = rec.get('image_path') or rec.get('image')
        ref_text = rec.get('text', '') or rec.get('description_english', '')
        prompt_text = "<image>\nExtract all prescription text, drug names, and medical details from this image accurately."
        
        if 'conversations' in rec:
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

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
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

        if idx % 10 == 0 or idx == 1 or idx == len(records):
            print(f"[{idx}/{len(records)}] CER: {cer:.4f} | WER: {wer:.4f} | Line Match: {line_struct['line_count_match']} | Dosage Acc: {dosage_acc:.2f}", flush=True)

    elapsed = time.time() - start_time
    avg_cer = total_cer / max(1, len(records))
    avg_wer = total_wer / max(1, len(records))
    exact_match_acc = (exact_matches / float(max(1, len(records)))) * 100.0
    line_match_acc = (line_count_matches / float(max(1, len(records)))) * 100.0
    avg_dosage_acc = (total_dosage_acc / float(max(1, len(records)))) * 100.0

    summary = {
        "manifest_name": manifest_key,
        "manifest_path": manifest_path,
        "total_evaluated": len(records),
        "evaluation_time_sec": round(elapsed, 2),
        "sec_per_sample": round(elapsed / max(1, len(records)), 3),
        "mean_cer": round(avg_cer, 4),
        "mean_wer": round(avg_wer, 4),
        "exact_match_accuracy_pct": round(exact_match_acc, 2),
        "line_count_structural_accuracy_pct": round(line_match_acc, 2),
        "dosage_syntax_accuracy_pct": round(avg_dosage_acc, 2)
    }

    return {
        "summary": summary,
        "details": results
    }

def main():
    max_samples = 100
    if len(sys.argv) > 1:
        try:
            max_samples = int(sys.argv[1])
        except ValueError:
            pass

    print("=" * 75, flush=True)
    print("STAGE 3 FULL-PAGE OCR TEST EVALUATION SUITE (`stage3_fullpage_best`)", flush=True)
    print("=" * 75, flush=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Device: {device} | Precision: {dtype}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[1/3] Loading Processor...", flush=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    print(f"\n[2/3] Loading Base Model from {BASE_MODEL_PATH}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    ckpt_path = Path(CHECKPOINT_PATH)
    if ckpt_path.exists() and (ckpt_path / "adapter_config.json").exists():
        print(f"Loading Stage 3 Master LoRA Adapter from {CHECKPOINT_PATH}...", flush=True)
        model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
    else:
        print(f"[Warning] LoRA Adapter not found at {CHECKPOINT_PATH}. Evaluating base model.", flush=True)

    model.to(device)
    model.eval()

    print("\n[3/3] Running Evaluation on Test Manifests...", flush=True)
    all_reports = {}

    for key, path in TEST_MANIFESTS.items():
        res = evaluate_dataset(key, path, model, processor, device, max_samples=max_samples)
        if res:
            all_reports[key] = res

    final_output = {
        "checkpoint": str(CHECKPOINT_PATH),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluations": {k: v["summary"] for k, v in all_reports.items()}
    }

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"summary": final_output, "details": {k: v["details"] for k, v in all_reports.items()}}, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print("FINAL ACCURACY EVALUATION SUITE BREAKDOWN (`stage3_fullpage_best`)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Metric':<38} | {'Prescription 600 Test':<20} | {'Master VLM Test Split':<20}", flush=True)
    print("-" * 85, flush=True)

    p600 = all_reports.get("Prescription_Dataset_600_Test", {}).get("summary", {})
    mvlm = all_reports.get("Master_VLM_Test_Split", {}).get("summary", {})

    print(f"{'Evaluated Samples':<38} | {str(p600.get('total_evaluated', 'N/A')):<20} | {str(mvlm.get('total_evaluated', 'N/A')):<20}", flush=True)
    print(f"{'Character Error Rate (CER)':<38} | {str(p600.get('mean_cer', 'N/A')):<20} | {str(mvlm.get('mean_cer', 'N/A')):<20}", flush=True)
    print(f"{'Word Error Rate (WER)':<38} | {str(p600.get('mean_wer', 'N/A')):<20} | {str(mvlm.get('mean_wer', 'N/A')):<20}", flush=True)
    print(f"{'Exact Match (EM) %':<38} | {str(p600.get('exact_match_accuracy_pct', 'N/A')) + '%':<20} | {str(mvlm.get('exact_match_accuracy_pct', 'N/A')) + '%':<20}", flush=True)
    print(f"{'Line Structural Match %':<38} | {str(p600.get('line_count_structural_accuracy_pct', 'N/A')) + '%':<20} | {str(mvlm.get('line_count_structural_accuracy_pct', 'N/A')) + '%':<20}", flush=True)
    print(f"{'Dosage Syntax Accuracy %':<38} | {str(p600.get('dosage_syntax_accuracy_pct', 'N/A')) + '%':<20} | {str(mvlm.get('dosage_syntax_accuracy_pct', 'N/A')) + '%':<20}", flush=True)
    print("=" * 80, flush=True)
    print(f"\nDetailed evaluation report saved to: {REPORT_PATH.resolve()}", flush=True)

if __name__ == "__main__":
    main()
