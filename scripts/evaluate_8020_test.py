#!/usr/bin/env python3
"""
evaluate_8020_test.py

Evaluates the fine-tuned Prescription OCR model on the 20% test split (120 test samples).
Calculates and outputs:
1. Character Error Rate (CER %)
2. Word Error Rate (WER %)
3. Token-level F1 Score (Precision, Recall, F1)
4. Jaccard Similarity (Token IoU)
5. Structural Output Quality / Exact Match %

Saves output report to reports/prescription_8020_test_results.json.
"""

import os
import sys
import json
import math
import re
import gc
from pathlib import Path
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

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

from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

def find_project_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    candidates = [current_dir.parent, Path.cwd()]
    for cand in candidates:
        if (cand / "processed" / "test_20.jsonl").exists() or (cand / "models" / "PaddleOCR-VL-1.6").exists():
            return cand
    return current_dir.parent

sys.path.insert(0, str(find_project_root()))
try:
    from scripts.dataset_loader import preprocess_image_aspect_ratio
except ImportError:
    def preprocess_image_aspect_ratio(image, max_dim=768):
        w, h = image.size
        scale = max_dim / float(max(w, h))
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return image


def compute_levenshtein(seq1, seq2):
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

def calculate_cer(ref: str, pred: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    dist = compute_levenshtein(list(ref), list(pred))
    return dist / float(len(ref))

def calculate_wer(ref: str, pred: str) -> float:
    ref_words = ref.strip().split()
    pred_words = pred.strip().split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    dist = compute_levenshtein(ref_words, pred_words)
    return dist / float(len(ref_words))

def tokenize_words(text: str):
    return [w.lower() for w in re.findall(r'\b\w+\b', text)]

def calculate_f1_score(ref: str, pred: str):
    ref_tokens = tokenize_words(ref)
    pred_tokens = tokenize_words(pred)

    if not ref_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0

    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    overlap = 0
    for t in pred_tokens:
        if ref_counts.get(t, 0) > 0:
            overlap += 1
            ref_counts[t] -= 1

    precision = overlap / float(len(pred_tokens))
    recall = overlap / float(len(ref_tokens))

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def calculate_jaccard_similarity(ref: str, pred: str) -> float:
    set_ref = set(tokenize_words(ref))
    set_pred = set(tokenize_words(pred))

    if not set_ref and not set_pred:
        return 1.0
    if not set_ref or not set_pred:
        return 0.0

    intersection = set_ref.intersection(set_pred)
    union = set_ref.union(set_pred)

    return len(intersection) / float(len(union))

def is_exact_match(ref: str, pred: str) -> bool:
    ref_clean = re.sub(r'\s+', ' ', ref.strip().lower())
    pred_clean = re.sub(r'\s+', ' ', pred.strip().lower())
    return ref_clean == pred_clean

def extract_and_parse_json(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    try:
        fixed = text.strip()
        if not fixed.endswith('}'):
            fixed += '}'
        return json.loads(fixed)
    except Exception:
        return None

def extract_medication_items(data):
    if isinstance(data, dict):
        if "medications" in data and isinstance(data["medications"], list):
            return data["medications"]
        elif "medication" in data:
            if isinstance(data["medication"], list):
                return data["medication"]
            elif isinstance(data["medication"], dict):
                return [data["medication"]]
    elif isinstance(data, list):
        return data
    return []

def is_semantic_json_match(ref: str, pred: str) -> bool:
    ref_obj = extract_and_parse_json(ref)
    pred_obj = extract_and_parse_json(pred)
    if ref_obj is not None and pred_obj is not None:
        return ref_obj == pred_obj
    return is_exact_match(ref, pred)

def calculate_field_entity_metrics(ref_text: str, pred_text: str):
    ref_obj = extract_and_parse_json(ref_text)
    pred_obj = extract_and_parse_json(pred_text)

    if not ref_obj and not pred_obj:
        return 1.0, 1.0, 1.0, 100.0
    if not ref_obj or not pred_obj:
        return 0.0, 0.0, 0.0, 0.0

    ref_meds = extract_medication_items(ref_obj)
    pred_meds = extract_medication_items(pred_obj)

    ref_tuples = []
    for item in ref_meds:
        if isinstance(item, dict):
            for k, v in item.items():
                if v is not None and str(v).lower() not in ["null", "not specified", "none", ""]:
                    norm_k = "dosage" if k in ["dose", "dosage"] else ("frequency" if k in ["freq", "frequency"] else k)
                    ref_tuples.append((norm_k, str(v).strip().lower()))

    pred_tuples = []
    for item in pred_meds:
        if isinstance(item, dict):
            for k, v in item.items():
                if v is not None and str(v).lower() not in ["null", "not specified", "none", ""]:
                    norm_k = "dosage" if k in ["dose", "dosage"] else ("frequency" if k in ["freq", "frequency"] else k)
                    pred_tuples.append((norm_k, str(v).strip().lower()))

    if not ref_tuples and not pred_tuples:
        return 1.0, 1.0, 1.0, 100.0
    if not ref_tuples or not pred_tuples:
        return 0.0, 0.0, 0.0, 0.0

    ref_counts = {}
    for t in ref_tuples:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    overlap = 0
    for t in pred_tuples:
        if ref_counts.get(t, 0) > 0:
            overlap += 1
            ref_counts[t] -= 1

    prec = overlap / float(len(pred_tuples))
    rec = overlap / float(len(ref_tuples))
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    field_acc = (overlap / float(max(len(ref_tuples), len(pred_tuples)))) * 100.0
    return prec, rec, f1, field_acc

def evaluate_test_split():
    root_dir = find_project_root()
    base_model_path = root_dir / "models" / "PaddleOCR-VL-1.6"
    
    adapter_path = root_dir / "checkpoints" / "prescription_ocr_8020_best"
    if not adapter_path.exists():
        adapter_path = root_dir / "checkpoints" / "stage3_fullpage_best"

    test_manifest_path = root_dir / "processed" / "test_20.jsonl"
    report_dir = root_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json_file = report_dir / "prescription_8020_test_results.json"
    report_csv_file = report_dir / "prescription_8020_test_results.csv"

    print("=" * 95, flush=True)
    print("  PRESCRIPTION OCR 20% TEST SPLIT EVALUATION SUITE (OPTIMIZED FIELD-LEVEL)", flush=True)
    print("=" * 95, flush=True)
    print(f"Base Model Path   : {base_model_path}", flush=True)
    print(f"Adapter Path      : {adapter_path}", flush=True)
    print(f"Test Manifest     : {test_manifest_path}", flush=True)
    print("-" * 95, flush=True)

    if not test_manifest_path.exists():
        print(f"[ERROR] Test manifest not found at: {test_manifest_path}", flush=True)
        sys.exit(1)

    records = []
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    print(f"Loaded {len(records)} test records from manifest.", flush=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print(f"Loading Base Model & Adapter into {device}...", flush=True)
    processor = AutoProcessor.from_pretrained(str(base_model_path), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    if adapter_path.exists():
        print(f"Attaching LoRA Adapter weights from '{adapter_path}'...", flush=True)
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    else:
        model = base_model

    model.to(device)
    model.eval()

    total_cer = 0.0
    total_wer = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_jaccard = 0.0
    exact_matches = 0

    total_field_prec = 0.0
    total_field_rec = 0.0
    total_field_f1 = 0.0
    total_field_acc = 0.0
    valid_json_count = 0

    sample_results = []
    prompt = "Extract all prescription text, drug names, and medical details from this image accurately."

    print("\nExecuting Test Set Inference with High-Res (max_dim=768), max_new_tokens=768, num_beams=1, repetition_penalty=1.05...", flush=True)
    for idx, rec in enumerate(records, start=1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        img_path = rec["image_path"]
        ref_text = rec.get("ocr_structured_json", "").strip()
        if not ref_text:
            ref_text = rec.get("structured_json", "").strip()
        if not ref_text:
            ref_text = rec.get("text", "").strip()

        img = Image.open(img_path).convert("RGB")
        img = preprocess_image_aspect_ratio(img, max_dim=512)

        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        if hasattr(processor, "apply_chat_template"):
            p_str = processor.apply_chat_template(msgs, add_generation_prompt=True)
        else:
            p_str = f"<image>\n{prompt}\nAssistant: "

        pred_text = ""
        try:
            inputs = processor(text=[p_str], images=[img], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            in_len = inputs["input_ids"].shape[1]
            eos_id = getattr(processor.tokenizer, "eos_token_id", 2)
            pad_id = getattr(processor.tokenizer, "pad_token_id", 0)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.05,
                    no_repeat_ngram_size=6,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id
                )

            new_ids = generated_ids[0][in_len:]
            pred_text = processor.decode(new_ids, skip_special_tokens=True).strip()

            del inputs, generated_ids
        except Exception as e:
            print(f"  [WARNING] Sample {idx} failed with error: {e}", flush=True)
            pred_text = ""

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cer = calculate_cer(ref_text, pred_text)
        wer = calculate_wer(ref_text, pred_text)
        prec, rec_val, f1 = calculate_f1_score(ref_text, pred_text)
        jaccard = calculate_jaccard_similarity(ref_text, pred_text)
        exact = is_semantic_json_match(ref_text, pred_text)

        f_prec, f_rec, f_f1, f_acc = calculate_field_entity_metrics(ref_text, pred_text)
        if extract_and_parse_json(pred_text) is not None:
            valid_json_count += 1

        total_cer += cer
        total_wer += wer
        total_precision += prec
        total_recall += rec_val
        total_f1 += f1
        total_jaccard += jaccard
        if exact:
            exact_matches += 1

        total_field_prec += f_prec
        total_field_rec += f_rec
        total_field_f1 += f_f1
        total_field_acc += f_acc

        if idx % 1 == 0 or idx == len(records):
            curr_cer = round((total_cer / idx) * 100.0, 2)
            curr_ff1 = round(total_field_f1 / idx, 4)
            print(f"  [{idx:3d}/{len(records)}] Running Avg CER: {curr_cer}% | Field Entity F1: {curr_ff1}", flush=True)

        sample_results.append({
            "sample_id": idx,
            "image": Path(img_path).name,
            "reference": ref_text,
            "prediction": pred_text,
            "cer_pct": round(cer * 100.0, 2),
            "wer_pct": round(wer * 100.0, 2),
            "token_f1_score": round(f1, 4),
            "field_entity_precision": round(f_prec, 4),
            "field_entity_recall": round(f_rec, 4),
            "field_entity_f1_score": round(f_f1, 4),
            "field_accuracy_pct": round(f_acc, 2),
            "jaccard_similarity": round(jaccard, 4),
            "exact_match": exact
        })

    num_samples = len(records)
    avg_cer_pct = (total_cer / max(1, num_samples)) * 100.0
    avg_wer_pct = (total_wer / max(1, num_samples)) * 100.0
    avg_precision = total_precision / max(1, num_samples)
    avg_recall = total_recall / max(1, num_samples)
    avg_f1 = total_f1 / max(1, num_samples)
    avg_jaccard = total_jaccard / max(1, num_samples)
    exact_match_pct = (exact_matches / max(1, num_samples)) * 100.0

    avg_field_prec = total_field_prec / max(1, num_samples)
    avg_field_rec = total_field_rec / max(1, num_samples)
    avg_field_f1 = total_field_f1 / max(1, num_samples)
    avg_field_acc = total_field_acc / max(1, num_samples)
    valid_json_pct = (valid_json_count / max(1, num_samples)) * 100.0

    summary_metrics = {
        "dataset_split": "20% Test Split (120 samples)",
        "total_test_samples": num_samples,
        "character_error_rate_cer_pct": round(avg_cer_pct, 2),
        "word_error_rate_wer_pct": round(avg_wer_pct, 2),
        "string_token_precision": round(avg_precision, 4),
        "string_token_recall": round(avg_recall, 4),
        "string_token_f1_score": round(avg_f1, 4),
        "field_entity_precision": round(avg_field_prec, 4),
        "field_entity_recall": round(avg_field_rec, 4),
        "field_entity_f1_score": round(avg_field_f1, 4),
        "field_accuracy_pct": round(avg_field_acc, 2),
        "valid_json_syntax_pct": round(valid_json_pct, 2),
        "jaccard_similarity": round(avg_jaccard, 4),
        "exact_match_structural_quality_pct": round(exact_match_pct, 2),
        "exact_matches_count": exact_matches
    }

    final_report = {
        "summary": summary_metrics,
        "samples": sample_results
    }

    with open(report_json_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    import csv
    with open(report_csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "image", "reference", "prediction",
            "cer_pct", "wer_pct", "token_f1_score",
            "field_entity_precision", "field_entity_recall", "field_entity_f1_score", "field_accuracy_pct",
            "jaccard_similarity", "exact_match"
        ])
        for s in sample_results:
            writer.writerow([
                s["sample_id"], s["image"], s["reference"], s["prediction"],
                s["cer_pct"], s["wer_pct"], s["token_f1_score"],
                s["field_entity_precision"], s["field_entity_recall"], s["field_entity_f1_score"], s["field_accuracy_pct"],
                s["jaccard_similarity"], s["exact_match"]
            ])

    print("\n" + "=" * 95, flush=True)
    print("  FINAL 20% TEST SPLIT VERIFICATION METRICS REPORT (OPTIMIZED FIELD-LEVEL)", flush=True)
    print("=" * 95, flush=True)
    print(f"Total Test Samples                  : {num_samples}", flush=True)
    print(f"Field Entity F1 Score               : {avg_field_f1:.4f}", flush=True)
    print(f"Field Entity Precision              : {avg_field_prec:.4f}", flush=True)
    print(f"Field Entity Recall                 : {avg_field_rec:.4f}", flush=True)
    print(f"Field Entity Accuracy (%)           : {avg_field_acc:.2f}%", flush=True)
    print(f"Valid JSON Syntax (%)               : {valid_json_pct:.2f}%", flush=True)
    print(f"Character Error Rate (CER %)        : {avg_cer_pct:.2f}%", flush=True)
    print(f"Word Error Rate (WER %)             : {avg_wer_pct:.2f}%", flush=True)
    print(f"Structural Quality / Exact Match    : {exact_match_pct:.2f}% ({exact_matches}/{num_samples})", flush=True)
    print(f"JSON Report Saved To                : {report_json_file.as_posix()}", flush=True)
    print(f"CSV Report Saved To                 : {report_csv_file.as_posix()}", flush=True)
    print("=" * 95, flush=True)

if __name__ == "__main__":
    evaluate_test_split()

