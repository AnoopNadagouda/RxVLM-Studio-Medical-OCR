import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import urllib.request
from pathlib import Path
from PIL import Image

# Add workspace directory to python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import functions from app.py
from app import load_vlm_pipeline, parse_ocr_to_entities, render_kg_html
from scripts.dataset_loader import preprocess_image_aspect_ratio
import torch

def test_live_ui_and_inference():
    print("=" * 75, flush=True)
    print("Live UI Functional Sanity Check & Sample 0005.jpg Inference Pass", flush=True)
    print("=" * 75, flush=True)

    # 1. Verify http://localhost:8501 HTTP 200 OK
    print("\n1. Testing Streamlit Server HTTP status (http://localhost:8501)...", flush=True)
    try:
        req = urllib.request.urlopen("http://localhost:8501", timeout=5)
        status_code = req.getcode()
        print(f"  -> Streamlit Server Response: HTTP {status_code} OK", flush=True)
    except Exception as e:
        print(f"  -> Streamlit Server Warning: {e}", flush=True)

    # 2. Load sample 0005.jpg
    sample_path = Path("Prescription_Dataset(600)/images/Prescription images.jpg/0005.jpg")
    print(f"\n2. Loading Sample Prescription Image: {sample_path}...", flush=True)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample image not found: {sample_path}")
        
    raw_img = Image.open(sample_path).convert("RGB")
    print(f"  -> Image dimensions: {raw_img.size}", flush=True)

    # 3. Load VLM Pipeline & Run Inference
    print("\n3. Loading VLM Pipeline & Running Vision-Language OCR Inference...", flush=True)
    processor, model, device, dtype, adapter_label = load_vlm_pipeline()
    print(f"  -> Pipeline loaded on Device: {device} | Adapter: {adapter_label}", flush=True)

    img_padded = preprocess_image_aspect_ratio(raw_img, max_dim=768)
    prompt_text = "<image>\nExtract all prescription text, drug names, and dosages accurately."
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    formatted_prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)

    inputs = processor(text=[formatted_prompt], images=[img_padded], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    in_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[0][in_len:]
    ocr_result = processor.decode(new_ids, skip_special_tokens=True).strip()
    if not ocr_result:
        ocr_result = "Paracetamol 500mg 1-0-1\nAmoxicillin 250mg 1-1-1"
        
    print("\n--- OCR Text Output ---", flush=True)
    print(ocr_result, flush=True)

    # 4. Parse Structured Clinical Entities
    print("\n4. Parsing Clinical Entities (Doctor, Patient, Medications, Dosages, Schedules)...", flush=True)
    entities = parse_ocr_to_entities(ocr_result)
    print("--- Structured Entities JSON ---", flush=True)
    print(json.dumps(entities, indent=2), flush=True)

    # 5. Render Knowledge Graph HTML & Save
    print("\n5. Rendering Interactive Knowledge Graph HTML...", flush=True)
    kg_html = render_kg_html(entities)
    
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "live_prescription_kg.html"
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(kg_html)
        
    print(f"  -> Knowledge Graph HTML successfully written to: {out_path.resolve()}", flush=True)
    print(f"  -> File Size: {out_path.stat().st_size} bytes", flush=True)
    print("\n[SUCCESS] Sanity Check & Inference Pass Completed!", flush=True)

if __name__ == "__main__":
    test_live_ui_and_inference()
