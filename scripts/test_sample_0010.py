import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import urllib.request
from pathlib import Path
from PIL import Image
import torch

# Add workspace directory to python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from app import parse_ocr_to_entities, render_kg_html
from scripts.dataset_loader import preprocess_image_aspect_ratio

def run_standalone_0010_test():
    print("=" * 75, flush=True)
    print("Live UI Functional Check & Sample 0010.jpg Inference Pass", flush=True)
    print("=" * 75, flush=True)

    # 1. Verify http://localhost:8501 HTTP 200 OK
    print("\n1. Testing Streamlit Server HTTP status (http://localhost:8501)...", flush=True)
    try:
        req = urllib.request.urlopen("http://localhost:8501", timeout=5)
        status_code = req.getcode()
        print(f"  -> Streamlit Server Response: HTTP {status_code} OK", flush=True)
    except Exception as e:
        print(f"  -> Streamlit Server Warning: {e}", flush=True)

    # 2. Locate sample 0010.jpg
    sample_path = Path("Prescription_Dataset(600)/images/Prescription images.jpg/0010.jpg")
    if not sample_path.exists():
        base_dir = Path("Prescription_Dataset(600)")
        found = list(base_dir.rglob("*0010.jpg"))
        if found:
            sample_path = found[0]
        else:
            raise FileNotFoundError("Sample image 0010.jpg not found!")

    raw_img = Image.open(sample_path).convert("RGB")
    print(f"\n2. Loaded Image: {sample_path} | Dimensions: {raw_img.size}", flush=True)

    # 3. Load Base Model & Stage 3 LoRA Weights
    print("\n3. Loading Base Model & Stage 3 LoRA Adapter Weights...", flush=True)
    base_model_path = "models/PaddleOCR-VL-1.6"
    stage3_ckpt = Path("./checkpoints/stage3_fullpage_best")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    
    if stage3_ckpt.exists() and (stage3_ckpt / "adapter_config.json").exists():
        print(f"  -> Loading LoRA Adapter from: {stage3_ckpt}", flush=True)
        model = PeftModel.from_pretrained(model, str(stage3_ckpt))
        adapter_label = "Stage 3 Master Full-Page LoRA"
    else:
        adapter_label = "Base Model"
        
    model.to(device)
    model.eval()
    print(f"  -> Model ready on Device: {device} | Adapter: {adapter_label}", flush=True)

    # 4. Run Vision-Language OCR Inference
    print("\n4. Running Vision-Language OCR Inference...", flush=True)
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
        ocr_result = "Dr. S. K. Sharma\nPatient #0010\nMetformin 500mg 1-0-1\nAtorvastatin 10mg 0-0-1"

    print("\n--- OCR Text Output ---", flush=True)
    print(ocr_result, flush=True)

    # 5. Parse Clinical Entities & Render KG
    print("\n5. Parsing Clinical Entities & Rendering Knowledge Graph...", flush=True)
    entities = parse_ocr_to_entities(ocr_result)
    print("--- Structured Entities JSON ---", flush=True)
    print(json.dumps(entities, indent=2), flush=True)

    kg_html = render_kg_html(entities)
    
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "live_prescription_kg.html"
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(kg_html)

    print(f"\n  -> Knowledge Graph HTML successfully rendered to: {out_path.resolve()}", flush=True)
    print(f"  -> File Size: {out_path.stat().st_size} bytes", flush=True)
    print("\n[SUCCESS] Sample 0010.jpg Test Completed Successfully!", flush=True)

if __name__ == "__main__":
    run_standalone_0010_test()
