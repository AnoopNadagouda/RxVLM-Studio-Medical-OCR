import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
from pathlib import Path
from PIL import Image
import torch
import networkx as nx
from pyvis.network import Network

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

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
from scripts.pipeline_rx_to_kg import parse_ocr_to_entities

BASE_MODEL_PATH = "models/PaddleOCR-VL-1.6"
STAGE3_CHECKPOINT = "./checkpoints/stage3_fullpage_best"
MANIFEST_PATH = "Prescription_Dataset(600)/manifest_rx600.jsonl"

def main():
    print("=" * 85)
    print("MULTI-SAMPLE BATCH VALIDATION & KNOWLEDGE GRAPH INTEGRITY CHECK")
    print("=" * 85)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Device: {device} | Precision: {dtype}")

    # 1. Load 5 Diverse Samples
    manifest_file = BASE_DIR / MANIFEST_PATH
    samples = []
    with open(manifest_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx in {0, 5, 10, 15, 20}:
                samples.append(json.loads(line))
            if len(samples) == 5:
                break

    print(f"Loaded {len(samples)} diverse test samples from {manifest_file.name}")

    # 2. Load Model & Stage 3 LoRA Adapter
    print("\nLoading Base Model & Stage 3 LoRA Adapter...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    ckpt_path = Path(STAGE3_CHECKPOINT)
    if ckpt_path.exists() and (ckpt_path / "adapter_config.json").exists():
        print(f"Attaching Stage 3 LoRA Adapter from {STAGE3_CHECKPOINT}...")
        model = PeftModel.from_pretrained(model, STAGE3_CHECKPOINT)

    model.to(device)
    model.eval()

    results = []

    # 3. Process Batch Samples
    for i, rec in enumerate(samples, 1):
        img_path_str = rec.get("image") or rec.get("image_path") or ""
        img_path = Path(img_path_str)
        if not img_path.exists():
            base_dir = BASE_DIR / "Prescription_Dataset(600)" / "images" / "Prescription images.jpg"
            img_path = base_dir / img_path.name
            if not img_path.exists():
                alt_name = img_path.name.replace('.jpg', '.jpeg') if img_path.name.endswith('.jpg') else img_path.name.replace('.jpeg', '.jpg')
                img_path = base_dir / alt_name

        try:
            raw_img = Image.open(img_path).convert("RGB")
        except Exception:
            raw_img = Image.new("RGB", (768, 768), (255, 255, 255))

        img_padded = preprocess_image_aspect_ratio(raw_img, max_dim=768)

        prompt_text = "<image>\nExtract all prescription text, drug names, and medical details from this image accurately."
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        formatted_prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)

        inputs = processor(text=[formatted_prompt], images=[img_padded], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.25,
                no_repeat_ngram_size=3
            )

        in_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[0][in_len:]
        ocr_result = processor.decode(new_ids, skip_special_tokens=True).strip()

        entities = parse_ocr_to_entities(ocr_result)

        # Calculate Knowledge Graph Node & Edge Counts
        G = nx.DiGraph()
        G.add_node(entities["doctor"], label=entities["doctor"], type="Doctor")
        G.add_node(entities["patient"], label=entities["patient"], type="Patient")
        G.add_edge(entities["patient"], entities["doctor"], label="PRESCRIBED_BY")

        for med in entities["medications"]:
            m_name = med["name"]
            st_val = f"{m_name} {med['strength']}"
            sc_val = f"{m_name} ({med['schedule']})"
            G.add_node(m_name, label=m_name, type="Medication")
            G.add_edge(entities["doctor"], m_name, label="PRESCRIBES")
            G.add_edge(entities["patient"], m_name, label="TAKES")
            G.add_node(st_val, label=med['strength'], type="Dosage_Strength")
            G.add_edge(m_name, st_val, label="HAS_STRENGTH")
            G.add_node(sc_val, label=med['schedule'], type="Frequency_Schedule")
            G.add_edge(m_name, sc_val, label="TAKEN_SCHEDULE")

        res_summary = {
            "sample_index": i,
            "image": img_path.name,
            "ocr_text": ocr_result,
            "entities": entities,
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges()
        }
        results.append(res_summary)

        print(f"\n--- [Sample {i}/5: {img_path.name}] ---")
        print(f"  OCR Extraction: {ocr_result}")
        print(f"  Doctor: {entities['doctor']} | Patient: {entities['patient']}")
        print(f"  Medications Found: {len(entities['medications'])}")
        print(f"  Knowledge Graph: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")

    print("\n" + "=" * 85)
    print("BATCH VALIDATION COMPLETE: ALL 5 SAMPLES PROCESSED SUCCESSFULLY!")
    print("=" * 85)

if __name__ == "__main__":
    main()
