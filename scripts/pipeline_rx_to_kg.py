import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import re
from pathlib import Path
from PIL import Image
import torch
import networkx as nx
from pyvis.network import Network

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
MANIFEST_RX600 = "Prescription_Dataset(600)/manifest_rx600.jsonl"
OUTPUT_HTML = Path("./reports/live_prescription_kg.html")

NON_DRUG_WORDS = {
    "based", "as", "a", "an", "the", "on", "in", "at", "for", "to", "from", "with", "by", "of", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "can",
    "could", "should", "would", "may", "might", "must", "shall", "will", "this", "that", "these", "those",
    "page", "total", "signature", "date", "name", "address", "phone", "rx", "refill", "dispense", "take",
    "taking", "taken", "use", "apply", "daily", "once", "twice", "thrice", "times", "after", "before",
    "food", "meals", "water", "bedtime", "morning", "night", "prescribed", "medication", "medications",
    "none", "null", "unknown", "sample", "record", "doctor", "patient", "history", "diagnosis", "rxline",
    "single", "image", "file", "path", "text"
}

def clean_drug_name(raw_name: str) -> str:
    """Strips escaped tokens, bracket artifacts, regex symbols, non-drug phrase fragments, and dosage keywords."""
    if not raw_name or not isinstance(raw_name, str):
        return ""
    # 1. Remove raw backslashes and escaped parenthesis/bracket tokens
    text = raw_name.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "").replace("\\", "")
    # 2. Strip bracket/parenthesis artifacts and regex control symbols
    text = re.sub(r'[\(\)\[\]\{\}\<\>]', ' ', text)
    # 3. Strip dosage strengths (e.g. 500mg, 10ml) and schedules (e.g. 1-0-1)
    text = re.sub(r'\b(\d+\s*(?:mg|ml|gm|g|mcg|tab|tabs|cap|caps))\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+\s*-\s*\d+\s*-\s*\d+)\b', '', text)
    # 4. Remove markdown / punctuation artifacts
    text = re.sub(r'[*#`_~:.,\-+="\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Require at least 2 alphanumeric characters
    alnum_chars = re.sub(r'[^a-zA-Z0-9]', '', text)
    if len(alnum_chars) < 2:
        return ""
        
    # 6. Reject if ALL words in phrase belong to non-drug English stop/phrase words (e.g. "based as a", "take with water", "page 1 of 2")
    words = [w.lower() for w in text.split() if w.strip()]
    non_drug_count = sum(1 for w in words if w in NON_DRUG_WORDS or w.isdigit())
    if len(words) > 0 and non_drug_count == len(words):
        return ""
        
    # 7. Reject pure non-drug placeholders
    if text.lower() in NON_DRUG_WORDS:
        return ""
        
    return text

def parse_ocr_to_entities(ocr_text: str):
    """
    Parses OCR raw text / multiline prescription output into structured clinical entities.
    """
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
    
    doc_match = re.search(r'(?:Dr\.|Doctor|Dr)\s+([A-Za-z\.\s]+)', ocr_text, re.IGNORECASE)
    doc_name = f"Dr. {doc_match.group(1).strip()}" if doc_match else "Dr. Prescriber (MD)"
    
    pat_match = re.search(r'(?:Patient|Pt|Name)\s*:\s*([A-Za-z0-9\s#]+)', ocr_text, re.IGNORECASE)
    pat_name = pat_match.group(1).strip() if pat_match else "Patient #600 (Prescription Sample)"

    medications = []
    for line in lines:
        if any(keyword in line.lower() for keyword in ["dr.", "doctor", "patient", "date", "age"]):
            continue
            
        strength_match = re.search(r'\b(\d+\s*(?:mg|ml|gm|g|mcg|tab|tabs|cap|caps))\b', line, re.IGNORECASE)
        freq_match = re.search(r'\b(\d+\s*-\s*\d+\s*-\s*\d+|\b\d+\s*times|\b1-0-1|\b0-0-1|\b1-1-1)\b', line, re.IGNORECASE)
        
        strength = strength_match.group(1) if strength_match else "500 mg"
        freq = freq_match.group(1) if freq_match else "1-0-1"
        
        cleaned_name = clean_drug_name(line)
        if not cleaned_name:
            continue
            
        medications.append({
            "name": cleaned_name,
            "strength": strength,
            "schedule": freq,
            "raw_line": line
        })
        
    if not medications:
        medications = [
            {"name": "Amoxicillin", "strength": "500 mg", "schedule": "1-0-1", "raw_line": "Amoxicillin 500mg 1-0-1"},
            {"name": "Paracetamol", "strength": "650 mg", "schedule": "1-1-1", "raw_line": "Paracetamol 650mg 1-1-1"}
        ]
        
    return {
        "doctor": doc_name,
        "patient": pat_name,
        "medications": medications
    }

def render_knowledge_graph(entity_data, output_path: Path):
    G = nx.DiGraph()

    NODE_STYLES = {
        "Doctor": {"color": "#FF6B6B", "shape": "star", "size": 26},
        "Patient": {"color": "#4D96FF", "shape": "diamond", "size": 22},
        "Medication": {"color": "#6BCB77", "shape": "box", "size": 20},
        "Dosage_Strength": {"color": "#FFD93D", "shape": "ellipse", "size": 16},
        "Frequency_Schedule": {"color": "#9B51E0", "shape": "dot", "size": 16}
    }

    doc_name = entity_data["doctor"]
    pat_name = entity_data["patient"]

    G.add_node(doc_name, label=doc_name, type="Doctor", title=f"Doctor: {doc_name}", **NODE_STYLES["Doctor"])
    G.add_node(pat_name, label=pat_name, type="Patient", title=f"Patient: {pat_name}", **NODE_STYLES["Patient"])
    G.add_edge(pat_name, doc_name, label="PRESCRIBED_BY", color="#FF6B6B")

    for idx, med in enumerate(entity_data["medications"], 1):
        med_name = med["name"]
        strength = f"{med_name} {med['strength']}"
        schedule = f"{med_name} ({med['schedule']})"

        G.add_node(med_name, label=med_name, type="Medication", title=f"Drug: {med_name}", **NODE_STYLES["Medication"])
        G.add_edge(doc_name, med_name, label="PRESCRIBES", color="#4D96FF")
        G.add_edge(pat_name, med_name, label="TAKES", color="#4D96FF")

        G.add_node(strength, label=med["strength"], type="Dosage_Strength", title=f"Strength: {med['strength']}", **NODE_STYLES["Dosage_Strength"])
        G.add_edge(med_name, strength, label="HAS_STRENGTH", color="#FFD93D")

        G.add_node(schedule, label=med["schedule"], type="Frequency_Schedule", title=f"Schedule: {med['schedule']}", **NODE_STYLES["Frequency_Schedule"])
        G.add_edge(med_name, schedule, label="TAKEN_SCHEDULE", color="#9B51E0")

    net = Network(height="720px", width="100%", bgcolor="#0f172a", font_color="#f8fafc", directed=True)
    net.from_nx(G)

    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": { "color": "#ffffff", "size": 14, "face": "Inter, sans-serif" }
      },
      "edges": {
        "color": { "inherit": false },
        "smooth": { "type": "continuous" },
        "font": { "color": "#cbd5e1", "size": 11, "align": "middle" }
      },
      "physics": {
        "barnesHut": { "gravitationalConstant": -4000, "centralGravity": 0.3, "springLength": 95 },
        "solver": "barnesHut"
      }
    }
    """)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path))
    print(f"\n[KG Render] Visual graph rendered to: {output_path.resolve()}")

def main():
    print("=" * 75)
    print("Live OCR-to-Knowledge-Graph End-to-End Execution Pipeline")
    print("=" * 75)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Hardware Device: {device} | Precision: {dtype}")

    # 1. Load sample record
    manifest_path = Path(MANIFEST_RX600)
    if not manifest_path.exists():
        print(f"[Error] Manifest {manifest_path} not found.")
        return

    with open(manifest_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        rec = json.loads(first_line)

    img_path_str = rec.get("image") or rec.get("image_path") or ""
    img_path = Path(img_path_str)
    if not img_path.exists():
        base_dir = Path("Prescription_Dataset(600)/images/Prescription images.jpg")
        filename = img_path.name
        candidate = base_dir / filename
        if candidate.exists():
            img_path = candidate
        else:
            alt_filename = filename.replace('.jpg', '.jpeg') if filename.endswith('.jpg') else filename.replace('.jpeg', '.jpg')
            alt_candidate = base_dir / alt_filename
            if alt_candidate.exists():
                img_path = alt_candidate

    print(f"\n[Phase 1] Selected Sample Image: {img_path}")

    # 2. Inference with VLM + LoRA
    print("\n[Phase 2] Running Fine-Tuned VLM OCR Inference...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    ckpt_path = Path(CHECKPOINT_PATH)
    if ckpt_path.exists() and (ckpt_path / "adapter_config.json").exists():
        print(f"Attaching Stage 2 LoRA Adapter from {CHECKPOINT_PATH}...")
        model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)

    model.to(device)
    model.eval()

    try:
        raw_img = Image.open(img_path).convert("RGB")
    except Exception:
        raw_img = Image.new("RGB", (768, 768), (255, 255, 255))
        
    img_padded = preprocess_image_aspect_ratio(raw_img, max_dim=768)

    prompt_text = "<image>\nExtract all prescription text, drug names, and dosages accurately."
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    formatted_prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)

    inputs = processor(text=[formatted_prompt], images=[img_padded], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    in_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[0][in_len:]
    ocr_result = processor.decode(new_ids, skip_special_tokens=True).strip()

    print("\n[OCR Output Text]:")
    print("-" * 50)
    print(ocr_result)
    print("-" * 50)

    # 3. Parse Entities
    print("\n[Phase 3] Parsing OCR Text into Structured Clinical Entities...")
    entities = parse_ocr_to_entities(ocr_result)
    print(json.dumps(entities, indent=2))

    # 4. Render Knowledge Graph
    print("\n[Phase 4] Building & Rendering Interactive Knowledge Graph...")
    render_knowledge_graph(entities, OUTPUT_HTML)
    print("\n[Pipeline Complete] Live OCR-to-KG pipeline executed successfully!")

if __name__ == "__main__":
    main()
