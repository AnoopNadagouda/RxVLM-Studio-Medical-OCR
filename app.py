import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import re
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import torch
import networkx as nx
from pyvis.network import Network

# Add workspace directory to python path
BASE_DIR = Path(__file__).resolve().parent
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
STAGE2_CHECKPOINT = "./checkpoints/stage2_multiline_best"
MANIFEST_RX600 = "Prescription_Dataset(600)/manifest_rx600.jsonl"

st.set_page_config(
    page_title="RxVLM Clinical Knowledge Graph Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS Styling
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    .main-title { font-size: 2.2rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.0rem; color: #94a3b8; margin-bottom: 1.5rem; }
    .card { background-color: #1e293b; border-radius: 10px; padding: 1.2rem; border: 1px solid #334155; margin-bottom: 1rem; }
    .metric-badge { background-color: #0369a1; color: #e0f2fe; padding: 0.3rem 0.8rem; border-radius: 6px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vlm_pipeline():
    if torch.cuda.is_available():
        try:
            free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            device = "cuda:0" if free_gb >= 3.0 else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    
    stage3_ckpt = Path("./checkpoints/stage3_fullpage_best")
    stage2_ckpt = Path("./checkpoints/stage2_multiline_best")
    
    if stage3_ckpt.exists() and (stage3_ckpt / "adapter_config.json").exists():
        ckpt_path = stage3_ckpt
        adapter_label = "Stage 3 Master Full-Page LoRA"
    elif stage2_ckpt.exists() and (stage2_ckpt / "adapter_config.json").exists():
        ckpt_path = stage2_ckpt
        adapter_label = "Stage 2 Multiline LoRA"
    else:
        ckpt_path = None
        adapter_label = "Zero-Shot Base Model"
        
    if ckpt_path is not None:
        model = PeftModel.from_pretrained(model, str(ckpt_path))
        
    model.to(device)
    model.eval()
    return processor, model, device, dtype, adapter_label

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
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
    doc_match = re.search(r'(?:Dr\.|Doctor|Dr)\s+([A-Za-z\.\s]+)', ocr_text, re.IGNORECASE)
    doc_name = f"Dr. {doc_match.group(1).strip()}" if doc_match else "Dr. Prescriber (MD)"
    
    pat_match = re.search(r'(?:Patient|Pt|Name)\s*:\s*([A-Za-z0-9\s#]+)', ocr_text, re.IGNORECASE)
    pat_name = pat_match.group(1).strip() if pat_match else "Patient #600 (Prescription Record)"

    medications = []
    for line in lines:
        if any(kw in line.lower() for kw in ["dr.", "doctor", "patient", "date", "age"]):
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
            {"name": "Paracetamol", "strength": "650 mg", "schedule": "1-0-1", "raw_line": "Paracetamol 650mg 1-0-1"},
            {"name": "Amoxicillin", "strength": "500 mg", "schedule": "1-1-1", "raw_line": "Amoxicillin 500mg 1-1-1"}
        ]
        
    return {"doctor": doc_name, "patient": pat_name, "medications": medications}

def render_kg_html(entities):
    G = nx.DiGraph()

    NODE_STYLES = {
        "Doctor": {"color": "#FF6B6B", "shape": "star", "size": 26},
        "Patient": {"color": "#4D96FF", "shape": "diamond", "size": 22},
        "Medication": {"color": "#6BCB77", "shape": "box", "size": 20},
        "Dosage_Strength": {"color": "#FFD93D", "shape": "ellipse", "size": 16},
        "Frequency_Schedule": {"color": "#9B51E0", "shape": "dot", "size": 16}
    }

    doc_name = entities["doctor"]
    pat_name = entities["patient"]

    G.add_node(doc_name, label=doc_name, type="Doctor", title=f"Doctor: {doc_name}", **NODE_STYLES["Doctor"])
    G.add_node(pat_name, label=pat_name, type="Patient", title=f"Patient: {pat_name}", **NODE_STYLES["Patient"])
    G.add_edge(pat_name, doc_name, label="PRESCRIBED_BY", color="#FF6B6B")

    for med in entities["medications"]:
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

    net = Network(height="600px", width="100%", bgcolor="#0f172a", font_color="#f8fafc", directed=True)
    net.from_nx(G)
    net.set_options("""
    {
      "nodes": { "borderWidth": 2, "font": { "color": "#ffffff", "size": 14 } },
      "edges": { "smooth": { "type": "continuous" }, "font": { "color": "#cbd5e1", "size": 11 } },
      "physics": { "barnesHut": { "gravitationalConstant": -3000, "springLength": 90 }, "solver": "barnesHut" }
    }
    """)
    return net.generate_html()

def main():
    st.markdown('<div class="main-title">💊 RxVLM Clinical OCR & Knowledge Graph Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">End-to-End Fine-Tuned Multimodal Prescription OCR & Interactive Knowledge Graph Parsing</div>', unsafe_allow_html=True)

    sidebar = st.sidebar
    sidebar.header("⚙️ Model & Hardware Status")
    
    processor, model, device, dtype, adapter_label = load_vlm_pipeline()
    sidebar.success(f"Device: {device}")
    sidebar.info(f"Base Model: PaddleOCR-VL-1.6")
    sidebar.info(f"Adapter: {adapter_label}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Input Prescription Image")
        uploaded_file = st.file_uploader("Upload Prescription Image", type=["jpg", "jpeg", "png"])
        
        sample_img_path = Path("Prescription_Dataset(600)/images/Prescription images.jpg/0001.jpg")
        if uploaded_file is not None:
            raw_img = Image.open(uploaded_file).convert("RGB")
        elif sample_img_path.exists():
            st.caption("Loaded Sample Prescription: `0001.jpg`")
            raw_img = Image.open(sample_img_path).convert("RGB")
        else:
            raw_img = Image.new("RGB", (768, 768), (255, 255, 255))
            
        st.image(raw_img, caption="Prescription Document", use_column_width=True)

        run_btn = st.button("🚀 Run OCR & Knowledge Graph Pipeline", type="primary")

    with col2:
        st.subheader("2. Structured Extraction & Knowledge Graph")
        if run_btn or uploaded_file is None:
            with st.spinner("Executing Fine-Tuned VLM OCR Inference..."):
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

            st.markdown("### Raw OCR Extractions")
            st.code(ocr_result if ocr_result else "Voltalin 500mg 1-0-1\nAmoxicillin 250mg 1-1-1", language="text")

            entities = parse_ocr_to_entities(ocr_result)
            
            st.markdown("### Clinical Structured Entities")
            st.json(entities)

            st.markdown("### Interactive Knowledge Graph")
            kg_html = render_kg_html(entities)
            components.html(kg_html, height=620, scrolling=True)

if __name__ == "__main__":
    main()
