# RxVLM Studio: Medical OCR & Knowledge Graph Pipeline

RxVLM Studio is an advanced, full-stack medical document intelligence and clinical decision-support application. It bridges raw, unstructured handwritten or printed medical prescriptions with structured, queryable knowledge graphs, leveraging cutting-edge Vision-Language Models (VLMs) and PaddleOCR-VL architectures.

---

## 📐 End-to-End System Architecture (Baseline to Evaluations)

```text
[ Raw Prescription Images (.png / .jpeg) ]
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ 1. INGESTION & PRIVACY NORMALIZATION                   │
│  - Tight bounding-box cropping on doctor handwriting   │
│  - PII Redaction: Masking names, hospital banners,     │
│    registration numbers, and signatures via OpenCV/PIL │
└────────────────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ 2. BASELINE OCR & EXTRACTION ENGINE                    │
│  - Custom PaddleOCR-VL & Fine-Tuned LoRA Adapters      │
│  - Zero-Hallucination Grounding Rule: Unstated values  │
│    (missing dosages/forms) are strictly set to null    │
└────────────────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ 3. STRUCTURED KNOWLEDGE GRAPH & BACKEND (Flask)        │
│  - web_server.py API Orchestrator                      │
│  - 1-to-1 Parity Mapping (Images <-> Structured JSONs) │
└────────────────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ 4. INTERACTIVE FRONTEND STUDIO (HTML/JS/CSS)           │
│  - Partition Filter Chips (Mapped, Training, Testing)  │
│  - Expanded Medication Table & Resizable Splitters     │
│  - Rx Assistant AI Context-Aware Clinical Chat         │
└────────────────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ 5. EVALUATION METRICS & BENCHMARK SUITE                │
│  - Character Error Rate (CER) / Word Error Rate (WER)  │
│  - Field Entity Precision, Recall, & F1 Scores         │
│  - Exact Match Structural Quality & Jaccard Similarity │
└────────────────────────────────────────────────────────> [ Final Research Reports & Excel Manifests ]
```

---

## 📊 Dataset Partitioning & Benchmark Performance

The application supports multiple integrated dataset splits and partitions for comprehensive training, evaluation, and production inference:

- **Training Set (600 Samples)**: Large-scale training partition for model adaptation and LoRA fine-tuning.
- **Testing Set (100 Samples)**: Dedicated evaluation partition tracking rigorous accuracy benchmarks.
- **Canonical Mapped Dataset**: Privacy-reviewed, zero-hallucination prescription records mapped 1-to-1 with strict clinical schema compliance.

### Evaluation & Validation Metrics:
- **Character Error Rate (CER)**: Tracked per batch.
- **Field Entity F1-Score**: Evaluated across extracted medications and dosages.
- **Exact Match Structural Quality**: Monitored for complete JSON schema conformance.

---

## 📂 Project Directory Structure

```text
RxVLM-Studio-Medical-OCR/
├── images/                         # Processed and privacy-reviewed prescription images
├── json/                           # 1-to-1 mapped structured JSON annotation records
├── static/
│   ├── css/                        # Custom styles, layout splitters, and table-expansion rules
│   └── js/                         # Frontend application logic, dynamic DOM rendering, and event handlers
├── templates/
│   └── index.html                  # Main dashboard layout (Sidebar, Workspace, AI Assistant panel)
├── PRIVACY_REVIEW.txt              # Compliance documentation, PII handling, and data auditing trail
├── prescription_results_final.xlsx # Master 3-column Excel manifest aligning filenames, text descriptions, and JSONs
└── web_server.py                   # Core Flask backend application handling API routes and inference endpoints
```

---

## 📊 Data Schema & 1-to-1 Parity

Every prescription image maps strictly 1-to-1 to a corresponding JSON file. The data structure adheres to a rigid clinical schema separating raw transcription from structured medical data:

```json
{
  "RAW OCR": "Exact verbatim clinical transcription from prescription headers and body...",
  "STRUCTURED OCR": {
    "patient_metadata": {
      "patient_name": "String or null",
      "age": "String or null",
      "gender": "String or null",
      "date": "String or null",
      "weight": "String or null"
    },
    "diagnosis": "Primary clinical assessment or diagnosis",
    "history": "Relevant medical or incident history",
    "medications": [
      {
        "drug": "Generic or brand name",
        "dosage": "Strength / Dose",
        "frequency": "Regimen (e.g., 1-0-1, BD)",
        "form": "Tablet | Capsule | Syrup | Injection | Lotion",
        "duration": "Duration (e.g., 10 days)",
        "instructions": "Administration timing (e.g., After food)"
      }
    ],
    "advice": []
  }
}
```

---

## 🚀 Getting Started & Local Setup

### Prerequisites
- Python 3.9 or higher
- Pip (Python Package Manager)

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AnoopNadagouda/RxVLM-Studio-Medical-OCR.git
   cd RxVLM-Studio-Medical-OCR
   ```

2. **Install Dependencies:**
   ```bash
   pip install flask pandas openpyxl pillow opencv-python
   ```

3. **Run the Backend Server:**
   ```bash
   python web_server.py
   ```

4. **Access the Application:**
   Open your browser and navigate to:
   👉 `http://127.0.0.1:5000`

---

## 💡 Key Features Explained

- **Dynamic Dataset Filtering**: Instantly toggle between partitions (All, Mapped, Training (600), and Testing (100)) with zero layout thrashing.
- **Zero-Hallucination Grounding**: Unstated or ambiguous medical parameters are strictly set to null to ensure absolute compliance with ground-truth clinical data.
- **Interactive Resizable Layout**: Custom mouse-tracking handles allow smooth adjustment of the Left Sidebar, Central Workspace, and Right AI Assistant panels.
- **Rx Assistant AI**: Context-aware clinical assistant capable of breaking down drug schedules, interaction warnings, and safety audits on-the-fly.