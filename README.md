# prescription-ocr-pipeline

Clinical Prescription OCR & Interactive Knowledge Graph AI System based on **PaddleOCR-VL-1.6**, PEFT LoRA, and Group Relative Policy Optimization (GRPO).

## 📌 Features
- **Multimodal VLM Architecture**: Fine-tuned `PaddleOCR-VL-1.6` for handwritten doctor prescription parsing.
- **PEFT LoRA Fine-Tuning**: Multi-stage curriculum training across single word, multiline, and full-page prescription datasets.
- **GRPO Reinforcement Learning**: Policy gradient preference optimization for clinical entity extraction.
- **Clean Post-Processing**: Phrase exclusion and regex artifact filtering via `clean_drug_name`.
- **Interactive UI & Dashboard**: Streamlit application (`app.py`) featuring prescription image uploading, structured JSON parsing, and dynamic PyVis Knowledge Graph rendering.

## 🚀 Quick Start

### 1. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 2. Run Live Inference Sanity Check
```bash
python scripts/test_live_inference.py
```

### 3. Joint SFT + GRPO Training
```bash
python scripts/train_grpo_prescription_ocr.py --epochs 3 --lora_r 16 --lora_alpha 32
```

## 📄 License
MIT License
