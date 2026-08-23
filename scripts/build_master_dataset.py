import os
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_DIR = BASE_DIR / "ocr_manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

MASTER_ALL_PATH = MANIFEST_DIR / "vlm_master_all_data.jsonl"
MASTER_TRAIN_PATH = MANIFEST_DIR / "vlm_master_train.jsonl"
MASTER_VAL_PATH = MANIFEST_DIR / "vlm_master_val.jsonl"

def collect_records_from_manifests():
    records = []
    seen_images = set()
    
    manifest_sources = [
        MANIFEST_DIR / "vlm_stage1_train.jsonl",
        MANIFEST_DIR / "vlm_stage1_val.jsonl",
        MANIFEST_DIR / "vlm_stage2_train.jsonl",
        MANIFEST_DIR / "vlm_stage2_val.jsonl",
        BASE_DIR / "Prescription_Dataset(600)" / "manifest_rx600.jsonl"
    ]
    
    for m_path in manifest_sources:
        if not m_path.exists():
            continue
        print(f"Reading source manifest: {m_path.name}...")
        with open(m_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                img_path = data.get("image") or data.get("image_path") or ""
                if not img_path:
                    continue
                    
                target_text = data.get("text", "")
                if not target_text:
                    for msg in data.get("conversations", []):
                        if msg.get("from") == "gpt":
                            target_text = msg.get("value", "")
                            
                target_text = target_text.strip()
                if not target_text:
                    continue
                    
                # Deduplicate by image path
                if img_path in seen_images:
                    continue
                seen_images.add(img_path)
                
                records.append({
                    "image_path": str(img_path),
                    "text": target_text
                })
                
    return records

def main():
    print("=" * 80)
    print("Consolidating Master Universal VLM Dataset Manifest")
    print("=" * 80)
    
    records = collect_records_from_manifests()
    print(f"\nTotal Valid Image-Label Pairs Consolidated: {len(records)}")
    
    # Shuffle deterministically
    random.seed(42)
    random.shuffle(records)
    
    # Write master all file
    with open(MASTER_ALL_PATH, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Exported Master All Data ({len(records)} samples) -> {MASTER_ALL_PATH.resolve()}")
    
    # 90 / 10 Train / Val Split
    split_idx = int(len(records) * 0.90)
    train_records = records[:split_idx]
    val_records = records[split_idx:]
    
    with open(MASTER_TRAIN_PATH, 'w', encoding='utf-8') as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(MASTER_VAL_PATH, 'w', encoding='utf-8') as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\n[Split Complete]")
    print(f"  - Train Split (90%): {len(train_records)} samples -> {MASTER_TRAIN_PATH.resolve()}")
    print(f"  - Val Split   (10%): {len(val_records)} samples -> {MASTER_VAL_PATH.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
