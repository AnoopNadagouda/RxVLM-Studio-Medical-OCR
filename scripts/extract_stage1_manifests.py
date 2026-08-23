import os
import json
from pathlib import Path

# Input & Output paths
MANIF_DIR = Path(r"C:\Users\ANOOP\Downloads\ocr_manifests")
LOCAL_MANIF_DIR = Path(r"ocr_manifests")
LOCAL_MANIF_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SOURCES = {'rxhandbd', 'kaggle_bd_rx', 'kalashsh'}

def extract_manifest(input_filename, output_filenames):
    input_path = MANIF_DIR / input_filename
    if not input_path.exists():
        input_path = LOCAL_MANIF_DIR / input_filename

    print(f"Reading from {input_path}...")
    filtered_records = []
    total_records = 0
    source_counts = {}

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total_records += 1
            data = json.loads(line)
            source = data.get('metadata', {}).get('source', '')
            if source in TARGET_SOURCES:
                filtered_records.append(data)
                source_counts[source] = source_counts.get(source, 0) + 1

    for out_path in output_filenames:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for rec in filtered_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"Saved {len(filtered_records)} records to {out_path}")
    
    print(f"Total read: {total_records}, Filtered: {len(filtered_records)}")
    print(f"Source breakdown: {source_counts}")

print("--- Processing Training Split ---")
extract_manifest(
    "vlm_train.jsonl",
    [MANIF_DIR / "vlm_stage1_train.jsonl", LOCAL_MANIF_DIR / "vlm_stage1_train.jsonl"]
)

print("\n--- Processing Validation Split ---")
extract_manifest(
    "vlm_val.jsonl",
    [MANIF_DIR / "vlm_stage1_val.jsonl", LOCAL_MANIF_DIR / "vlm_stage1_val.jsonl"]
)
