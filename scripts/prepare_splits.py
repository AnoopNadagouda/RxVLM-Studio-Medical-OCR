#!/usr/bin/env python3
"""
prepare_splits.py

Loads, cleans, handles extension mismatches, and splits the Prescription Dataset into
80% Training, 10% Validation, and 10% Testing sets.

Saves clean .jsonl files in the processed/ directory:
- processed/train.jsonl
- processed/val.jsonl
- processed/test.jsonl
"""

import os
import csv
import json
import random
import sys
from pathlib import Path


def find_project_root() -> Path:
    """Locate root directory containing annotations and Prescription_Dataset(600)."""
    current_dir = Path(__file__).resolve().parent
    candidates = [current_dir.parent, Path.cwd()]
    for cand in candidates:
        if (cand / "annotations" / "Prescription Dataset.csv").exists() or (cand / "Prescription_Dataset(600)").exists():
            return cand
    return current_dir.parent


def prepare_dataset_splits(seed: int = 42, train_ratio: float = 0.80, val_ratio: float = 0.10):
    root_dir = find_project_root()
    csv_path = root_dir / "annotations" / "Prescription Dataset.csv"
    images_dir = root_dir / "Prescription_Dataset(600)"
    processed_dir = root_dir / "processed"

    print("=" * 80)
    print("  PRESCRIPTION DATASET SPLIT PREPARATION")
    print("=" * 80)
    print(f"Project Root     : {root_dir}")
    print(f"Annotation CSV   : {csv_path}")
    print(f"Images Directory : {images_dir}")
    print(f"Output Directory : {processed_dir}")
    print(f"Random Seed      : {seed}")
    print(f"Split Ratios     : Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={(1 - train_ratio - val_ratio)*100:.0f}%")
    print("-" * 80)

    if not csv_path.exists():
        print(f"[ERROR] Annotation CSV not found at: {csv_path}")
        sys.exit(1)

    if not images_dir.exists():
        print(f"[ERROR] Images directory not found at: {images_dir}")
        sys.exit(1)

    processed_dir.mkdir(parents=True, exist_ok=True)

    # Crawl images directory to map all available files by exact name & stem
    disk_files_exact = {}
    disk_files_stem = {}

    for root, _, files in os.walk(images_dir):
        for fname in files:
            full_path = Path(root) / fname
            disk_files_exact[fname] = full_path
            disk_files_stem[full_path.stem.lower()] = full_path

    print(f"Discovered {len(disk_files_exact)} total image files inside Prescription_Dataset(600)")

    # Read and clean CSV records
    cleaned_records = []
    skipped_empty = 0
    skipped_missing = 0
    ext_mismatch_count = 0

    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = []

        for row_idx, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                skipped_empty += 1
                continue

            img_filename = row[0].strip()
            if not img_filename:
                skipped_empty += 1
                continue

            # Resolve actual image file path on disk
            target_path = None
            if img_filename in disk_files_exact:
                target_path = disk_files_exact[img_filename]
            else:
                stem = Path(img_filename).stem.lower()
                if stem in disk_files_stem:
                    target_path = disk_files_stem[stem]
                    ext_mismatch_count += 1

            if not target_path or not target_path.exists():
                skipped_missing += 1
                print(f"  [Row {row_idx:4d}] Filtered out missing file entry: '{img_filename}'")
                continue

            # Extract fields
            desc_english = row[1].strip() if len(row) > 1 else ""
            structured_json_text = row[2].strip() if len(row) > 2 else ""
            ocr_structured_json = row[3].strip() if len(row) > 3 else ""

            # Normalize image path for JSON output (forward slashes)
            norm_path = target_path.as_posix()

            record = {
                "image_path": norm_path,
                "image_filename": target_path.name,
                "csv_filename": img_filename,
                "description_english": desc_english,
                "structured_json": structured_json_text,
                "ocr_structured_json": ocr_structured_json,
                "text": structured_json_text if structured_json_text else desc_english
            }

            cleaned_records.append(record)

    print("-" * 80)
    print(f"CSV Cleaning Summary:")
    print(f"  |- Total valid Records Processed : {len(cleaned_records)}")
    print(f"  |- Extension Mismatches Handled : {ext_mismatch_count} (mapped .jpg -> .jpeg on disk)")
    print(f"  |- Skipped Empty / Malformed    : {skipped_empty}")
    print(f"  |- Skipped Missing Image Rows   : {skipped_missing}")

    if not cleaned_records:
        print("[ERROR] No valid records found to split!")
        sys.exit(1)

    # Shuffle deterministically with seed
    random.seed(seed)
    random.shuffle(cleaned_records)

    total_n = len(cleaned_records)
    n_train = int(total_n * train_ratio)
    n_val = int(total_n * val_ratio)
    n_test = total_n - n_train - n_val

    train_set = cleaned_records[:n_train]
    val_set = cleaned_records[n_train:n_train + n_val]
    test_set = cleaned_records[n_train + n_val:]

    print("-" * 80)
    print("Dataset Split Allocation:")
    print(f"  |- Training Set   (80%): {len(train_set)} records")
    print(f"  |- Validation Set (10%): {len(val_set)} records")
    print(f"  |- Test Set       (10%): {len(test_set)} records")

    # Save to JSONL manifests
    splits = {
        "train.jsonl": train_set,
        "val.jsonl": val_set,
        "test.jsonl": test_set
    }

    print("-" * 80)
    print("Saving JSONL Manifests:")

    for out_name, records in splits.items():
        out_file = processed_dir / out_name
        with open(out_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [OK] Saved {len(records):3d} records -> {out_file.as_posix()}")

    print("\nSample JSON Object Format:")
    print(json.dumps(train_set[0], indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    prepare_dataset_splits(seed=42)
