#!/usr/bin/env python3
"""
verify_pipeline.py

Verifies the data pipeline by:
1. Loading processed/val.jsonl.
2. Randomly selecting 2 records from the validation set.
3. Verifying that the image path in each record points to a valid, readable image file on disk.
4. Printing the image filename, description, and structured JSON snippet to confirm full pipeline operational status.
"""

import os
import json
import random
import sys
from pathlib import Path
from PIL import Image


def find_project_root() -> Path:
    """Locate root directory containing processed/val.jsonl."""
    current_dir = Path(__file__).resolve().parent
    candidates = [current_dir.parent, Path.cwd()]
    for cand in candidates:
        if (cand / "processed" / "val.jsonl").exists():
            return cand
    return current_dir.parent


def verify_pipeline(num_samples: int = 2, seed: int = None):
    root_dir = find_project_root()
    val_jsonl_path = root_dir / "processed" / "val.jsonl"

    print("=" * 80)
    print("  DATA PIPELINE VERIFICATION REPORT")
    print("=" * 80)
    print(f"Project Root    : {root_dir}")
    print(f"Validation Manifest : {val_jsonl_path}")
    print("-" * 80)

    if not val_jsonl_path.exists():
        print(f"[ERROR] Validation manifest not found at: {val_jsonl_path}")
        sys.exit(1)

    # 1. Load records from processed/val.jsonl
    records = []
    with open(val_jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if line_str:
                try:
                    record = json.loads(line_str)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Line {line_idx} is not valid JSON: {e}")

    print(f"Successfully loaded {len(records)} validation records from {val_jsonl_path.name}")
    print("-" * 80)

    if len(records) < num_samples:
        print(f"[WARNING] Requested {num_samples} samples, but manifest only contains {len(records)} records.")
        num_samples = len(records)

    # 2. Randomly select 2 records
    if seed is not None:
        random.seed(seed)
    
    selected_records = random.sample(records, num_samples)

    # 3 & 4. Verify readable image on disk & print detailed snippet
    print(f"Inspecting {num_samples} Randomly Selected Records for Demo Verification:\n")

    all_valid = True

    for i, rec in enumerate(selected_records, start=1):
        img_path_str = rec.get("image_path", "")
        img_path = Path(img_path_str)
        img_filename = rec.get("image_filename", img_path.name)
        description = rec.get("description_english", "")
        structured_json_raw = rec.get("structured_json", "")

        print(f"--- SAMPLE {i} / {num_samples} ---")
        print(f"Filename       : {img_filename}")
        print(f"Image Path     : {img_path_str}")

        # 3. Image Readability Verification
        if not img_path.exists():
            print(f"Image Status   : [FAILED] File does not exist on disk!")
            all_valid = False
        else:
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify image integrity
                # Re-open for metadata check after verify()
                with Image.open(img_path) as img:
                    width, height = img.size
                    fmt = img.format
                    mode = img.mode
                    file_size_kb = os.path.getsize(img_path) / 1024.0
                    print(f"Image Status   : [VERIFIED READABLE] {width}x{height} px | Format: {fmt} | Mode: {mode} | Size: {file_size_kb:.1f} KB")
            except Exception as ex:
                print(f"Image Status   : [FAILED READABLE] Corrupted or unreadable image file: {ex}")
                all_valid = False

        # Description
        print(f"Description    : {description}")

        # Structured JSON Snippet
        print("Structured JSON Snippet:")
        if structured_json_raw:
            try:
                parsed_json = json.loads(structured_json_raw)
                formatted_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                # Print up to 15 lines of formatted snippet
                json_lines = formatted_json.splitlines()
                if len(json_lines) > 15:
                    snippet = "\n".join(json_lines[:15]) + f"\n  ... ({len(json_lines) - 15} lines truncated)"
                else:
                    snippet = formatted_json
                print("  " + snippet.replace("\n", "\n  "))
            except Exception:
                # If raw string is not valid JSON object, print truncated string
                snippet = structured_json_raw[:200] + ("..." if len(structured_json_raw) > 200 else "")
                print(f"  {snippet}")
        else:
            print("  (Empty or None)")

        print("-" * 80)

    print("=" * 80)
    if all_valid:
        print("RESULT: PIPELINE VERIFICATION SUCCESSFUL - DATA IS FULLY OPERATIONAL FOR DEMO")
    else:
        print("RESULT: PIPELINE VERIFICATION FAILED - ONE OR MORE IMAGES UNREADABLE")
    print("=" * 80)


if __name__ == "__main__":
    verify_pipeline(num_samples=2)
