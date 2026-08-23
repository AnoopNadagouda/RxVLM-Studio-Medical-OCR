#!/usr/bin/env python3
"""
audit_dataset.py

Audits annotations/Prescription Dataset.csv against the Prescription_Dataset(600) directory.

Tasks:
1. Loads annotations/Prescription Dataset.csv.
2. Checks that every image filename listed in the first column actually exists inside Prescription_Dataset(600).
3. Prints a clear, detailed summary report showing:
   - Total valid image-annotation pairs found.
   - Any missing image files.
   - The count of any empty or malformed rows.
"""

import os
import csv
import sys
from pathlib import Path


def find_project_root() -> Path:
    """Locate root directory containing annotations and Prescription_Dataset(600)."""
    current_dir = Path(__file__).resolve().parent
    # Check parent directory (project root) or current working directory
    candidates = [current_dir.parent, Path.cwd()]
    for cand in candidates:
        if (cand / "annotations" / "Prescription Dataset.csv").exists() or (cand / "Prescription_Dataset(600)").exists():
            return cand
    return current_dir.parent


def audit_prescription_dataset():
    root_dir = find_project_root()
    csv_path = root_dir / "annotations" / "Prescription Dataset.csv"
    images_dir = root_dir / "Prescription_Dataset(600)"

    print("=" * 80)
    print("  PRESCRIPTION DATASET AUDIT REPORT")
    print("=" * 80)
    print(f"Project Root     : {root_dir}")
    print(f"Annotation CSV   : {csv_path}")
    print(f"Images Directory : {images_dir}")
    print("-" * 80)

    if not csv_path.exists():
        print(f"[ERROR] Annotation CSV not found at: {csv_path}")
        sys.exit(1)

    if not images_dir.exists():
        print(f"[ERROR] Images directory not found at: {images_dir}")
        sys.exit(1)

    # Crawl images directory to map all available filenames
    all_disk_files_exact = {}
    all_disk_files_stem = {}
    total_images_on_disk = 0

    for root, _, files in os.walk(images_dir):
        for fname in files:
            full_p = Path(root) / fname
            all_disk_files_exact[fname] = full_p
            all_disk_files_stem[Path(fname).stem.lower()] = fname
            total_images_on_disk += 1

    print(f"Total image files discovered on disk inside Prescription_Dataset(600): {total_images_on_disk}")
    print("-" * 80)

    valid_pairs_exact = 0
    valid_pairs_alt_ext = 0
    missing_files = []
    malformed_rows = []
    total_rows = 0

    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = []

        for row_idx, row in enumerate(reader, start=2):
            total_rows += 1
            if not row or not any(cell.strip() for cell in row):
                malformed_rows.append((row_idx, "Empty row"))
                continue

            img_filename = row[0].strip()
            if not img_filename:
                malformed_rows.append((row_idx, "Empty image filename column"))
                continue

            # Check exact filename match
            if img_filename in all_disk_files_exact:
                valid_pairs_exact += 1
            else:
                # Check stem match (e.g., .jpg in CSV vs .jpeg on disk)
                stem = Path(img_filename).stem.lower()
                if stem in all_disk_files_stem:
                    actual_fname = all_disk_files_stem[stem]
                    valid_pairs_alt_ext += 1
                    missing_files.append((row_idx, img_filename, f"Extension mismatch (found on disk as '{actual_fname}')"))
                else:
                    missing_files.append((row_idx, img_filename, "File missing on disk"))

    total_valid_pairs = valid_pairs_exact + valid_pairs_alt_ext

    # Detailed Printout
    print("\n--- AUDIT SUMMARY ---")
    print(f"Total CSV Data Rows Processed     : {total_rows}")
    print(f"Total Valid Image-Annotation Pairs : {total_valid_pairs}")
    print(f"  |- Exact Filename Matches       : {valid_pairs_exact}")
    print(f"  |- Extension Mismatch Matches   : {valid_pairs_alt_ext} (e.g. .jpg in CSV vs .jpeg on disk)")
    print(f"Total Missing Image Files         : {len(missing_files)}")
    print(f"Total Empty or Malformed Rows     : {len(malformed_rows)}")
    print("-" * 80)

    if malformed_rows:
        print(f"\n[!] MALFORMED / EMPTY ROWS ({len(malformed_rows)} found):")
        for row_num, reason in malformed_rows:
            print(f"  Row {row_num:4d}: {reason}")
    else:
        print("\n[OK] No empty or malformed rows found.")

    if missing_files:
        print(f"\n[!] MISSING / MISMATCHED IMAGE FILES ({len(missing_files)} found):")
        # Print breakdown summary of missing files
        strictly_missing = [m for m in missing_files if "missing on disk" in m[2]]
        ext_mismatch = [m for m in missing_files if "Extension mismatch" in m[2]]

        if ext_mismatch:
            print(f"\n  -- Extension Mismatches ({len(ext_mismatch)} files):")
            for row_num, fname, note in ext_mismatch[:10]:
                print(f"     Row {row_num:4d}: CSV lists '{fname}' -> {note}")
            if len(ext_mismatch) > 10:
                print(f"     ... and {len(ext_mismatch) - 10} more extension mismatches.")

        if strictly_missing:
            print(f"\n  -- Strictly Missing Files ({len(strictly_missing)} files):")
            for row_num, fname, note in strictly_missing:
                print(f"     Row {row_num:4d}: '{fname}' -> {note}")
        else:
            print("\n  -- Strictly Missing Files: 0 (All CSV image stems present on disk)")
    else:
        print("\n[OK] Every image filename listed in CSV exists on disk.")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    audit_prescription_dataset()
