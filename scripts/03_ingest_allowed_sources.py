#!/usr/bin/env python3
"""
RxHandIndia — Word Image Bank Ingestion (ALLOWED SOURCES ONLY)

Ingests ONLY the 6 allowed sources per dataaugment.md:
  Source 1: kalashsh — pre-cropped word images with CSV labels (~2,500)
  Source 2: mehaksingal — full page images (detect->annotate pipeline, marked pending)
  Source 3: chaithanyakota — full page images (detect->annotate pipeline, marked pending)
  Source 4: avi-kai — labeled word images by filename (~46)
  Source 5: chhavimohitkar — pre-cropped word images (no labels, marked pending) + full pages
  Source 6: font rendering fallback (separate script)

Output:
  - output/word_bank_v2/*.png        — standardized word images
  - output/word_bank_registry_v2.jsonl — registry entries
"""

import os
import sys
import json
import hashlib
import random
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
from collections import Counter, defaultdict

BASE = Path(r"C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump")
OUT = BASE / "output"
OUT.mkdir(parents=True, exist_ok=True)

WORD_BANK_DIR = OUT / "word_bank_v2"
WORD_BANK_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W, CANVAS_H = 512, 128
RANDOM_SEED = 42

registry = []
stats = defaultdict(int)
seen_hashes = set()


def standardize_image(img_path, out_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        stats["standardize_failed"] += 1
        return False

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
    w, h = img.size
    scale = min((CANVAS_W - 8) / w, (CANVAS_H - 8) / h)
    if scale != 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    offset_x = (CANVAS_W - img.size[0]) // 2
    offset_y = (CANVAS_H - img.size[1]) // 2
    canvas.paste(img, (offset_x, offset_y))
    canvas.save(out_path)
    return True


def img_hash(path):
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()[:12]
    except Exception:
        return None


def find_images(directory):
    img_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]:
        img_files.extend(directory.rglob(ext))
    return img_files


def add_entry(source, text, img_path, split="train", generic_name="",
              annotation_status="confirmed", extra=None):
    if not text or text.strip() == "" or text.strip().lower() in ("nan", "none", ""):
        stats["skipped_empty_text"] += 1
        return False

    text = text.strip()

    h = img_hash(img_path)
    if h is None:
        stats["hash_failed"] += 1
        return False

    if h in seen_hashes:
        stats["skipped_duplicate"] += 1
        return False
    seen_hashes.add(h)

    image_id = f"{source}_{h}"
    out_name = f"{image_id}.png"
    out_path = WORD_BANK_DIR / out_name

    if not standardize_image(img_path, out_path):
        return False

    entry = {
        "image_id": image_id,
        "source": source,
        "text": text,
        "split": split,
        "generic_name": generic_name,
        "annotation_status": annotation_status,
        "img_path": str(out_path),
    }
    if extra:
        entry.update(extra)

    registry.append(entry)
    stats[f"source_{source}"] += 1
    return True


def clean_kalashsh_label(raw_text):
    text = raw_text.strip()
    text = re.sub(r'^\([a-z]\)\s*', '', text)
    text = text.strip()
    return text


# ──────────────────────────────────────────────────
# Source 1: kalashsh/handwritten-medicine-data
# ──────────────────────────────────────────────────
def process_kalashsh():
    src = BASE / "kalashsh"
    print("\n[Source 1] Processing kalashsh...")

    if not src.exists():
        print("  [SKIP] Directory not found")
        return

    img_dir = src / "images" / "images"
    if not img_dir.exists():
        img_dir = src / "images"
        if not img_dir.exists():
            print("  [SKIP] Image directory not found")
            return

    for csv_name, split_name in [("train2.csv", "train"), ("test2.csv", "test"), ("validate2.csv", "val")]:
        csv_path = src / csv_name
        if not csv_path.exists():
            print(f"  [WARN] {csv_name} not found, skipping {split_name}")
            continue

        df = pd.read_csv(csv_path)
        print(f"  {csv_name}: {len(df)} entries")

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  kalashsh/{split_name}"):
            fname = str(row["FILENAME"]).strip()
            raw_identity = str(row["IDENTITY"]).strip()

            text = clean_kalashsh_label(raw_identity)

            found_img = None
            for subdir in ["train", "test", "validate"]:
                candidate = img_dir / subdir / fname
                if candidate.exists():
                    found_img = candidate
                    break

            if found_img is None:
                candidate = img_dir / fname
                if candidate.exists():
                    found_img = candidate

            if found_img is None:
                stats["kalashsh_img_not_found"] += 1
                continue

            add_entry("kalashsh", text, found_img, split=split_name,
                      annotation_status="confirmed")


# ──────────────────────────────────────────────────
# Source 2: mehaksingal/illegible-medical-prescription-images-dataset
# Full pages — needs detect->annotate. Register as pending.
# ──────────────────────────────────────────────────
def process_mehaksingal():
    src = BASE / "mehaksingal"
    print("\n[Source 2] Processing mehaksingal (detect->annotate)...")

    data_dir = src / "data"
    if not data_dir.exists():
        for d in src.iterdir():
            if d.is_dir():
                data_dir = d
                break
        if not data_dir.exists():
            print("  [SKIP] Directory not found")
            return

    img_files = find_images(data_dir)
    print(f"  Found {len(img_files)} page images (need detect->annotate pipeline)")

    detect_dir = OUT / "detection_crops" / "mehaksingal"
    detect_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for img_path in tqdm(img_files, desc="  mehaksingal/register_pages"):
        manifest.append({
            "source": "mehaksingal",
            "page_path": str(img_path),
            "page_name": img_path.name,
            "status": "needs_detection",
        })

    manifest_path = OUT / "mehaksingal_pages.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    stats["mehaksingal_pages"] = len(manifest)
    print(f"  Registered {len(manifest)} pages for detection -> {manifest_path}")


# ──────────────────────────────────────────────────
# Source 3: chaithanyakota/100-handwritten-medical-records
# Full pages with partial labels — needs detect->annotate.
# ──────────────────────────────────────────────────
def process_chaithanyakota():
    src = BASE / "chaithanyakota"
    print("\n[Source 3] Processing chaithanyakota (detect->annotate)...")

    if not src.exists():
        print("  [SKIP] Directory not found")
        return

    img_files = find_images(src)
    page_imgs = [f for f in img_files if f.parent == src or "image" in str(f.parent).lower()]
    print(f"  Found {len(page_imgs)} page images (need detect->annotate pipeline)")

    csv_path = None
    for candidate in src.glob("*.csv"):
        csv_path = candidate
        break

    drug_labels = {}
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"  CSV: {len(df)} rows, columns: {list(df.columns)}")
        for col in df.columns:
            sample = df[col].dropna().head(3).tolist()
            print(f"    {col}: {sample}")

    detect_dir = OUT / "detection_crops" / "chaithanyakota"
    detect_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for img_path in tqdm(page_imgs, desc="  chaithanyakota/register_pages"):
        manifest.append({
            "source": "chaithanyakota",
            "page_path": str(img_path),
            "page_name": img_path.name,
            "status": "needs_detection",
        })

    manifest_path = OUT / "chaithanyakota_pages.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    stats["chaithanyakota_pages"] = len(manifest)
    print(f"  Registered {len(manifest)} pages for detection -> {manifest_path}")


# ──────────────────────────────────────────────────
# Source 4: avi-kai/Medical_Prescription_Handwritten_Words
# Labeled word images — filenames are labels.
# ──────────────────────────────────────────────────
def process_avi_kai():
    src = BASE / "avi-kai" / "Prescription_Content" / "images"
    print("\n[Source 4] Processing avi-kai...")

    if not src.exists():
        print("  [SKIP] Directory not found")
        return

    img_files = find_images(src)
    print(f"  Found {len(img_files)} images")

    for img_path in tqdm(img_files, desc="  avi-kai"):
        stem = img_path.stem

        if stem.isdigit():
            stats["avi_kai_skipped_numeric"] += 1
            continue

        label = stem.replace("_", " ").replace("-", " ").strip()

        skip_names = {"After", "Before", "Blood", "Breakfast", "Sample"}
        if label in skip_names:
            stats["avi_kai_skipped_non_drug"] += 1
            continue

        if not label:
            stats["avi_kai_skipped_empty"] += 1
            continue

        add_entry("avi_kai", label, img_path, split="train",
                  annotation_status="confirmed")


# ──────────────────────────────────────────────────
# Source 5: chhavimohitkar/handwritten_medical_prescription_training_dataset
# Pre-cropped word images (data*.png) — no labels, need annotation.
# Full page images (Sample*.png) — need detect->annotate.
# ──────────────────────────────────────────────────
def process_chhavimohitkar():
    src = BASE / "chhavimohitkar"
    print("\n[Source 5] Processing chhavimohitkar...")

    if not src.exists():
        print("  [SKIP] Directory not found")
        return

    data_crops = sorted(src.glob("data*.png"))
    sample_pages = sorted(src.glob("Sample*.png"))

    print(f"  Found {len(data_crops)} pre-cropped word images (need annotation)")
    print(f"  Found {len(sample_pages)} full page images (need detect->annotate)")

    crop_manifest = []
    for img_path in tqdm(data_crops, desc="  chhavimohitkar/crops"):
        stem = img_path.stem
        parts = stem.replace("data", "").split(".")
        prescription_id = parts[0] if parts else "0"
        word_idx = parts[1] if len(parts) > 1 else "0"

        crop_manifest.append({
            "source": "chhavimohitkar",
            "crop_path": str(img_path),
            "crop_name": img_path.name,
            "prescription_id": prescription_id,
            "word_index": word_idx,
            "status": "needs_annotation",
            "label": "",
        })

    crop_manifest_path = OUT / "chhavimohitkar_crops.jsonl"
    with open(crop_manifest_path, "w", encoding="utf-8") as f:
        for entry in crop_manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    page_manifest = []
    for img_path in tqdm(sample_pages, desc="  chhavimohitkar/pages"):
        page_manifest.append({
            "source": "chhavimohitkar",
            "page_path": str(img_path),
            "page_name": img_path.name,
            "status": "needs_detection",
        })

    page_manifest_path = OUT / "chhavimohitkar_pages.jsonl"
    with open(page_manifest_path, "w", encoding="utf-8") as f:
        for entry in page_manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    stats["chhavimohitkar_crops"] = len(crop_manifest)
    stats["chhavimohitkar_pages"] = len(page_manifest)
    print(f"  Registered {len(crop_manifest)} crops -> {crop_manifest_path}")
    print(f"  Registered {len(page_manifest)} pages -> {page_manifest_path}")


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("RxHandIndia - Word Bank Ingestion (Allowed Sources Only)")
    print("=" * 60)
    print(f"DataDump: {BASE}")
    print(f"Output:   {WORD_BANK_DIR}")
    print(f"Canvas:   {CANVAS_W}x{CANVAS_H}")
    print()
    print("Allowed sources: kalashsh, mehaksingal, chaithanyakota,")
    print("                  avi-kai, chhavimohitkar, font-render")
    print()

    process_kalashsh()
    process_mehaksingal()
    process_chaithanyakota()
    process_avi_kai()
    process_chhavimohitkar()

    registry_path = OUT / "word_bank_registry_v2.jsonl"
    with open(registry_path, "w", encoding="utf-8") as f:
        for entry in registry:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total word images in registry: {len(registry)}")

    confirmed = [e for e in registry if e["annotation_status"] == "confirmed"]
    pending = [e for e in registry if e["annotation_status"] == "pending"]
    print(f"  Confirmed (labeled): {len(confirmed)}")
    print(f"  Pending annotation:  {len(pending)}")

    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    texts = [e["text"] for e in registry]
    text_counts = Counter(texts)
    print(f"\nUnique drug names: {len(text_counts)}")
    print("Top 20 drug names:")
    for text, count in text_counts.most_common(20):
        print(f"    {count:4d}  {text}")

    source_counts = Counter(e["source"] for e in registry)
    print(f"\nBy source:")
    for source, count in source_counts.most_common():
        print(f"    {count:5d}  {source}")

    split_counts = Counter(e["split"] for e in registry)
    print(f"\nBy split:")
    for split, count in split_counts.most_common():
        print(f"    {count:5d}  {split}")

    print(f"\nRegistry saved to: {registry_path}")
    print(f"Images saved to:   {WORD_BANK_DIR}")

    pending_files = list(OUT.glob("*_pages.jsonl")) + list(OUT.glob("*_crops.jsonl"))
    if pending_files:
        print(f"\nPending annotation manifests:")
        for pf in pending_files:
            with open(pf) as f:
                count = sum(1 for _ in f)
            print(f"  {pf.name}: {count} items")


if __name__ == "__main__":
    main()