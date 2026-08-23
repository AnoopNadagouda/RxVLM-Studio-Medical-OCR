#!/usr/bin/env python3
"""
RxHandIndia -- Word Detection on Prescription Pages

Runs PaddleOCR text detection on full-page prescription images from:
  - mehaksingal (129 pages)
  - chaithanyakota (100 pages)
  - chhavimohitkar (25 Sample pages)

Outputs cropped word images to output/detection_crops/{source}/
and detection manifest JSONL files for Label Studio annotation.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BASE = Path(r"C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump")
OUT = BASE / "output"
CROP_BASE = OUT / "detection_crops"

CANVAS_W, CANVAS_H = 512, 128
MIN_CROP_W = 20
MIN_CROP_H = 10
PADDING = 5

seen_hashes = set()


def standardize_crop(img, out_path):
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
    w, h = img.size
    scale = min((CANVAS_W - 8) / w, (CANVAS_H - 8) / h, 4.0)
    if scale != 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    offset_x = (CANVAS_W - img.size[0]) // 2
    offset_y = (CANVAS_H - img.size[1]) // 2
    canvas.paste(img, (offset_x, offset_y))
    canvas.save(out_path)
    return True


def img_hash(pil_img):
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()[:12]


def detect_and_crop(source_name, page_manifest_path, ocr):
    manifest_path = Path(page_manifest_path)
    if not manifest_path.exists():
        print(f"  [SKIP] Manifest not found: {manifest_path}")
        return 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f if line.strip()]

    print(f"  Processing {len(pages)} pages from {source_name}...")

    source_crop_dir = CROP_BASE / source_name
    source_crop_dir.mkdir(parents=True, exist_ok=True)

    all_crops = []
    skipped = 0

    for page_entry in tqdm(pages, desc=f"  {source_name}/detect"):
        page_path = Path(page_entry["page_path"])
        if not page_path.exists():
            skipped += 1
            continue

        try:
            page_img = Image.open(page_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        try:
            result = ocr.ocr(str(page_path))
        except Exception as e:
            print(f"    [WARN] OCR failed for {page_path.name}: {e}")
            skipped += 1
            continue

        if not result or not result[0]:
            continue

        r0 = result[0]
        polys = r0['dt_polys']
        rec_texts = r0['rec_texts']

        if not polys:
            continue

        page_w, page_h = page_img.size
        crop_idx = 0

        for i, bbox in enumerate(polys):
            bbox = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
            text_guess = rec_texts[i] if i < len(rec_texts) else ""

            if not bbox or len(bbox) < 4:
                continue

            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x1 = max(0, int(min(xs)) - PADDING)
            y1 = max(0, int(min(ys)) - PADDING)
            x2 = min(page_w, int(max(xs)) + PADDING)
            y2 = min(page_h, int(max(ys)) + PADDING)

            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_w < MIN_CROP_W or crop_h < MIN_CROP_H:
                continue

            crop = page_img.crop((x1, y1, x2, y2))

            h = img_hash(crop)
            if h in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(h)

            crop_id = f"{source_name}_{page_path.stem}_{crop_idx}"
            crop_out_name = f"{crop_id}.png"
            crop_out_path = source_crop_dir / crop_out_name

            standardize_crop(crop, crop_out_path)

            all_crops.append({
                "crop_id": crop_id,
                "source": source_name,
                "page_path": str(page_path),
                "page_name": page_path.name,
                "bbox": [x1, y1, x2, y2],
                "ocr_text_guess": text_guess,
                "status": "needs_annotation",
                "crop_path": str(crop_out_path),
                "crop_name": crop_out_name,
            })
            crop_idx += 1

    crops_manifest_path = OUT / f"{source_name}_detection_crops.jsonl"
    with open(crops_manifest_path, "w", encoding="utf-8") as f:
        for entry in all_crops:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Detected {len(all_crops)} word crops ({skipped} skipped)")
    print(f"  Saved to: {crops_manifest_path}")
    return len(all_crops)


def main():
    print("=" * 60)
    print("RxHandIndia - Word Detection on Prescription Pages")
    print("=" * 60)

    from paddleocr import PaddleOCR
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang="en")
    print("PaddleOCR ready.\n")

    total_crops = 0

    for source_name, manifest_file in [
        ("mehaksingal", "mehaksingal_pages.jsonl"),
        ("chaithanyakota", "chaithanyakota_pages.jsonl"),
        ("chhavimohitkar", "chhavimohitkar_pages.jsonl"),
    ]:
        print(f"\n[{source_name}]")
        manifest_path = OUT / manifest_file
        count = detect_and_crop(source_name, manifest_path, ocr)
        total_crops += count

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_crops} word crops detected across all sources")
    print("=" * 60)

    for source_name in ["mehaksingal", "chaithanyakota", "chhavimohitkar"]:
        crop_manifest = OUT / f"{source_name}_detection_crops.jsonl"
        if crop_manifest.exists():
            with open(crop_manifest) as f:
                count = sum(1 for _ in f)
            print(f"  {source_name}: {count} crops")


if __name__ == "__main__":
    main()