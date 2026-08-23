#!/usr/bin/env python3
"""
Extract individual letter crops from word-crop handwriting datasets using
horizontal projection profile analysis. Optionally uses PaddleOCR-VL-1.5
spotting to detect text regions first.

Datasets: kaggle_bd_rx, rxhandbd, kalashsh, avi-kai
(No eka* datasets, no composite_medocr)
"""

import argparse
import csv
import glob
import logging
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(r"C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump")
OUTPUT_DIR = BASE_DIR / "letter_dataset"
LETTERS_DIR = OUTPUT_DIR / "letters"

# ---------------------------------------------------------------------------
# PaddleOCR-VL-1.5 spotting helpers
# ---------------------------------------------------------------------------

_model = None
_processor = None
_device = None

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


def load_paddleocr_model(gpu_id: int = 0):
    """Load PaddleOCR-VL-1.5 model once."""
    global _model, _processor, _device
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    model_path = "PaddlePaddle/PaddleOCR-VL-1.5"
    log.info("Loading PaddleOCR-VL-1.5 on %s ...", _device)
    _model = (
        AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        .to(_device)
        .eval()
    )
    _processor = AutoProcessor.from_pretrained(model_path)
    log.info("Processor size: %s", _processor.image_processor.size)
    log.info("PaddleOCR model loaded.")


def paddleocr_spot(image_path: str, task: str = "spotting"):
    """Run spotting (or other task) on a single image. Returns list of (box, text)."""
    import torch

    assert _model is not None, "Model not loaded"
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    spotting_upscale_threshold = 1500
    upscale_factor = 1
    if task == "spotting" and orig_w < spotting_upscale_threshold and orig_h < spotting_upscale_threshold:
        process_w, process_h = orig_w * 2, orig_h * 2
        upscale_factor = 2
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        image = image.resize((process_w, process_h), resample_filter)
    max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPTS[task]},
            ],
        }
    ]
    inputs = _processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": _processor.image_processor.size["shortest_edge"],
                "longest_edge": max_pixels,
            }
        },
    ).to(_model.device)
    outputs = _model.generate(**inputs, max_new_tokens=512)
    result = _processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])

    # Parse: <box>x1,y1,x2,y2</box>text<box>x1,y1,x2,y2</box>text...
    boxes_texts = []
    pattern = r"<box>(\d+),(\d+),(\d+),(\d+)</box>([^<]+)"
    for m in re.finditer(pattern, result):
        x1, y1, x2, y2, text = (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
            m.group(5).strip(),
        )
        # If we upscaled, scale box back
        if upscale_factor > 1:
            x1 //= upscale_factor
            y1 //= upscale_factor
            x2 //= upscale_factor
            y2 //= upscale_factor
        boxes_texts.append(((x1, y1, x2, y2), text))
    return boxes_texts


# ---------------------------------------------------------------------------
# Letter segmentation via horizontal projection profile
# ---------------------------------------------------------------------------


def segment_letters(image_np: np.ndarray, gap_threshold_ratio: float = 0.15):
    """
    Segment a word image into individual letter bounding boxes using
    horizontal projection profile analysis.

    Uses CLAHE contrast enhancement, adaptive thresholding, and morphological
    cleanup to handle low-contrast handwritten word images robustly.

    Returns list of (x1, y1, x2, y2) crop boxes.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np.copy()

    h, w = gray.shape[:2]

    # 1. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 2. Adaptive thresholding (better than Otsu for uneven illumination)
    block_size = max(11, h // 5)
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block_size, 10,
    )

    # 3. Morphological cleanup: opening to remove noise, then dilation to reconnect strokes
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    binary = cv2.dilate(binary, dilate_kernel, iterations=1)

    # 4. Horizontal projection profile
    h_proj = np.sum(binary, axis=0)  # shape: (width,)

    # Smarter gap detection: use median of non-zero column sums
    nonzero_proj = h_proj[h_proj > 0]
    if len(nonzero_proj) == 0:
        return []
    median_val = np.median(nonzero_proj)
    gap_thresh = gap_threshold_ratio * median_val
    if gap_thresh < 1:
        gap_thresh = 1

    # Find runs of non-gap columns (letter regions), with minimum gap width
    width = binary.shape[1]
    in_letter = False
    letter_regions = []
    start = 0
    gap_start = -1
    for col in range(width):
        if h_proj[col] > gap_thresh:
            if not in_letter:
                # Check if the gap we just exited was wide enough (>=2 cols) to split
                in_letter = True
                start = col
            gap_start = -1
        else:
            if in_letter:
                if gap_start == -1:
                    gap_start = col
                # Only end the letter region if gap spans at least 2 consecutive columns
                if col - gap_start + 1 >= 2:
                    letter_regions.append((start, gap_start))
                    in_letter = False
                    gap_start = -1
    if in_letter:
        if gap_start >= 0:
            letter_regions.append((start, gap_start))
        else:
            letter_regions.append((start, width))

    # 6. Filter out segments narrower than 3 pixels (noise)
    letter_regions = [(s, e) for s, e in letter_regions if (e - s) >= 3]

    # Convert to full bounding boxes (include vertical extent)
    boxes = []
    for x_start, x_end in letter_regions:
        col_slice = binary[:, x_start:x_end]
        v_proj = np.sum(col_slice, axis=1)
        y_positions = np.where(v_proj > 0)[0]
        if len(y_positions) == 0:
            continue
        y1 = int(y_positions[0])
        y2 = int(y_positions[-1]) + 1
        boxes.append((x_start, y1, x_end, y2))

    return boxes


# ---------------------------------------------------------------------------
# Crop and normalize a letter
# ---------------------------------------------------------------------------


def crop_and_normalize_letter(
    image_np: np.ndarray, box: tuple, padding: int = 5, target_size: int = 64
):
    """
    Crop letter from image, add padding, normalize to square + resize.
    Returns the normalized PIL Image or None if too small.
    """
    x1, y1, x2, y2 = box
    h, w = image_np.shape[:2]
    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(w, x2 + padding)
    y2p = min(h, y2 + padding)

    crop = image_np[y1p:y2p, x1p:x2p]
    ch, cw = crop.shape[:2]

    # Minimum size filter
    if cw < 5 or ch < 5:
        return None

    # Pad to square with white background
    if len(crop.shape) == 2:
        # Grayscale
        max_side = max(ch, cw)
        pad_top = (max_side - ch) // 2
        pad_bottom = max_side - ch - pad_top
        pad_left = (max_side - cw) // 2
        pad_right = max_side - cw - pad_left
        square = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=255,
        )
        pil_img = Image.fromarray(square, mode="L")
    else:
        max_side = max(ch, cw)
        pad_top = (max_side - ch) // 2
        pad_bottom = max_side - ch - pad_top
        pad_left = (max_side - cw) // 2
        pad_right = max_side - cw - pad_left
        square = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        pil_img = Image.fromarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))

    pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return pil_img


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


def find_rxhandbd_base():
    """Find the rxhandbd dataset base directory (handles Unicode apostrophe)."""
    parent = BASE_DIR / "rxhandbd" / "doctors-handwritten-prescription-bd-dataset"
    # Glob for the inner directory
    matches = list(parent.glob("Doctor*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find rxhandbd inner directory under {parent}")


def load_kaggle_bd_rx():
    """Load kaggle_bd_rx dataset entries: list of (image_path, label, source_image_name)."""
    entries = []
    base = BASE_DIR / "kaggle_bd_rx"

    # Train
    train_csv = base / "Train_Label.csv"
    train_img_dir = base / "Train_Set"
    if train_csv.exists():
        with open(train_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["Images"]
                label = row["Text"]
                img_path = train_img_dir / img_name
                if img_path.exists():
                    entries.append((str(img_path), label, img_name))

    # Test
    test_csv = base / "Test_Labels.csv"
    test_img_dir = base / "Test_Set"
    if test_csv.exists():
        with open(test_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["Images"]
                label = row["Text"]
                img_path = test_img_dir / img_name
                if img_path.exists():
                    entries.append((str(img_path), label, img_name))

    return entries


def load_rxhandbd():
    """Load rxhandbd dataset entries."""
    entries = []
    base = find_rxhandbd_base()

    splits = [
        ("Training", "training_words", "training_labels.csv"),
        ("Testing", "testing_words", "testing_labels.csv"),
        ("Validation", "validation_words", "validation_labels.csv"),
    ]
    for split_name, img_subdir, csv_name in splits:
        csv_path = base / split_name / csv_name
        img_dir = base / split_name / img_subdir
        if not csv_path.exists():
            log.warning("rxhandbd CSV not found: %s", csv_path)
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Columns: IMAGE, MEDICINE_NAME, GENERIC_NAME
                img_name = row["IMAGE"]
                # Use MEDICINE_NAME as label (primary), fallback to GENERIC_NAME
                label = row.get("MEDICINE_NAME", "").strip()
                if not label:
                    label = row.get("GENERIC_NAME", "").strip()
                if not label:
                    continue
                # IMAGE column might just be filename like "0.png"
                img_path = img_dir / img_name
                if img_path.exists():
                    entries.append((str(img_path), label, img_name))

    return entries


def load_kalashsh():
    """Load kalashsh dataset entries using train2.csv, test2.csv, validate2.csv."""
    entries = []
    base = BASE_DIR / "kalashsh"
    img_base = base / "images" / "images"

    splits = [
        ("train", "train2.csv"),
        ("test", "test2.csv"),
        ("validate", "validate2.csv"),
    ]
    for split_dir, csv_name in splits:
        csv_path = base / csv_name
        img_dir = img_base / split_dir
        if not csv_path.exists():
            log.warning("kalashsh CSV not found: %s", csv_path)
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["FILENAME"].strip()
                label = row["IDENTITY"].strip()
                img_path = img_dir / img_name
                if img_path.exists():
                    entries.append((str(img_path), label, img_name))

    return entries


def load_avi_kai():
    """Load avi-kai dataset entries. Label = filename stem."""
    entries = []
    img_dir = BASE_DIR / "avi-kai" / "Prescription_Content" / "images"
    if not img_dir.exists():
        log.warning("avi-kai image dir not found: %s", img_dir)
        return entries

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"):
        for img_path in sorted(img_dir.glob(ext)):
            label = img_path.stem  # e.g. "Amoxicillin" from "Amoxicillin.png"
            entries.append((str(img_path), label, img_path.name))

    return entries


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def filter_label(label: str) -> str:
    """Keep only alphanumeric chars, remove spaces."""
    return re.sub(r"[^A-Za-z0-9]", "", label)


def assign_segments_to_chars_proportional(letter_boxes, word_label):
    """
    Greedy proportional assignment of segments to characters.

    Even if the number of segments doesn't match the number of characters,
    assign each segment to the character whose proportional range contains
    the segment's center.

    Returns list of (box, char) tuples.
    """
    n_chars = len(word_label)
    n_segs = len(letter_boxes)
    if n_chars == 0 or n_segs == 0:
        return []

    # Total width spanned by all segments
    total_width = sum(x2 - x1 for x1, y1, x2, y2 in letter_boxes)
    if total_width == 0:
        return []

    # Compute cumulative proportion boundaries for each character
    # char i occupies [i/n_chars, (i+1)/n_chars) of total width
    char_boundaries = []
    for i in range(n_chars):
        start_frac = i / n_chars
        end_frac = (i + 1) / n_chars
        char_boundaries.append((start_frac * total_width, end_frac * total_width))

    # For each segment, find which character its center falls into
    result = []
    for box in letter_boxes:
        x1, y1, x2, y2 = box
        seg_center = (x1 + x2) / 2.0
        # Find the character whose range contains this center
        assigned_char_idx = 0
        for ci, (c_start, c_end) in enumerate(char_boundaries):
            if c_start <= seg_center < c_end:
                assigned_char_idx = ci
                break
        else:
            # Center falls past last boundary — assign to last char
            assigned_char_idx = n_chars - 1
        result.append((box, word_label[assigned_char_idx]))

    return result


def process_image(
    img_path: str,
    label: str,
    source_name: str,
    dataset_prefix: str,
    skip_spotting: bool,
    labels_rows: list,
    skipped_rows: list,
    letter_counter: dict,
    min_segment_ratio: float = 0.5,
    gap_threshold_ratio: float = 0.15,
):
    """
    Process a single image: segment letters, assign labels, save crops.
    """
    try:
        image_pil = Image.open(img_path).convert("RGB")
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        log.warning("Failed to open %s: %s", img_path, e)
        skipped_rows.append((source_name, f"Failed to open image: {e}"))
        return

    clean_label = filter_label(label)
    if not clean_label:
        skipped_rows.append((source_name, "Empty label after filtering"))
        return

    if skip_spotting:
        # Treat the entire image as one word
        word_boxes = [((0, 0, image_np.shape[1], image_np.shape[0]), label)]
    else:
        # Use PaddleOCR spotting to detect word regions
        try:
            spotted = paddleocr_spot(img_path)
        except Exception as e:
            log.warning("PaddleOCR spotting failed for %s: %s", img_path, e)
            skipped_rows.append((source_name, f"PaddleOCR spotting failed: {e}"))
            return
        if not spotted:
            # Fallback: treat entire image as one word
            word_boxes = [((0, 0, image_np.shape[1], image_np.shape[0]), label)]
        else:
            # Map spotted text back to label characters
            # Join all spotted text and try to match with label
            word_boxes = spotted

    # Process each word box
    for word_idx, (box, word_text) in enumerate(word_boxes):
        x1, y1, x2, y2 = box
        # Clamp coordinates
        h, w = image_np.shape[:2]
        x1c = max(0, min(x1, w))
        y1c = max(0, min(y1, h))
        x2c = max(0, min(x2, w))
        y2c = max(0, min(y2, h))

        if x2c <= x1c or y2c <= y1c:
            continue

        word_crop = image_np[y1c:y2c, x1c:x2c]

        # Segment letters within this word crop
        letter_boxes = segment_letters(word_crop, gap_threshold_ratio=gap_threshold_ratio)

        if not skip_spotting:
            # When using spotting, the word_text comes from OCR
            # We need to map it to the known label
            # Use the spotted word text filtered as the label for this segment
            word_label = filter_label(word_text)
        else:
            # The whole image is one word, use the full label
            word_label = clean_label

        if not word_label:
            continue

        n_segs = len(letter_boxes)
        n_chars = len(word_label)

        # Check segment/char count ratio — skip if too far off
        if n_segs > 0 and n_chars > 0:
            ratio = n_segs / n_chars
            if ratio < min_segment_ratio or ratio > (1.0 / min_segment_ratio):
                log.warning(
                    "Segment/label ratio too far off for %s (word %d): %d segments vs %d chars '%s' (ratio=%.2f)",
                    source_name, word_idx, n_segs, n_chars, word_label, ratio,
                )
                skipped_rows.append((
                    source_name,
                    f"Segment/label ratio off: {n_segs} segments vs {n_chars} chars for '{word_label}' (ratio={ratio:.2f})",
                ))
                continue

        if n_segs != n_chars:
            log.warning(
                "Segment/label mismatch for %s (word %d): %d segments vs %d chars '%s' — using proportional assignment",
                source_name, word_idx, n_segs, n_chars, word_label,
            )

        # Assign segments to characters using greedy proportional assignment
        seg_char_pairs = assign_segments_to_chars_proportional(letter_boxes, word_label)

        # Save each assigned letter crop
        for seg_idx, (lbox, char) in enumerate(seg_char_pairs):
            char_lower = char.lower()
            # Offset letter box coords to full image coords
            full_box = (lbox[0] + x1c, lbox[1] + y1c, lbox[2] + x1c, lbox[3] + y1c)

            pil_letter = crop_and_normalize_letter(image_np, full_box)
            if pil_letter is None:
                continue

            # Generate filename: dataset_prefix_source_segidx.png
            # Sanitize source_name for use in filename
            safe_source = Path(source_name).stem
            fname = f"{dataset_prefix}_{safe_source}_{word_idx}_{seg_idx}.png"
            letter_dir = LETTERS_DIR / char_lower
            letter_dir.mkdir(parents=True, exist_ok=True)
            save_path = letter_dir / fname
            pil_letter.save(str(save_path))

            crop_rel = f"letters/{char_lower}/{fname}"
            labels_rows.append((char_lower, source_name, crop_rel))
            letter_counter[char_lower] = letter_counter.get(char_lower, 0) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Extract letter crops from word-crop datasets")
    parser.add_argument(
        "--dataset",
        choices=["all", "kaggle_bd_rx", "rxhandbd", "kalashsh", "avi-kai"],
        default="all",
        help="Which dataset(s) to process",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--skip-spotting",
        action="store_true",
        help="Skip PaddleOCR spotting; segment word images directly",
    )
    parser.add_argument(
        "--min-segment-ratio",
        type=float,
        default=0.5,
        help="Minimum segment/char ratio; skip image if num_segments/num_chars or num_chars/num_segments falls below this (default: 0.5)",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.15,
        help="Gap threshold ratio for segment_letters; columns with projection sum below ratio*median are gaps (default: 0.15)",
    )
    args = parser.parse_args()

    datasets_to_run = []
    if args.dataset == "all":
        datasets_to_run = ["kaggle_bd_rx", "rxhandbd", "kalashsh", "avi-kai"]
    else:
        datasets_to_run = [args.dataset]

    # Create output dirs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LETTERS_DIR.mkdir(parents=True, exist_ok=True)

    # Load PaddleOCR model if needed
    if not args.skip_spotting:
        load_paddleocr_model(args.gpu)

    # Load datasets
    dataset_entries = {}
    loaders = {
        "kaggle_bd_rx": load_kaggle_bd_rx,
        "rxhandbd": load_rxhandbd,
        "kalashsh": load_kalashsh,
        "avi-kai": load_avi_kai,
    }
    for ds_name in datasets_to_run:
        log.info("Loading dataset: %s", ds_name)
        entries = loaders[ds_name]()
        log.info("  %s: %d images", ds_name, len(entries))
        dataset_entries[ds_name] = entries

    # Process
    all_labels = []
    all_skipped = []
    letter_counter = {}

    for ds_name, entries in dataset_entries.items():
        prefix = ds_name.replace("-", "_")
        log.info("Processing dataset: %s (%d images)", ds_name, len(entries))
        for i, (img_path, label, source_name) in enumerate(entries):
            process_image(
                img_path=img_path,
                label=label,
                source_name=source_name,
                dataset_prefix=prefix,
                skip_spotting=args.skip_spotting,
                labels_rows=all_labels,
                skipped_rows=all_skipped,
                letter_counter=letter_counter,
                min_segment_ratio=args.min_segment_ratio,
                gap_threshold_ratio=args.gap_threshold,
            )
            if (i + 1) % 100 == 0:
                log.info("  [%s] Processed %d/%d images", ds_name, i + 1, len(entries))

    # Write labels.csv
    labels_csv_path = OUTPUT_DIR / "labels.csv"
    with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["letter", "source_image", "letter_crop_path"])
        writer.writerows(all_labels)
    log.info("Wrote %d entries to %s", len(all_labels), labels_csv_path)

    # Write skipped.csv
    skipped_csv_path = OUTPUT_DIR / "skipped.csv"
    with open(skipped_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_image", "reason"])
        writer.writerows(all_skipped)
    log.info("Wrote %d skipped entries to %s", len(all_skipped), skipped_csv_path)

    # Summary
    log.info("=== Summary ===")
    log.info("Total letter crops: %d", len(all_labels))
    log.info("Total skipped: %d", len(all_skipped))
    log.info("Letter distribution:")
    for ch in sorted(letter_counter.keys()):
        log.info("  %s: %d", ch, letter_counter[ch])


if __name__ == "__main__":
    main()