"""
Source 6: Font rendering fallback for RxHandIndia.
Generates synthetic handwritten word images for Layer 1 drugs NOT covered by real images from Sources 1-5.
"""

import csv
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

BASE_DIR = Path(r"C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump")
VOCAB_PATH = BASE_DIR / "layer1_vocabulary.csv"
REGISTRY_PATH = BASE_DIR / "output" / "word_bank_registry_v2.jsonl"
DETECTION_MANIFESTS = {
    "mehaksingal": BASE_DIR / "output" / "mehaksingal_detection_crops.jsonl",
    "chaithanyakota": BASE_DIR / "output" / "chaithanyakota_detection_crops.jsonl",
    "chhavimohitkar": BASE_DIR / "output" / "chhavimohitkar_detection_crops.jsonl",
}
CHHAVI_PRECROP_PATH = BASE_DIR / "output" / "chhavimohitkar_crops.jsonl"
OUTPUT_DIR = BASE_DIR / "output" / "font_renders"
FONTS_DIR = Path(r"C:\Windows\Fonts")

CANVAS_W, CANVAS_H = 512, 128
MAX_RENDER_DRUGS = 5000


def normalize_name(name):
    return name.lower().strip().replace(" ", "").replace("-", "").replace(".", "").replace("/", "")


def build_prefix_trie(strings):
    trie = {}
    for s in strings:
        node = trie
        for c in s:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['__end__'] = True
    return trie


def is_prefix_match(trie, word, min_len=4):
    if len(word) < min_len:
        return False
    node = trie
    for i, c in enumerate(word):
        if '__end__' in node and i >= min_len:
            return True
        if c not in node:
            return False
        node = node[c]
    return len(word) >= min_len


def load_vocabulary_drugs():
    drugs = set()
    original_case = {}
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("brand_name", "").strip()
            if name:
                norm = normalize_name(name)
                drugs.add(norm)
                original_case[norm] = name
    return drugs, original_case


def load_confirmed_drugs():
    confirmed = set()
    if not REGISTRY_PATH.exists():
        return confirmed
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "").strip()
            if text:
                confirmed.add(normalize_name(text))
    return confirmed


def load_detection_ocr_guesses():
    guesses = set()
    for src_name, manifest_path in DETECTION_MANIFESTS.items():
        if not manifest_path.exists():
            print(f"  [WARN] Manifest not found: {manifest_path}")
            continue
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ocr_text = rec.get("ocr_text_guess", "").strip()
                if ocr_text:
                    guesses.add(normalize_name(ocr_text))
    return guesses


def find_covered_drugs(vocab_norms, confirmed_norms, ocr_guess_norms):
    covered = set(vocab_norms & confirmed_norms)
    covered |= vocab_norms & ocr_guess_norms
    remaining = vocab_norms - covered
    if not remaining or not ocr_guess_norms:
        return covered

    ocr_trie = build_prefix_trie(ocr_guess_norms)
    vocab_trie = build_prefix_trie(remaining)
    vocab_remaining = remaining

    for vnorm in list(vocab_remaining):
        if vnorm in covered:
            continue
        if is_prefix_match(ocr_trie, vnorm):
            covered.add(vnorm)

    for gnorm in ocr_guess_norms:
        if len(gnorm) < 4:
            continue
        node = vocab_trie
        prefix = ""
        for c in gnorm:
            prefix += c
            if c not in node:
                break
            node = node[c]
            if '__end__' in node and len(prefix) >= 4:
                if prefix in vocab_remaining:
                    covered.add(prefix)

    return covered


def discover_fonts():
    preferred_keywords = ["brush", "script", "cursive", "handwriting", "comic", "bradley", "segoesc", "segoepr"]
    available = []
    try:
        for entry in os.listdir(FONTS_DIR):
            entry_lower = entry.lower()
            if entry_lower.endswith((".ttf", ".otf", ".ttc")):
                for kw in preferred_keywords:
                    if kw in entry_lower:
                        available.append(str(FONTS_DIR / entry))
                        break
    except Exception as e:
        print(f"  [WARN] Error scanning fonts: {e}")

    fallback_fonts = []
    for name in ["ariali.ttf", "arial.ttf", "segoeuii.ttf", "segoeui.ttf"]:
        p = FONTS_DIR / name
        if p.exists():
            fallback_fonts.append(str(p))

    if available:
        print(f"  Found {len(available)} handwriting/cursive font(s)")
        return available, fallback_fonts
    else:
        print("  No handwriting fonts found, using fallback fonts")
        return fallback_fonts, []


def try_load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return None


def apply_degradation(img):
    arr = np.array(img).astype(np.float32)
    sigma = random.uniform(3, 8)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    contrast_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    if random.random() < 0.5:
        radius = random.uniform(0.5, 1.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def render_variant(text, font_path, size, variant, fallback_font_path=None):
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = try_load_font(font_path, size)
    if font is None:
        if fallback_font_path:
            font = try_load_font(fallback_font_path, size)
        if font is None:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if variant == "A":
        x = (CANVAS_W - tw) // 2
        y = (CANVAS_H - th) // 2
        draw.text((x, y), text, fill=(0, 0, 0), font=font)

    elif variant == "B":
        angle = random.uniform(-3, 3)
        txt_img = Image.new("RGBA", (CANVAS_W, CANVAS_H), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        bx = txt_draw.textbbox((0, 0), text, font=font)
        tw2, th2 = bx[2] - bx[0], bx[3] - bx[1]
        tx = (CANVAS_W - tw2) // 2
        ty = (CANVAS_H - th2) // 2
        txt_draw.text((tx, ty), text, fill=(0, 0, 0, 255), font=font)
        txt_img = txt_img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255, 0))
        img.paste(txt_img.convert("RGB"), mask=txt_img.split()[3])

    elif variant == "C":
        use_italic = fallback_font_path and fallback_font_path != font_path
        if use_italic:
            font_c = try_load_font(fallback_font_path, size)
            if font_c is None:
                font_c = font
        else:
            font_c = font
        txt_img = Image.new("RGBA", (CANVAS_W * 2, CANVAS_H * 2), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        bx = txt_draw.textbbox((0, 0), text, font=font_c)
        tw2, th2 = bx[2] - bx[0], bx[3] - bx[1]
        stretch_x = random.uniform(1.05, 1.15)
        new_w = int(CANVAS_W * stretch_x)
        txt_draw.text(((CANVAS_W * 2 - tw2) // 2, (CANVAS_H * 2 - th2) // 2), text, fill=(0, 0, 0, 255), font=font_c)
        txt_img = txt_img.resize((new_w, CANVAS_H), Image.LANCZOS)
        paste_x = (CANVAS_W - new_w) // 2
        if paste_x < 0:
            paste_x = 0
        img = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
        img.paste(txt_img.convert("RGB"), (paste_x, 0))

    img = apply_degradation(img)
    return img


def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Source 6: Font Render Fallback for RxHandIndia")
    print("=" * 60)

    # 1. Load vocabulary
    print("\n[1] Loading Layer 1 vocabulary...")
    vocab_norms, original_case = load_vocabulary_drugs()
    print(f"  Total vocabulary drugs: {len(vocab_norms)}")

    # 2. Load confirmed drugs from registry
    print("\n[2] Loading confirmed drugs from registry...")
    confirmed_norms = load_confirmed_drugs()
    print(f"  Confirmed unique drug names (normalized): {len(confirmed_norms)}")

    # 3. Load OCR guesses from detection crops
    print("\n[3] Loading OCR guesses from detection crop manifests...")
    ocr_guess_norms = load_detection_ocr_guesses()
    print(f"  Unique OCR guess texts (normalized): {len(ocr_guess_norms)}")

    # 4. Find covered drugs
    print("\n[4] Finding covered drugs (exact + prefix matching)...")
    covered = find_covered_drugs(vocab_norms, confirmed_norms, ocr_guess_norms)
    uncovered = vocab_norms - covered
    print(f"  Covered by real sources: {len(covered)}")
    print(f"  Uncovered (need font render): {len(uncovered)}")

    if not uncovered:
        print("\n  All drugs are covered! No font renders needed.")
        return

    render_drugs = uncovered
    if len(render_drugs) > MAX_RENDER_DRUGS:
        print(f"  Capping renders from {len(render_drugs)} to {MAX_RENDER_DRUGS} (sorted by name)")
        render_drugs = set(sorted(render_drugs)[:MAX_RENDER_DRUGS])

    # 5. Discover fonts
    print("\n[5] Discovering fonts...")
    primary_fonts, fallback_fonts = discover_fonts()
    font_a_path = primary_fonts[0] if primary_fonts else (fallback_fonts[0] if fallback_fonts else None)
    font_b_path = primary_fonts[0] if primary_fonts else (fallback_fonts[0] if fallback_fonts else None)
    font_c_alt = None
    if len(primary_fonts) > 1:
        font_c_alt = primary_fonts[1]
    elif len(fallback_fonts) > 1:
        font_c_alt = fallback_fonts[1]
    elif fallback_fonts:
        font_c_alt = fallback_fonts[0]

    if font_a_path is None:
        print("  [ERROR] No fonts available at all!")
        sys.exit(1)

    print(f"  Primary font: {Path(font_a_path).name}")
    if font_c_alt:
        print(f"  Alt font (variant C): {Path(font_c_alt).name}")

    # 6. Create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 7. Render
    render_list = sorted(render_drugs)
    print(f"\n[6] Rendering {len(render_list)} drugs x 3 variants = {len(render_list) * 3} images...")
    new_entries = []
    fonts_used = set()
    fonts_used.add(Path(font_a_path).name)
    if font_c_alt:
        fonts_used.add(Path(font_c_alt).name)

    for i, norm_name in enumerate(render_list):
        drug_display = original_case.get(norm_name, norm_name)
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in norm_name)

        for variant, size, fpath, alt_fpath in [
            ("A", 48, font_a_path, None),
            ("B", 40, font_b_path, None),
            ("C", 44, font_a_path, font_c_alt),
        ]:
            img = render_variant(drug_display, fpath, size, variant, fallback_font_path=alt_fpath)
            fname = f"fontrender_{safe_name}_{variant}.png"
            img_path = OUTPUT_DIR / fname
            img.save(str(img_path))

            entry = {
                "image_id": f"fontrender_{safe_name}_{variant}",
                "source": "font_render",
                "text": drug_display,
                "split": "train",
                "generic_name": drug_display,
                "annotation_status": "synthetic",
                "img_path": str(img_path.resolve()),
            }
            new_entries.append(entry)

        if (i + 1) % 50 == 0 or (i + 1) == len(render_list):
            print(f"  Rendered {i + 1}/{len(render_list)} drugs...")

    # 8. Append to registry
    print(f"\n[7] Appending {len(new_entries)} entries to registry...")
    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 9. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total vocabulary drugs:         {len(vocab_norms)}")
    print(f"  Covered by real sources:        {len(covered)}")
    print(f"  Uncovered (font render):        {len(uncovered)}")
    print(f"  Total renders generated:        {len(new_entries)}")
    print(f"  Font(s) used:                   {', '.join(sorted(fonts_used))}")
    print(f"  Output dir:                     {OUTPUT_DIR}")
    print(f"  Registry updated:               {REGISTRY_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()