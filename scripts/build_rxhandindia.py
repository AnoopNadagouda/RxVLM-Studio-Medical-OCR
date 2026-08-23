#!/usr/bin/env python3
"""
RxHandIndia — Complete Dataset Construction Pipeline
=====================================================
End-to-end pipeline that builds the RxHandIndia dataset following
the guidelines in dataaugment.md.

Stages:
  1. Layer 1 Vocabulary — merge Indian drug databases
  2. Co-occurrence Graph — build from composition fields
  3. Word Bank — standardize all word-level images
  4. Layer 2 Visual Vocabulary — image-backed drug entries
  5. Drug Grouping — 5-strategy cascade
  6. Font Rendering — synthetic fallback for uncovered drugs
  7. Augmentation — 3x expand real handwritten crops
  8. Composite RxPage — assemble multi-line prescription images
  9. Train/Val/Test Split — 80/10/10, val/test real-only
  10. Manifest — JSONL output compatible with RxHandBD
  11. Quality Checks — coverage, duplicates, synthetic ratio

Usage:
  python build_rxhandindia.py [--stage STAGE] [--skip-font] [--skip-augment]
"""

import os
import sys
import json
import hashlib
import random
import re
import math
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

try:
    from trdg.generators import GeneratorFromStrings as TrdgGenerator
    HAS_TRDG = True
except ImportError:
    HAS_TRDG = False

from tqdm import tqdm

N_WORKERS = max(1, multiprocessing.cpu_count() - 1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rxhandindia")

BASE = Path(r"C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump")
OUT = BASE / "output"
FONTS_DIR = BASE / "fonts"

CANVAS_W, CANVAS_H = 512, 128
WORD_PAD = 12
LINE_PAD = 16
COMPOSITE_MAX_LINES = 5
COMPOSITES_PER_GROUP = 4
AUGMENT_MULTIPLIER = 3

random.seed(42)
np.random.seed(42)

DOSAGE_FORMS = ["tab", "cap", "syp", "inj", "gel", "oint", "drop", "inh", "susp", "sachet"]
MODIFIERS = [
    "500mg", "250mg", "650mg", "1g", "BD", "OD", "TDS", "1-0-1", "1-0-0",
    "0-0-1", "HS", "SOS", "PRN", "2x", "3x", "x 3 days", "x 5 days",
    "x 7 days", "x 10 days", "after food", "before food",
]

FONT_URLS = {
    "HomemadeApple-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/homemadeapple/HomemadeApple-Regular.ttf",
    "RockSalt-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/rocksalt/RockSalt-Regular.ttf",
    "MsMadi-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/msmadi/MsMadi-Regular.ttf",
    "PermanentMarker-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/permanentmarker/PermanentMarker-Regular.ttf",
    "ReenieBeanie-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/reeniebeanie/ReenieBeanie-Regular.ttf",
    "Caveat-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat%5Bwdth%2Cwght%5D.ttf",
    "KleeOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kleeone/KleeOne-Regular.ttf",
    "PatrickHand-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf",
}


# ═══════════════════════════════════════════════════════════════════
# STAGE 1: Layer 1 Vocabulary
# ═══════════════════════════════════════════════════════════════════

def build_layer1_vocabulary():
    log.info("Stage 1: Building Layer 1 Vocabulary")
    vocab_path = OUT / "layer1_vocabulary.csv"

    records = []
    seen_names = set()

    def add_record(brand, generic="", dosage_form="", atc_class="", manufacturer="", source=""):
        key = brand.strip().lower()
        if not key or key in ("nan", "none", ""):
            return
        if key in seen_names:
            return
        seen_names.add(key)
        records.append({
            "brand_name": brand.strip(),
            "generic_composition": generic.strip(),
            "dosage_form": dosage_form.strip(),
            "atc_class": atc_class.strip(),
            "manufacturer": manufacturer.strip(),
            "source": source,
        })

    # Source 1: All India Drug Bank (az-medicine)
    src1 = BASE / "corpora" / "az-medicine-dataset-of-india" / "A_Z_medicines_dataset_of_India.csv"
    if src1.exists():
        log.info(f"  Loading {src1.name}")
        df = pd.read_csv(src1, low_memory=False)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  az-medicine"):
            name = str(row.get("name", "")).strip()
            comp1 = str(row.get("short_composition1", "")).strip()
            comp2 = str(row.get("short_composition2", "")).strip()
            comp = "; ".join(filter(lambda x: x and x != "nan", [comp1, comp2]))
            dtype = str(row.get("type", "")).strip()
            mfr = str(row.get("manufacturer_name", "")).strip()
            add_record(name, generic=comp, dosage_form=dtype, manufacturer=mfr, source="az_medicine")

    # Source 2: Extensive A-Z Medicines
    src2 = BASE / "corpora" / "az_india_ext" / "Extensive_A_Z_medicines_dataset_of_India.csv"
    if src2.exists():
        log.info(f"  Loading {src2.name}")
        df = pd.read_csv(src2, low_memory=False)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  az_india_ext"):
            name = str(row.get("name", "")).strip()
        comp1 = str(row.get("short_composition1", "")).strip()
        comp2 = str(row.get("short_composition2", "")).strip()
        comp = "; ".join(filter(lambda x: x and x != "nan", [comp1, comp2]))
        atc = str(row.get("Therapeutic Class", "")).strip()
        action = str(row.get("Action Class", "")).strip()
        atc_full = "; ".join(filter(lambda x: x and x != "nan", [atc, action]))
        mfr = str(row.get("manufacturer_name", "")).strip()
        add_record(name, generic=comp, atc_class=atc_full, manufacturer=mfr, source="az_india_ext")

    # Source 3: All India Drug Bank (different CSV)
    src3 = BASE / "corpora" / "all_india_drugbank" / "all_medicine databased.csv"
    if src3.exists():
        log.info(f"  Loading {src3.name}")
        df = pd.read_csv(src3, low_memory=False)
        cols = list(df.columns)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  all_india_drugbank"):
            name = str(row.iloc[1]).strip() if len(cols) > 1 else ""
            comp1 = str(row.get("short_composition1", str(row.iloc[7]) if len(cols) > 7 else "")).strip()
            comp2 = str(row.get("short_composition2", str(row.iloc[8]) if len(cols) > 8 else "")).strip()
            comp = "; ".join(filter(lambda x: x and x != "nan", [comp1, comp2]))
            atc = str(row.get("Therapeutic Class", "")).strip()
            add_record(name, generic=comp, atc_class=atc, source="all_india_drugbank")

    # Source 4: junioralive
    src4 = BASE / "corpora" / "junioralive" / "DATA" / "indian_medicine_data.csv"
    if src4.exists():
        log.info(f"  Loading {src4.name}")
        df = pd.read_csv(src4, low_memory=False)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  junioralive"):
            name = str(row.get("name", "")).strip()
            comp1 = str(row.get("short_composition1", "")).strip()
            comp2 = str(row.get("short_composition2", "")).strip()
            comp = "; ".join(filter(lambda x: x and x != "nan", [comp1, comp2]))
            mfr = str(row.get("manufacturer_name", "")).strip()
            add_record(name, generic=comp, manufacturer=mfr, source="junioralive")

    vocab_df = pd.DataFrame(records)
    vocab_df.to_csv(vocab_path, index=False)
    log.info(f"  Layer 1 vocabulary: {len(vocab_df)} unique drug entries → {vocab_path}")
    return vocab_df


# ═══════════════════════════════════════════════════════════════════
# STAGE 2: Co-occurrence Graph
# ═══════════════════════════════════════════════════════════════════

def build_cooccurrence_graph(vocab_df):
    """Build drug co-occurrence graph from composition overlap."""
    log.info("Stage 2: Building Co-occurrence Graph")
    graph_path = OUT / "co_occurrence_graph.json"

    composition_map = defaultdict(list)
    for _, row in vocab_df.iterrows():
        name = row["brand_name"].strip().lower()
        comp = str(row.get("generic_composition", "")).strip()
        if not comp or comp == "nan":
            continue
        ingredients = re.split(r"[;,+&]", comp)
        ingredients = [ing.strip().split("(")[0].strip().lower() for ing in ingredients if ing.strip()]
        for ing in ingredients:
            if ing:
                composition_map[ing].append(name)

    co_occurrence = defaultdict(lambda: defaultdict(int))
    ingredient_drug_count = {k: len(set(v)) for k, v in composition_map.items()}

    for ing, drugs in composition_map.items():
        unique_drugs = list(set(drugs))
        for i in range(len(unique_drugs)):
            for j in range(i + 1, min(len(unique_drugs), 200)):
                a, b = unique_drugs[i], unique_drugs[j]
                co_occurrence[a][b] += 1
                co_occurrence[b][a] += 1

    graph = {}
    threshold = 0.30
    for drug_a, partners in co_occurrence.items():
        total_a = sum(partners.values())
        if total_a == 0:
            continue
        significant = {}
        for drug_b, count in sorted(partners.items(), key=lambda x: -x[1])[:50]:
            cond_prob = count / total_a
            if cond_prob >= threshold:
                significant[drug_b] = {"count": count, "conditional_prob": round(cond_prob, 3)}
        if significant:
            graph[drug_a] = significant

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    log.info(f"  Co-occurrence graph: {len(graph)} drugs with significant partners → {graph_path}")
    return graph


# ═══════════════════════════════════════════════════════════════════
# STAGE 3: Word Bank
# ═══════════════════════════════════════════════════════════════════

def img_hash(path):
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()[:12]
    except Exception:
        return None


def _enhance_image_quality(img):
    """Enhance image quality: contrast, levels, denoise. Preserve handwriting."""
    # Convert to 8-bit if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Convert to numpy for auto-levels
    img_arr = np.array(img, dtype=np.float32)
    
    # Auto-levels: stretch histogram to use full dynamic range
    # But be conservative to preserve text darkness
    for c in range(3):
        channel = img_arr[:, :, c]
        p2, p98 = np.percentile(channel, [2, 98])
        if p98 > p2 + 10:  # Only if there's meaningful contrast
            channel = (channel - p2) / (p98 - p2 + 1e-5) * 255
            img_arr[:, :, c] = channel
    
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    
    # Enhance contrast moderately for handwriting clarity
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.4)  # 40% contrast boost
    
    # Enhance sharpness for crisp edges
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)  # 20% sharpness boost
    
    # Light median filter for noise reduction (must be odd size)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    return img


def standardize_image(img_path, out_path, canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return False
    
    # Enhance quality
    img = _enhance_image_quality(img)
    
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    w, h = img.size
    if w == 0 or h == 0:
        return False
    scale = min((canvas_w - 8) / w, (canvas_h - 8) / h)
    if scale != 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    offset_x = (canvas_w - img.size[0]) // 2
    offset_y = (canvas_h - img.size[1]) // 2
    canvas.paste(img, (offset_x, offset_y))
    canvas.save(out_path)
    return True


def _standardize_worker(args):
    img_path, out_path, canvas_w, canvas_h = args
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None
    
    # Enhance quality
    img = _enhance_image_quality(img)
    
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    w, h = img.size
    if w == 0 or h == 0:
        return None
    scale = min((canvas_w - 8) / w, (canvas_h - 8) / h)
    if scale != 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    offset_x = (canvas_w - img.size[0]) // 2
    offset_y = (canvas_h - img.size[1]) // 2
    canvas.paste(img, (offset_x, offset_y))
    canvas.save(out_path)
    return out_path


def find_images(directory):
    img_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]:
        img_files.extend(directory.rglob(ext))
    return img_files


def process_word_bank():
    """Process all image sources: word-level go to word bank, full-page go to full_pages for detect→annotate."""
    log.info("Stage 3: Building Word Bank + Registering Full-Page Sources")
    wb_dir = OUT / "word_bank"
    wb_dir.mkdir(parents=True, exist_ok=True)
    fp_dir = OUT / "full_pages"
    fp_dir.mkdir(parents=True, exist_ok=True)
    registry_path = OUT / "word_bank_registry.jsonl"
    fullpages_path = OUT / "full_pages_registry.jsonl"

    existing_registry = []
    existing_hashes = set()
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    existing_registry.append(entry)
                    h = entry.get("image_id", "").split("_")[-1] if "_" in entry.get("image_id", "") else ""
                    if h:
                        existing_hashes.add(h)
        log.info(f"  Existing word bank registry: {len(existing_registry)} entries")

    registry = list(existing_registry)
    full_pages_registry = []
    seen_hashes = set(existing_hashes)
    stats = defaultdict(int)

    def add_entry(source, text, img_path, split="train", generic_name="", synthetic=False):
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
        out_path = wb_dir / out_name
        if not standardize_image(img_path, out_path):
            stats["standardize_failed"] += 1
            return False
        entry = {
            "image_id": image_id,
            "source": source,
            "text": text,
            "split": split,
            "generic_name": generic_name,
            "synthetic": synthetic,
            "img_path": str(out_path),
        }
        registry.append(entry)
        stats[f"source_{source}"] += 1
        return True

    def add_full_page(source, img_path, label=""):
        h = img_hash(img_path)
        if h is None:
            stats[f"{source}_hash_failed"] += 1
            return
        image_id = f"{source}_{h}"
        out_name = f"{image_id}{img_path.suffix}"
        out_path = fp_dir / out_name
        if not out_path.exists():
            import shutil
            shutil.copy2(str(img_path), str(out_path))
        entry = {
            "image_id": image_id,
            "source": source,
            "original_path": str(img_path),
            "full_page_path": str(out_path),
            "label": label.strip() if label else "",
            "status": "detect_pending",
        }
        full_pages_registry.append(entry)
        stats[f"source_{source}_fullpage"] += 1

    # Source 1: kalashsh (highest quality, pre-cropped word images)
    src = BASE / "kalashsh"
    if src.exists():
        log.info("  Processing kalashsh (Source 1: pre-cropped labeled words)...")
        for split_name, csv_name in [("train", "train2.csv"), ("test", "test2.csv"), ("val", "validate2.csv")]:
            csv_path = src / csv_name
            if not csv_path.exists():
                csv_path = src / csv_name.replace("2", "")
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            label_map = {}
            for _, row in df.iterrows():
                fname = str(row.iloc[0]).strip()
                text = str(row.iloc[1]).strip() if len(row) > 1 else ""
                label_map[fname] = text
            img_dirs = [src / "images" / "images" / split_name, src / "images" / split_name]
            img_files = []
            for d in img_dirs:
                if d.exists():
                    img_files = find_images(d)
                    break
            if not img_files:
                all_imgs = find_images(src / "images")
                img_files = [f for f in all_imgs if split_name in str(f) or split_name == "val" and "validate" in str(f)]
            for img_path in tqdm(img_files, desc=f"  kalashsh/{split_name}"):
                fname = img_path.name
                text = label_map.get(fname) or label_map.get(img_path.stem + ".png")
                if text is None:
                    stem = img_path.stem
                    for k, v in label_map.items():
                        if Path(k).stem == stem:
                            text = v
                            break
                if text is None or text == "nan":
                    stats["kalashsh_unmatched"] += 1
                    continue
                add_entry("kalashsh", text, img_path, split=split_name)

    # Source 2: mehaksingal (full prescription pages — detect→annotate needed)
    src = BASE / "mehaksingal" / "data"
    if src.exists():
        log.info("  Processing mehaksingal (Source 2: full-page prescriptions)...")
        img_files = find_images(src)
        for img_path in tqdm(img_files, desc="  mehaksingal"):
            add_full_page("mehaksingal", img_path)

    # Source 3: chaithanyakota (full prescription pages — detect→annotate, has CSV labels)
    src = BASE / "chaithanyakota"
    if src.exists():
        log.info("  Processing chaithanyakota (Source 3: full-page prescriptions with labels)...")
        csv_path = src / "chaithanyakota.csv"
        img_dir = src / "images"
        label_map = {}
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    fname = str(row.iloc[0]).strip()
                    meds = str(row.iloc[1]).strip() if len(row) > 1 else ""
                    if fname and meds and meds.lower() not in ("nan", "none", ""):
                        label_map[fname] = meds
            except Exception:
                pass
        if img_dir.exists():
            img_files = find_images(img_dir)
            for img_path in tqdm(img_files, desc="  chaithanyakota"):
                meds = label_map.get(img_path.name) or label_map.get(img_path.stem + ".jpg")
                add_full_page("chaithanyakota", img_path, label=meds or "")

    # Source 4: avi-kai (labeled word images)
    src = BASE / "avi-kai" / "Prescription_Content" / "images"
    if src.exists():
        log.info("  Processing avi-kai...")
        img_files = find_images(src)
        for img_path in tqdm(img_files, desc="  avi-kai"):
            stem = img_path.stem
            if stem.isdigit():
                stats["avi_kai_skipped_page"] += 1
                continue
            label = stem.replace("_", " ").replace("-", " ").strip()
            if not label:
                continue
            add_entry("avi_kai", label, img_path, split="train")

    # Source 5: chhavimohitkar (prescription pages — detect→annotate needed, no labels)
    src = BASE / "chhavimohitkar"
    if src.exists():
        log.info("  Processing chhavimohitkar (Source 5: full-page prescriptions, no labels)...")
        img_files = find_images(src)
        for img_path in tqdm(img_files, desc="  chhavimohitkar"):
            add_full_page("chhavimohitkar", img_path)

    # Save registries
    with open(registry_path, "w", encoding="utf-8") as f:
        for entry in registry:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(fullpages_path, "w", encoding="utf-8") as f:
        for entry in full_pages_registry:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log.info(f"  Word bank: {len(registry)} word-level images")
    log.info(f"  Full pages: {len(full_pages_registry)} images (awaiting detect→annotate)")
    for k, v in sorted(stats.items()):
        log.info(f"    {k}: {v}")

    source_counts = Counter(e["source"] for e in registry)
    log.info("  Word bank by source:")
    for source, count in source_counts.most_common():
        log.info(f"    {count:5d}  {source}")

    if full_pages_registry:
        fp_counts = Counter(e["source"] for e in full_pages_registry)
        log.info("  Full pages by source:")
        for source, count in fp_counts.most_common():
            log.info(f"    {count:5d}  {source}")

    return registry, full_pages_registry


# ═══════════════════════════════════════════════════════════════════
# STAGE 4: Layer 2 Visual Vocabulary
# ═══════════════════════════════════════════════════════════════════

def build_layer2_vocabulary(registry, vocab_df):
    """Determine which Layer 1 drugs have image backing (Layer 2)."""
    log.info("Stage 4: Building Layer 2 Visual Vocabulary")
    l2_path = OUT / "layer2_visual_vocabulary.json"

    text_to_entries = defaultdict(list)
    for entry in registry:
        text_to_entries[entry["text"].lower()].append(entry)

    brand_name_set = set(vocab_df["brand_name"].str.lower().values)
    generic_map = {}
    for _, row in vocab_df.iterrows():
        brand = row["brand_name"].strip().lower()
        comp = str(row.get("generic_composition", "")).strip().lower()
        if comp and comp != "nan":
            generic_map[brand] = comp

    visual_vocab = {}
    for drug_lower in brand_name_set:
        tier = None
        if drug_lower in text_to_entries:
            tier = 1
        else:
            comp = generic_map.get(drug_lower, "")
            if comp:
                ingredients = [ing.strip().split("(")[0].strip().lower() for ing in re.split(r"[;,+&]", comp) if ing.strip()]
                for ing in ingredients:
                    if ing and ing in text_to_entries:
                        tier = 2
                        break
        if tier:
            entries = text_to_entries.get(drug_lower, [])
            visual_vocab[drug_lower] = {
                "tier": tier,
                "image_count": len(entries),
                "brand_name": drug_lower,
                "sources": list(set(e["source"] for e in entries)) if entries else [],
            }

    with open(l2_path, "w", encoding="utf-8") as f:
        json.dump(visual_vocab, f, ensure_ascii=False, indent=2)

    tier_counts = Counter(v["tier"] for v in visual_vocab.values())
    log.info(f"  Layer 2: {len(visual_vocab)} image-backed drugs")
    for tier, count in sorted(tier_counts.items()):
        log.info(f"    Tier {tier}: {count} drugs")

    return visual_vocab


# ═══════════════════════════════════════════════════════════════════
# STAGE 5: Drug Grouping (5-strategy cascade)
# ═══════════════════════════════════════════════════════════════════

def build_drug_groupings(vocab_df, co_occurrence_graph, visual_vocab):
    """Apply 5-strategy grouping cascade to create clinically coherent drug groups."""
    log.info("Stage 5: Building Drug Groupings")
    groups_path = OUT / "drug_groups.json"

    image_drugs = set(visual_vocab.keys())
    if not image_drugs:
        log.warning("  No image-backed drugs; using all vocabulary entries")
        image_drugs = set(vocab_df["brand_name"].str.lower().values)

    groups = []
    assigned = set()

    def add_group(members, strategy):
        members_lower = [m.lower() for m in members if m.lower() in image_drugs and m.lower() not in assigned]
        if len(members_lower) >= 2:
            groups.append({"strategy": strategy, "drugs": members_lower})
            assigned.update(members_lower)

    # Strategy 1: ATC Therapeutic Class
    atc_groups = defaultdict(list)
    for _, row in vocab_df.iterrows():
        brand = row["brand_name"].strip().lower()
        atc = str(row.get("atc_class", "")).strip()
        if brand in image_drugs and atc and atc != "nan":
            atc_groups[atc].append(brand)
    for atc, drugs in atc_groups.items():
        if len(drugs) >= 2:
            add_group(drugs, "atc_class")

    # Strategy 2: Generic / Active Ingredient
    generic_groups = defaultdict(list)
    for _, row in vocab_df.iterrows():
        brand = row["brand_name"].strip().lower()
        comp = str(row.get("generic_composition", "")).strip().lower()
        if brand in image_drugs and comp and comp != "nan":
            primary_ingredient = re.split(r"[;,+&]", comp)[0].strip().split("(")[0].strip()
            if primary_ingredient:
                generic_groups[primary_ingredient].append(brand)
    for ingredient, drugs in generic_groups.items():
        if len(drugs) >= 2:
            add_group(drugs, "generic_ingredient")

    # Strategy 3: Composition Match
    comp_groups = defaultdict(list)
    for _, row in vocab_df.iterrows():
        brand = row["brand_name"].strip().lower()
        comp = str(row.get("generic_composition", "")).strip().lower()
        if brand in image_drugs and comp and comp != "nan":
            ingredients = set()
            for part in re.split(r"[;,+&]", comp):
                ing = part.strip().split("(")[0].strip().lower()
                if ing:
                    ingredients.add(ing)
            key = ";".join(sorted(ingredients))
            comp_groups[key].append(brand)
    for key, drugs in comp_groups.items():
        if len(drugs) >= 2:
            add_group(drugs, "composition_match")

    # Strategy 4: Co-occurrence Graph
    for drug_a, partners in co_occurrence_graph.items():
        if drug_a not in image_drugs or drug_a in assigned:
            continue
        co_drugs = [p for p in partners.keys() if p in image_drugs and p not in assigned]
        if len(co_drugs) >= 1:
            add_group([drug_a] + co_drugs[:COMPOSITE_MAX_LINES - 1], "co_occurrence")

    # Strategy 5: 4-Character Prefix
    prefix_groups = defaultdict(list)
    for drug in image_drugs:
        if drug not in assigned:
            prefix = drug[:4].lower()
            prefix_groups[prefix].append(drug)
    for prefix, drugs in prefix_groups.items():
        if len(drugs) >= 2:
            add_group(drugs, "prefix4")

    # Remainder: chunk unassigned drugs
    unassigned = [d for d in image_drugs if d not in assigned]
    for i in range(0, len(unassigned), COMPOSITE_MAX_LINES):
        chunk = unassigned[i:i + COMPOSITE_MAX_LINES]
        if len(chunk) >= 2:
            groups.append({"strategy": "remainder", "drugs": chunk})

    with open(groups_path, "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)

    strategy_counts = Counter(g["strategy"] for g in groups)
    log.info(f"  Groups: {len(groups)} total")
    for strat, count in strategy_counts.most_common():
        log.info(f"    {strat}: {count}")
    log.info(f"  Assigned drugs: {len(assigned)}, Unassigned: {len(unassigned)}")

    return groups


# ═══════════════════════════════════════════════════════════════════
# STAGE 6: Font Rendering (Synthetic Fallback)
# ═══════════════════════════════════════════════════════════════════

def download_fonts():
    """Download Google Fonts for handwriting rendering."""
    import requests
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Stage 6a: Downloading fonts")
    for fname, url in FONT_URLS.items():
        out_path = FONTS_DIR / fname
        if out_path.exists():
            log.info(f"  {fname} already exists, skipping")
            continue
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            log.info(f"  Downloaded {fname}")
        except Exception as e:
            log.warning(f"  Failed to download {fname}: {e}")
    return FONTS_DIR


def load_fonts():
    fonts = []
    if not FONTS_DIR.exists():
        return fonts
    for f in FONTS_DIR.rglob("*.ttf"):
        try:
            for size in [36, 40, 44, 48]:
                font = ImageFont.truetype(str(f), size)
                fonts.append((f.name, font))
        except Exception:
            pass
    for f in FONTS_DIR.rglob("*.otf"):
        try:
            for size in [36, 40, 44, 48]:
                font = ImageFont.truetype(str(f), size)
                fonts.append((f.name, font))
        except Exception:
            pass
    return fonts


def apply_geometric_distortions(img):
    w, h = img.size
    # 1. Rotation ±15°
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

    # 2. Horizontal shear ±0.25 (70%)
    if random.random() < 0.7:
        shear = random.uniform(-0.25, 0.25)
        img = img.transform((w, h), Image.AFFINE, (1, shear, 0, 0, 1, 0), fillcolor=(255, 255, 255))

    # 3. Elastic warp (50%) - simplified using small displacements
    if random.random() < 0.5:
        dx = np.random.uniform(-2, 2, (h, w)).astype(np.float32)
        dy = np.random.uniform(-2, 2, (h, w)).astype(np.float32)
        from PIL import ImageFilter
        dx = np.array(Image.fromarray(dx).filter(ImageFilter.GaussianBlur(radius=3)))
        dy = np.array(Image.fromarray(dy).filter(ImageFilter.GaussianBlur(radius=3)))
        # Skip full elastic warp for performance - use subtle random offset instead
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-2, 2)
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(img, (offset_x, offset_y))
        img = canvas

    return img


def apply_degradation(img):
    # 4. Directional smudge / motion blur (50%)
    if random.random() < 0.5:
        kernel_size = random.randint(2, 6)
        angle = random.uniform(0, 180)
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        dx = int(np.cos(np.radians(angle)) * center)
        dy = int(np.sin(np.radians(angle)) * center)
        for i in range(kernel_size):
            t = i / max(kernel_size - 1, 1)
            x = int(center + t * dx)
            y = int(center + t * dy)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
        if kernel.sum() > 0:
            kernel /= kernel.sum()
            arr = np.array(img).astype(np.float32)
            from scipy.ndimage import convolve
            for c in range(3):
                arr[:, :, c] = convolve(arr[:, :, c], kernel, mode="constant", cval=255)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 5. Gaussian noise (always)
    sigma = random.uniform(3, 12)
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # 6. Gaussian blur (40%)
    if random.random() < 0.4:
        radius = random.uniform(0.4, 1.2)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 7. Paper texture overlay (40%)
    if random.random() < 0.4:
        texture = np.random.randint(230, 256, (CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        texture_img = Image.fromarray(texture)
        arr_img = np.array(img).astype(np.float32)
        arr_tex = np.array(texture_img).astype(np.float32)
        blended = (arr_img * arr_tex / 255.0).astype(np.uint8)
        img = Image.fromarray(blended)

    return img


def render_drug_name(text, font_name, font, variants=4):
    images = []
    for variant_idx in range(variants):
        bg_r = random.randint(245, 255)
        bg_g = random.randint(245, 255)
        bg_b = random.randint(240, 252)
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (bg_r, bg_g, bg_b))

        ink_r = random.randint(0, 40)
        ink_g = random.randint(0, 40)
        ink_b = random.randint(20, 80)

        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (CANVAS_W - tw) // 2 + random.randint(-5, 5)
        y = (CANVAS_H - th) // 2 + random.randint(-3, 3)

        draw.text((x, y), text, fill=(ink_r, ink_g, ink_b), font=font)

        canvas = apply_geometric_distortions(canvas)
        canvas = apply_degradation(canvas)

        images.append(canvas)

    return images


def _render_drug_batch(args):
    drug, font_paths, variants = args
    results = []
    display_name = drug.title()
    font_choices = random.sample(font_paths, min(2, len(font_paths)))
    for fp in font_choices:
        try:
            for size in [36, 40, 44, 48]:
                font = ImageFont.truetype(fp, size)
                font_name = Path(fp).name
                images = render_drug_name(display_name, font_name, font, variants=variants)
                for img in images:
                    h = hashlib.md5(img.tobytes()).hexdigest()[:12]
                    image_id = f"synth_{h}"
                    results.append((image_id, display_name, font_name, img))
                break
        except Exception:
            continue
    return results


def _generate_trdg_batch(args):
    drug_name, font_paths, variants_per_font, count = args
    results = []
    display_name = drug_name.title()
    font_subset = random.sample(font_paths, min(2, len(font_paths))) if font_paths else []
    try:
        gen = TrdgGenerator(
            strings=[display_name],
            count=count,
            fonts=font_subset,
            language="en",
            size=random.choice([36, 40, 44, 48]),
            skewing_angle=8,
            random_skew=True,
            blur=2,
            random_blur=True,
            background_type=random.choice([0, 1]),
            distorsion_type=random.choice([0, 1, 2, 3]),
            distorsion_orientation=random.choice([0, 1]),
            width=-1,
            fit=True,
            text_color="#1a1a2e",
        )
        for img, lbl in gen:
            canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
            w, h = img.size
            if w > 0 and h > 0:
                scale = min((CANVAS_W - 8) / w, (CANVAS_H - 8) / h, 1.0)
                if scale < 1.0:
                    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                ox = (CANVAS_W - img.size[0]) // 2
                oy = (CANVAS_H - img.size[1]) // 2
                canvas.paste(img, (ox, oy))
            h = hashlib.md5(canvas.tobytes()).hexdigest()[:12]
            image_id = f"trdg_{h}"
            results.append((image_id, display_name, "trdg", canvas))
    except Exception as e:
        import traceback
        traceback.print_exc()
    return results


def generate_synthetic_images(vocab_df, visual_vocab, registry, skip=False):
    log.info("Stage 6: Generating Synthetic Font-Rendered Images")
    synth_dir = OUT / "word_bank_synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)

    if skip:
        log.info("  Skipped (--skip-font)")
        return []

    text_to_entries = defaultdict(list)
    for entry in registry:
        text_to_entries[entry["text"].lower()].append(entry)

    covered_drugs = set(text_to_entries.keys())

    brand_names = vocab_df["brand_name"].str.lower().unique()
    uncovered = [b for b in brand_names if b not in covered_drugs]

    uncovered.sort(key=lambda x: len(x))
    uncovered = uncovered[:5000]

    synthetic_entries = []

    if HAS_TRDG:
        log.info("  Using TRDG (TextRecognitionDataGenerator) for synthetic rendering")
        font_paths = []
        if FONTS_DIR.exists():
            font_paths = [str(f) for f in FONTS_DIR.rglob("*.ttf")] + [str(f) for f in FONTS_DIR.rglob("*.otf")]
        if not font_paths:
            log.info("  No custom fonts found; TRDG will use built-in fonts")
        else:
            log.info(f"  Using {len(font_paths)} custom fonts from {FONTS_DIR}")

        variants_per_drug = 4
        batch_args = [(drug, font_paths, variants_per_drug, variants_per_drug) for drug in uncovered]

        log.info(f"  Rendering {len(uncovered)} drugs with TRDG ({N_WORKERS} workers)...")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for results in tqdm(pool.map(_generate_trdg_batch, batch_args), total=len(batch_args), desc="  TRDG rendering"):
                for image_id, display_name, font_name, img in results:
                    out_name = f"{image_id}.png"
                    out_path = synth_dir / out_name
                    img.save(out_path)
                    entry = {
                        "image_id": image_id,
                        "source": "trdg",
                        "text": display_name,
                        "split": "train",
                        "generic_name": "",
                        "synthetic": True,
                        "font": font_name,
                        "img_path": str(out_path),
                    }
                    synthetic_entries.append(entry)
    else:
        log.warning("  TRDG not installed; falling back to PIL rendering")
        fonts = load_fonts()
        if not fonts:
            log.warning("  No fonts loaded; downloading...")
            download_fonts()
            fonts = load_fonts()

        if not fonts:
            log.error("  No fonts available, cannot generate synthetic images")
            return []

        log.info(f"  Loaded {len(fonts)} font variants (PIL fallback)")

        font_paths = list(set(str(f) for f in FONTS_DIR.rglob("*.ttf")))
        if not font_paths:
            font_paths = list(set(str(f) for f in FONTS_DIR.rglob("*.otf")))

        batch_args = [(drug, font_paths, 2) for drug in uncovered]

        log.info(f"  Rendering {len(uncovered)} drugs with {N_WORKERS} workers...")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for results in tqdm(pool.map(_render_drug_batch, batch_args), total=len(batch_args), desc="  Rendering synthetic"):
                for image_id, display_name, font_name, img in results:
                    out_name = f"{image_id}.png"
                    out_path = synth_dir / out_name
                    img.save(out_path)
                    entry = {
                        "image_id": image_id,
                        "source": "font",
                        "text": display_name,
                        "split": "train",
                        "generic_name": "",
                        "synthetic": True,
                        "font": font_name,
                        "img_path": str(out_path),
                    }
                    synthetic_entries.append(entry)

    log.info(f"  Generated {len(synthetic_entries)} synthetic images for {len(uncovered)} drugs")
    return synthetic_entries


# ═══════════════════════════════════════════════════════════════════
# STAGE 7: Augmentation (Real Images Only)
# ═══════════════════════════════════════════════════════════════════

def _augment_single_entry_pil(args):
    entry, aug_dir, has_alb = args
    results = []
    img_path = Path(entry["img_path"])
    if not img_path.exists():
        return results
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return results

    if has_alb:
        transform = A.Compose([
            A.Rotate(limit=10, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.GaussNoise(var_limit=(10, 40), p=0.4),
            A.ElasticTransform(alpha=1, sigma=10, p=0.3, border_mode=0, value=255),
        ])

    for aug_idx in range(AUGMENT_MULTIPLIER - 1):
        arr = np.array(img)
        if has_alb:
            try:
                transformed = transform(image=arr)
                aug_arr = transformed["image"]
            except Exception:
                aug_arr = arr
        else:
            aug_img = img.copy()
            if random.random() < 0.7:
                angle = random.uniform(-10, 10)
                aug_img = aug_img.rotate(angle, fillcolor=(255, 255, 255))
            if random.random() < 0.5:
                enhancer = ImageEnhance.Brightness(aug_img)
                aug_img = enhancer.enhance(random.uniform(0.8, 1.2))
                enhancer = ImageEnhance.Contrast(aug_img)
                aug_img = enhancer.enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.4:
                aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(3, 5)))
            if random.random() < 0.4:
                noise = np.random.normal(0, random.uniform(10, 40), np.array(aug_img).shape)
                aug_arr = np.clip(np.array(aug_img) + noise, 0, 255).astype(np.uint8)
                aug_img = Image.fromarray(aug_arr)
            aug_arr = np.array(aug_img)

        aug_img = Image.fromarray(aug_arr)
        h = hashlib.md5(aug_img.tobytes()).hexdigest()[:12]
        image_id = f"aug_{entry['image_id']}_{aug_idx}_{h}"
        out_name = f"{image_id}.png"
        out_path = Path(aug_dir) / out_name
        aug_img.save(out_path)

        aug_entry = dict(entry)
        aug_entry["image_id"] = image_id
        aug_entry["img_path"] = str(out_path)
        aug_entry["source"] = f"aug_{entry.get('source', 'unknown')}"
        aug_entry["synthetic"] = False
        results.append(aug_entry)

    return results


def _augment_single_entry_trdg(args):
    entry, aug_dir, font_paths = args
    results = []
    text = entry.get("text", "").strip()
    if not text:
        return results

    try:
        gen = TrdgGenerator(
            strings=[text],
            count=AUGMENT_MULTIPLIER - 1,
            fonts=random.sample(font_paths, min(2, len(font_paths))) if font_paths else [],
            language="en",
            size=random.choice([48, 52, 56, 60]),
            skewing_angle=8,
            random_skew=True,
            blur=2,
            random_blur=True,
            background_type=random.choice([0, 1]),
            distorsion_type=random.choice([0, 1, 2, 3]),
            distorsion_orientation=random.choice([0, 1]),
            width=-1,
            fit=True,
            text_color="#1a1a2e",
        )
        for aug_idx, (img, lbl) in enumerate(gen):
            if img.size[0] <= 0 or img.size[1] <= 0:
                continue
            h = hashlib.md5(img.tobytes()).hexdigest()[:12]
            image_id = f"trdg_aug_{entry['image_id']}_{aug_idx}_{h}"
            out_name = f"{image_id}.png"
            out_path = Path(aug_dir) / out_name
            img.save(out_path)

            aug_entry = dict(entry)
            aug_entry["image_id"] = image_id
            aug_entry["img_path"] = str(out_path)
            aug_entry["source"] = f"trdg_aug_{entry.get('source', 'unknown')}"
            aug_entry["synthetic"] = False
            results.append(aug_entry)
    except Exception as e:
        import traceback
        traceback.print_exc()

    return results


def augment_real_images(registry, skip=False):
    log.info("Stage 7: Augmenting Real Images")
    aug_dir = OUT / "word_bank_augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)

    if skip:
        log.info("  Skipped (--skip-augment)")
        return []

    real_entries = [e for e in registry if not e.get("synthetic", False)]
    log.info(f"  Augmenting {len(real_entries)} real images x{AUGMENT_MULTIPLIER} with {N_WORKERS} workers")

    augmented_entries = []

    if HAS_TRDG:
        log.info("  Using TRDG for augmentation (re-render text labels with distortions)")
        font_paths = []
        if FONTS_DIR.exists():
            font_paths = [str(f) for f in FONTS_DIR.rglob("*.ttf")] + [str(f) for f in FONTS_DIR.rglob("*.otf")]
        if not font_paths:
            log.info("  No custom fonts found; TRDG will use built-in fonts")

        batch_args = [(e, str(aug_dir), font_paths) for e in real_entries]

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for results in tqdm(pool.map(_augment_single_entry_trdg, batch_args, chunksize=32), total=len(batch_args), desc="  TRDG augmenting"):
                augmented_entries.extend(results)
    else:
        log.warning("  TRDG not installed; falling back to PIL augmentation")
        batch_args = [(e, str(aug_dir), HAS_ALBUMENTATIONS) for e in real_entries]

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for results in tqdm(pool.map(_augment_single_entry_pil, batch_args, chunksize=32), total=len(batch_args), desc="  Augmenting"):
                augmented_entries.extend(results)

    log.info(f"  Generated {len(augmented_entries)} augmented images")
    return augmented_entries


# ═══════════════════════════════════════════════════════════════════
# STAGE 8: Composite RxPage Construction
# ═══════════════════════════════════════════════════════════════════

def create_prescription_line(drug_name, dosage_form, modifier, word_images_by_text, word_canvas=CANVAS_W, word_h=CANVAS_H):
    """Create a single prescription line image: [dosage_form] [drug_name] [modifier?]"""
    line_parts = []
    for part in [dosage_form, drug_name, modifier]:
        if not part:
            continue
        part_lower = part.lower()
        # Try to find a real word image for this part
        candidates = word_images_by_text.get(part_lower, [])
        if candidates:
            chosen = random.choice(candidates)
            try:
                img = Image.open(chosen["img_path"]).convert("RGB")
                img = img.resize((word_canvas, word_h), Image.LANCZOS)
                line_parts.append(img)
                continue
            except Exception:
                pass
        # Fallback: render with default font
        try:
            font = ImageFont.load_default(size=28)
        except TypeError:
            font = ImageFont.load_default()
        part_img = Image.new("RGB", (word_canvas, word_h), (255, 255, 255))
        draw = ImageDraw.Draw(part_img)
        bbox = draw.textbbox((0, 0), part, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (word_canvas - tw) // 2
        y = (word_h - th) // 2
        draw.text((x, y), part, fill=(0, 0, 50), font=font)
        line_parts.append(part_img)

    if not line_parts:
        return None

    # Concatenate horizontally with padding
    total_w = sum(p.size[0] for p in line_parts) + WORD_PAD * (len(line_parts) - 1)
    line_h = max(p.size[1] for p in line_parts)
    line_img = Image.new("RGB", (total_w, line_h), (255, 255, 255))
    x_offset = 0
    for part in line_parts:
        y_offset = (line_h - part.size[1]) // 2
        line_img.paste(part, (x_offset, y_offset))
        x_offset += part.size[0] + WORD_PAD

    return line_img


def _build_single_rxpage(args):
    group, word_images_by_text, dosage_form_map, rxpage_id = args
    drugs = group["drugs"]
    n_lines = min(len(drugs), COMPOSITE_MAX_LINES)
    if n_lines < 2:
        return []

    results = []
    for comp_idx in range(COMPOSITES_PER_GROUP):
        selected = random.sample(drugs, min(n_lines, len(drugs)))
        lines = []
        labels = []

        for drug in selected:
            dosage = dosage_form_map.get(drug, random.choice(DOSAGE_FORMS))
            modifier = random.choice(MODIFIERS) if random.random() < 0.5 else ""
            display_name = drug.title()

            line_img = create_prescription_line(
                display_name, dosage, modifier, word_images_by_text
            )
            if line_img is None:
                continue

            line_label_parts = [p for p in [dosage, display_name, modifier] if p]
            lines.append(line_img)
            labels.append(" ".join(line_label_parts))

        if len(lines) < 2:
            continue

        max_w = max(l.size[0] for l in lines)
        total_h = sum(l.size[1] for l in lines) + LINE_PAD * (len(lines) - 1)
        rxpage_img = Image.new("RGB", (max_w, total_h), (255, 255, 255))

        y_offset = 0
        for line_img in lines:
            x_offset = (max_w - line_img.size[0]) // 2
            rxpage_img.paste(line_img, (x_offset, y_offset))
            y_offset += line_img.size[1] + LINE_PAD

        image_id = f"rxpage_{rxpage_id:06d}_{comp_idx}"
        results.append((image_id, rxpage_img, "\\n".join(labels), group["strategy"]))

    return results


def build_composite_rxpages(groups, all_entries, vocab_df):
    log.info("Stage 8: Building Composite RxPage Images")
    rxpage_dir = OUT / "rxpage"
    rxpage_dir.mkdir(parents=True, exist_ok=True)

    word_images_by_text = defaultdict(list)
    for entry in all_entries:
        word_images_by_text[entry["text"].lower()].append(entry)

    dosage_form_map = {}
    for _, row in vocab_df.iterrows():
        brand = row["brand_name"].strip().lower()
        dtype = str(row.get("dosage_form", "")).strip().lower()
        if dtype and dtype != "nan":
            name_lower = brand
            for df in DOSAGE_FORMS:
                if df in dtype or df in name_lower:
                    dosage_form_map[brand] = df
                    break

    batch_args = [(g, word_images_by_text, dosage_form_map, i) for i, g in enumerate(groups)]
    rxpage_entries = []
    rxpage_id = 0

    log.info(f"  Building rxpages with {N_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for results in tqdm(pool.map(_build_single_rxpage, batch_args, chunksize=16), total=len(batch_args), desc="  Building rxpages"):
            for image_id, rxpage_img, label, strategy in results:
                out_name = f"{image_id}.png"
                out_path = rxpage_dir / out_name
                rxpage_img.save(out_path)

                entry = {
                    "image_id": image_id,
                    "source": "composite",
                    "text": label,
                    "type": "rxpage",
                    "split": "train",
                    "synthetic": False,
                    "img_path": str(out_path),
                }
                rxpage_entries.append(entry)
                rxpage_id += 1

    log.info(f"  Generated {len(rxpage_entries)} rxpage composite images")
    return rxpage_entries


# ═══════════════════════════════════════════════════════════════════
# STAGE 9: Train/Val/Test Split
# ═══════════════════════════════════════════════════════════════════

def assign_splits(all_entries):
    """Assign train/val/test splits. Val and test are real-only. 80/10/10."""
    log.info("Stage 9: Assigning Train/Val/Test Splits")

    real_entries = [e for e in all_entries if not e.get("synthetic", False)]
    synth_entries = [e for e in all_entries if e.get("synthetic", False)]
    rxpage_entries = [e for e in all_entries if e.get("type") == "rxpage"]

    single_real = [e for e in real_entries if e.get("type") != "rxpage"]

    # Group by drug name to prevent data leakage
    drug_groups = defaultdict(list)
    for entry in single_real:
        drug_key = entry["text"].lower().strip()
        drug_groups[drug_key].append(entry)

    drug_names = list(drug_groups.keys())
    random.shuffle(drug_names)

    n = len(drug_names)
    n_test = max(1, int(n * 0.10))
    n_val = max(1, int(n * 0.10))
    n_train = n - n_test - n_val

    test_drugs = set(drug_names[:n_test])
    val_drugs = set(drug_names[n_test:n_test + n_val])
    train_drugs = set(drug_names[n_test + n_val:])

    # Ensure test drugs don't appear in synthetic train
    test_drug_names = test_drugs

    for entry in single_real:
        drug_key = entry["text"].lower().strip()
        if drug_key in test_drugs:
            entry["split"] = "test"
        elif drug_key in val_drugs:
            entry["split"] = "val"
        else:
            entry["split"] = "train"

    # Synthetic entries: train only, exclude test drugs
    for entry in synth_entries:
        drug_key = entry["text"].lower().strip()
        if drug_key in test_drug_names:
            entry["split"] = "exclude"
        else:
            entry["split"] = "train"

    # Rxpage entries: train only
    for entry in rxpage_entries:
        entry["split"] = "train"

    # Filter out excluded
    valid_entries = [e for e in single_real + synth_entries + rxpage_entries if e.get("split") != "exclude"]

    split_counts = Counter(e["split"] for e in valid_entries)
    log.info(f"  Split distribution:")
    for split, count in sorted(split_counts.items()):
        log.info(f"    {split}: {count}")

    type_counts = Counter(e.get("type", "single") for e in valid_entries)
    log.info(f"  Type distribution:")
    for t, count in sorted(type_counts.items()):
        log.info(f"    {t}: {count}")

    return valid_entries


# ═══════════════════════════════════════════════════════════════════
# STAGE 10: Manifest
# ═══════════════════════════════════════════════════════════════════

def write_manifest(entries):
    """Write final JSONL manifest compatible with RxHandBD format."""
    log.info("Stage 10: Writing Manifest")
    manifest_path = OUT / "manifest.jsonl"

    # Make paths relative to dataset root
    out_root = str(OUT).replace("\\", "/")

    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            img_path = entry.get("img_path", "").replace("\\", "/")
            # Make relative
            if out_root in img_path:
                img_path = img_path.replace(out_root + "/", "")

            manifest_entry = {
                "image": img_path,
                "text": entry.get("text", ""),
                "type": entry.get("type", "single"),
                "split": entry.get("split", "train"),
                "synthetic": entry.get("synthetic", False),
            }
            if entry.get("source"):
                manifest_entry["source"] = entry["source"]
            if entry.get("font"):
                manifest_entry["font"] = entry["font"]

            f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

    log.info(f"  Manifest written: {manifest_path} ({len(entries)} entries)")


# ═══════════════════════════════════════════════════════════════════
# STAGE 11: Quality Checks
# ═══════════════════════════════════════════════════════════════════

def run_quality_checks(entries, vocab_df, visual_vocab):
    """Run dataset quality checks from Section 10."""
    log.info("Stage 11: Quality Checks")

    # 1. Label distribution
    text_lengths = [len(e["text"]) for e in entries]
    outliers_long = [e for e in entries if len(e["text"]) > 40]
    outliers_short = [e for e in entries if len(e["text"]) < 3]
    log.info(f"  Label length: min={min(text_lengths)}, max={max(text_lengths)}, mean={np.mean(text_lengths):.1f}")
    log.info(f"  Outliers (>40 chars): {len(outliers_long)}")
    log.info(f"  Outliers (<3 chars): {len(outliers_short)}")

    # 2. Duplicate detection
    hash_map = defaultdict(list)
    for entry in entries:
        p = entry.get("img_path", "")
        if Path(p).exists():
            h = img_hash(p)
            if h:
                hash_map[h].append(entry["image_id"])
    duplicates = {h: ids for h, ids in hash_map.items() if len(ids) > 1}
    log.info(f"  Exact duplicates: {len(duplicates)}")

    # 3. Synthetic ratio
    train_entries = [e for e in entries if e["split"] == "train"]
    synth_train = [e for e in train_entries if e.get("synthetic", False)]
    synth_ratio = len(synth_train) / max(len(train_entries), 1)
    log.info(f"  Synthetic ratio in train: {synth_ratio:.1%} (target ≤ 40%)")
    if synth_ratio > 0.40:
        log.warning(f"  ⚠ Synthetic ratio {synth_ratio:.1%} exceeds 40% target!")

    # 4. Cross-contamination check
    test_drugs = set(e["text"].lower() for e in entries if e["split"] == "test")
    synth_train_drugs = set(e["text"].lower() for e in synth_train)
    contamination = test_drugs & synth_train_drugs
    log.info(f"  Test drugs in synthetic train: {len(contamination)}")
    if contamination:
        log.warning(f"  ⚠ Cross-contamination: {list(contamination)[:10]}...")

    # 5. Vocabulary coverage
    top_500_drugs = vocab_df["brand_name"].str.lower().value_counts().head(500).index.tolist()
    covered = sum(1 for d in top_500_drugs if d in visual_vocab)
    coverage = covered / max(len(top_500_drugs), 1)
    log.info(f"  Top-500 drug coverage (≥1 image): {covered}/{len(top_500_drugs)} = {coverage:.1%}")
    log.info(f"  Target: ≥80%")

    # Summary
    source_counts = Counter(e.get("source", "unknown") for e in entries)
    log.info("  Final source breakdown:")
    for src, count in source_counts.most_common():
        log.info(f"    {count:5d}  {src}")

    split_type = defaultdict(lambda: defaultdict(int))
    for e in entries:
        split_type[e["split"]][e.get("type", "single")] += 1
    log.info("  Split x Type:")
    for split in ["train", "val", "test"]:
        for t in ["single", "rxpage"]:
            log.info(f"    {split}/{t}: {split_type[split][t]}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RxHandIndia Dataset Builder")
    parser.add_argument("--stage", type=int, default=0, help="Run only this stage (1-11), 0=all")
    parser.add_argument("--skip-font", action="store_true", help="Skip font rendering stage")
    parser.add_argument("--skip-augment", action="store_true", help="Skip augmentation stage")
    parser.add_argument("--skip-composite", action="store_true", help="Skip rxpage composite stage")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RxHandIndia — Complete Dataset Construction Pipeline")
    print("=" * 60)
    print(f"DataDump: {BASE}")
    print(f"Output:   {OUT}")

    # Stage 1: Vocabulary
    if args.stage in (0, 1):
        vocab_df = build_layer1_vocabulary()
    else:
        vocab_path = OUT / "layer1_vocabulary.csv"
        if vocab_path.exists():
            vocab_df = pd.read_csv(vocab_path)
        else:
            log.error("Run stage 1 first"); return

    # Stage 2: Co-occurrence
    if args.stage in (0, 2):
        co_graph = build_cooccurrence_graph(vocab_df)
    else:
        graph_path = OUT / "co_occurrence_graph.json"
        co_graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.exists() else {}

    # Stage 3: Word Bank + Full-Page Registration
    if args.stage in (0, 3):
        registry, full_pages_registry = process_word_bank()
    else:
        reg_path = OUT / "word_bank_registry.jsonl"
        fp_path = OUT / "full_pages_registry.jsonl"
        registry = []
        full_pages_registry = []
        if reg_path.exists():
            with open(reg_path) as f:
                for line in f:
                    if line.strip():
                        registry.append(json.loads(line))
        if fp_path.exists():
            with open(fp_path) as f:
                for line in f:
                    if line.strip():
                        full_pages_registry.append(json.loads(line))

    # Stage 4: Visual Vocabulary
    if args.stage in (0, 4):
        visual_vocab = build_layer2_vocabulary(registry, vocab_df)
    else:
        l2_path = OUT / "layer2_visual_vocabulary.json"
        visual_vocab = json.loads(l2_path.read_text(encoding="utf-8")) if l2_path.exists() else {}

    # Stage 5: Drug Grouping
    if args.stage in (0, 5):
        groups = build_drug_groupings(vocab_df, co_graph, visual_vocab)
    else:
        groups_path = OUT / "drug_groups.json"
        groups = json.loads(groups_path.read_text(encoding="utf-8")) if groups_path.exists() else []

    # Stage 6: Font Rendering
    if args.stage in (0, 6):
        synthetic_entries = generate_synthetic_images(vocab_df, visual_vocab, registry, skip=args.skip_font)
    else:
        synthetic_entries = []

    # Stage 7: Augmentation
    if args.stage in (0, 7):
        augmented_entries = augment_real_images(registry, skip=args.skip_augment)
    else:
        augmented_entries = []

    # Combine all entries for next stages
    all_entries = []
    for e in registry:
        e["type"] = "single"
        all_entries.append(e)
    all_entries.extend(synthetic_entries)
    all_entries.extend(augmented_entries)

    # Stage 8: Composite RxPage
    if args.stage in (0, 8) and not args.skip_composite:
        rxpage_entries = build_composite_rxpages(groups, all_entries, vocab_df)
        all_entries.extend(rxpage_entries)
    elif args.stage == 8:
        rxpage_entries = build_composite_rxpages(groups, all_entries, vocab_df)
        all_entries.extend(rxpage_entries)

    # Stage 9: Split
    if args.stage in (0, 9):
        final_entries = assign_splits(all_entries)
    else:
        final_entries = all_entries

    # Stage 10: Manifest
    if args.stage in (0, 10):
        write_manifest(final_entries)
    elif args.stage == 10:
        write_manifest(final_entries)

    # Stage 11: Quality Checks
    if args.stage in (0, 11):
        run_quality_checks(final_entries, vocab_df, visual_vocab)
    elif args.stage == 11:
        run_quality_checks(final_entries, vocab_df, visual_vocab)

    print("\n" + "=" * 60)
    print("DONE — RxHandIndia dataset built successfully")
    print(f"Output directory: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()