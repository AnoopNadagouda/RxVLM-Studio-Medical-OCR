#!/usr/bin/env python3
"""
RxHandIndia Dataset Builder (v2) -- 6 Allowed Sources Only
===========================================================
Stages:
  1. Assemble confirmed word images (kalashsh + avi-kai from registry v2)
  2. Assemble synthetic images (font_render from registry v2)
  3. Split assignment (80/10/10 real, train-only synthetic)
  4. Drug grouping (5 strategies from dataaugment.md)
  5. Co-occurrence graph (from prescription page sources)
  6. Type A sample generation (single word)
  7. Type B sample generation (composite rxpage)
  8. Dataset output (structured dirs + metadata)
  9. Statistics and validation

Usage:
  python 07_build_rxhandindia.py
"""

import os
import sys
import json
import hashlib
import random
import re
import shutil
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rxhandindia_v2")

BASE = Path(r"C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump")
OUT = BASE / "output"
DATASET_DIR = OUT / "rxhandindia_dataset"

ALLOWED_SOURCES = {"kalashsh", "mehaksingal", "chaithanyakota", "avi-kai", "avi_kai", "chhavimohitkar", "font_render"}
PRESCRIPTION_PAGE_SOURCES = {"mehaksingal", "chaithanyakota", "chhavimohitkar"}

CANVAS_W, CANVAS_H = 512, 128
WORD_PAD = 12
LINE_PAD = 16
COMPOSITE_MAX_LINES = 5
COMPOSITES_PER_GROUP = 4

random.seed(42)
np.random.seed(42)

DOSAGE_FORMS = ["tab", "cap", "syp", "inj", "gel", "oint", "drop", "inh", "susp", "sachet"]
MODIFIERS = [
    "500mg", "250mg", "650mg", "1g", "BD", "OD", "TDS", "1-0-1", "1-0-0",
    "0-0-1", "HS", "SOS", "PRN", "2x", "3x", "x 3 days", "x 5 days",
    "x 7 days", "x 10 days", "after food", "before food",
]


# ======================================================================
# STAGE 1: Assemble confirmed word images
# ======================================================================

def load_registry_v2():
    reg_path = OUT / "word_bank_registry_v2.jsonl"
    if not reg_path.exists():
        log.error("word_bank_registry_v2.jsonl not found -- run 03_ingest_allowed_sources.py first")
        sys.exit(1)
    entries = []
    with open(reg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("source") in ALLOWED_SOURCES:
                entries.append(e)
    return entries


def load_detection_crops():
    crops = {}
    for source in PRESCRIPTION_PAGE_SOURCES:
        manifest_path = OUT / f"{source}_detection_crops.jsonl"
        if manifest_path.exists():
            source_crops = []
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    source_crops.append(json.loads(line))
            crops[source] = source_crops
            log.info(f"  Loaded {len(source_crops)} detection crops for {source}")
    return crops


def load_precrops():
    pre_path = OUT / "chhavimohitkar_crops.jsonl"
    if not pre_path.exists():
        return []
    precrs = []
    with open(pre_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            precrs.append(json.loads(line))
    return precrs


def load_label_studio_annotations():
    ann_path = OUT / "label_studio_annotations.json"
    if not ann_path.exists():
        log.info("  No Label Studio annotations found -- detection crops excluded from dataset")
        return {}
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    annotations = {}
    if isinstance(data, list):
        for task in data:
            task_id = task.get("id", "")
            anns = task.get("annotations", [])
            if anns:
                results = []
                for ann in anns:
                    for r in ann.get("result", []):
                        val = r.get("value", {})
                        text = val.get("text", [val.get("labels", [""])])[0] if isinstance(val.get("text"), list) else val.get("text", "")
                        if isinstance(text, list):
                            text = text[0] if text else ""
                        results.append(text.strip())
                if results:
                    annotations[str(task_id)] = results
    log.info(f"  Loaded {len(annotations)} annotated tasks from Label Studio")
    return annotations


def stage1_assemble_confirmed(registry):
    log.info("Stage 1: Assembling confirmed word images")
    confirmed = [e for e in registry if e.get("annotation_status") == "confirmed"]
    for e in confirmed:
        e["sample_type"] = "type_a"
        e["synthetic"] = False
    source_counts = Counter(e["source"] for e in confirmed)
    log.info(f"  Confirmed images: {len(confirmed)}")
    for src, cnt in source_counts.most_common():
        log.info(f"    {src}: {cnt}")
    return confirmed


def merge_annotated_crops(confirmed, detection_crops, precrops, ls_annotations):
    if not ls_annotations:
        log.info("  No Label Studio annotations -- skipping crop merge")
        return confirmed

    merged = list(confirmed)
    added = 0

    for source, crops in detection_crops.items():
        for crop in crops:
            crop_id = crop.get("crop_id", "")
            if crop_id in ls_annotations:
                labels = ls_annotations[crop_id]
                for label in labels:
                    if not label or label.lower() in ("__illegible__", ""):
                        continue
                    entry = {
                        "image_id": crop_id,
                        "source": source,
                        "text": label,
                        "split": "",
                        "generic_name": "",
                        "annotation_status": "annotated",
                        "img_path": crop.get("crop_path", ""),
                        "sample_type": "type_a",
                        "synthetic": False,
                    }
                    merged.append(entry)
                    added += 1

    for crop in precrops:
        crop_id = crop.get("crop_name", crop.get("crop_path", ""))
        crop_key = crop.get("crop_name", "")
        if crop_key in ls_annotations:
            labels = ls_annotations[crop_key]
            for label in labels:
                if not label or label.lower() in ("__illegible__", ""):
                    continue
                entry = {
                    "image_id": f"chhavimohitkar_precrop_{crop.get('prescription_id', '')}_{crop.get('word_index', '')}",
                    "source": "chhavimohitkar",
                    "text": label,
                    "split": "",
                    "generic_name": "",
                    "annotation_status": "annotated",
                    "img_path": crop.get("crop_path", ""),
                    "sample_type": "type_a",
                    "synthetic": False,
                }
                merged.append(entry)
                added += 1

    log.info(f"  Merged {added} annotated crops into dataset")
    return merged


# ======================================================================
# STAGE 2: Assemble synthetic images
# ======================================================================

def stage2_assemble_synthetic(registry):
    log.info("Stage 2: Assembling synthetic images")
    synthetic = [e for e in registry if e.get("source") == "font_render"]
    for e in synthetic:
        e["sample_type"] = "type_a"
        e["synthetic"] = True
    log.info(f"  Synthetic images: {len(synthetic)}")
    return synthetic


# ======================================================================
# STAGE 3: Split assignment
# ======================================================================

def stage3_assign_splits(real_entries, synthetic_entries):
    log.info("Stage 3: Assigning train/val/test splits")

    drug_groups = defaultdict(list)
    for entry in real_entries:
        key = entry["text"].lower().strip()
        drug_groups[key].append(entry)

    drug_names = sorted(drug_groups.keys())
    random.shuffle(drug_names)

    n = len(drug_names)
    n_test = max(1, int(n * 0.10))
    n_val = max(1, int(n * 0.10))

    test_drugs = set(drug_names[:n_test])
    val_drugs = set(drug_names[n_test:n_test + n_val])
    train_drugs = set(drug_names[n_test + n_val:])

    for entry in real_entries:
        key = entry["text"].lower().strip()
        if key in test_drugs:
            entry["split"] = "test"
        elif key in val_drugs:
            entry["split"] = "val"
        else:
            entry["split"] = "train"

    for entry in synthetic_entries:
        key = entry["text"].lower().strip()
        if key in test_drugs:
            entry["split"] = "exclude"
        else:
            entry["split"] = "train"

    valid_real = [e for e in real_entries if e.get("split") != "exclude"]
    valid_synth = [e for e in synthetic_entries if e.get("split") != "exclude"]

    split_counts = Counter(e["split"] for e in valid_real + valid_synth)
    log.info(f"  Real: {len(valid_real)}, Synthetic: {len(valid_synth)}")
    for split, cnt in sorted(split_counts.items()):
        log.info(f"    {split}: {cnt}")

    return valid_real, valid_synth, test_drugs


# ======================================================================
# STAGE 4: Drug grouping strategies
# ======================================================================

def load_drug_vocabulary():
    vocab_records = []
    seen_names = {}

    def add_record(brand, generic="", dosage_form="", atc_class="", manufacturer="", source=""):
        key = brand.strip().lower()
        if not key or key in ("nan", "none", ""):
            return
        rec = {
            "brand_name": brand.strip(),
            "generic_composition": generic.strip(),
            "dosage_form": dosage_form.strip(),
            "atc_class": atc_class.strip(),
            "manufacturer": manufacturer.strip(),
            "source_db": source,
        }
        if key in seen_names:
            idx = seen_names[key]
            existing = vocab_records[idx]
            for field in ("generic_composition", "dosage_form", "atc_class", "manufacturer"):
                if not existing.get(field, "").strip() and rec.get(field, "").strip():
                    existing[field] = rec[field]
            return
        seen_names[key] = len(vocab_records)
        vocab_records.append(rec)

    src1 = BASE / "corpora" / "az-medicine-dataset-of-india" / "A_Z_medicines_dataset_of_India.csv"
    if src1.exists():
        log.info(f"  Loading {src1.name}")
        df = pd.read_csv(src1, low_memory=False)
        for _, row in df.iterrows():
            name = str(row.get("name", "")).strip()
            comp1 = str(row.get("short_composition1", "")).strip()
            comp2 = str(row.get("short_composition2", "")).strip()
            comp = "; ".join(filter(lambda x: x and x != "nan", [comp1, comp2]))
            dtype = str(row.get("type", "")).strip()
            mfr = str(row.get("manufacturer_name", "")).strip()
            add_record(name, generic=comp, dosage_form=dtype, manufacturer=mfr, source="az_medicine")

    src2 = BASE / "corpora" / "az_india_ext" / "Extensive_A_Z_medicines_dataset_of_India.csv"
    if src2.exists():
        log.info(f"  Loading {src2.name}")
        df = pd.read_csv(src2, low_memory=False)
        for _, row in df.iterrows():
            name = str(row.get("name", "")).strip()
            comp1 = str(row.get("short_composition1", "")).strip()
            comp2 = str(row.get("short_composition2", "")).strip()
            comp = "; ".join(filter(lambda x: x and x != "nan", [comp1, comp2]))
            atc = str(row.get("Therapeutic Class", "")).strip()
            action = str(row.get("Action Class", "")).strip()
            atc_full = "; ".join(filter(lambda x: x and x != "nan", [atc, action]))
            mfr = str(row.get("manufacturer_name", "")).strip()
            add_record(name, generic=comp, atc_class=atc_full, manufacturer=mfr, source="az_india_ext")

    src3 = BASE / "corpora" / "all_india_drugbank" / "all_medicine databased.csv"
    if src3.exists():
        log.info(f"  Loading {src3.name}")
        df = pd.read_csv(src3, low_memory=False)
        cols = list(df.columns)
        for _, row in df.iterrows():
            name = str(row.iloc[1]).strip() if len(cols) > 1 else ""
            comp1 = str(row.get("short_composition1", str(row.iloc[7]) if len(cols) > 7 else "")).strip()
            comp2 = str(row.get("short_composition2", str(row.iloc[8]) if len(cols) > 8 else "")).strip()
            comp = "; ".join(filter(lambda x: x and x != "nan", [comp1, comp2]))
            atc = str(row.get("Therapeutic Class", "")).strip()
            add_record(name, generic=comp, atc_class=atc, source="all_india_drugbank")

    src4 = BASE / "corpora" / "junioralive" / "DATA" / "indian_medicine_data.csv"
    if src4.exists():
        log.info(f"  Loading {src4.name}")
        df = pd.read_csv(src4, low_memory=False)
        for _, row in df.iterrows():
            name = str(row.get("name", "")).strip()
            comp1 = str(row.get("short_composition1", "")).strip()
            comp2 = str(row.get("short_composition2", "")).strip()
            comp = "; ".join(filter(lambda x: x and x != "nan", [comp1, comp2]))
            mfr = str(row.get("manufacturer_name", "")).strip()
            add_record(name, generic=comp, manufacturer=mfr, source="junioralive")

    vocab_df = pd.DataFrame(vocab_records)
    log.info(f"  Layer 1 vocabulary: {len(vocab_df)} unique drug entries")
    return vocab_df


def stage4_drug_grouping(real_entries, synthetic_entries, vocab_df, co_occurrence_graph):
    log.info("Stage 4: Building drug groupings (5 strategies)")

    image_drugs = set()
    for e in real_entries + synthetic_entries:
        t = e["text"].lower().strip()
        if t:
            image_drugs.add(t)

    if not image_drugs:
        log.warning("  No image-backed drugs found; using vocabulary names")
        image_drugs = set(vocab_df["brand_name"].str.lower().values) if len(vocab_df) > 0 else set()

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
    log.info(f"    Strategy 1 (ATC): {sum(1 for g in groups if g['strategy'] == 'atc_class')} groups")

    # Strategy 2: Generic / Active Ingredient
    generic_groups = defaultdict(list)
    for _, row in vocab_df.iterrows():
        brand = row["brand_name"].strip().lower()
        comp = str(row.get("generic_composition", "")).strip().lower()
        if brand in image_drugs and comp and comp != "nan":
            primary = re.split(r"[;,+&]", comp)[0].strip().split("(")[0].strip()
            if primary:
                generic_groups[primary].append(brand)
    for ingredient, drugs in generic_groups.items():
        if len(drugs) >= 2:
            add_group(drugs, "generic_ingredient")
    log.info(f"    Strategy 2 (Generic): {sum(1 for g in groups if g['strategy'] == 'generic_ingredient')} groups")

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
    log.info(f"    Strategy 3 (Composition): {sum(1 for g in groups if g['strategy'] == 'composition_match')} groups")

    # Strategy 4: Co-occurrence Graph
    co_groups_added = 0
    for drug_a, partners in co_occurrence_graph.items():
        if drug_a not in image_drugs or drug_a in assigned:
            continue
        co_drugs = [p for p in partners.keys() if p in image_drugs and p not in assigned]
        if len(co_drugs) >= 1:
            add_group([drug_a] + co_drugs[:COMPOSITE_MAX_LINES - 1], "co_occurrence")
            co_groups_added += 1
    log.info(f"    Strategy 4 (Co-occurrence): {co_groups_added} groups")

    # Strategy 5: 4-Character Prefix
    prefix_groups = defaultdict(list)
    for drug in image_drugs:
        if drug not in assigned:
            prefix = drug[:4].lower()
            prefix_groups[prefix].append(drug)
    for prefix, drugs in prefix_groups.items():
        if len(drugs) >= 2:
            add_group(drugs, "prefix4")
    log.info(f"    Strategy 5 (Prefix4): {sum(1 for g in groups if g['strategy'] == 'prefix4')} groups")

    # Remainder
    unassigned = [d for d in sorted(image_drugs) if d not in assigned]
    for i in range(0, len(unassigned), COMPOSITE_MAX_LINES):
        chunk = unassigned[i:i + COMPOSITE_MAX_LINES]
        if len(chunk) >= 2:
            groups.append({"strategy": "remainder", "drugs": chunk})
    log.info(f"    Remainder: {sum(1 for g in groups if g['strategy'] == 'remainder')} groups")

    log.info(f"  Total groups: {len(groups)}, Assigned drugs: {len(assigned)}, Unassigned: {len(unassigned)}")
    return groups


# ======================================================================
# STAGE 5: Co-occurrence graph
# ======================================================================

def stage5_cooccurrence_graph(vocab_df, detection_crops):
    log.info("Stage 5: Building co-occurrence graph")

    graph_path = OUT / "co_occurrence_graph.json"
    if graph_path.exists():
        log.info("  Loading existing co-occurrence graph")
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
        log.info(f"  Graph: {len(graph)} drugs with significant partners")
        return graph

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

    page_crops = defaultdict(list)
    for source, crops in detection_crops.items():
        for crop in crops:
            page_key = crop.get("page_name", "")
            if page_key:
                page_crops[(source, page_key)].append(crop)

    log.info(f"  Prescription pages with crops: {len(page_crops)}")

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    log.info(f"  Co-occurrence graph: {len(graph)} drugs -> {graph_path}")
    return graph


# ======================================================================
# STAGE 6: Type A sample generation
# ======================================================================

def stage6_type_a(real_entries, synthetic_entries):
    log.info("Stage 6: Generating Type A samples (single word)")

    type_a = []
    for e in real_entries + synthetic_entries:
        img_path = e.get("img_path", "")
        if not img_path or not Path(img_path).exists():
            continue
        type_a.append({
            "image_id": e.get("image_id", ""),
            "source": e.get("source", ""),
            "text": e.get("text", ""),
            "generic_name": e.get("generic_name", ""),
            "split": e.get("split", ""),
            "synthetic": e.get("synthetic", False),
            "sample_type": "type_a",
            "img_path": img_path,
        })

    log.info(f"  Type A samples: {len(type_a)}")
    source_counts = Counter(e["source"] for e in type_a)
    for src, cnt in source_counts.most_common():
        log.info(f"    {src}: {cnt}")
    return type_a


# ======================================================================
# STAGE 7: Type B sample generation (composite rxpage)
# ======================================================================

def create_prescription_line(drug_name, dosage_form, modifier, word_images_by_text):
    line_parts = []
    for part in [dosage_form, drug_name, modifier]:
        if not part:
            continue
        part_lower = part.lower()
        candidates = word_images_by_text.get(part_lower, [])
        if candidates:
            chosen = random.choice(candidates)
            try:
                img = Image.open(chosen["img_path"]).convert("RGB")
                img = img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
                line_parts.append(img)
                continue
            except Exception:
                pass
        try:
            font = ImageFont.load_default(28)
        except TypeError:
            font = ImageFont.load_default()
        part_img = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
        draw = ImageDraw.Draw(part_img)
        bbox = draw.textbbox((0, 0), part, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (CANVAS_W - tw) // 2
        y = (CANVAS_H - th) // 2
        draw.text((x, y), part, fill=(0, 0, 50), font=font)
        line_parts.append(part_img)

    if not line_parts:
        return None, ""

    total_w = sum(p.size[0] for p in line_parts) + WORD_PAD * (len(line_parts) - 1)
    line_h = max(p.size[1] for p in line_parts)
    line_img = Image.new("RGB", (total_w, line_h), (255, 255, 255))
    x_offset = 0
    for part in line_parts:
        y_offset = (line_h - part.size[1]) // 2
        line_img.paste(part, (x_offset, y_offset))
        x_offset += part.size[0] + WORD_PAD

    label_parts = [p for p in [dosage_form, drug_name, modifier] if p]
    label = " ".join(label_parts)
    return line_img, label


def stage7_type_b(groups, real_entries, synthetic_entries, vocab_df):
    log.info("Stage 7: Generating Type B samples (composite rxpage)")

    word_images_by_text = defaultdict(list)
    for e in real_entries + synthetic_entries:
        t = e.get("text", "").lower().strip()
        p = e.get("img_path", "")
        if t and p and Path(p).exists():
            word_images_by_text[t].append(e)

    dosage_form_map = {}
    for _, row in vocab_df.iterrows():
        brand = row["brand_name"].strip().lower()
        dtype = str(row.get("dosage_form", "")).strip().lower()
        if dtype and dtype != "nan":
            for df in DOSAGE_FORMS:
                if df in dtype or df in brand:
                    dosage_form_map[brand] = df
                    break

    type_b = []
    rxpage_id = 0

    for group in groups:
        drugs = group["drugs"]
        n_lines = min(len(drugs), COMPOSITE_MAX_LINES)
        if n_lines < 2:
            continue

        for _ in range(COMPOSITES_PER_GROUP):
            selected = random.sample(drugs, min(n_lines, len(drugs)))
            lines = []
            labels = []

            for drug in selected:
                dosage = dosage_form_map.get(drug, random.choice(DOSAGE_FORMS))
                modifier = random.choice(MODIFIERS) if random.random() < 0.5 else ""
                display_name = drug.title()

                line_img, line_label = create_prescription_line(
                    display_name, dosage, modifier, word_images_by_text
                )
                if line_img is None:
                    continue
                lines.append(line_img)
                labels.append(line_label)

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

            image_id = f"rxpage_{rxpage_id:06d}"
            full_label = "\\n".join(labels)

            out_name = f"{image_id}.png"
            out_path = DATASET_DIR / "train" / "type_b" / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rxpage_img.save(out_path)

            type_b.append({
                "image_id": image_id,
                "source": "composite",
                "text": full_label,
                "generic_name": "",
                "split": "train",
                "synthetic": False,
                "sample_type": "type_b",
                "img_path": str(out_path),
                "grouping_strategy": group["strategy"],
            })
            rxpage_id += 1

    log.info(f"  Type B samples: {len(type_b)}")
    return type_b


# ======================================================================
# STAGE 8: Dataset output
# ======================================================================

def stage8_output(real_entries, synthetic_entries, type_a, type_b, groups, co_graph, test_drugs):
    log.info("Stage 8: Writing dataset output")

    for split in ["train", "val", "test"]:
        for stype in ["type_a", "type_b"]:
            (DATASET_DIR / split / stype).mkdir(parents=True, exist_ok=True)

    def copy_to_dataset(entry, split, stype):
        src = Path(entry["img_path"])
        if not src.exists():
            return None
        dest_dir = DATASET_DIR / split / stype
        dest_name = f"{entry['image_id']}.png"
        dest = dest_dir / dest_name
        shutil.copy2(src, dest)
        rel_path = f"{split}/{stype}/{dest_name}"
        return rel_path

    all_samples = []
    for e in type_a:
        split = e.get("split", "train")
        stype = "type_a"
        rel_path = copy_to_dataset(e, split, stype)
        if rel_path is None:
            continue
        sample = {
            "image": rel_path,
            "text": e["text"],
            "type": "single",
            "split": split,
            "synthetic": e.get("synthetic", False),
            "source": e["source"],
            "image_id": e["image_id"],
            "generic_name": e.get("generic_name", ""),
        }
        all_samples.append(sample)

    for e in type_b:
        split = e.get("split", "train")
        stype = "type_b"
        dest_dir = DATASET_DIR / split / stype
        src = Path(e["img_path"])
        if src.exists() and dest_dir.exists():
            dest = dest_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            rel_path = f"{split}/{stype}/{src.name}"
        else:
            rel_path = e["img_path"]
        sample = {
            "image": rel_path,
            "text": e["text"],
            "type": "rxpage",
            "split": split,
            "synthetic": e.get("synthetic", False),
            "source": e.get("source", "composite"),
            "image_id": e["image_id"],
            "grouping_strategy": e.get("grouping_strategy", ""),
        }
        all_samples.append(sample)

    metadata_path = DATASET_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    log.info(f"  metadata.json: {len(all_samples)} samples")

    drug_groups_path = DATASET_DIR / "drug_groups.json"
    with open(drug_groups_path, "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
    log.info(f"  drug_groups.json: {len(groups)} groups")

    coocc_path = DATASET_DIR / "cooccurrence_graph.json"
    with open(coocc_path, "w", encoding="utf-8") as f:
        json.dump(co_graph, f, ensure_ascii=False, indent=2)
    log.info(f"  cooccurrence_graph.json: {len(co_graph)} entries")

    return all_samples


# ======================================================================
# STAGE 9: Statistics and validation
# ======================================================================

def stage9_statistics(all_samples, real_entries, synthetic_entries, test_drugs):
    log.info("Stage 9: Statistics and validation")

    stats = {}

    # Source validation
    sources_in_data = set(e.get("source", "") for e in real_entries + synthetic_entries)
    disallowed = sources_in_data - ALLOWED_SOURCES - {"composite"}
    if disallowed:
        log.error(f"  DISALLOWED SOURCES FOUND: {disallowed}")
    else:
        log.info("  Source validation: PASS (no disallowed sources)")
    stats["disallowed_sources"] = list(disallowed)

    # Split counts
    split_counts = Counter(s["split"] for s in all_samples)
    stats["split_counts"] = dict(split_counts)
    log.info("  Split distribution:")
    for split, cnt in sorted(split_counts.items()):
        log.info(f"    {split}: {cnt}")

    # Type counts
    type_counts = Counter(s["type"] for s in all_samples)
    stats["type_counts"] = dict(type_counts)
    log.info("  Type distribution:")
    for t, cnt in sorted(type_counts.items()):
        log.info(f"    {t}: {cnt}")

    # Split x Type
    split_type = defaultdict(lambda: defaultdict(int))
    for s in all_samples:
        split_type[s["split"]][s["type"]] += 1
    stats["split_type"] = {k: dict(v) for k, v in split_type.items()}
    log.info("  Split x Type:")
    for split in ["train", "val", "test"]:
        for t in ["single", "rxpage"]:
            log.info(f"    {split}/{t}: {split_type[split][t]}")

    # Synthetic in val/test check
    synth_in_val = [s for s in all_samples if s.get("synthetic") and s["split"] in ("val", "test")]
    if synth_in_val:
        log.error(f"  VALIDATION FAIL: {len(synth_in_val)} synthetic images in val/test!")
    else:
        log.info("  Synthetic val/test check: PASS")
    stats["synthetic_in_val_test"] = len(synth_in_val)

    # Source breakdown
    source_counts = Counter(s.get("source", "unknown") for s in all_samples)
    stats["source_counts"] = dict(source_counts.most_common())
    log.info("  Source breakdown:")
    for src, cnt in source_counts.most_common():
        log.info(f"    {src}: {cnt}")

    # Split ratios
    real_split = Counter(e["split"] for e in real_entries)
    total_real = sum(real_split.values())
    if total_real > 0:
        train_pct = real_split.get("train", 0) / total_real * 100
        val_pct = real_split.get("val", 0) / total_real * 100
        test_pct = real_split.get("test", 0) / total_real * 100
        log.info(f"  Real split ratios: train={train_pct:.1f}%, val={val_pct:.1f}%, test={test_pct:.1f}%")
        stats["real_split_ratios"] = {"train": train_pct, "val": val_pct, "test": test_pct}

    # Total counts
    stats["total_samples"] = len(all_samples)
    stats["real_images"] = len(real_entries)
    stats["synthetic_images"] = len(synthetic_entries)
    stats["type_a_samples"] = type_counts.get("single", 0)
    stats["type_b_samples"] = type_counts.get("rxpage", 0)

    # Unique drug names
    unique_drugs = set(s["text"].lower().strip() for s in all_samples if s["type"] == "single")
    stats["unique_drug_names"] = len(unique_drugs)
    log.info(f"  Unique drug names: {len(unique_drugs)}")

    # Synthetic ratio in train
    train_samples = [s for s in all_samples if s["split"] == "train"]
    synth_train = [s for s in train_samples if s.get("synthetic")]
    synth_ratio = len(synth_train) / max(len(train_samples), 1)
    log.info(f"  Synthetic ratio in train: {synth_ratio:.1%} (target <= 40%)")
    stats["synthetic_ratio_train"] = synth_ratio

    stats_path = DATASET_DIR / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log.info(f"  statistics.json written")

    return stats


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="RxHandIndia Dataset Builder v2 (6 allowed sources)")
    parser.add_argument("--skip-type-b", action="store_true", help="Skip Type B composite generation")
    args = parser.parse_args()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RxHandIndia Dataset Builder v2")
    print("Allowed sources: kalashsh, mehaksingal, chaithanyakota,")
    print("                 avi-kai, chhavimohitkar, font_render")
    print("=" * 60)
    print(f"DataDump: {BASE}")
    print(f"Output:   {DATASET_DIR}")

    # Stage 1: Confirmed word images
    registry = load_registry_v2()
    confirmed = stage1_assemble_confirmed(registry)

    # Load detection crops and Label Studio annotations
    detection_crops = load_detection_crops()
    precrops = load_precrops()
    ls_annotations = load_label_studio_annotations()

    # Merge annotated crops if available
    real_entries = merge_annotated_crops(confirmed, detection_crops, precrops, ls_annotations)

    # Stage 2: Synthetic images
    synthetic_entries = stage2_assemble_synthetic(registry)

    # Stage 3: Split assignment
    real_entries, synthetic_entries, test_drugs = stage3_assign_splits(real_entries, synthetic_entries)

    # Stage 4: Drug vocabulary (for grouping)
    vocab_df = load_drug_vocabulary()

    # Stage 5: Co-occurrence graph
    co_graph = stage5_cooccurrence_graph(vocab_df, detection_crops)

    # Stage 4: Drug grouping
    groups = stage4_drug_grouping(real_entries, synthetic_entries, vocab_df, co_graph)

    # Stage 6: Type A samples
    type_a = stage6_type_a(real_entries, synthetic_entries)

    # Stage 7: Type B samples
    type_b = []
    if not args.skip_type_b:
        type_b = stage7_type_b(groups, real_entries, synthetic_entries, vocab_df)
    else:
        log.info("Stage 7: Skipping Type B (--skip-type-b)")

    # Stage 8: Dataset output
    all_samples = stage8_output(real_entries, synthetic_entries, type_a, type_b, groups, co_graph, test_drugs)

    # Stage 9: Statistics and validation
    stats = stage9_statistics(all_samples, real_entries, synthetic_entries, test_drugs)

    print("\n" + "=" * 60)
    print("DONE -- RxHandIndia dataset built successfully")
    print(f"Output: {DATASET_DIR}")
    print(f"Total samples: {stats.get('total_samples', 0)}")
    print(f"  Type A (single word): {stats.get('type_a_samples', 0)}")
    print(f"  Type B (rxpage):      {stats.get('type_b_samples', 0)}")
    print(f"  Real images:          {stats.get('real_images', 0)}")
    print(f"  Synthetic images:     {stats.get('synthetic_images', 0)}")
    print(f"  Unique drug names:    {stats.get('unique_drug_names', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()