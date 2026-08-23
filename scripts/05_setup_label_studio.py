#!/usr/bin/env python3
"""
RxHandIndia -- Label Studio Setup for Word Crop Annotation

Reads detection crop manifests (JSONL) from all 3 sources plus the
pre-cropped chhavimohitkar images, and produces a single JSON file
that can be imported into a Label Studio project.

Also prints instructions for starting Label Studio and importing tasks.
"""

import csv
import json
import os
import sys

BASE = r"C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump"
OUT = os.path.join(BASE, "output")
VOCAB_PATH = os.path.join(BASE, "layer1_vocabulary.csv")

DETECTION_MANIFESTS = {
    "mehaksingal": os.path.join(OUT, "mehaksingal_detection_crops.jsonl"),
    "chaithanyakota": os.path.join(OUT, "chaithanyakota_detection_crops.jsonl"),
    "chhavimohitkar": os.path.join(OUT, "chhavimohitkar_detection_crops.jsonl"),
}

PRECROP_MANIFEST = os.path.join(OUT, "chhavimohitkar_crops.jsonl")

IMPORT_OUTPUT = os.path.join(OUT, "label_studio_import.json")

LABEL_STUDIO_CONFIG = """
<View>
  <Image name="crop_image" value="$image"/>
  <Header value="Drug Name (type or select from suggestions)"/>
  <TextArea name="drug_name" value="$ocr_hint"
            placeholder="Enter drug name..."
            maxUsages="1"/>
  <Header value="Skip if not a drug"/>
  <Choices name="skip_choice" toName="crop_image" choice="single-radio">
    <Choice value="skip" alias="skip"/>
  </Choices>
</View>
"""


def load_vocabulary(vocab_path):
    names = set()
    with open(vocab_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            brand = row.get("brand_name", "").strip()
            if brand:
                names.add(brand.lower())
    return sorted(names)


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_detection_task(entry):
    crop_path = entry.get("crop_path", "")
    if not os.path.isabs(crop_path):
        crop_path = os.path.abspath(os.path.join(BASE, crop_path))

    return {
        "data": {
            "image": crop_path,
            "source": entry.get("source", "unknown"),
            "ocr_hint": entry.get("ocr_text_guess", ""),
            "crop_id": entry.get("crop_id", ""),
        }
    }


def build_precrop_task(entry):
    crop_path = entry.get("crop_path", "")
    if not os.path.isabs(crop_path):
        crop_path = os.path.abspath(os.path.join(BASE, crop_path))

    return {
        "data": {
            "image": crop_path,
            "source": entry.get("source", "chhavimohitkar_precrop"),
            "ocr_hint": "",
            "crop_id": "precrop_{}_{}".format(
                entry.get("prescription_id", ""),
                entry.get("word_index", "")
            ),
        }
    }


def main():
    print("=" * 60)
    print("RxHandIndia - Label Studio Setup")
    print("=" * 60)

    # --- Load vocabulary ---
    print("\n[1/4] Loading drug vocabulary...")
    vocab = load_vocabulary(VOCAB_PATH)
    print("  Loaded {} unique drug names".format(len(vocab)))

    vocab_output = os.path.join(OUT, "label_studio_vocabulary.json")
    with open(vocab_output, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print("  Saved vocabulary to: {}".format(vocab_output))

    # --- Load detection crops ---
    print("\n[2/4] Loading detection crop manifests...")
    all_tasks = []
    source_counts = {}

    for source_name, manifest_path in DETECTION_MANIFESTS.items():
        if not os.path.exists(manifest_path):
            print("  WARNING: Manifest not found: {}".format(manifest_path))
            continue
        entries = load_jsonl(manifest_path)
        count = 0
        for entry in entries:
            all_tasks.append(build_detection_task(entry))
            count += 1
        source_counts[source_name] = count
        print("  {}: {} crops".format(source_name, count))

    # --- Load pre-cropped chhavimohitkar images ---
    print("\n[3/4] Loading pre-cropped chhavimohitkar images...")
    if os.path.exists(PRECROP_MANIFEST):
        precrop_entries = load_jsonl(PRECROP_MANIFEST)
        precrop_count = 0
        for entry in precrop_entries:
            all_tasks.append(build_precrop_task(entry))
            precrop_count += 1
        source_counts["chhavimohitkar_precrop"] = precrop_count
        print("  chhavimohitkar (pre-crops): {} crops".format(precrop_count))
    else:
        print("  WARNING: Pre-crop manifest not found: {}".format(PRECROP_MANIFEST))

    total = len(all_tasks)
    print("\n  TOTAL tasks: {}".format(total))

    # --- Write import JSON ---
    print("\n[4/4] Writing Label Studio import JSON...")
    with open(IMPORT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)
    print("  Saved to: {}".format(IMPORT_OUTPUT))
    print("  File size: {:.1f} MB".format(
        os.path.getsize(IMPORT_OUTPUT) / (1024 * 1024)
    ))

    # --- Save labeling config ---
    config_output = os.path.join(OUT, "label_studio_config.xml")
    with open(config_output, "w", encoding="utf-8") as f:
        f.write(LABEL_STUDIO_CONFIG.strip())
    print("  Saved labeling config to: {}".format(config_output))

    # --- Print instructions ---
    print("\n" + "=" * 60)
    print("LABEL STUDIO INSTRUCTIONS")
    print("=" * 60)
    print("""
1. START LABEL STUDIO:
   > label-studio start

   This opens a browser at http://localhost:8080
   First time: create an account (any email/password).

2. CREATE THE PROJECT:
   - Click "Create Project"
   - Name: "RxHandIndia - Word Crop Annotation"
   - Go to "Labeling Interface" tab
   - Click "Code" toggle
   - Paste the XML from: {config}
   - Click "Save"

3. IMPORT TASKS:
   - Open the project
   - Click "Import" button
   - Upload the file: {import_file}
   - Confirm import

4. ANNOTATE:
   - For each crop, type the drug name in the text box
     (OCR hint is pre-filled as a starting point)
   - If the crop is NOT a drug name (header, patient info, etc.)
     check the "skip" radio button
   - Click "Submit" to move to next task

5. EXPORT ANNOTATIONS:
   - In the project, click "Export"
   - Choose format: "JSON" or "CSV"
   - The exported file contains all annotations with
     drug_name and skip_choice for each task

SOURCE SUMMARY:""".format(
        config=config_output,
        import_file=IMPORT_OUTPUT,
    ))

    for src, cnt in sorted(source_counts.items()):
        print("   {}: {} tasks".format(src, cnt))
    print("   TOTAL: {} tasks".format(total))

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())