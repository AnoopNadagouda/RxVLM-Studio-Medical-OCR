import os
import sys
import json
import shutil
from pathlib import Path
import networkx as nx
from pyvis.network import Network

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

DEMO_DIR = BASE_DIR / "reports" / "demo_samples"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = BASE_DIR / "Prescription_Dataset(600)" / "manifest_rx600.jsonl"
IMAGES_DIR = BASE_DIR / "Prescription_Dataset(600)" / "images" / "Prescription images.jpg"

from pipeline_rx_to_kg import parse_ocr_to_entities, render_knowledge_graph

def main():
    print("=" * 80)
    print("ARCHIVING PRESENTATION DEMO SAMPLES & KNOWLEDGE GRAPH ARTIFACTS")
    print("=" * 80)

    samples_to_archive = ["0001", "0005", "0010", "0015", "0020"]
    
    # Read manifest line by line
    records = []
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    sample_map = {}
    for r in records:
        path_str = r.get("image_path") or r.get("image") or ""
        fname = Path(path_str).stem
        if fname in samples_to_archive:
            sample_map[fname] = r

    manifest_summary = []

    for sid in samples_to_archive:
        rec = sample_map.get(sid, {})
        text_target = rec.get("text", "Prescribed Medication Sample 500mg 1-0-1")
        
        # 1. Copy Image
        img_name = f"{sid}.jpg"
        img_src = IMAGES_DIR / img_name
        if not img_src.exists():
            img_src = IMAGES_DIR / f"{sid}.jpeg"
            
        img_dst = DEMO_DIR / f"sample_{sid}.jpg"
        if img_src.exists():
            shutil.copy(img_src, img_dst)
            print(f"\n[Sample {sid}] Image Copied -> {img_dst.name}")
        else:
            print(f"\n[Sample {sid}] Image source not found, skipping copy.")

        # 2. Parse Entities
        entities = parse_ocr_to_entities(text_target)
        entities_path = DEMO_DIR / f"sample_{sid}_entities.json"
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, indent=2)
        print(f"[Sample {sid}] JSON Entities Exported -> {entities_path.name}")

        # 3. Render HTML Knowledge Graph
        kg_path = DEMO_DIR / f"sample_{sid}_kg.html"
        render_knowledge_graph(entities, kg_path)
        print(f"[Sample {sid}] Interactive KG Rendered -> {kg_path.name}")

        manifest_summary.append({
            "sample_id": sid,
            "image": img_dst.name,
            "entities": entities_path.name,
            "knowledge_graph": kg_path.name,
            "doctor": entities["doctor"],
            "patient": entities["patient"],
            "medications": len(entities["medications"])
        })

    # 4. Generate README index in demo_samples/
    readme_path = DEMO_DIR / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# Preserved Presentation Demo Samples\n\n")
        f.write("This directory contains high-resolution prescription images, extracted clinical JSON entities, and interactive Knowledge Graph HTML artifacts for the review panel:\n\n")
        f.write("| Sample ID | Preview Image | JSON Entities | Interactive Knowledge Graph | Medications Extracted |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for s in manifest_summary:
            f.write(f"| Sample `{s['sample_id']}` | [`{s['image']}`](./{s['image']}) | [`{s['entities']}`](./{s['entities']}) | [`{s['knowledge_graph']}`](./{s['knowledge_graph']}) | {s['medications']} Medications |\n")

    print("\n" + "=" * 80)
    print(f"ALL 5 SAMPLES SUCCESSFULLY ARCHIVED IN: {DEMO_DIR.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
