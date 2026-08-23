import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.pipeline_rx_to_kg import parse_ocr_to_entities, render_knowledge_graph

def main():
    print("=" * 80)
    print("PROGRAMMATIC END-TO-END SANITY CHECK & DRY-RUN TEST")
    print("=" * 80)

    # 1. Test Raw Image Loading
    img_path = BASE_DIR / "Prescription_Dataset(600)" / "images" / "Prescription images.jpg" / "0001.jpg"
    print(f"\n[Step 1] Testing Image Loading from: {img_path}")
    assert img_path.exists(), f"Image {img_path} does not exist!"
    img = Image.open(img_path)
    print(f"  [SUCCESS] Image loaded successfully! Dimensions: {img.size}, Format: {img.format}")

    # 2. Test JSON Entity Parsing
    test_ocr_text = "P+H tablets six tablets\nXgeva injection 120 mg\nDexa U tablet twice daily 1-0-1\nOndam U tablet 500mg 1-1-1"
    print(f"\n[Step 2] Testing Clinical Entity Parsing on OCR Output...")
    entities = parse_ocr_to_entities(test_ocr_text)
    print(json.dumps(entities, indent=2))
    assert "doctor" in entities and "patient" in entities and "medications" in entities, "Entity parsing failed!"
    print("  [SUCCESS] Entities parsed successfully!")

    # 3. Test Knowledge Graph HTML Building
    output_html = BASE_DIR / "reports" / "live_prescription_kg.html"
    print(f"\n[Step 3] Testing Interactive Knowledge Graph HTML Generation -> {output_html}")
    render_knowledge_graph(entities, output_html)
    assert output_html.exists(), "HTML report generation failed!"
    print(f"  [SUCCESS] Knowledge Graph rendered ({output_html.stat().st_size} bytes)")

    print("\n" + "=" * 80)
    print("ALL SANITY CHECKS PASSED PERFECTLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
