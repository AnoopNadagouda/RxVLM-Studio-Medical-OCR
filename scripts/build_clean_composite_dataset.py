import os
import re
import json
import random
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps

# Set fixed seed for 100% reproducible zero-leakage splits
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR_TRAIN = BASE_DIR / "data" / "clean_composite_medocr" / "train"
OUTPUT_DIR_VAL = BASE_DIR / "data" / "clean_composite_medocr" / "val"
MANIFEST_DIR = BASE_DIR / "ocr_manifests"

OUTPUT_DIR_TRAIN.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_VAL.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MANIFEST_PATH = MANIFEST_DIR / "clean_composite_train.jsonl"
VAL_MANIFEST_PATH = MANIFEST_DIR / "clean_composite_val.jsonl"

def sanitize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def ingest_kaggle_bd_rx(base_dir):
    samples = []
    kaggle_dir = base_dir / "kaggle_bd_rx"
    
    train_csv = kaggle_dir / "Train_Label.csv"
    train_img_dir = kaggle_dir / "Train_Set"
    if train_csv.exists() and train_img_dir.exists():
        df = pd.read_csv(train_csv)
        for _, row in df.iterrows():
            img_name = str(row['Images']).strip()
            text = sanitize_text(row['Text'])
            img_path = train_img_dir / img_name
            if text and img_path.exists():
                samples.append((str(img_path), text))
                
    test_csv = kaggle_dir / "Test_Labels.csv"
    test_img_dir = kaggle_dir / "Test_Set"
    if test_csv.exists() and test_img_dir.exists():
        df = pd.read_csv(test_csv)
        for _, row in df.iterrows():
            img_name = str(row['Images']).strip()
            text = sanitize_text(row['Text'])
            img_path = test_img_dir / img_name
            if text and img_path.exists():
                samples.append((str(img_path), text))
                
    print(f"[Kaggle BD Rx] Ingested {len(samples)} valid word samples.")
    return samples

def ingest_rxhandbd(base_dir):
    samples = []
    root_rx = base_dir / "rxhandbd" / "doctors-handwritten-prescription-bd-dataset" / "Doctor’s Handwritten Prescription BD dataset"
    
    split_info = [
        ("Training", "training_labels.csv", "training_words"),
        ("Validation", "validation_labels.csv", "validation_words"),
        ("Testing", "testing_labels.csv", "testing_words"),
    ]
    
    for folder, csv_name, img_subfolder in split_info:
        csv_path = root_rx / folder / csv_name
        img_dir = root_rx / folder / img_subfolder
        if csv_path.exists() and img_dir.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_name = str(row['IMAGE']).strip()
                # Prefer MEDICINE_NAME, fallback to GENERIC_NAME
                text = sanitize_text(row.get('MEDICINE_NAME', row.get('GENERIC_NAME', '')))
                img_path = img_dir / img_name
                if text and img_path.exists():
                    samples.append((str(img_path), text))
                    
    print(f"[RxHandBD] Ingested {len(samples)} valid word samples.")
    return samples

def synthesize_multi_word_line(word_samples, target_height=64, max_words=4):
    """
    Concatenates 1 to max_words cropped images horizontally with variable spacing.
    Returns (composite_PIL_image, transcription_text)
    """
    n_words = random.randint(1, max_words)
    chosen = [random.choice(word_samples) for _ in range(n_words)]
    
    images = []
    texts = []
    
    for img_path, text in chosen:
        try:
            img = Image.open(img_path).convert("RGB")
            # Scale proportionally to target_height
            w, h = img.size
            if h > 0:
                scale = target_height / float(h)
                new_w = max(4, int(w * scale))
                img_resized = img.resize((new_w, target_height), Image.Resampling.BILINEAR)
                images.append(img_resized)
                texts.append(text)
        except Exception:
            continue
            
    if not images:
        # Fallback if image opening fails
        fallback_img = Image.new("RGB", (128, target_height), color=(255, 255, 255))
        return fallback_img, "Medicine"
        
    # Calculate composite dimensions with random spacing
    spacings = [random.randint(12, 28) for _ in range(len(images) - 1)] + [0]
    total_width = sum(img.width for img in images) + sum(spacings)
    
    # White background composite image
    composite = Image.new("RGB", (total_width + 16, target_height + 12), color=(255, 255, 255))
    
    x_offset = 8
    y_offset = 6
    for img, sp in zip(images, spacings):
        composite.paste(img, (x_offset, y_offset))
        x_offset += img.width + sp
        
    full_text = " ".join(texts)
    return composite, full_text

def build_dataset():
    print("=" * 70)
    print("Building Clean Composite MedOCR Dataset (Zero-Leakage Guarantee)")
    print("=" * 70)
    
    # Step 1: Gather raw word samples from Bucket A
    all_raw_samples = []
    all_raw_samples.extend(ingest_kaggle_bd_rx(BASE_DIR))
    all_raw_samples.extend(ingest_rxhandbd(BASE_DIR))
    
    print(f"\nTotal raw word samples ingested: {len(all_raw_samples)}")
    
    # Step 2: Extract all individual atomic words across all raw labels
    all_atomic_words = set()
    sample_word_list = []
    for img_path, text in all_raw_samples:
        words = [w.strip() for w in text.strip().split() if w.strip()]
        if not words:
            continue
        all_atomic_words.update(words)
        sample_word_list.append((img_path, text, words))
        
    unique_words = sorted(list(all_atomic_words))
    print(f"Total unique atomic words across all labels: {len(unique_words)}")
    
    # Step 3: Enforce strict 85% Train / 15% Val atomic word split
    random.shuffle(unique_words)
    val_cutoff = int(len(unique_words) * 0.15)
    
    val_vocab = set(unique_words[:val_cutoff])
    train_vocab = set(unique_words[val_cutoff:])
    
    # Absolute zero leakage assertion on vocabulary sets
    vocab_overlap = train_vocab & val_vocab
    assert len(vocab_overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping words: {vocab_overlap}"
    print(f"\nSplit Vocabulary: Train Words = {len(train_vocab)} | Val Words = {len(val_vocab)} | Overlap = {len(vocab_overlap)}")
    
    # Step 4: Partition raw image samples based on atomic words
    train_word_samples = []
    val_word_samples = []
    discarded_samples = 0
    
    for img_path, text, words in sample_word_list:
        # Check if ALL words in this sample are in train_vocab
        if all(w in train_vocab for w in words):
            train_word_samples.append((img_path, text))
        # Check if ALL words in this sample are in val_vocab
        elif all(w in val_vocab for w in words):
            val_word_samples.append((img_path, text))
        else:
            discarded_samples += 1
            
    print(f"Pool Sizes: Train Word Samples = {len(train_word_samples)} | Val Word Samples = {len(val_word_samples)} | Discarded Mixed Samples = {discarded_samples}")
    
    # Step 5: Generate Synthetic Multi-Word Composite Datasets
    NUM_TRAIN_SAMPLES = 3000
    NUM_VAL_SAMPLES = 500
    
    print(f"\nSynthesizing {NUM_TRAIN_SAMPLES} train composite lines...")
    train_manifest_entries = []
    for idx in range(1, NUM_TRAIN_SAMPLES + 1):
        img_composite, text = synthesize_multi_word_line(train_word_samples, target_height=64, max_words=4)
        filename = f"train_{idx:05d}.png"
        save_path = OUTPUT_DIR_TRAIN / filename
        img_composite.save(save_path)
        
        rel_path = str(save_path.relative_to(BASE_DIR)).replace("\\", "/")
        entry = {
            "id": f"clean_train_{idx:05d}",
            "image": rel_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nTranscribe the prescription text in the image accurately."
                },
                {
                    "from": "gpt",
                    "value": text
                }
            ]
        }
        train_manifest_entries.append(entry)
        
    print(f"Synthesizing {NUM_VAL_SAMPLES} val composite lines...")
    val_manifest_entries = []
    for idx in range(1, NUM_VAL_SAMPLES + 1):
        img_composite, text = synthesize_multi_word_line(val_word_samples, target_height=64, max_words=4)
        filename = f"val_{idx:05d}.png"
        save_path = OUTPUT_DIR_VAL / filename
        img_composite.save(save_path)
        
        rel_path = str(save_path.relative_to(BASE_DIR)).replace("\\", "/")
        entry = {
            "id": f"clean_val_{idx:05d}",
            "image": rel_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nTranscribe the prescription text in the image accurately."
                },
                {
                    "from": "gpt",
                    "value": text
                }
            ]
        }
        val_manifest_entries.append(entry)
        
    # Write JSONL manifests
    with open(TRAIN_MANIFEST_PATH, "w", encoding="utf-8") as f:
        for entry in train_manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open(VAL_MANIFEST_PATH, "w", encoding="utf-8") as f:
        for entry in val_manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"\nManifests saved cleanly to:")
    print(f"  - Train Manifest: {TRAIN_MANIFEST_PATH}")
    print(f"  - Val Manifest:   {VAL_MANIFEST_PATH}")
    
    # Step 6: Post-generation Verification & Assertion
    train_manifest_words = set()
    for e in train_manifest_entries:
        text = e['conversations'][1]['value']
        for w in text.split():
            train_manifest_words.add(w)
            
    val_manifest_words = set()
    for e in val_manifest_entries:
        text = e['conversations'][1]['value']
        for w in text.split():
            val_manifest_words.add(w)
            
    final_overlap = train_manifest_words & val_manifest_words
    print(f"\nFinal Manifest Verification:")
    print(f"  - Train Manifest Vocabulary Size: {len(train_manifest_words)} unique words")
    print(f"  - Val Manifest Vocabulary Size:   {len(val_manifest_words)} unique words")
    print(f"  - Final Vocabulary Overlap:       {len(final_overlap)} words")
    
    assert len(final_overlap) == 0, f"VERIFICATION FAILED: Overlap found: {final_overlap}"
    print("\n[OK] ZERO DATA LEAKAGE ASSERTION PASSED CONFIDENTLY!")

if __name__ == "__main__":
    build_dataset()
