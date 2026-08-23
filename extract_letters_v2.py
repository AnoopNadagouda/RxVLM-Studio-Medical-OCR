#!/usr/bin/env python3
import os
import sys
import csv
import json
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

os.environ.setdefault('NVIDIA_VISIBLE_DEVICES', '0')
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

from paddleocr import PaddleOCR

LETTER_SIZE = 64
MIN_LETTER_PX = 5
OUTPUT_DIR = Path('/data/letter_dataset')
SKIPPED_CSV = OUTPUT_DIR / 'skipped.csv'
LABELS_CSV = OUTPUT_DIR / 'labels.csv'
WORD_BOXES_JSON = OUTPUT_DIR / 'word_boxes.json'
LETTER_SEGMENTS_JSON = OUTPUT_DIR / 'letter_segments.json'

OCR = None

def get_ocr():
    global OCR
    if OCR is None:
        OCR = PaddleOCR(lang='en', use_textline_orientation=True, enable_mkldnn=True)
    return OCR


def preprocess_for_projection(img_pil):
    gray = img_pil.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    arr = np.array(gray)
    _, binary = cv2_threshold(arr)
    return binary


def cv2_threshold(arr):
    mean_val = arr.mean()
    std_val = arr.std()
    if std_val < 20:
        thresh = mean_val
    else:
        thresh = mean_val - 0.5 * std_val
    binary = (arr < thresh).astype(np.uint8) * 255
    return arr, binary


def find_char_boundaries_profile(binary_img, num_chars):
    h_proj = binary_img.sum(axis=0)
    if h_proj.max() == 0:
        return None

    total_width = len(h_proj)
    if num_chars <= 0 or total_width == 0:
        return None

    min_gap = max(2, total_width // (num_chars * 4))
    threshold = h_proj.mean() * 0.15

    in_char = False
    segments = []
    start = 0
    for i, v in enumerate(h_proj):
        if not in_char and v > threshold:
            in_char = True
            start = i
        elif in_char and v <= threshold:
            in_char = False
            if i - start >= 2:
                segments.append((start, i))

    if in_char:
        segments.append((start, total_width))

    return segments


def merge_small_segments(segments, target_n):
    if len(segments) <= target_n or target_n <= 0:
        return segments

    while len(segments) > target_n:
        min_gap = float('inf')
        min_idx = -1
        for i in range(len(segments) - 1):
            gap = segments[i + 1][0] - segments[i][1]
            if gap < min_gap:
                min_gap = gap
                min_idx = i
        if min_idx < 0:
            break
        merged = (segments[min_idx][0], segments[min_idx + 1][1])
        segments = segments[:min_idx] + [merged] + segments[min_idx + 2:]

    return segments


def equal_width_splits(width, num_chars):
    if num_chars <= 0:
        return []
    cw = width / num_chars
    return [(int(i * cw), int((i + 1) * cw)) for i in range(num_chars)]


def segment_word_into_letters(word_img_pil, label):
    clean_label = label.strip()
    num_chars = len(clean_label)
    if num_chars == 0:
        return [], "empty_label"

    w, h = word_img_pil.size
    if w < MIN_LETTER_PX or h < MIN_LETTER_PX:
        return [], "too_small"

    binary = preprocess_for_projection(word_img_pil)
    segments = find_char_boundaries_profile(binary, num_chars)

    if segments is not None and len(segments) == num_chars:
        reason = "profile_exact"
    elif segments is not None and len(segments) > num_chars:
        segments = merge_small_segments(segments, num_chars)
        if segments is not None and len(segments) == num_chars:
            reason = "profile_merged"
        else:
            segments = equal_width_splits(w, num_chars)
            reason = "equal_width_fallback"
    else:
        segments = equal_width_splits(w, num_chars)
        reason = "equal_width_fallback"

    letters = []
    for i, (x1, x2) in enumerate(segments):
        x1c = max(0, x1)
        x2c = min(w, x2)
        if x2c - x1c < MIN_LETTER_PX:
            continue
        char_img = word_img_pil.crop((x1c, 0, x2c, h))
        char = clean_label[i] if i < len(clean_label) else '?'
        letters.append((char, char_img, x1c, x2c))

    return letters, reason


def pad_to_square(img, size=64, bg_color=255):
    img = img.convert('L')
    w, h = img.size
    max_dim = max(w, h)
    padded = Image.new('L', (max_dim, max_dim), bg_color)
    offset_x = (max_dim - w) // 2
    offset_y = (max_dim - h) // 2
    padded.paste(img, (offset_x, offset_y))
    return padded.resize((size, size), Image.LANCZOS)


def run_ocr_batch(ocr, img_paths, already_done=None):
    if already_done is None:
        already_done = set()
    results_map = {}
    batch_size = 32
    todo = [(i, p) for i, p in enumerate(img_paths) if p not in already_done]
    total_todo = len(todo)
    print(f"  OCR: {len(already_done)} cached, {total_todo} to process")
    t0 = time.time()
    for batch_idx, start in enumerate(range(0, total_todo, batch_size)):
        batch_items = todo[start:start + batch_size]
        batch = [p for _, p in batch_items]
        try:
            batch_results = list(ocr.predict([str(p) for p in batch]))
            for (_, orig_path), res in zip(batch_items, batch_results):
                results_map[orig_path] = res
        except Exception as e:
            print(f"  OCR batch failed at offset {start}: {e}, falling back to single")
            for _, orig_path in batch_items:
                try:
                    single_res = list(ocr.predict(str(orig_path)))
                    results_map[orig_path] = single_res[0] if single_res else {}
                except Exception as e2:
                    print(f"  OCR single failed for {orig_path}: {e2}")
                    results_map[orig_path] = {}
        done_count = len(already_done) + min(start + batch_size, total_todo)
        elapsed = time.time() - t0
        if batch_idx % 10 == 0 or batch_idx == 0:
            rate = done_count / elapsed if elapsed > 0 else 0
            print(f"  OCR progress: {done_count}/{len(img_paths)} ({rate:.1f}/s)")
    return results_map


def extract_word_boxes(img_path, ocr_result):
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size

    word_boxes = []
    texts = ocr_result.get('rec_texts', [])
    scores = ocr_result.get('rec_scores', [])
    polys = ocr_result.get('rec_polys', [])
    for txt, sc, poly in zip(texts, scores, polys):
        poly_arr = np.array(poly)
        xs, ys = poly_arr[:, 0], poly_arr[:, 1]
        x_min, y_min = int(xs.min()), int(ys.min())
        x_max, y_max = int(xs.max()), int(ys.max())
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)
        word_boxes.append({'x1': x_min, 'y1': y_min, 'x2': x_max, 'y2': y_max, 'ocr_text': txt, 'ocr_conf': float(sc)})

    if len(word_boxes) == 0:
        word_boxes = [{'x1': 0, 'y1': 0, 'x2': img_w, 'y2': img_h, 'ocr_text': '', 'ocr_conf': 0.0}]

    return word_boxes


def process_image_segmentation(img_path, label, source_stem, dataset_name, word_boxes):
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size
    clean_label = label.strip()
    num_label_chars = len(clean_label)

    if len(word_boxes) == 1:
        wb = word_boxes[0]
        bx1, by1, bx2, by2 = wb['x1'], wb['y1'], wb['x2'], wb['y2']
        bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
        by1, by2 = min(by1, by2), max(by1, by2)
        bx1 = 0 if bx1 < 0 else bx1
        by1 = 0 if by1 < 0 else by1
        bx2, by2 = min(bx2, img_w), min(by2, img_h)
        if bx2 - bx1 < MIN_LETTER_PX or by2 - by1 < MIN_LETTER_PX:
            return None, f"word_box_too_small: ({bx1},{by1})-({bx2},{by2}) in {img_w}x{img_h}", "single_box"
        word_img = img.crop((bx1, by1, bx2, by2))
        seg_label = wb['ocr_text'] if (not clean_label and wb.get('ocr_text')) else clean_label
        letters, method = segment_word_into_letters(word_img, seg_label)
        num_label_chars = len(seg_label.strip())
        abs_offset_x = bx1
        abs_offset_y = by1
    else:
        all_letters = []
        method = "multi_box"
        for wb in word_boxes:
            bx1, by1, bx2, by2 = wb['x1'], wb['y1'], wb['x2'], wb['y2']
            bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
            by1, by2 = min(by1, by2), max(by1, by2)
            bx1 = 0 if bx1 < 0 else bx1
            by1 = 0 if by1 < 0 else by1
            bx2, by2 = min(bx2, img_w), min(by2, img_h)
            if bx2 - bx1 < MIN_LETTER_PX or by2 - by1 < MIN_LETTER_PX:
                continue
            box_img = img.crop((bx1, by1, bx2, by2))
            box_label = wb['ocr_text'] if wb['ocr_text'] else clean_label
            box_letters, _ = segment_word_into_letters(box_img, box_label)
            for char, char_img, lx1, lx2 in box_letters:
                all_letters.append((char, char_img, bx1 + lx1, bx1 + lx2, by1, by2))
        letters = all_letters
        abs_offset_x = 0
        abs_offset_y = 0

    if len(letters) != num_label_chars:
        diff = len(letters) - num_label_chars
        if abs(diff) == 1 and len(word_boxes) == 1:
            if diff == 1:
                thinnest_idx = min(range(len(letters)), key=lambda i: letters[i][3] - letters[i][2])
                neighbor_idx = thinnest_idx + 1 if thinnest_idx < len(letters) - 1 else thinnest_idx - 1
                keep_idx = min(thinnest_idx, neighbor_idx)
                remove_idx = max(thinnest_idx, neighbor_idx)
                kept = letters[keep_idx]
                removed = letters[remove_idx]
                new_x1 = min(kept[2], removed[2])
                new_x2 = max(kept[3], removed[3])
                new_char_img = word_img.crop((new_x1, 0, new_x2, word_img.size[1]))
                merged = (kept[0], new_char_img, new_x1, new_x2)
                letters = [l for i, l in enumerate(letters) if i != keep_idx and i != remove_idx]
                letters.insert(keep_idx, merged)
                letters = [(seg_label[i] if i < len(seg_label) else '?', l[1], l[2], l[3]) for i, l in enumerate(letters)]
                method = method + "_tol_merge"
            elif diff == -1:
                widest_idx = max(range(len(letters)), key=lambda i: letters[i][3] - letters[i][2])
                w = letters[widest_idx]
                mid = (w[2] + w[3]) // 2
                left_img = word_img.crop((w[2], 0, mid, word_img.size[1]))
                right_img = word_img.crop((mid, 0, w[3], word_img.size[1]))
                left = (seg_label[widest_idx] if widest_idx < len(seg_label) else '?', left_img, w[2], mid)
                right = (seg_label[widest_idx + 1] if widest_idx + 1 < len(seg_label) else '?', right_img, mid, w[3])
                letters = letters[:widest_idx] + [left, right] + letters[widest_idx + 1:]
                method = method + "_tol_split"
        if len(letters) != num_label_chars:
            return None, f"segment_mismatch: got {len(letters)} segments, expected {num_label_chars} chars (method={method})", method

    saved = []
    seg_boundaries = []
    for i, entry in enumerate(letters):
        if len(entry) == 6:
            char, char_img, abs_x1, abs_x2, abs_y1, abs_y2 = entry
        else:
            char, char_img, lx1, lx2 = entry
            abs_x1 = abs_offset_x + lx1
            abs_x2 = abs_offset_x + lx2
            abs_y1 = abs_offset_y
            abs_y2 = abs_offset_y + (by2 - by1)

        if char_img.size[0] < MIN_LETTER_PX or char_img.size[1] < MIN_LETTER_PX:
            return None, f"letter_too_small: char '{char}' at idx {i} is {char_img.size}", method

        letter_dir = OUTPUT_DIR / 'letters' / char
        letter_dir.mkdir(parents=True, exist_ok=True)

        safe_stem = source_stem.replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"{safe_stem}_{i}.png"
        out_path = letter_dir / filename

        square_img = pad_to_square(char_img, LETTER_SIZE)
        square_img.save(out_path)

        saved.append({
            'letter': char,
            'source_file': source_stem,
            'source_dataset': dataset_name,
            'letter_idx': i,
            'method': method,
            'out_path': str(out_path),
        })

        seg_boundaries.append({
            'char': char,
            'x1': abs_x1,
            'y1': abs_y1,
            'x2': abs_x2,
            'y2': abs_y2,
            'out_path': str(out_path),
        })

    return saved, None, method, seg_boundaries


def load_avi_kai():
    img_dir = Path('/data/avi-kai/Prescription_Content/images')
    if not img_dir.exists():
        print(f"  avi-kai: dir not found {img_dir}")
        return []
    items = []
    for f in sorted(img_dir.glob('*.png')):
        label = f.stem
        items.append((str(f), label, f.stem, 'avi-kai'))
    print(f"  avi-kai: {len(items)} images")
    return items


def load_kaggle_bd_rx():
    base = Path('/data/kaggle_bd_rx')
    if not base.exists():
        print(f"  kaggle_bd_rx: dir not found")
        return []
    items = []
    for csv_name in ['Train_Label.csv']:
        csv_path = base / csv_name
        if not csv_path.exists():
            continue
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row['Images'].strip()
                label = row['Text'].strip()
                for subdir in ['Train_Set', 'Test_Set']:
                    p = base / subdir / fname
                    if p.exists():
                        items.append((str(p), label, p.stem, 'kaggle_bd_rx'))
                        break
    print(f"  kaggle_bd_rx: {len(items)} images")
    return items


def load_rxhandbd():
    base = Path('/data/rxhandbd')
    if not base.exists():
        print(f"  rxhandbd: dir not found")
        return []
    actual_base = None
    for child in base.iterdir():
        if child.is_dir():
            for inner in child.iterdir():
                if inner.is_dir() and (inner / 'Training').exists():
                    actual_base = inner
                    break
            if actual_base:
                break
    if not actual_base:
        for split in ['Training', 'Testing', 'Validation']:
            if (base / split).exists():
                actual_base = base
                break
    if not actual_base:
        print(f"  rxhandbd: no split dirs found")
        return []
    items = []
    for split in ['Training', 'Testing', 'Validation']:
        csv_name = f'{split.lower()}_labels.csv'
        csv_path = actual_base / split / csv_name
        img_dir = actual_base / split / f'{split.lower()}_words'
        if not csv_path.exists():
            continue
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row['IMAGE'].strip()
                label = row.get('MEDICINE_NAME', '').strip()
                if not label:
                    label = row.get('GENERIC_NAME', '').strip()
                p = img_dir / fname
                if p.exists():
                    items.append((str(p), label, p.stem, 'rxhandbd'))
    print(f"  rxhandbd: {len(items)} images")
    return items


def load_kalashsh():
    base = Path('/data/kalashsh')
    if not base.exists():
        print(f"  kalashsh: dir not found")
        return []
    items = []
    for split in ['train', 'test', 'validate']:
        csv_path = base / f'{split}2.csv'
        img_dir = base / 'images' / 'images' / split
        if not csv_path.exists():
            continue
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row['FILENAME'].strip()
                label = row['IDENTITY'].strip()
                p = img_dir / fname
                if p.exists():
                    items.append((str(p), label, p.stem, 'kalashsh'))
    print(f"  kalashsh: {len(items)} images")
    return items


def load_mehaksingal():
    img_dir = Path('/data/mehaksingal/data/')
    if not img_dir.exists():
        print(f"  mehaksingal: dir not found {img_dir}")
        return []
    items = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for f in sorted(img_dir.glob(ext)):
            items.append((str(f), '', f.stem, 'mehaksingal'))
    print(f"  mehaksingal: {len(items)} images")
    return items


def load_chhavimohitkar():
    base = Path('/data/chhavimohitkar/')
    if not base.exists():
        print(f"  chhavimohitkar: dir not found")
        return []
    items = []
    for f in sorted(base.iterdir()):
        if not f.is_file():
            continue
        name = f.name
        if name.startswith('Sample') or name.endswith('.zip'):
            continue
        parts = name.split('.')
        if len(parts) >= 3 and parts[0].startswith('data'):
            try:
                int(parts[0][4:])
                int(parts[1])
                if parts[-1].lower() == 'png':
                    items.append((str(f), '', f.stem, 'chhavimohitkar'))
            except ValueError:
                continue
    print(f"  chhavimohitkar: {len(items)} images")
    return items


def load_chinmays():
    base = Path('/data/chinmays/chinmays_rx_hf/')
    if not base.exists():
        print(f"  chinmays: dir not found {base}")
        return []
    items = []
    for split in ['train', 'test']:
        img_dir = base / split / 'images'
        ann_dir = base / split / 'annotations'
        if not img_dir.exists():
            continue
        for f in sorted(img_dir.glob('prescription_*.png')):
            ann_path = ann_dir / (f.stem + '.json')
            label = ''
            if ann_path.exists():
                try:
                    with open(ann_path, 'r', encoding='utf-8') as jf:
                        ann = json.load(jf)
                    gt = ann.get('ground_truth', '')
                    for token in ['<s_ocr>', '</s>']:
                        gt = gt.replace(token, '')
                    label = gt.strip()
                except Exception:
                    pass
            stem = f"{split}_{f.stem}"
            items.append((str(f), '', stem, 'chinmays'))
    print(f"  chinmays: {len(items)} images")
    return items


def load_chaithanyakota():
    base = Path('/data/chaithanyakota/')
    if not base.exists():
        print(f"  chaithanyakota: dir not found")
        return []
    csv_path = base / 'chaithaanyakota.csv'
    if not csv_path.exists():
        print(f"  chaithanyakota: csv not found {csv_path}")
        return []
    img_dir = base / 'chaithanyakota' / 'images'
    if not img_dir.exists():
        img_dir = base / 'images'
    if not img_dir.exists():
        print(f"  chaithanyakota: img dir not found")
        return []
    items = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get('filename', '').strip()
            if not fname:
                continue
            p = img_dir / fname
            if p.exists():
                stem = f"chaithanyakota_{p.stem}"
                items.append((str(p), '', stem, 'chaithanyakota'))
    print(f"  chaithanyakota: {len(items)} images")
    return items


def load_tavishm():
    arrow_dir = Path('/data/tavishm/tavishm/train/')
    out_dir = Path('/data/tavishm/extracted_images/')
    if not arrow_dir.exists():
        print(f"  tavishm: dir not found {arrow_dir}")
        return []
    arrow_files = list(arrow_dir.glob('*.arrow'))
    if not arrow_files:
        print(f"  tavishm: no .arrow files found")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    try:
        import pyarrow.ipc
        for arrow_file in arrow_files:
            reader = pyarrow.ipc.open_file(arrow_file)
            table = reader.read_all()
            if 'image' not in table.column_names:
                continue
            col = table.column('image')
            for i in range(len(col)):
                entry = col[i].as_py()
                img_bytes = None
                if isinstance(entry, dict):
                    img_bytes = entry.get('bytes', None)
                elif isinstance(entry, (bytes, bytearray)):
                    img_bytes = bytes(entry)
                if img_bytes is None:
                    continue
                stem = f"tavishm_{i:04d}"
                out_path = out_dir / f"{stem}.jpg"
                if out_path.exists():
                    items.append((str(out_path), '', stem, 'tavishm'))
                    continue
                try:
                    from PIL import Image as _Img
                    import io
                    img = _Img.open(io.BytesIO(img_bytes))
                    img.save(out_path, 'JPEG')
                    items.append((str(out_path), '', stem, 'tavishm'))
                except Exception as e:
                    print(f"  tavishm: failed to save image {i}: {e}")
        print(f"  tavishm: {len(items)} images")
        return items
    except ImportError:
        pass
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(arrow_dir.parent))
        for i in range(len(ds)):
            entry = ds[i].get('image', None)
            if entry is None:
                continue
            stem = f"tavishm_{i:04d}"
            out_path = out_dir / f"{stem}.jpg"
            if out_path.exists():
                items.append((str(out_path), '', stem, 'tavishm'))
                continue
            try:
                entry.save(out_path, 'JPEG')
                items.append((str(out_path), '', stem, 'tavishm'))
            except Exception as e:
                print(f"  tavishm: failed to save image {i}: {e}")
        print(f"  tavishm: {len(items)} images")
        return items
    except ImportError:
        pass
    print(f"  tavishm: neither pyarrow nor datasets available, skipping")
    return []


DATASET_LOADERS = {
    'avi-kai': load_avi_kai,
    'kaggle_bd_rx': load_kaggle_bd_rx,
    'rxhandbd': load_rxhandbd,
    'kalashsh': load_kalashsh,
    'mehaksingal': load_mehaksingal,
    'chhavimohitkar': load_chhavimohitkar,
    'chinmays': load_chinmays,
    'chaithanyakota': load_chaithanyakota,
    'tavishm': load_tavishm,
}


CHECKPOINT_INTERVAL = 500


def load_json_checkpoint(path):
    if path.exists():
        print(f"  Checkpoint found: {path} ({path.stat().st_size / 1e6:.1f} MB)")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_labels_csv(rows):
    with open(LABELS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['letter', 'source_file', 'source_dataset', 'letter_idx', 'method', 'out_path'])
        writer.writeheader()
        writer.writerows(rows)


def save_skipped_csv(rows):
    with open(SKIPPED_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source_file', 'source_dataset', 'label', 'reason'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='append', choices=list(DATASET_LOADERS.keys()))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='Limit per dataset (0=no limit)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint files')
    args = parser.parse_args()

    if args.all:
        datasets = list(DATASET_LOADERS.keys())
    elif args.dataset:
        datasets = args.dataset
    else:
        print("Specify --dataset <name> or --all")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_items = []
    for ds in datasets:
        print(f"\nLoading {ds}...")
        items = DATASET_LOADERS[ds]()
        if args.limit > 0:
            items = items[:args.limit]
        all_items.extend(items)

    total = len(all_items)
    print(f"\nTotal images to process: {total}")

    print("Initializing PaddleOCR...")
    ocr = get_ocr()
    print("OCR ready.")

    # --- Phase 1: OCR with resume ---
    print(f"\nPhase 1: Running OCR on {total} images (batched)...")
    img_paths = [item[0] for item in all_items]
    ocr_results = {}

    existing_wb = load_json_checkpoint(WORD_BOXES_JSON)
    if existing_wb and args.resume:
        cached_paths = set()
        for stem, info in existing_wb.items():
            p = info.get('img_path', '')
            cached_paths.add(p)
        ocr_results = {p: {} for p in cached_paths}
        print(f"  Resuming OCR: {len(cached_paths)} cached from word_boxes.json")
    else:
        existing_wb = None

    new_ocr = run_ocr_batch(ocr, img_paths, already_done=set(ocr_results.keys()))
    ocr_results.update(new_ocr)
    print(f"OCR complete. Got results for {len(ocr_results)} images.")

    # --- Phase 2: Word boxes ---
    print("\nPhase 2: Extracting word boxes...")
    word_boxes_map = existing_wb if existing_wb else {}
    stems_to_process = [stem for _, _, stem, _ in all_items if stem not in word_boxes_map]
    print(f"  {len(word_boxes_map)} cached, {len(stems_to_process)} to process")

    path_to_item = {item[0]: item for item in all_items}
    for stem in stems_to_process:
        info = None
        for img_path, label, s, dataset in all_items:
            if s == stem:
                info = (img_path, label, stem, dataset)
                break
        if not info:
            continue
        img_path, label, _, dataset = info
        ocr_result = ocr_results.get(img_path, {})
        word_boxes = extract_word_boxes(img_path, ocr_result)
        word_boxes_map[stem] = {
            'img_path': img_path,
            'label': label,
            'dataset': dataset,
            'word_boxes': word_boxes,
        }

    with open(WORD_BOXES_JSON, 'w', encoding='utf-8') as f:
        json.dump(word_boxes_map, f, indent=2)
    print(f"Word boxes saved to {WORD_BOXES_JSON} ({len(word_boxes_map)} entries)")

    # --- Phase 3: Segmentation with resume + periodic saves ---
    print("\nPhase 3: Segmenting letters...")
    labels_rows = []
    skipped_rows = []
    letter_segments_map = {}
    success_count = 0
    skip_count = 0

    existing_seg = load_json_checkpoint(LETTER_SEGMENTS_JSON)
    if existing_seg and args.resume:
        for stem, info in existing_seg.items():
            if info.get('skipped', False):
                skipped_rows.append({
                    'source_file': stem,
                    'source_dataset': info.get('dataset', ''),
                    'label': info.get('label', ''),
                    'reason': info.get('skip_reason', ''),
                })
                skip_count += 1
            else:
                segs = info.get('segments', [])
                for seg in segs:
                    labels_rows.append({
                        'letter': seg['char'],
                        'source_file': stem,
                        'source_dataset': info.get('dataset', ''),
                        'letter_idx': seg.get('letter_idx', 0),
                        'method': info.get('method', ''),
                        'out_path': seg.get('out_path', ''),
                    })
                success_count += 1
        letter_segments_map = dict(existing_seg)
        print(f"  Resumed: {success_count} success, {skip_count} skipped from checkpoint")
    else:
        existing_seg = None

    already_segmented = set(letter_segments_map.keys())
    t0 = time.time()
    processed_this_run = 0

    for idx, (img_path, label, stem, dataset) in enumerate(all_items):
        if stem in already_segmented:
            continue

        elapsed = time.time() - t0
        done = success_count + skip_count
        remaining = total - done
        if processed_this_run > 0:
            rate = processed_this_run / elapsed if elapsed > 0 else 0
            eta = remaining / rate if rate > 0 else 0
        else:
            rate = 0
            eta = 0

        if processed_this_run % 100 == 0 or processed_this_run < 5:
            print(f"  [{done}/{total}] {stem} (label='{label}') rate={rate:.1f}/s eta={eta:.0f}s")

        word_boxes = word_boxes_map.get(stem, {}).get('word_boxes', [])
        if not word_boxes:
            if stem in word_boxes_map:
                word_boxes = word_boxes_map[stem].get('word_boxes', [])
            else:
                word_boxes = [{'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1, 'ocr_text': '', 'ocr_conf': 0.0}]

        result = process_image_segmentation(img_path, label, stem, dataset, word_boxes)

        if result[0] is not None:
            saved = result[0]
            method = result[2]
            seg_boundaries = result[3]
            labels_rows.extend(saved)
            success_count += 1
            letter_segments_map[stem] = {
                'label': label,
                'dataset': dataset,
                'method': method,
                'segments': seg_boundaries,
                'skipped': False,
                'skip_reason': None,
            }
        else:
            error = result[1]
            method = result[2] if len(result) > 2 else 'unknown'
            skipped_rows.append({
                'source_file': stem,
                'source_dataset': dataset,
                'label': label,
                'reason': error,
            })
            skip_count += 1
            letter_segments_map[stem] = {
                'label': label,
                'dataset': dataset,
                'method': method,
                'segments': [],
                'skipped': True,
                'skip_reason': error,
            }

        processed_this_run += 1

        if processed_this_run > 0 and processed_this_run % CHECKPOINT_INTERVAL == 0:
            save_labels_csv(labels_rows)
            save_skipped_csv(skipped_rows)
            with open(LETTER_SEGMENTS_JSON, 'w', encoding='utf-8') as f:
                json.dump(letter_segments_map, f, indent=2)
            print(f"  *** Checkpoint saved at {done+1}/{total} ({success_count} ok, {skip_count} skip) ***")

    save_labels_csv(labels_rows)
    save_skipped_csv(skipped_rows)
    with open(LETTER_SEGMENTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(letter_segments_map, f, indent=2)

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.1f}s! Success: {success_count}, Skipped: {skip_count}")
    print(f"Labels: {LABELS_CSV} ({len(labels_rows)} rows)")
    print(f"Skipped: {SKIPPED_CSV} ({len(skipped_rows)} rows)")
    print(f"Word boxes: {WORD_BOXES_JSON}")
    print(f"Letter segments: {LETTER_SEGMENTS_JSON}")


if __name__ == '__main__':
    main()