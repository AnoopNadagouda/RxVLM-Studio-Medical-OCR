#!/usr/bin/env python3
import os, glob, csv
os.environ['NVIDIA_VISIBLE_DEVICES'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

ocr = PaddleOCR(lang='en', use_textline_orientation=True)

def test_image(img_path, label=""):
    img = Image.open(img_path)
    w, h = img.size
    print(f'  Size: {w}x{h}')
    result = list(ocr.predict(img_path))
    for res in result:
        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])
        polys = res.get('rec_polys', [])
        print(f'  Detections: {len(texts)}')
        for i, (txt, sc, poly) in enumerate(zip(texts, scores, polys)):
            poly_arr = np.array(poly)
            xs, ys = poly_arr[:, 0], poly_arr[:, 1]
            x_min, y_min = int(xs.min()), int(ys.min())
            x_max, y_max = int(xs.max()), int(ys.max())
            print(f'  [{i}] ocr="{txt}" conf={sc:.3f} box=({x_min},{y_min})-({x_max},{y_max})')
    return result

print("=== kaggle_bd_rx ===")
csv_path = '/data/kaggle_bd_rx/Train_Label.csv'
if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)[:5]
    for row in rows:
        fname = row['Images']
        txt = row['Text']
        for subdir in ['Train_Set', 'Test_Set']:
            p = f'/data/kaggle_bd_rx/{subdir}/{fname}'
            if os.path.exists(p):
                print(f'\n{fname} label="{txt}"')
                test_image(p, txt)
                break

print("\n=== rxhandbd ===")
for split in ['Training', 'Testing', 'Validation']:
    csv_path = f'/data/rxhandbd/{split}/{split.lower()}_labels.csv'
    img_dir = f'/data/rxhandbd/{split}/{split.lower()}_words'
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)[:3]
        for row in rows:
            fname = row['IMAGE']
            txt = row.get('MEDICINE_NAME', row.get('GENERIC_NAME', ''))
            p = os.path.join(img_dir, fname)
            if os.path.exists(p):
                print(f'\n{fname} label="{txt}"')
                test_image(p, txt)

print("\n=== kalashsh ===")
for split in ['train', 'test', 'validate']:
    csv_path = f'/data/kalashsh/{split}2.csv'
    img_dir = f'/data/kalashsh/images/images/{split}'
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)[:3]
        for row in rows:
            fname = row['FILENAME']
            txt = row['IDENTITY']
            p = os.path.join(img_dir, fname)
            if os.path.exists(p):
                print(f'\n{fname} label="{txt}"')
                test_image(p, txt)