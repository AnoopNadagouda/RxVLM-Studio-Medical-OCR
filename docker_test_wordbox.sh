#!/usr/bin/env python3
import os
os.environ['NVIDIA_VISIBLE_DEVICES'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

ocr_wb = PaddleOCR(lang='en', use_textline_orientation=True, return_word_box=True)

img_dir = '/data/kalashsh/images/images/train'
imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

for img_name in imgs[:5]:
    img_path = os.path.join(img_dir, img_name)
    result = list(ocr_wb.predict(img_path))
    print(f'\n=== {img_name} (return_word_box=True) ===')
    for res in result:
        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])
        rec_boxes = res.get('rec_boxes', None)
        polys = res.get('rec_polys', [])
        print(f'  n_texts={len(texts)} rec_boxes={rec_boxes.shape if rec_boxes is not None else None}')
        for i, (txt, sc) in enumerate(zip(texts, scores)):
            poly_arr = np.array(polys[i])
            xs, ys = poly_arr[:, 0], poly_arr[:, 1]
            x_min, y_min = int(xs.min()), int(ys.min())
            x_max, y_max = int(xs.max()), int(ys.max())
            box_str = f'({x_min},{y_min})-({x_max},{y_max})'
            if rec_boxes is not None and len(rec_boxes.shape) == 3:
                wb = rec_boxes[i]
                print(f'  [{i}] text="{txt}" conf={sc:.3f} word_box={box_str} n_word_boxes={len(wb)} word_boxes={wb.tolist()}')
            else:
                print(f'  [{i}] text="{txt}" conf={sc:.3f} word_box={box_str}')