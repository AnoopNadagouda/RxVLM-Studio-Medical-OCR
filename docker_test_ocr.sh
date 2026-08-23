#!/usr/bin/env python3
import os
import numpy as np

os.environ['NVIDIA_VISIBLE_DEVICES'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='en', use_textline_orientation=True, return_word_box=True)

img_dir = '/data/avi-kai/Prescription_Content/images'
imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])[:10]

for img_name in imgs:
    img_path = os.path.join(img_dir, img_name)
    result = list(ocr.predict(img_path))
    print(f'\n=== {img_name} ===')
    for res in result:
        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])
        polys = res.get('rec_polys', [])
        dt_polys = res.get('dt_polys', [])
        word_box = res.get('return_word_box', False)
        rec_boxes = res.get('rec_boxes', None)
        print(f'  return_word_box={word_box}, n_texts={len(texts)}, n_polys={len(polys)}, n_dt_polys={len(dt_polys)}')
        if rec_boxes is not None:
            print(f'  rec_boxes shape={rec_boxes.shape}, dtype={rec_boxes.dtype}')
        for i, (txt, sc, poly) in enumerate(zip(texts, scores, polys)):
            poly_arr = np.array(poly)
            xs = poly_arr[:, 0]
            ys = poly_arr[:, 1]
            x_min, y_min = xs.min(), ys.min()
            x_max, y_max = xs.max(), ys.max()
            print(f'  [{i}] text="{txt}" conf={sc:.3f} box=({x_min},{y_min})-({x_max},{y_max}) pts={len(poly)}')