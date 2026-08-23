#!/usr/bin/env python3
import os
os.environ['NVIDIA_VISIBLE_DEVICES'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

ocr = PaddleOCR(lang='en', use_textline_orientation=True)

img_path = '/data/kalashsh/images/images/train/(a) Benzyl Penicillin-3.png'
result = list(ocr.predict(img_path))

res = result[0]
print("All keys:", list(res.keys()))
for k in res.keys():
    v = res[k]
    if isinstance(v, (str, int, float, bool, type(None))):
        print(f'  {k}: {v}')
    elif isinstance(v, np.ndarray):
        print(f'  {k}: ndarray shape={v.shape} dtype={v.dtype}')
    elif isinstance(v, list):
        print(f'  {k}: list len={len(v)}')
        if len(v) > 0 and len(v) <= 5:
            for i, item in enumerate(v):
                if isinstance(item, np.ndarray):
                    print(f'    [{i}] ndarray shape={item.shape} dtype={item.dtype} data={item.flatten()[:20]}')
                elif isinstance(item, str):
                    print(f'    [{i}] str="{item}"')
                elif isinstance(item, (int, float)):
                    print(f'    [{i}] {item}')
                else:
                    print(f'    [{i}] type={type(item)}')
    elif isinstance(v, dict):
        print(f'  {k}: dict keys={list(v.keys())}')

# Check the doc_preprocessor_res
dpr = res.get('doc_preprocessor_res')
if dpr is not None:
    print("\ndoc_preprocessor_res keys:", list(dpr.keys()) if hasattr(dpr, 'keys') else dir(dpr))

# Check model_settings
ms = res.get('model_settings', {})
print("\nmodel_settings:", ms)

# Check text_det_params
tdp = res.get('text_det_params', {})
print("\ntext_det_params:", tdp)