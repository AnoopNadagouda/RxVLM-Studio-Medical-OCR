from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
import re
import csv
import hashlib
import base64
from urllib.parse import urlparse, parse_qs, unquote

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = os.path.join(BASE_DIR, "models", "PaddleOCR-VL-1.5")

print(f"Loading PaddleOCR-VL-1.5 model on {DEVICE}...")
_processor = AutoProcessor.from_pretrained(MODEL_ID)
_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to(DEVICE).eval()
print("PaddleOCR-VL-1.5 model loaded successfully.")
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
CROPPED_DIR = os.path.join(BASE_DIR, 'cropped')
SPOTTINGS_CSV = os.path.join(BASE_DIR, 'spottings.csv')
ANNOTATIONS_DIR = os.path.join(BASE_DIR, 'annotations')

os.makedirs(CROPPED_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

SPOT_PATTERN = re.compile(
    r'(.+?)\s*<\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
)

def load_spottings_for_json():
    files_map = {}
    if not os.path.isfile(SPOTTINGS_CSV):
        return {"files": []}
    with open(SPOTTINGS_CSV, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            f = row["file"]
            if f not in files_map:
                files_map[f] = {"file": f, "url": "/" + f, "words": [], "image_labels": [], "image_ocr": ""}
            files_map[f]["words"].append({
                "text": row["text"], "type": row["type"],
                "x": int(row["x"]), "y": int(row["y"]),
                "w": int(row["w"]), "h": int(row["h"]),
                "shape": row.get("shape", "rect") or "rect",
                "angle": float(row.get("angle", 0) or 0),
                "points": row.get("points", "") or "",
            })
    for f, entry in files_map.items():
        rh = hashlib.md5(f.encode()).hexdigest()
        ann_path = os.path.join(ANNOTATIONS_DIR, f"{rh}.json")
        if os.path.isfile(ann_path):
            try:
                saved = json.loads(open(ann_path, encoding="utf-8").read())
                entry["image_labels"] = saved.get("image_labels", [])
                entry["words"] = saved.get("words", entry["words"])
                entry["image_ocr"] = saved.get("image_ocr", "")
                entry["saved"] = True
            except Exception:
                pass
    return {"files": list(files_map.values())}

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/spottings.json':
            self._json(load_spottings_for_json())
        elif parsed.path == '/api/med-names':
            texts = set()
            med = os.path.join(BASE_DIR, 'med_names.txt')
            if os.path.isfile(med):
                for l in open(med, encoding='utf-8').read().splitlines():
                    if l.strip():
                        texts.add(l.strip())
            if os.path.isfile(SPOTTINGS_CSV):
                with open(SPOTTINGS_CSV, encoding='utf-8', newline="") as fh:
                    for row in csv.DictReader(fh):
                        t = row.get("text", "").strip()
                        if t:
                            texts.add(t)
            if os.path.isdir(ANNOTATIONS_DIR):
                for af in os.listdir(ANNOTATIONS_DIR):
                    if af.endswith('.json'):
                        try:
                            data = json.loads(open(os.path.join(ANNOTATIONS_DIR, af), encoding='utf-8').read())
                            for word in data.get("words", []):
                                t = word.get("text", "").strip() if word.get("text") else ""
                                if t:
                                    texts.add(t)
                        except Exception:
                            pass
            self._json(sorted(texts))
        elif parsed.path == '/api/list':
            qs = parse_qs(parsed.query)
            folder = qs.get('dir', [''])[0]
            source = qs.get('source', ['raw'])[0]
            if source == 'cropped':
                results = []
                for f in sorted(os.listdir(CROPPED_DIR)):
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                        results.append('/cropped/' + f)
                self._json(sorted(results))
            else:
                folder_path = os.path.join(BASE_DIR, folder)
                results = []
                for root, dirs, files in os.walk(folder_path):
                    skip = [d for d in dirs if d.startswith('.') or d == '__pycache__' or d == '.venv']
                    for d in skip:
                        dirs.remove(d)
                    for f in files:
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                            rel = os.path.relpath(os.path.join(root, f), BASE_DIR)
                            results.append('/' + rel.replace('\\', '/'))
                self._json(sorted(results))
        elif parsed.path == '/api/next-num':
            existing = [f for f in os.listdir(CROPPED_DIR) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
            nums = []
            for f in existing:
                try:
                    nums.append(int(os.path.splitext(f)[0]))
                except ValueError:
                    pass
            nxt = max(nums) + 1 if nums else 1
            self._json({'next': nxt})
        elif parsed.path == '/api/annotations':
            anno_file = os.path.join(BASE_DIR, 'annotations.json')
            data = {}
            if os.path.exists(anno_file):
                with open(anno_file, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
            self._json(data)
        elif parsed.path == '/api/detect':
            qs = parse_qs(parsed.query)
            img_path = qs.get('img', [''])[0]
            if not img_path:
                self._json({'error': 'no img param'})
                return
            full_path = os.path.normpath(os.path.join(BASE_DIR, img_path.lstrip('/')))
            if not os.path.isfile(full_path):
                parent_path = os.path.normpath(os.path.join(BASE_DIR, '..', img_path.lstrip('/')))
                if os.path.isfile(parent_path):
                    full_path = parent_path
                else:
                    self._json({'error': 'file not found: ' + full_path})
                    return
            try:
                processor, model = _processor, _model

                img = Image.open(full_path).convert('RGB')
                orig_w, orig_h = img.size
                upscaled = False
                if orig_w < 1500 and orig_h < 1500:
                    img = img.resize((orig_w * 2, orig_h * 2), Image.LANCZOS)
                    upscaled = True
                proc_w, proc_h = img.size

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Spotting:"},
                        ],
                    }
                ]
                chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[chat_text],
                    images=[img],
                    max_pixels=2048 * 28 * 28,
                    return_tensors="pt",
                ).to(DEVICE)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=4096,
                    )
                generated_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
                output_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

                del inputs, generated_ids
                torch.cuda.empty_cache()

                scale_x = orig_w / proc_w
                scale_y = orig_h / proc_h

                boxes = []
                for m in SPOT_PATTERN.finditer(output_text):
                    text = m.group(1).strip()
                    vals = [int(m.group(i)) for i in range(2, 10)]
                    x1, y1, x2, y2, x3, y3, x4, y4 = vals

                    px1 = x1 / 1000.0 * proc_w * scale_x
                    py1 = y1 / 1000.0 * proc_h * scale_y
                    px2 = x2 / 1000.0 * proc_w * scale_x
                    py2 = y2 / 1000.0 * proc_h * scale_y
                    px3 = x3 / 1000.0 * proc_w * scale_x
                    py3 = y3 / 1000.0 * proc_h * scale_y
                    px4 = x4 / 1000.0 * proc_w * scale_x
                    py4 = y4 / 1000.0 * proc_h * scale_y

                    min_x = min(px1, px2, px3, px4)
                    min_y = min(py1, py2, py3, py4)
                    max_x = max(px1, px2, px3, px4)
                    max_y = max(py1, py2, py3, py4)

                    bw = int(max_x - min_x)
                    bh = int(max_y - min_y)

                    if bw < 10 or bh < 10:
                        continue

                    boxes.append({
                        'x': int(min_x),
                        'y': int(min_y),
                        'w': bw,
                        'h': bh,
                        'text': text,
                    })

                self._json({'boxes': boxes})
            except Exception as ex:
                import traceback
                traceback.print_exc()
                self._json({'error': str(ex)})
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/crop':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body)
            img_data = base64.b64decode(data['image'].split(',')[1])
            num = data.get('num', 1)
            ext = data.get('ext', 'png')
            filepath = os.path.join(CROPPED_DIR, f'{num}.{ext}')
            with open(filepath, 'wb') as fh:
                fh.write(img_data)
            self._json({'ok': True, 'path': '/cropped/' + f'{num}.{ext}'})
        elif parsed.path == '/api/annotations':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body)
            anno_file = os.path.join(BASE_DIR, 'annotations.json')
            with open(anno_file, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            self._json({'ok': True})
        else:
            self.send_error(404)

    def _json(self, obj):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    httpd = HTTPServer(('0.0.0.0', 8090), Handler)
    print(f'Serving on http://localhost:8090 (device={DEVICE})')
    httpd.serve_forever()