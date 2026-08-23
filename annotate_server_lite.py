from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
import csv
import hashlib
import re
import base64
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
CROPPED_DIR = os.path.join(BASE_DIR, 'cropped')
SPOTTINGS_CSV = os.path.join(BASE_DIR, 'spottings.csv')
ANNOTATIONS_DIR = os.path.join(BASE_DIR, 'annotations')

os.makedirs(CROPPED_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

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
                entry["crop_files"] = saved.get("crop_files", [])
                entry["saved"] = True
            except Exception:
                pass
    return {"files": list(files_map.values())}

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.client_address[0]} - {format%args}")

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
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/crop':
            try:
                data = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                fp = data['file']
                src_path = os.path.join(BASE_DIR, fp.lstrip('/'))
                if not os.path.isfile(src_path):
                    src_path = os.path.normpath(os.path.join(BASE_DIR, '..', fp.lstrip('/')))
                if not os.path.isfile(src_path):
                    self._json({'ok': False, 'error': 'file not found'})
                    return
                img = Image.open(src_path).convert('RGB')
                iw, ih = img.size
                crops = []
                for i, w in enumerate(data.get('words', [])):
                    txt = w.get('text', '').strip()
                    if not txt:
                        crops.append('')
                        continue
                    x, y, bw, bh = int(w.get('x', 0)), int(w.get('y', 0)), int(w.get('w', 0)), int(w.get('h', 0))
                    if bw < 1 or bh < 1:
                        crops.append('')
                        continue
                    shape = w.get('shape', 'rect') or 'rect'
                    if shape == 'rrect':
                        angle = float(w.get('angle', 0))
                        cx = x + bw / 2
                        cy = y + bh / 2
                        rotated = img.rotate(-angle, center=(cx, cy), expand=False)
                        riw, rih = rotated.size
                        x1 = max(0, x)
                        y1 = max(0, y)
                        x2 = min(riw, x + bw)
                        y2 = min(rih, y + bh)
                        if x2 <= x1 or y2 <= y1:
                            crops.append('')
                            continue
                        crop = rotated.crop((x1, y1, x2, y2))
                    elif shape == 'poly':
                        try:
                            pts = json.loads(w.get('points', '[]'))
                        except (json.JSONDecodeError, TypeError):
                            pts = []
                        if not pts or len(pts) < 3:
                            x1 = max(0, x)
                            y1 = max(0, y)
                            x2 = min(iw, x + bw)
                            y2 = min(ih, y + bh)
                            if x2 <= x1 or y2 <= y1:
                                crops.append('')
                                continue
                            crop = img.crop((x1, y1, x2, y2))
                        else:
                            min_x = min(int(p[0]) for p in pts)
                            min_y = min(int(p[1]) for p in pts)
                            max_x = max(int(p[0]) for p in pts)
                            max_y = max(int(p[1]) for p in pts)
                            x1 = max(0, min_x)
                            y1 = max(0, min_y)
                            x2 = min(iw, max_x)
                            y2 = min(ih, max_y)
                            if x2 <= x1 or y2 <= y1:
                                crops.append('')
                                continue
                            crop = img.crop((x1, y1, x2, y2))
                    else:
                        x1 = max(0, x)
                        y1 = max(0, y)
                        x2 = min(iw, x + bw)
                        y2 = min(ih, y + bh)
                        if x2 <= x1 or y2 <= y1:
                            crops.append('')
                            continue
                        crop = img.crop((x1, y1, x2, y2))
                    rel_dir = os.path.dirname(fp)
                    out_dir = os.path.join(CROPPED_DIR, rel_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    ext = os.path.splitext(fp)[1]
                    wtype = w.get('type', 'other')
                    if wtype == 'other' and w.get('custom_type'):
                        wtype = w['custom_type']
                    out_name = f'{stem}_{i}_{wtype}{ext}'
                    crop.save(os.path.join(out_dir, out_name))
                    crops.append(f'cropped/{rel_dir}/{out_name}')
                self._json({'ok': True, 'crops': crops})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json({'ok': False, 'error': str(e)}, 400)
        elif parsed.path == '/api/save':
            try:
                data = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                fp = data['file']
                rh = hashlib.md5(fp.encode()).hexdigest()
                ann_path = os.path.join(ANNOTATIONS_DIR, f"{rh}.json")
                with open(ann_path, 'w', encoding='utf-8') as fh:
                    json.dump({
                        "file": fp,
                        "image_labels": data.get("image_labels", []),
                        "words": data.get("words", []),
                        "image_ocr": data.get("image_ocr", ""),
                        "crop_files": data.get("crop_files", []),
                        "saved_at": datetime.now().isoformat(),
                    }, fh, ensure_ascii=False, indent=2)
                self._json({"ok": True})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 400)
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

    def _json(self, obj, status=200):
        self.send_response(status)
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
    print(f'Serving on http://localhost:8090')
    httpd.serve_forever()