import torch
import gc
import time
import os
import re
import csv
import json
import hashlib
import traceback
from datetime import datetime
from urllib.parse import urlparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(BASE_DIR, "models", "PaddleOCR-VL-1.5")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
SPOTTINGS_CSV = os.path.join(BASE_DIR, "spottings.csv")
MASTER_CSV = os.path.join(BASE_DIR, "annotations.csv")
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
SPOT_FOLDERS = ['mehaksingal', 'chaithanyakota', 'chhavimohitkar']
VRAM_BUDGET_MB = 5530

MAX_PIXELS = 1003520
MAX_NEW_TOKENS = 1024
UPSCALE_THRESHOLD = 1500
MAX_IMAGE_PIXELS = 40_000_000

SPOT_PATTERN = re.compile(
    r'(\S+?)'
    r'<\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
    r'<\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
    r'<\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
    r'<\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.makedirs(ANNOTATIONS_DIR, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vram = torch.cuda.memory_allocated() // 1048576 if torch.cuda.is_available() else 0
    print(f"[{ts}] [VRAM {vram}MB] {msg}")


def load_model():
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def preprocess_one(path, processor):
    img = Image.open(path).convert('RGB')
    orig_w, orig_h = img.size
    if orig_w < UPSCALE_THRESHOLD and orig_h < UPSCALE_THRESHOLD:
        img = img.resize((orig_w * 2, orig_h * 2), Image.LANCZOS)
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Spotting:"}]}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_text], images=[img], max_pixels=MAX_PIXELS, return_tensors="pt",
    )
    return inputs, (orig_w, orig_h)


def parse_spots(raw_text, orig_w, orig_h):
    results = []
    for m in SPOT_PATTERN.finditer(raw_text):
        text = m.group(1)
        vals = [int(m.group(i)) for i in range(2, 10)]
        xs = [vals[i] / 1000.0 * orig_w for i in range(0, 8, 2)]
        ys = [vals[i] / 1000.0 * orig_h for i in range(1, 8, 2)]
        min_x = max(0, min(xs))
        min_y = max(0, min(ys))
        max_x = min(orig_w, max(xs))
        max_y = min(orig_h, max(ys))
        bw = int(max_x - min_x)
        bh = int(max_y - min_y)
        if bw < 5 or bh < 5:
            continue
        results.append({'text': text, 'type': '', 'x': int(min_x), 'y': int(min_y), 'w': bw, 'h': bh})
    return results


def infer_one(model, processor, gpu_inputs, orig_w, orig_h):
    input_len = gpu_inputs['input_ids'].shape[-1]
    with torch.no_grad():
        gen_ids = model.generate(
            **gpu_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    new_tokens = gen_ids[0, input_len:].cpu()
    del gen_ids
    torch.cuda.empty_cache()
    decoded = processor.decode(new_tokens, skip_special_tokens=False)
    del new_tokens
    spots = parse_spots(decoded, orig_w, orig_h)
    return spots


def collect_images():
    images = []
    for folder in SPOT_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for root, dirs, files in os.walk(folder_path):
            skip = [d for d in dirs if d.startswith('.') or d == '__pycache__' or d == '.venv']
            for d in skip:
                dirs.remove(d)
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, BASE_DIR).replace('\\', '/')
                    images.append((full, rel))
    return sorted(images, key=lambda x: x[1])


def load_existing_spottings():
    done = set()
    if not os.path.exists(SPOTTINGS_CSV):
        return done
    try:
        with open(SPOTTINGS_CSV, 'r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                done.add(row['file'])
    except Exception:
        return set()
    return done


def save_spottings(rows):
    tmp = SPOTTINGS_CSV + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['file', 'text', 'type', 'x', 'y', 'w', 'h'])
        writer.writeheader()
        writer.writerows(rows)
    try:
        if os.path.exists(SPOTTINGS_CSV):
            os.remove(SPOTTINGS_CSV)
    except PermissionError:
        pass
    try:
        os.replace(tmp, SPOTTINGS_CSV)
    except PermissionError:
        with open(SPOTTINGS_CSV, 'w', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=['file', 'text', 'type', 'x', 'y', 'w', 'h'])
            writer.writeheader()
            writer.writerows(rows)
        try:
            os.remove(tmp)
        except OSError:
            pass


def warmup(model, processor):
    tiny = Image.new("RGB", (64, 64), (255, 255, 255))
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Spotting:"}]}],
        tokenize=False, add_generation_prompt=True,
    )
    for i in range(2):
        try:
            inputs = processor(text=[chat_text], images=[tiny], max_pixels=MAX_PIXELS, return_tensors="pt")
            gpu_inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            del inputs
            with torch.no_grad():
                model.generate(**gpu_inputs, max_new_tokens=8, pad_token_id=processor.tokenizer.eos_token_id)
            del gpu_inputs
            torch.cuda.empty_cache()
            gc.collect()
            log(f"Warmup {i+1}/2 done")
        except Exception as e:
            log(f"Warmup {i+1}/2 failed: {e}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
    log("Ready.")


def load_spottings_for_json():
    files_map = {}
    if not os.path.exists(SPOTTINGS_CSV):
        return {'files': []}
    with open(SPOTTINGS_CSV, 'r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            f = row['file']
            if f not in files_map:
                files_map[f] = {'file': f, 'url': '/' + f, 'words': []}
            files_map[f]['words'].append({
                'text': row['text'],
                'type': row['type'],
                'x': int(row['x']),
                'y': int(row['y']),
                'w': int(row['w']),
                'h': int(row['h']),
            })
    return {'files': list(files_map.values())}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == '/' or path == '/annotate.html':
            html_path = os.path.join(BASE_DIR, 'annotate.html')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            with open(html_path, 'rb') as fh:
                self.wfile.write(fh.read())
        elif path == '/spottings.json':
            self._json(load_spottings_for_json())
        elif path == '/api/med-names':
            med_path = os.path.join(BASE_DIR, 'med_names.txt')
            names = []
            if os.path.exists(med_path):
                with open(med_path, 'r', encoding='utf-8') as fh:
                    names = [line.strip() for line in fh if line.strip()]
            self._json(names)
        elif path.startswith('/cropped/'):
            cropped_path = os.path.join(BASE_DIR, path.lstrip('/'))
            if os.path.isfile(cropped_path):
                self.send_response(200)
                ext = os.path.splitext(cropped_path)[1].lower().lstrip('.')
                ct = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'bmp': 'image/bmp', 'webp': 'image/webp'}.get(ext, 'application/octet-stream')
                self.send_header('Content-Type', ct)
                self.end_headers()
                with open(cropped_path, 'rb') as fh:
                    self.wfile.write(fh.read())
            else:
                self.send_error(404)
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/save':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body)
            file_path = data['file']
            image_labels = data.get('image_labels', [])
            words = data.get('words', [])
            rel_hash = hashlib.md5(file_path.encode()).hexdigest()
            json_path = os.path.join(ANNOTATIONS_DIR, f'{rel_hash}.json')
            per_image = {'file': file_path, 'image_labels': image_labels, 'words': words}
            with open(json_path, 'w', encoding='utf-8') as fh:
                json.dump(per_image, fh, ensure_ascii=False, indent=2)
            master_rows = []
            if os.path.exists(MASTER_CSV):
                with open(MASTER_CSV, 'r', encoding='utf-8', newline='') as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        if row['file'] != file_path:
                            master_rows.append(row)
            tags_str = ';'.join(il.get('tag', '') for il in image_labels)
            for w in words:
                master_rows.append({
                    'file': file_path,
                    'image_tags': tags_str,
                    'word_text': w.get('text', ''),
                    'word_type': w.get('type', ''),
                    'x': w.get('x', ''),
                    'y': w.get('y', ''),
                    'w': w.get('w', ''),
                    'h': w.get('h', ''),
                })
            tmp = MASTER_CSV + '.tmp'
            with open(tmp, 'w', encoding='utf-8', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=['file', 'image_tags', 'word_text', 'word_type', 'x', 'y', 'w', 'h'])
                writer.writeheader()
                writer.writerows(master_rows)
            os.replace(tmp, MASTER_CSV)
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
    log(f"Loading PaddleOCR-VL-1.5 on {DEVICE}...")
    model, processor = load_model()
    torch.cuda.synchronize()
    log(f"Model loaded. VRAM: {torch.cuda.memory_allocated() // 1048576}MB")

    all_images = collect_images()
    done_files = load_existing_spottings()
    todo = [(full, rel) for full, rel in all_images if rel not in done_files]

    def _img_area(item):
        try:
            with Image.open(item[0]) as im:
                return im.size[0] * im.size[1]
        except Exception:
            return 0

    todo.sort(key=_img_area, reverse=True)
    log(f"{len(all_images)} images found, {len(all_images) - len(todo)} done, {len(todo)} to process (largest first)")

    spotting_rows = []
    if os.path.exists(SPOTTINGS_CSV):
        with open(SPOTTINGS_CSV, 'r', encoding='utf-8', newline='') as fh:
            spotting_rows = list(csv.DictReader(fh))

    try:
        if todo:
            warmup(model, processor)

            pbar = tqdm(total=len(todo), desc="Spotting", unit="img")
            t0_all = time.time()

            for i, (full_path, rel_path) in enumerate(todo):
                gpu_inputs = None
                inputs = None
                try:
                    allocated = torch.cuda.memory_allocated() // 1048576
                    if allocated > VRAM_BUDGET_MB:
                        log(f"VRAM {allocated}MB > budget, empty_cache and retry")
                        torch.cuda.empty_cache()
                        gc.collect()
                        allocated = torch.cuda.memory_allocated() // 1048576
                        if allocated > VRAM_BUDGET_MB:
                            log(f"VRAM still {allocated}MB after cleanup, skipping {rel_path}")
                            pbar.update(1)
                            continue

                    try:
                        with Image.open(full_path) as _im:
                            _px = _im.size[0] * _im.size[1]
                        if _px > MAX_IMAGE_PIXELS:
                            log(f"Image too large ({_px}px), skipping {rel_path}")
                            pbar.update(1)
                            continue
                    except Exception:
                        pass

                    inputs, (orig_w, orig_h) = preprocess_one(full_path, processor)
                    gpu_inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    del inputs
                    inputs = None

                    spots = infer_one(model, processor, gpu_inputs, orig_w, orig_h)
                    for w in spots:
                        spotting_rows.append({'file': rel_path, 'text': w['text'], 'type': w['type'], 'x': w['x'], 'y': w['y'], 'w': w['w'], 'h': w['h']})

                except (RuntimeError, torch.AcceleratorError) as e:
                    log(f"GPU error on {rel_path}: {e}")
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                except Exception as e:
                    log(f"Error on {rel_path}: {e}")
                finally:
                    if gpu_inputs is not None:
                        try:
                            del gpu_inputs
                        except Exception:
                            pass
                    if inputs is not None:
                        try:
                            del inputs
                        except Exception:
                            pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()
                    save_spottings(spotting_rows)
                    pbar.update(1)

                if (i + 1) % 25 == 0:
                    allocated = torch.cuda.memory_allocated() // 1048576
                    reserved = torch.cuda.memory_reserved() // 1048576
                    log(f"Progress {i+1}/{len(todo)} | {len(spotting_rows)} spots | alloc={allocated}MB reserved={reserved}MB")

            pbar.close()
            elapsed = time.time() - t0_all
            log(f"Spotting done. {len(spotting_rows)} spots total in {elapsed:.1f}s")
    except Exception as e:
        log(f"Spotting loop crashed: {e}")
        traceback.print_exc()
        try:
            save_spottings(spotting_rows)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        log("Saved progress, continuing to HTTP server...")

    log("Starting HTTP server on 0.0.0.0:8090")
    httpd = HTTPServer(('0.0.0.0', 8090), Handler)
    log(f"http://localhost:8090 (device={DEVICE})")
    httpd.serve_forever()