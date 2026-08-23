import torch
import gc
import time
import os
import re
import csv
import json
import hashlib
import traceback
import threading
import queue
from datetime import datetime
from urllib.parse import urlparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

# ── CRITICAL: tell PyTorch NOT to over-reserve VRAM on Windows ──
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH   = os.path.join(BASE_DIR, "models", "PaddleOCR-VL-1.5")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
SPOTTINGS_CSV   = os.path.join(BASE_DIR, "spottings.csv")
MASTER_CSV      = os.path.join(BASE_DIR, "annotations.csv")
IMAGE_EXTS      = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
SPOT_FOLDERS    = ["mehaksingal", "chaithanyakota", "chhavimohitkar"]

# ── FIXED CONSTANTS (root cause of your OOM) ──
MAX_PIXELS      = 501760        # was 1003520 — halved; spotting needs less than OCR
MAX_NEW_TOKENS  = 512           # was 1024 — spotting outputs are token-heavy but 512 is enough
UPSCALE_THRESHOLD = 0           # was 1500 — DISABLED: upscaling was doubling pixel count into OOM
MAX_IMAGE_PIXELS  = 20_000_000  # was 40M — skip truly giant images

# ── Speed settings ──
PREFETCH = 3                    # images to preprocess ahead in RAM
SAVE_EVERY = 10                 # save CSV every N images

torch.backends.cudnn.benchmark      = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32     = True

os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

SPOT_PATTERN = re.compile(
    r'(\S+?)'
    r'<\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
    r'<\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
    r'<\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
    r'<\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
)

def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vram = torch.cuda.memory_allocated() // 1048576 if torch.cuda.is_available() else 0
    print(f"[{ts}] [VRAM {vram}MB] {msg}")

def vram_free_mb():
    props = torch.cuda.get_device_properties(0)
    return (props.total_memory - torch.cuda.memory_reserved()) // 1048576

# ────────────────────────────────────────────
def load_model():
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16          # fixed deprecation warning
    ).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor

# ────────────────────────────────────────────
def preprocess_one(path, processor):
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size

    # ── UPSCALING REMOVED: was the main OOM trigger ──
    # Only upscale if image is TINY (< 100px) — otherwise keep native res
    if orig_w < 100 and orig_h < 100:
        img = img.resize((orig_w * 2, orig_h * 2), Image.LANCZOS)

    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Spotting:"}]}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_text], images=[img],
        max_pixels=MAX_PIXELS,
        return_tensors="pt",
    )
    return inputs, (orig_w, orig_h)

def preprocess_and_pin(path, processor):
    """Preprocess + move to pinned RAM for fast async GPU transfer"""
    inputs, dims = preprocess_one(path, processor)
    pinned = {
        k: v.pin_memory() if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    return pinned, dims

# ────────────────────────────────────────────
def parse_spots(raw_text, orig_w, orig_h):
    results = []
    for m in SPOT_PATTERN.finditer(raw_text):
        text = m.group(1)
        vals = [int(m.group(i)) for i in range(2, 10)]
        xs   = [vals[i] / 1000.0 * orig_w for i in range(0, 8, 2)]
        ys   = [vals[i] / 1000.0 * orig_h for i in range(1, 8, 2)]
        min_x = max(0, min(xs));  max_x = min(orig_w, max(xs))
        min_y = max(0, min(ys));  max_y = min(orig_h, max(ys))
        bw = int(max_x - min_x);  bh = int(max_y - min_y)
        if bw < 5 or bh < 5:
            continue
        results.append({"text": text, "type": "", "x": int(min_x), "y": int(min_y), "w": bw, "h": bh})
    return results

# ────────────────────────────────────────────
def infer_one(model, processor, gpu_inputs, orig_w, orig_h):
    input_len = gpu_inputs["input_ids"].shape[-1]
    with torch.no_grad():
        gen_ids = model.generate(
            **gpu_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            early_stopping=True,
        )
    torch.cuda.synchronize()
    new_tokens = gen_ids[0, input_len:].cpu()
    del gen_ids
    torch.cuda.empty_cache()
    decoded = processor.decode(new_tokens, skip_special_tokens=False)
    del new_tokens
    return parse_spots(decoded, orig_w, orig_h)

def safe_infer(model, processor, gpu_inputs, orig_w, orig_h, rel_path):
    """OOM-safe wrapper: retries at half resolution on failure"""
    try:
        return infer_one(model, processor, gpu_inputs, orig_w, orig_h)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" not in str(e).lower():
            raise
        log(f"OOM on {rel_path} — retrying at half tokens")
        torch.cuda.empty_cache(); gc.collect()
        try:
            input_len = gpu_inputs["input_ids"].shape[-1]
            with torch.no_grad():
                gen_ids = model.generate(
                    **gpu_inputs,
                    max_new_tokens=MAX_NEW_TOKENS // 2,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    early_stopping=True,
                )
            torch.cuda.synchronize()
            new_tokens = gen_ids[0, input_len:].cpu()
            del gen_ids; torch.cuda.empty_cache()
            decoded = processor.decode(new_tokens, skip_special_tokens=False)
            del new_tokens
            return parse_spots(decoded, orig_w, orig_h)
        except Exception as e2:
            log(f"Retry also failed for {rel_path}: {e2}")
            torch.cuda.empty_cache(); gc.collect()
            return []

# ────────────────────────────────────────────
def collect_images():
    images = []
    for folder in SPOT_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", ".venv")]
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    full = os.path.join(root, f)
                    rel  = os.path.relpath(full, BASE_DIR).replace("\\", "/")
                    images.append((full, rel))
    return sorted(images, key=lambda x: x[1])

def load_existing_spottings():
    done = set()
    if not os.path.exists(SPOTTINGS_CSV):
        return done
    try:
        with open(SPOTTINGS_CSV, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                done.add(row["file"])
    except Exception:
        return set()
    return done

def save_spottings(rows):
    tmp = SPOTTINGS_CSV + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "text", "type", "x", "y", "w", "h"])
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(3):
        try:
            if os.path.exists(SPOTTINGS_CSV):
                os.remove(SPOTTINGS_CSV)
            os.replace(tmp, SPOTTINGS_CSV)
            return
        except PermissionError:
            time.sleep(0.2)
    # fallback direct write
    with open(SPOTTINGS_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "text", "type", "x", "y", "w", "h"])
        writer.writeheader()
        writer.writerows(rows)
    try: os.remove(tmp)
    except OSError: pass

# ────────────────────────────────────────────
def warmup(model, processor):
    """Use a real tiny white image — not a large sample image"""
    tiny = Image.new("RGB", (128, 128), (255, 255, 255))
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Spotting:"}]}],
        tokenize=False, add_generation_prompt=True,
    )
    for i in range(2):
        try:
            torch.cuda.empty_cache(); gc.collect()
            inputs = processor(text=[chat_text], images=[tiny], max_pixels=MAX_PIXELS, return_tensors="pt")
            gpu_inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            del inputs
            with torch.no_grad():
                model.generate(**gpu_inputs, max_new_tokens=8,
                               pad_token_id=processor.tokenizer.eos_token_id)
            del gpu_inputs
            torch.cuda.empty_cache(); gc.collect()
            log(f"Warmup {i+1}/2 done")
        except Exception as e:
            log(f"Warmup {i+1}/2 failed (non-fatal): {e}")
            torch.cuda.empty_cache(); gc.collect()
    log(f"Ready. Free VRAM: {vram_free_mb()}MB")

# ────────────────────────────────────────────
def load_spottings_for_json():
    files_map = {}
    if not os.path.exists(SPOTTINGS_CSV):
        return {"files": []}
    with open(SPOTTINGS_CSV, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            f = row["file"]
            if f not in files_map:
                files_map[f] = {"file": f, "url": "/" + f, "words": []}
            files_map[f]["words"].append({
                "text": row["text"], "type": row["type"],
                "x": int(row["x"]), "y": int(row["y"]),
                "w": int(row["w"]), "h": int(row["h"]),
            })
    return {"files": list(files_map.values())}

# ────────────────────────────────────────────
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        if path in ("/", "/annotate.html"):
            html_path = os.path.join(BASE_DIR, "annotate.html")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open(html_path, "rb") as fh:
                self.wfile.write(fh.read())
        elif path == "/spottings.json":
            self._json(load_spottings_for_json())
        elif path == "/api/med-names":
            med_path = os.path.join(BASE_DIR, "med_names.txt")
            names = []
            if os.path.exists(med_path):
                with open(med_path, "r", encoding="utf-8") as fh:
                    names = [l.strip() for l in fh if l.strip()]
            self._json(names)
        elif path.startswith("/cropped/"):
            cropped_path = os.path.join(BASE_DIR, path.lstrip("/"))
            if os.path.isfile(cropped_path):
                self.send_response(200)
                ext = os.path.splitext(cropped_path)[1].lower().lstrip(".")
                ct  = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                       "bmp": "image/bmp", "webp": "image/webp"}.get(ext, "application/octet-stream")
                self.send_header("Content-Type", ct)
                self.end_headers()
                with open(cropped_path, "rb") as fh:
                    self.wfile.write(fh.read())
            else:
                self.send_error(404)
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            data   = json.loads(self.rfile.read(length))
            file_path    = data["file"]
            image_labels = data.get("image_labels", [])
            words        = data.get("words", [])
            rel_hash     = hashlib.md5(file_path.encode()).hexdigest()
            json_path    = os.path.join(ANNOTATIONS_DIR, f"{rel_hash}.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump({"file": file_path, "image_labels": image_labels, "words": words},
                          fh, ensure_ascii=False, indent=2)
            master_rows = []
            if os.path.exists(MASTER_CSV):
                with open(MASTER_CSV, "r", encoding="utf-8", newline="") as fh:
                    master_rows = [r for r in csv.DictReader(fh) if r["file"] != file_path]
            tags_str = ";".join(il.get("tag", "") for il in image_labels)
            for w in words:
                master_rows.append({
                    "file": file_path, "image_tags": tags_str,
                    "word_text": w.get("text", ""), "word_type": w.get("type", ""),
                    "x": w.get("x", ""), "y": w.get("y", ""),
                    "w": w.get("w", ""), "h": w.get("h", ""),
                })
            tmp = MASTER_CSV + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["file","image_tags","word_text","word_type","x","y","w","h"])
                writer.writeheader(); writer.writerows(master_rows)
            os.replace(tmp, MASTER_CSV)
            self._json({"ok": True})
        else:
            self.send_error(404)

    def _json(self, obj):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200); self.end_headers()

    def log_message(self, format, *args):
        pass  # silence HTTP request spam in terminal

# ════════════════════════════════════════════
if __name__ == "__main__":
    log(f"Loading PaddleOCR-VL-1.5 on {DEVICE}...")
    model, processor = load_model()
    torch.cuda.synchronize()
    log(f"Model loaded. VRAM: {torch.cuda.memory_allocated() // 1048576}MB | Free: {vram_free_mb()}MB")

    all_images  = collect_images()
    done_files  = load_existing_spottings()
    todo        = [(full, rel) for full, rel in all_images if rel not in done_files]

    # Sort SMALLEST first — better for memory stability (large images at end when cache is warm)
    def _img_area(item):
        try:
            with Image.open(item[0]) as im: return im.size[0] * im.size[1]
        except Exception: return 0

    todo.sort(key=_img_area)   # smallest first (was reverse=True — that was feeding you OOMs immediately)
    log(f"{len(all_images)} images found, {len(all_images)-len(todo)} done, {len(todo)} to process")

    spotting_rows = []
    if os.path.exists(SPOTTINGS_CSV):
        with open(SPOTTINGS_CSV, "r", encoding="utf-8", newline="") as fh:
            spotting_rows = list(csv.DictReader(fh))

    try:
        if todo:
            warmup(model, processor)

        # ── Prefetch queue: CPU preprocesses PREFETCH images ahead ──
        prefetch_q = queue.Queue(maxsize=PREFETCH)

        def producer():
            for full_path, rel_path in todo:
                try:
                    with Image.open(full_path) as _im:
                        if _im.size[0] * _im.size[1] > MAX_IMAGE_PIXELS:
                            prefetch_q.put((full_path, rel_path, None, None, "toobig"))
                            continue
                    pinned, dims = preprocess_and_pin(full_path, processor)
                    prefetch_q.put((full_path, rel_path, pinned, dims, None))
                except Exception as e:
                    prefetch_q.put((full_path, rel_path, None, None, str(e)))
            prefetch_q.put(None)  # sentinel

        prod_thread = threading.Thread(target=producer, daemon=True)
        prod_thread.start()

        pbar  = tqdm(total=len(todo), desc="Spotting", unit="img")
        t0    = time.time()

        while True:
            item = prefetch_q.get()
            if item is None:
                break

            full_path, rel_path, pinned, dims, err = item
            gpu_inputs = None

            try:
                if err == "toobig":
                    log(f"Skipping (too large): {rel_path}")
                    pbar.update(1); continue
                if err:
                    log(f"Preprocess error on {rel_path}: {err}")
                    pbar.update(1); continue

                orig_w, orig_h = dims
                gpu_inputs = {
                    k: v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in pinned.items()
                }
                torch.cuda.synchronize()
                del pinned

                spots = safe_infer(model, processor, gpu_inputs, orig_w, orig_h, rel_path)
                for w in spots:
                    spotting_rows.append({
                        "file": rel_path, "text": w["text"], "type": w["type"],
                        "x": w["x"], "y": w["y"], "w": w["w"], "h": w["h"],
                    })

            except Exception as e:
                log(f"Error on {rel_path}: {e}")
            finally:
                if gpu_inputs is not None:
                    try: del gpu_inputs
                    except Exception: pass
                torch.cuda.empty_cache()
                gc.collect()

            pbar.update(1)
            idx = pbar.n

            if idx % SAVE_EVERY == 0:
                save_spottings(spotting_rows)
                free = vram_free_mb()
                log(f"Progress {idx}/{len(todo)} | {len(spotting_rows)} spots | Free VRAM: {free}MB")

        pbar.close()
        save_spottings(spotting_rows)
        elapsed = time.time() - t0
        log(f"Spotting done. {len(spotting_rows)} spots in {elapsed:.1f}s ({elapsed/max(len(todo),1):.1f}s/img)")

    except Exception as e:
        log(f"Spotting loop crashed: {e}")
        traceback.print_exc()
        try: save_spottings(spotting_rows)
        except Exception: pass
        torch.cuda.empty_cache(); gc.collect()
        log("Saved progress, continuing to HTTP server...")

    log("Starting HTTP server on 0.0.0.0:8090")
    httpd = HTTPServer(("0.0.0.0", 8090), Handler)
    log(f"http://localhost:8090 (device={DEVICE})")
    httpd.serve_forever()
