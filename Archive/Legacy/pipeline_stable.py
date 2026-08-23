import os
import sys
import csv
import json
import time
import logging
import hashlib
import sqlite3
import subprocess
import threading
import queue
from pathlib import Path
from datetime import datetime
from PIL import Image
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

BASE_DIR        = Path(__file__).parent
DATA_DIR        = Path(os.environ.get("DATA_DIR", str(BASE_DIR)))
SPOTTINGS_CSV   = BASE_DIR / "spottings.csv"
MASTER_CSV      = DATA_DIR / "annotations.csv"
DB_PATH         = DATA_DIR / "annotations.db"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
CROPPED_DIR     = DATA_DIR / "cropped"
IMAGE_EXTS      = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
SPOT_FOLDERS    = ["mehaksingal", "chaithanyakota", "chhavimohitkar"]
WORKER_SCRIPT   = BASE_DIR / "_worker.py"
IMAGES_DIR      = Path(os.environ.get("IMAGES_DIR", str(BASE_DIR)))
MODEL_DIR       = Path(os.environ.get("MODEL_DIR", str(BASE_DIR / "models" / "PaddleOCR-VL-1.5")))

BATCH_SIZE = 25

ANNOTATIONS_DIR.mkdir(exist_ok=True)
CROPPED_DIR.mkdir(exist_ok=True)

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS spottings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            file      TEXT NOT NULL,
            text      TEXT,
            type      TEXT,
            x         INTEGER,
            y         INTEGER,
            w         INTEGER,
            h         INTEGER,
            shape     TEXT DEFAULT 'rect',
            angle     REAL DEFAULT 0,
            points    TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS annotations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            file          TEXT NOT NULL,
            image_tags    TEXT DEFAULT '',
            word_text     TEXT DEFAULT '',
            word_type     TEXT DEFAULT '',
            custom_type   TEXT DEFAULT '',
            x             INTEGER,
            y             INTEGER,
            w             INTEGER,
            h             INTEGER,
            shape         TEXT DEFAULT 'rect',
            angle         REAL DEFAULT 0,
            points        TEXT DEFAULT '',
            cropped_file  TEXT DEFAULT '',
            source_image  TEXT DEFAULT '',
            image_ocr     TEXT DEFAULT '',
            is_file_row   INTEGER DEFAULT 0,
            saved_at      TEXT DEFAULT '',
            UNIQUE(file, word_text, word_type, x, y, w, h)
        );
        CREATE INDEX IF NOT EXISTS idx_spottings_file ON spottings(file);
        CREATE INDEX IF NOT EXISTS idx_annotations_file ON annotations(file);
    """)
    # Migrate existing tables: add shape/angle/points columns if missing
    for table, cols in [("spottings", ["shape", "angle", "points"]), ("annotations", ["shape", "angle", "points"])]:
        try:
            existing_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        except Exception:
            existing_cols = set()
        if "shape" not in existing_cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN shape TEXT DEFAULT 'rect'")
        if "angle" not in existing_cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN angle REAL DEFAULT 0")
        if "points" not in existing_cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN points TEXT DEFAULT ''")
    conn.commit()
    conn.close()

def log(msg):
    logging.info(msg)

def collect_images():
    images = []
    for folder in SPOT_FOLDERS:
        fp = IMAGES_DIR / folder
        if not fp.is_dir():
            continue
        for f in sorted(fp.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS:
                rel = f.relative_to(IMAGES_DIR).as_posix()
                images.append((str(f), rel))
    return sorted(images, key=lambda x: x[1])

def load_done():
    done = set()
    if not SPOTTINGS_CSV.exists():
        return done
    try:
        with open(SPOTTINGS_CSV, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                done.add(row["file"])
    except Exception:
        pass
    return done

def save_new_rows(new_rows):
    existing = []
    if SPOTTINGS_CSV.exists():
        try:
            with open(SPOTTINGS_CSV, encoding="utf-8", newline="") as fh:
                existing = list(csv.DictReader(fh))
        except Exception:
            pass
    all_rows = existing + new_rows
    tmp = str(SPOTTINGS_CSV) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file","text","type","x","y","w","h","shape","angle","points"])
        w.writeheader(); w.writerows(all_rows)
    for _ in range(5):
        try:
            if SPOTTINGS_CSV.exists():
                SPOTTINGS_CSV.unlink()
            Path(tmp).replace(SPOTTINGS_CSV)
            break
        except PermissionError:
            time.sleep(0.5)
    conn = get_db()
    conn.execute("DELETE FROM spottings")
    conn.executemany(
        "INSERT INTO spottings (file,text,type,x,y,w,h,shape,angle,points) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(r["file"], r["text"], r["type"], r["x"], r["y"], r["w"], r["h"],
          r.get("shape","rect"), r.get("angle",0), r.get("points","")) for r in all_rows]
    )
    conn.commit()
    conn.close()

# ── HTTP server ────────────────────────────────────────────────────────────────
def load_spottings_for_json():
    files_map = {}
    if not SPOTTINGS_CSV.exists():
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
        ann_path = ANNOTATIONS_DIR / f"{rh}.json"
        if ann_path.exists():
            try:
                saved = json.loads(ann_path.read_text(encoding="utf-8"))
                entry["image_labels"] = saved.get("image_labels", [])
                entry["words"] = saved.get("words", entry["words"])
                entry["image_ocr"] = saved.get("image_ocr", "")
                entry["saved"] = True
            except Exception:
                pass
    return {"files": list(files_map.values())}

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(BASE_DIR), **kw)
    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/annotate.html"):
            html = BASE_DIR / "annotate.html"
            self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
            self.wfile.write(html.read_bytes())
        elif p == "/spottings.json":
            self._json(load_spottings_for_json())
        elif p == "/api/med-names":
            med = BASE_DIR / "med_names.txt"
            names = [l.strip() for l in med.read_text(encoding="utf-8").splitlines() if l.strip()] if med.exists() else []
            self._json(names)
        elif p.startswith("/cropped/"):
            cp = CROPPED_DIR / p[len("/cropped/"):].lstrip("/")
            if cp.is_file():
                self.send_response(200)
                ext = cp.suffix.lower().lstrip(".")
                ct = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg","bmp":"image/bmp","webp":"image/webp"}.get(ext,"application/octet-stream")
                self.send_header("Content-Type", ct); self.end_headers()
                self.wfile.write(cp.read_bytes())
            else: self.send_error(404)
        elif any(p.startswith(f"/{folder}/") or p == f"/{folder}" for folder in SPOT_FOLDERS):
            ip = IMAGES_DIR / p.lstrip("/")
            if ip.is_file():
                ext = ip.suffix.lower().lstrip(".")
                ct = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg","bmp":"image/bmp","webp":"image/webp"}.get(ext,"application/octet-stream")
                self.send_response(200); self.send_header("Content-Type", ct); self.end_headers()
                self.wfile.write(ip.read_bytes())
            else:
                self.send_error(404)
        else: super().do_GET()
    def do_POST(self):
        if urlparse(self.path).path == "/api/save":
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))))
                fp = data["file"]
                rh = hashlib.md5(fp.encode()).hexdigest()
                (ANNOTATIONS_DIR / f"{rh}.json").write_text(
                    json.dumps({"file":fp,"image_labels":data.get("image_labels",[]),"words":data.get("words",[]),"image_ocr":data.get("image_ocr",""),"saved_at":datetime.now().isoformat()},
                               ensure_ascii=False, indent=2), encoding="utf-8")
                crop_names = data.get("crop_files", [])
                master = []
                if MASTER_CSV.exists():
                    with open(MASTER_CSV, encoding="utf-8", newline="") as fh:
                        master = [r for r in csv.DictReader(fh) if r["file"] != fp]
                tags = ";".join((il.get("tag","") if isinstance(il, dict) else str(il)) for il in data.get("image_labels",[]))
                ocr = data.get("image_ocr","")
                file_row = {"file":fp,"image_tags":tags,"word_text":"","word_type":"","custom_type":"","x":"","y":"","w":"","h":"","shape":"","angle":0,"points":"","cropped_file":"","source_image":str((IMAGES_DIR / fp).resolve()),"image_ocr":ocr}
                master.append(file_row)
                for i,w in enumerate(data.get("words",[])):
                    wtype = w.get("type","other")
                    if wtype == "other" and w.get("custom_type"):
                        wtype = w["custom_type"]
                    crop_name = crop_names[i] if i < len(crop_names) else ""
                    src_abs = str((IMAGES_DIR / fp).resolve())
                    row = {"file":fp,"image_tags":tags,"word_text":w.get("text",""),
                        "word_type":w.get("type",""),"custom_type":w.get("custom_type",""),
                        "x":w.get("x",""),"y":w.get("y",""),"w":w.get("w",""),"h":w.get("h",""),
                        "shape":w.get("shape","rect"),"angle":w.get("angle",0),"points":w.get("points",""),
                        "cropped_file":crop_name,"source_image":src_abs,"image_ocr":""}
                    master.append(row)
                tmp = str(MASTER_CSV)+".tmp"
                with open(tmp,"w",encoding="utf-8",newline="") as fh:
                    wr = csv.DictWriter(fh,fieldnames=["file","image_tags","word_text","word_type","custom_type","x","y","w","h","shape","angle","points","cropped_file","source_image","image_ocr"])
                    wr.writeheader(); wr.writerows(master)
                try:
                    Path(tmp).replace(MASTER_CSV)
                except PermissionError:
                    with open(MASTER_CSV,"w",encoding="utf-8",newline="") as fh2:
                        wr2 = csv.DictWriter(fh2,fieldnames=["file","image_tags","word_text","word_type","custom_type","x","y","w","h","shape","angle","points","cropped_file","source_image","image_ocr"])
                        wr2.writeheader(); wr2.writerows(master)
                saved_at = datetime.now().isoformat()
                conn = get_db()
                conn.execute("DELETE FROM annotations WHERE file = ?", (fp,))
                conn.execute(
                    "INSERT INTO annotations (file,image_tags,word_text,word_type,custom_type,x,y,w,h,shape,angle,points,cropped_file,source_image,image_ocr,is_file_row,saved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (fp, tags, "", "", "", None, None, None, None, "", 0, "", "", str((IMAGES_DIR / fp).resolve()), ocr, 1, saved_at)
                )
                for i, w in enumerate(data.get("words", [])):
                    crop_name = crop_names[i] if i < len(crop_names) else ""
                    conn.execute(
                        "INSERT INTO annotations (file,image_tags,word_text,word_type,custom_type,x,y,w,h,shape,angle,points,cropped_file,source_image,image_ocr,is_file_row,saved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (fp, tags, w.get("text",""), w.get("type",""), w.get("custom_type",""),
                         w.get("x"), w.get("y"), w.get("w"), w.get("h"),
                         w.get("shape","rect"), w.get("angle",0), w.get("points",""),
                         crop_name, str((IMAGES_DIR / fp).resolve()), "", 0, saved_at)
                    )
                conn.commit()
                conn.close()
                self._json({"ok":True})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 400)
        elif urlparse(self.path).path == "/api/crop":
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))))
                fp = data["file"]
                src_path = IMAGES_DIR / fp
                if not src_path.is_file():
                    self._json({"ok": False, "error": "file not found"})
                    return
                img = Image.open(src_path).convert("RGB")
                iw, ih = img.size
                crops = []
                for i, w in enumerate(data.get("words", [])):
                    txt = w.get("text", "").strip()
                    if not txt:
                        crops.append("")
                        continue
                    x, y, bw, bh = int(w.get("x",0)), int(w.get("y",0)), int(w.get("w",0)), int(w.get("h",0))
                    if bw < 1 or bh < 1:
                        crops.append("")
                        continue
                    shape = w.get("shape", "rect") or "rect"
                    if shape == "rrect":
                        angle = float(w.get("angle", 0))
                        cx = x + bw / 2
                        cy = y + bh / 2
                        rotated = img.rotate(-angle, center=(cx, cy), expand=False)
                        riw, rih = rotated.size
                        x1 = max(0, x)
                        y1 = max(0, y)
                        x2 = min(riw, x + bw)
                        y2 = min(rih, y + bh)
                        if x2 <= x1 or y2 <= y1:
                            crops.append("")
                            continue
                        crop = rotated.crop((x1, y1, x2, y2))
                    elif shape == "poly":
                        try:
                            pts = json.loads(w.get("points", "[]"))
                        except (json.JSONDecodeError, TypeError):
                            pts = []
                        if not pts or len(pts) < 3:
                            # fallback to rect
                            x1 = max(0, x)
                            y1 = max(0, y)
                            x2 = min(iw, x + bw)
                            y2 = min(ih, y + bh)
                            if x2 <= x1 or y2 <= y1:
                                crops.append("")
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
                                crops.append("")
                                continue
                            crop = img.crop((x1, y1, x2, y2))
                    else:
                        x1 = max(0, x)
                        y1 = max(0, y)
                        x2 = min(iw, x + bw)
                        y2 = min(ih, y + bh)
                        if x2 <= x1 or y2 <= y1:
                            crops.append("")
                            continue
                        crop = img.crop((x1, y1, x2, y2))
                    rel = Path(fp)
                    out_dir = CROPPED_DIR / rel.parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                    stem = rel.stem
                    ext = rel.suffix
                    wtype = w.get("type", "other")
                    if wtype == "other" and w.get("custom_type"):
                        wtype = w["custom_type"]
                    out_name = f"{stem}_{i}_{wtype}{ext}"
                    crop.save(out_dir / out_name)
                    crops.append(f"cropped/{rel.parent}/{out_name}")
                self._json({"ok": True, "crops": crops})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 400)
        else: self.send_error(404)

    def _json(self, obj, status=200):
        self.send_response(status); self.send_header("Content-Type","application/json"); self.end_headers()
        self.wfile.write(json.dumps(obj).encode())
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin","*"); super().end_headers()
    def do_OPTIONS(self): self.send_response(200); self.end_headers()
    def log_message(self, *a): pass

# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-spotting", action="store_true", help="Skip spotting, just serve HTTP")
    parser.add_argument("--port", type=int, default=8090, help="HTTP port (default 8090)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host (default 0.0.0.0)")
    args = parser.parse_args()

    init_db()

    if not args.skip_spotting:
        all_images = collect_images()
        done       = load_done()
        todo       = [(f, r) for f, r in all_images if r not in done]

        log(f"{len(all_images)} images | {len(done)} done | {len(todo)} to process")

        if not todo:
            log("All images already spotted — GPU not needed")
        else:
            log(f"Strategy: fresh subprocess every {BATCH_SIZE} images (guaranteed stable)")
            chunks = [todo[i:i+BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
            t0 = time.time()
            processed = 0

            for ci, chunk in enumerate(chunks):
                log(f"Batch {ci+1}/{len(chunks)} — images {processed+1}–{processed+len(chunk)}")

                batch_file = BASE_DIR / "_batch.json"
                batch_file.write_text(json.dumps([r for _, r in chunk]), encoding="utf-8")

                log(f"Launching worker subprocess for {len(chunk)} images...")

                log_q = queue.Queue()

                def _reader(p, q):
                    try:
                        for line in p.stdout:
                            q.put(line)
                    except Exception:
                        pass
                    finally:
                        q.put(None)

                worker_env = os.environ.copy()
                worker_env["IMAGES_DIR"] = str(IMAGES_DIR)
                worker_env["MODEL_DIR"] = str(MODEL_DIR)

                proc = subprocess.Popen(
                    [sys.executable, "-u", str(WORKER_SCRIPT)],
                    cwd=str(BASE_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=worker_env,
                )

                t = threading.Thread(target=_reader, args=(proc, log_q), daemon=True)
                t.start()

                try:
                    while True:
                        try:
                            line = log_q.get(timeout=0.5)
                        except queue.Empty:
                            if proc.poll() is not None:
                                break
                            continue
                        if line is None:
                            break
                        print(line.rstrip(), flush=True)
                except KeyboardInterrupt:
                    log("Interrupted — killing worker...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    sys.exit(1)

                proc.wait()
                if proc.returncode != 0:
                    log(f"Batch {ci+1} exited with code {proc.returncode} — continuing")

                out_file = BASE_DIR / "_batch_out.json"
                if out_file.exists():
                    try:
                        new_rows = json.loads(out_file.read_text(encoding="utf-8"))
                        save_new_rows(new_rows)
                        out_file.unlink(missing_ok=True)
                        processed += len(chunk)
                        elapsed = time.time() - t0
                        avg = elapsed / processed
                        remaining = (len(todo) - processed) * avg
                        log(f"Done {processed}/{len(todo)} | avg {avg:.1f}s/img | ETA {remaining/60:.1f}min")
                    except Exception as e:
                        log(f"Failed reading batch output: {e}")

                batch_file.unlink(missing_ok=True)
    else:
        log("Skipping spotting (--skip-spotting)")

    log(f"Starting HTTP server on {args.host}:{args.port}")
    httpd = HTTPServer((args.host, args.port), Handler)
    log(f"http://localhost:{args.port}")
    httpd.serve_forever()
