"""
Worker script — loaded fresh per batch by pipeline.py
Loads model, processes images listed in _batch.json, writes _batch_out.json, exits.
"""
import logging

logging.basicConfig(level=logging.DEBUG, format="[worker] %(asctime)s %(message)s", datefmt="%H:%M:%S")

logging.info("Starting imports...")
import os
import sys
import gc
import re
import json
import time
from pathlib import Path
from PIL import Image
logging.info("Importing torch...")
import torch
logging.info("Importing transformers...")
from transformers import AutoProcessor, AutoModelForImageTextToText
logging.info("All imports done.")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "PaddleOCR-VL-1.5"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

MAX_PIXELS     = 313600   # 560x560 fixed tile
MIN_PIXELS     = 313600
MAX_NEW_TOKENS = 300

torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

SPOT_PATTERN = re.compile(
    r'(\S+?)'
    r'<\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
    r'<\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
    r'<\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
    r'<\|LOC_(\d+)\|><\|LOC_(\d+)\|>'
)

TARGET_W = 560
TARGET_H = 560

def resize_letterbox(img):
    orig_w, orig_h = img.size
    scale = min(TARGET_W / orig_w, TARGET_H / orig_h)
    nw, nh = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))
    canvas.paste(img, ((TARGET_W - nw) // 2, (TARGET_H - nh) // 2))
    return canvas

def parse_spots(text, ow, oh):
    results = []
    for m in SPOT_PATTERN.finditer(text):
        word = m.group(1)
        vals = [int(m.group(i)) for i in range(2, 10)]
        xs   = [vals[i] / 1000.0 * ow for i in range(0, 8, 2)]
        ys   = [vals[i] / 1000.0 * oh for i in range(1, 8, 2)]
        x1, x2 = max(0, min(xs)), min(ow, max(xs))
        y1, y2 = max(0, min(ys)), min(oh, max(ys))
        bw, bh = int(x2-x1), int(y2-y1)
        if bw >= 5 and bh >= 5:
            results.append({"text": word, "type": "", "x": int(x1), "y": int(y1), "w": bw, "h": bh})
    return results

def main():
    batch_file = BASE_DIR / "_batch.json"
    if not batch_file.exists():
        logging.error("No _batch.json found")
        return

    rel_paths = json.loads(batch_file.read_text(encoding="utf-8"))
    logging.info(f"Loading model for {len(rel_paths)} images...")

    # Load model fresh — no state from previous batch
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Spotting:"}]}],
        tokenize=False, add_generation_prompt=True,
    )

    # Warmup with tiny blank image
    tiny = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))
    inp  = processor(text=[chat_text], images=[tiny], min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, return_tensors="pt")
    gi   = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
    with torch.no_grad():
        model.generate(**gi, max_new_tokens=8,
                       pad_token_id=processor.tokenizer.eos_token_id,
                       do_sample=False, use_cache=True)
    del gi, inp; torch.cuda.empty_cache(); gc.collect()
    logging.info("Warmup done")

    out_rows = []
    for i, rel in enumerate(rel_paths):
        full = BASE_DIR / rel
        t0 = time.time()
        try:
            img = Image.open(full).convert("RGB")
            orig_w, orig_h = img.size
            img = resize_letterbox(img)

            inp = processor(text=[chat_text], images=[img],
                            min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, return_tensors="pt")
            gi  = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            del inp

            with torch.no_grad():
                gen = model.generate(
                    **gi,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    do_sample=False,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            ilen = gi["input_ids"].shape[-1]
            tokens = gen[0, ilen:].cpu()
            del gen, gi
            torch.cuda.empty_cache(); gc.collect()

            decoded = processor.decode(tokens, skip_special_tokens=False)
            del tokens
            spots = parse_spots(decoded, orig_w, orig_h)
            for s in spots:
                out_rows.append({"file": rel, **s})

            elapsed = time.time() - t0
            logging.info(f"{i+1}/{len(rel_paths)} {rel} — {len(spots)} spots in {elapsed:.1f}s")

        except Exception as e:
            logging.error(f"ERROR {rel}: {e}")
            try: torch.cuda.empty_cache(); gc.collect()
            except Exception: pass

    # Write results — coordinator picks these up
    out_file = BASE_DIR / "_batch_out.json"
    out_file.write_text(json.dumps(out_rows), encoding="utf-8")
    logging.info(f"Done. {len(out_rows)} spot rows written.")

if __name__ == "__main__":
    main()