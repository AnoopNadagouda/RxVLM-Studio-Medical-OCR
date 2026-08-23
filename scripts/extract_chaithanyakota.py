#!/usr/bin/env python3
"""Extract chaithanyakota parquet images to disk + CSV."""
import io
import pandas as pd
from PIL import Image
from pathlib import Path

BASE = Path(r"C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump\chaithanyakota")
PARQUET = BASE / "data" / "train-00000-of-00001.parquet"
OUT_DIR = BASE / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = BASE / "chaithanyakota.csv"

df = pd.read_parquet(PARQUET)
print(f"Loaded {len(df)} rows")

records = []
for i, row in df.iterrows():
    img_dict = row["image"]
    medicines = row["medicines"]
    raw_bytes = img_dict["bytes"]
    fname = f"chaithanyakota_{i:04d}.jpg"
    out_path = OUT_DIR / fname
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.save(out_path, "JPEG", quality=95)
    except Exception as e:
        print(f"  Row {i} failed: {e}")
        continue
    records.append({"filename": fname, "medicines": medicines})

out_df = pd.DataFrame(records)
out_df.to_csv(CSV_PATH, index=False)
print(f"Extracted {len(records)} images -> {OUT_DIR}")
print(f"CSV -> {CSV_PATH}")
print(out_df.head())