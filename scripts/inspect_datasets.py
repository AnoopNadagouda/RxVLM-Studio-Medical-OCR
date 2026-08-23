#!/usr/bin/env python3
"""Inspect HuggingFace datasets: chinmays, tavishm, eka."""
from datasets import load_from_disk
from pathlib import Path
import json

BASE = Path(r"C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump")

for name, rel_path in [
    ("chinmays", "chinmays/chinmays_rx"),
    ("tavishm", "tavishm/tavishm"),
    ("eka_test", "eka/eka_test"),
    ("eka_mirage_train", "eka/mirage_train"),
    ("eka_mirage_test", "eka/mirage_test"),
]:
    ds_path = BASE / rel_path
    if not ds_path.exists():
        print(f"\n=== {name}: NOT FOUND at {ds_path}")
        continue
    print(f"\n=== {name} ===")
    try:
        ds = load_from_disk(str(ds_path))
        if hasattr(ds, 'keys'):
            for split in ds:
                print(f"  Split: {split}, rows: {len(ds[split])}, cols: {ds[split].column_names}")
                row = ds[split][0]
                for k, v in row.items():
                    if k != 'image':
                        print(f"    {k} = {str(v)[:200]}")
                    else:
                        if isinstance(v, dict):
                            print(f"    image keys = {list(v.keys())}, bytes_len = {len(v.get('bytes', b''))}")
                        else:
                            print(f"    image type = {type(v)}")
                break
        else:
            print(f"  Rows: {len(ds)}, cols: {ds.column_names}")
            row = ds[0]
            for k, v in row.items():
                if k != 'image':
                    print(f"    {k} = {str(v)[:200]}")
                else:
                    if isinstance(v, dict):
                        print(f"    image keys = {list(v.keys())}, bytes_len = {len(v.get('bytes', b''))}")
                    else:
                        print(f"    image type = {type(v)}")
    except Exception as e:
        print(f"  ERROR: {e}")