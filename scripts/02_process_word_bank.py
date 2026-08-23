#!/usr/bin/env python3
"""
RxHandIndia — Word Image Bank Processor
Processes all image sources, normalizes to 512x128 RGB white bg.
Builds the word bank registry JSONL + full_pages registry JSONL.

Data sources:
  Source 1: kalashsh      — ~2506 pre-cropped word images + CSV labels
  Source 2: mehaksingal   — 129 full prescription pages (detect→annotate)
  Source 3: chaithanyakota — 100 full prescription pages + CSV labels (detect→annotate)
  Source 4: avi-kai       — ~46 labeled word images (filenames = labels)
  Source 5: chhavimohitkar — 204 prescription page images (detect→annotate, no labels)
  Source 6: font rendering — synthetic cursive fallback (handled in Stage 6)

Output:
  output/word_bank/*.png               — standardized word images
  output/word_bank_registry.jsonl      — word-level entries
  output/full_pages/*.png|*.jpg         — full-page images awaiting detect→annotate
  output/full_pages_registry.jsonl     — full-page entries with status
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_rxhandindia import process_word_bank, OUT, BASE

if __name__ == "__main__":
    print("=" * 60)
    print("RxHandIndia — Word Image Bank Processor")
    print("=" * 60)
    print(f"DataDump: {BASE}")
    print(f"Output:   {OUT}")
    print()
    print("Delegating to build_rxhandindia.py Stage 3 (process_word_bank)...")
    print()

    registry, full_pages_registry = process_word_bank()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Word-level images:  {len(registry)}")
    print(f"  Full-page images:   {len(full_pages_registry)}")