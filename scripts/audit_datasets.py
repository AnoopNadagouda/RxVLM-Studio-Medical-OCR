import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SKIP_DIRS = {'__pycache__', 'checkpoints', 'reports', 'models', 'ocr_manifests', 'fonts', 'docker-data', 'node_modules', '.git', '.opencode', 'Archive', 'lib'}

def audit_directory(dir_path: Path):
    img_count = 0
    ann_files = []
    
    for root, dirs, files in os.walk(dir_path):
        # Prune skip dirs
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in IMAGE_EXTS:
                img_count += 1
            elif ext in {'.json', '.jsonl', '.csv', '.txt', '.xml'}:
                ann_files.append(f)
                
    return img_count, len(ann_files), set(Path(f).suffix.lower() for f in ann_files)

def main():
    print("=" * 85)
    print("DATADUMP UNIVERSAL DATASET DIRECTORY AUDIT")
    print("=" * 85)
    
    target_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith('.')]
    
    print(f"{'Folder Name':<32} | {'Total Images':<14} | {'Ann Files':<10} | {'Formats Found':<15} | {'Status'}")
    print("-" * 85)
    
    total_images_all = 0
    
    for sd in sorted(target_dirs, key=lambda x: x.name.lower()):
        img_count, ann_count, formats = audit_directory(sd)
        total_images_all += img_count
        fmt_str = ", ".join(formats) if formats else "None"
        status = "Included" if img_count > 0 else "No Images"
        print(f"{sd.name:<32} | {img_count:<14} | {ann_count:<10} | {fmt_str:<15} | {status}")
        
    print("-" * 85)
    print(f"Total Images Discovered Across Workspace: {total_images_all}")
    print("=" * 85)

if __name__ == "__main__":
    main()
