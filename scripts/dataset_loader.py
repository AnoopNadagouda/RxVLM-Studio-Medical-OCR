import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

DEFAULT_PROMPT_TEXT = "<image>\nExtract all prescription text, drug names, and medical details from this image accurately."

def preprocess_image_aspect_ratio(img: Image.Image, max_dim: int = 768, fill_color=(255, 255, 255)) -> Image.Image:
    """
    Dynamic Aspect-Ratio Resizing:
    Scale image so max dimension is 768 while preserving aspect ratio.
    Pad remaining area with neutral background (255, 255, 255).
    """
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (max_dim, max_dim), fill_color)
    scale = max_dim / float(max(w, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (max_dim, max_dim), fill_color)
    paste_x = (max_dim - new_w) // 2
    paste_y = (max_dim - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas

def get_train_transforms():
    """
    Train split augmentation (p=0.3):
    - Light brightness/contrast jitter (+/- 0.1 factor)
    - Minor affine rotation (-5 to +5 deg)
    - NO destructive binarization/thresholding
    """
    return A.Compose([
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0, p=0.3),
        A.Affine(rotate=(-5, 5), p=0.3)
    ])

class MedicineWordDataset(Dataset):
    def __init__(self, manifest_path: str, is_train: bool = True, max_dim: int = 768):
        self.manifest_path = Path(manifest_path)
        self.is_train = is_train
        self.max_dim = max_dim
        self.transform = get_train_transforms() if is_train else None
        
        self.records = []
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        print(f"Loaded {len(self.records)} records from {self.manifest_path} (is_train={is_train})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path_str = rec.get('image') or rec.get('image_path') or ""
        img_path = Path(img_path_str)
        
        if not img_path.exists():
            base_dir = Path("Prescription_Dataset(600)/images/Prescription images.jpg")
            filename = img_path.name
            candidate = base_dir / filename
            if candidate.exists():
                img_path = candidate
            else:
                alt_filename = filename.replace('.jpg', '.jpeg') if filename.endswith('.jpg') else filename.replace('.jpeg', '.jpg')
                alt_candidate = base_dir / alt_filename
                if alt_candidate.exists():
                    img_path = alt_candidate
        
        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            img = Image.new("RGB", (self.max_dim, self.max_dim), (255, 255, 255))
        
        # 1. Dynamic Aspect-Ratio Resizing with Neutral Padding
        img_padded = preprocess_image_aspect_ratio(img, max_dim=self.max_dim)
        
        # 2. Augmentations (Train split only)
        if self.is_train and self.transform is not None:
            img_np = np.array(img_padded)
            augmented = self.transform(image=img_np)['image']
            img_padded = Image.fromarray(augmented)
        # Parse prompt & target text
        prompt = DEFAULT_PROMPT_TEXT
        target_text = rec.get('text', '')
        if not target_text:
            for msg in rec.get('conversations', []):
                if msg.get('from') == 'human':
                    prompt = msg.get('value', prompt)
                elif msg.get('from') == 'gpt':
                    target_text = msg.get('value', '')

        if target_text:
            target_text = target_text.strip()

        return {
            "image": img_padded,
            "prompt": prompt,
            "target_text": target_text,
            "source": rec.get('metadata', {}).get('source', rec.get('type', ''))
        }
