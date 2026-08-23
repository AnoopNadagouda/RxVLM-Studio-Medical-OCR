import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.preprocessing import normalize
import hdbscan
import umap

BASE = Path(r"C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump")
PROCESSED = BASE / "processed"

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class WordImageDataset(Dataset):
    def __init__(self, image_paths):
        self.paths = image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        rgb = bg.convert("RGB")
        return TRANSFORM(rgb), str(self.paths[idx])


def build_encoder():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(encoder, dataloader, device):
    all_embeddings = []
    all_paths = []
    for batch_imgs, batch_paths in dataloader:
        batch_imgs = batch_imgs.to(device)
        emb = encoder(batch_imgs)
        emb = emb.view(emb.size(0), -1)
        all_embeddings.append(emb.cpu().numpy())
        all_paths.extend(batch_paths)
    return np.vstack(all_embeddings), all_paths


def extract_style_features(image_path):
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)
    w, h = img.size
    total = w * h
    if total == 0:
        return np.zeros(20)
    alpha = arr[:, :, 3]
    opaque = alpha == 255
    n_opaque = int(np.sum(opaque))
    if n_opaque == 0:
        return np.zeros(20)
    ink_ratio = n_opaque / total
    aspect = w / h if h > 0 else 1
    ys, xs = np.where(opaque)
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    bbox_area = bbox_w * bbox_h if bbox_w > 0 and bbox_h > 0 else 1
    density = n_opaque / bbox_area if bbox_area > 0 else 0
    bbox_aspect = bbox_w / bbox_h if bbox_h > 0 else 1
    cx = np.mean(xs) / w if w > 0 else 0.5
    cy = np.mean(ys) / h if h > 0 else 0.5
    spread_x = np.std(xs) / w if w > 0 else 0
    spread_y = np.std(ys) / h if h > 0 else 0
    row_inks = np.sum(opaque, axis=1) / w
    col_inks = np.sum(opaque, axis=0) / h
    row_mean = float(np.mean(row_inks))
    row_std = float(np.std(row_inks))
    col_mean = float(np.mean(col_inks))
    col_std = float(np.std(col_inks))
    h_runs = []
    for r in range(h):
        row_alpha = alpha[r, :]
        changes = np.diff(row_alpha.astype(np.int8))
        starts = np.where((changes == 1) & (row_alpha[:-1] == 0))[0] + 1
        if row_alpha[0] == 255:
            starts = np.concatenate([[0], starts])
        ends = np.where((changes == -1) & (row_alpha[:-1] == 255))[0] + 1
        if row_alpha[-1] == 255:
            ends = np.concatenate([ends, [w]])
        for s, e in zip(starts, ends):
            h_runs.append(int(e - s))
    v_runs = []
    for c in range(w):
        col_alpha = alpha[:, c]
        changes = np.diff(col_alpha.astype(np.int8))
        starts = np.where((changes == 1) & (col_alpha[:-1] == 0))[0] + 1
        if col_alpha[0] == 255:
            starts = np.concatenate([[0], starts])
        ends = np.where((changes == -1) & (col_alpha[:-1] == 255))[0] + 1
        if col_alpha[-1] == 255:
            ends = np.concatenate([ends, [h]])
        for s, e in zip(starts, ends):
            v_runs.append(int(e - s))
    h_run_mean = float(np.mean(h_runs)) if h_runs else 0
    h_run_std = float(np.std(h_runs)) if h_runs else 0
    v_run_mean = float(np.mean(v_runs)) if v_runs else 0
    v_run_std = float(np.std(v_runs)) if v_runs else 0
    features = [
        ink_ratio, aspect, density, bbox_aspect,
        cx, cy, spread_x, spread_y,
        row_mean, row_std, col_mean, col_std,
        h_run_mean, h_run_std, v_run_mean, v_run_std,
        bbox_w / 100, bbox_h / 100, w / 100, h / 100,
    ]
    return np.array(features, dtype=np.float32)


def find_clusters(embeddings, style_feats, min_cluster_size, min_samples):
    normed_cnn = normalize(embeddings)
    style_normed = normalize(style_feats)
    combined = np.hstack([normed_cnn * 0.6, style_normed * 0.4])
    combined = normalize(combined)
    print(f"Combined features shape: {combined.shape}")
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=15,
        n_neighbors=10,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(combined)
    print(f"UMAP output shape: {reduced.shape}")
    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rxhandbd", "kaggle_bd_rx", "all"], default="all")
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    datasets = []
    if args.dataset in ("rxhandbd", "all"):
        for split in ["Training", "Validation", "Testing"]:
            d = PROCESSED / "rxhandbd" / split
            if d.exists():
                datasets.append(d)
    if args.dataset in ("kaggle_bd_rx", "all"):
        for split in ["Train", "Test"]:
            d = PROCESSED / "kaggle_bd_rx" / split
            if d.exists():
                datasets.append(d)

    all_paths = []
    for d in datasets:
        all_paths.extend(sorted(d.glob("*.png")))

    print(f"Found {len(all_paths)} images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = build_encoder().to(device)
    dataset = WordImageDataset(all_paths)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    print("Extracting embeddings...")
    embeddings, paths = extract_embeddings(encoder, dataloader, device)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Extracting style features...")
    style_feats = np.vstack([extract_style_features(p) for p in all_paths])
    print(f"Style features shape: {style_feats.shape}")

    print("Clustering...")
    labels = find_clusters(embeddings, style_feats, args.min_cluster_size, args.min_samples)

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    print(f"Clusters: {n_clusters}, Noise: {n_noise}")

    results = {}
    for path, label in zip(paths, labels):
        p = Path(path)
        rel = p.relative_to(PROCESSED)
        parts = rel.parts
        ds = parts[0]
        split = parts[1]
        fname = parts[2]
        key = f"{ds}/{split}/{fname}"
        results[key] = int(label)

    out_file = BASE / "cluster_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")

    from collections import Counter
    cluster_counts = Counter(labels)
    print("\nCluster sizes (top 20):")
    for cid, count in cluster_counts.most_common(20):
        tag = "noise" if cid == -1 else f"cluster_{cid}"
        print(f"  {tag}: {count}")

    out_dir = BASE / "clustered"
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)

    import shutil as sh
    counter = {}
    for path_str, label in zip(paths, labels):
        if label == -1:
            dest_dir = BASE / "clustered" / "noise"
        else:
            dest_dir = BASE / "clustered" / f"cluster_{label:03d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = Path(path_str)
        stem = src.stem
        key = (dest_dir, stem)
        counter[key] = counter.get(key, 0) + 1
        n = counter[key]
        dest_name = f"{stem}_{n}{src.suffix}" if n > 1 else src.name
        sh.copy2(src, dest_dir / dest_name)

    print(f"\nCopied to {out_dir}")


if __name__ == "__main__":
    main()