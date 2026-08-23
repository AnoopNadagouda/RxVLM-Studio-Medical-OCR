import argparse
import re
from pathlib import Path
from csv import DictReader
from PIL import Image


BASE = Path(r"C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump")
OUTPUT = BASE / "processed"

_rxhandbd_outer = BASE / "rxhandbd" / "doctors-handwritten-prescription-bd-dataset"
RXHANDBD_BASE = next(_rxhandbd_outer.iterdir())
RXHANDBD_SPLITS = {
    "Training": {"images": "Training/training_words", "labels": "Training/training_labels.csv"},
    "Validation": {"images": "Validation/validation_words", "labels": "Validation/validation_labels.csv"},
    "Testing": {"images": "Testing/testing_words", "labels": "Testing/testing_labels.csv"},
}

KAGGLE_BASE = BASE / "kaggle_bd_rx"
KAGGLE_SPLITS = {
    "Train": {"images": "Train_Set", "labels": "Train_Label.csv"},
    "Test": {"images": "Test_Set", "labels": "Test_Labels.csv"},
}


def sanitize_label(label):
    s = label.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s


def otsu_threshold(histogram, total):
    sum_total = sum(i * histogram[i] for i in range(256))
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    best = 0
    for t in range(256):
        weight_bg += histogram[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * histogram[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            best = t
    return best


def remove_white_bg(img, threshold):
    gray = img.convert("L")
    data = list(gray.getdata())
    histogram = [0] * 256
    for b in data:
        histogram[b] += 1
    t = otsu_threshold(histogram, len(data))
    below = sum(histogram[i] for i in range(t + 1))
    above = len(data) - below
    invert = below > above
    new_data = []
    for brightness in data:
        if invert:
            if brightness <= t:
                new_data.append((0, 0, 0, 0))
            else:
                new_data.append((0, 0, 0, 255))
        else:
            if brightness > t:
                new_data.append((0, 0, 0, 0))
            else:
                new_data.append((0, 0, 0, 255))
    out = Image.new("RGBA", img.size)
    out.putdata(new_data)
    return out


def unique_path(directory, stem, suffix):
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def process_rxhandbd(split_name, threshold):
    split_info = RXHANDBD_SPLITS[split_name]
    img_dir = RXHANDBD_BASE / split_info["images"]
    label_csv = RXHANDBD_BASE / split_info["labels"]
    out_dir = OUTPUT / "rxhandbd" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    dupes = 0
    missing = 0

    with open(label_csv, newline="", encoding="utf-8") as f:
        reader = DictReader(f)
        for row in reader:
            filename = row["IMAGE"]
            label = row["MEDICINE_NAME"]
            img_path = img_dir / filename
            if not img_path.exists():
                print(f"WARNING: missing image {img_path}")
                missing += 1
                continue
            img = Image.open(img_path)
            img = remove_white_bg(img, threshold)
            stem = sanitize_label(label)
            dest = unique_path(out_dir, stem, ".png")
            if dest.name != f"{stem}.png":
                dupes += 1
            img.save(dest, "PNG")
            count += 1
            if count % 100 == 0:
                print(f"  [{split_name}] processed {count} images")

    return count, dupes, missing


def process_kaggle(split_name, threshold):
    split_info = KAGGLE_SPLITS[split_name]
    img_dir = KAGGLE_BASE / split_info["images"]
    label_csv = KAGGLE_BASE / split_info["labels"]
    out_dir = OUTPUT / "kaggle_bd_rx" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    dupes = 0
    missing = 0

    with open(label_csv, newline="", encoding="utf-8") as f:
        reader = DictReader(f)
        for row in reader:
            filename = row["Images"]
            label = row["Text"]
            img_path = img_dir / filename
            if not img_path.exists():
                print(f"WARNING: missing image {img_path}")
                missing += 1
                continue
            img = Image.open(img_path)
            img = remove_white_bg(img, threshold)
            stem = sanitize_label(label)
            dest = unique_path(out_dir, stem, ".png")
            if dest.name != f"{stem}.png":
                dupes += 1
            img.save(dest, "PNG")
            count += 1
            if count % 100 == 0:
                print(f"  [{split_name}] processed {count} images")

    return count, dupes, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rxhandbd", "kaggle_bd_rx", "all"], default="all")
    parser.add_argument("--split", default="all")
    parser.add_argument("--threshold", type=int, default=200)
    args = parser.parse_args()

    total_count = 0
    total_dupes = 0
    total_missing = 0

    datasets = []
    if args.dataset in ("rxhandbd", "all"):
        datasets.append(("rxhandbd", RXHANDBD_SPLITS, process_rxhandbd))
    if args.dataset in ("kaggle_bd_rx", "all"):
        datasets.append(("kaggle_bd_rx", KAGGLE_SPLITS, process_kaggle))

    for ds_name, splits, processor in datasets:
        splits_to_run = list(splits.keys()) if args.split == "all" else [args.split]
        for sp in splits_to_run:
            if sp not in splits:
                print(f"WARNING: split '{sp}' not found in {ds_name}, skipping")
                continue
            print(f"Processing {ds_name} / {sp} ...")
            c, d, m = processor(sp, args.threshold)
            total_count += c
            total_dupes += d
            total_missing += m

    print(f"\nDone. Processed: {total_count}, Duplicates handled: {total_dupes}, Missing: {total_missing}")


if __name__ == "__main__":
    main()