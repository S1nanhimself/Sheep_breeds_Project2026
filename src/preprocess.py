"""
Deduplicate, normalise class names, resize, and split raw images into
data/processed/{train,val,test}/ClassName/

Usage:
    python src/preprocess.py --config configs/config.yaml
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import imagehash
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Map raw folder names (various capitalisations / typos) to canonical names
NAME_MAP = {
    "naeimi":  "Naeimi",
    "najdi":   "Najdi",
    "harri":   "Harri",
    "sawakni": "Sawakni",
    "roman":   "Roman",
    "barbari": "Barbari",
    "barbari goat": "Barbari",   # Roboflow sometimes labels this way
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def canonical_class(folder_name: str) -> str | None:
    return NAME_MAP.get(folder_name.lower().strip())


def collect_images(raw_dir: Path) -> dict[str, list[Path]]:
    """Walk raw_dir recursively; map folder name → list of image paths."""
    class_images: dict[str, list[Path]] = defaultdict(list)

    for img_path in raw_dir.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # The containing folder is the class label
        cls = canonical_class(img_path.parent.name)
        if cls is None:
            # Try one level up (some datasets nest train/val subfolders)
            cls = canonical_class(img_path.parent.parent.name)
        if cls is None:
            continue
        class_images[cls].append(img_path)

    return class_images


def deduplicate(paths: list[Path], hash_size: int = 8) -> list[Path]:
    """Remove perceptual duplicates using average hash."""
    seen: set[str] = set()
    unique: list[Path] = []
    for p in tqdm(paths, desc="Deduplicating", leave=False):
        try:
            h = str(imagehash.average_hash(Image.open(p), hash_size=hash_size))
        except Exception:
            continue
        if h not in seen:
            seen.add(h)
            unique.append(p)
    return unique


def resize_and_copy(src: Path, dst: Path, size: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(dst, quality=95)
    except Exception as e:
        print(f"  Warning: could not process {src}: {e}")


def preprocess(cfg: dict):
    raw_dir       = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    image_size    = cfg["data"]["image_size"]
    train_split   = cfg["data"]["train_split"]
    val_split     = cfg["data"]["val_split"]
    seed          = cfg["training"]["seed"]

    print(f"Scanning {raw_dir} for images...")
    class_images = collect_images(raw_dir)

    if not class_images:
        raise FileNotFoundError(
            f"No labelled images found in {raw_dir}.\n"
            "Run src/download_data.py first."
        )

    print("\nClass counts before dedup:")
    for cls, paths in sorted(class_images.items()):
        print(f"  {cls}: {len(paths)}")

    # Deduplicate per class
    deduped: dict[str, list[Path]] = {}
    for cls, paths in class_images.items():
        deduped[cls] = deduplicate(paths)

    print("\nClass counts after dedup:")
    total = 0
    for cls, paths in sorted(deduped.items()):
        print(f"  {cls}: {len(paths)}")
        total += len(paths)
    print(f"  TOTAL: {total}")

    # Stratified split
    all_paths, all_labels = [], []
    for cls, paths in deduped.items():
        all_paths.extend(paths)
        all_labels.extend([cls] * len(paths))

    test_size = 1.0 - train_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=seed,
    )
    # Split temp into val / test with correct ratio
    val_ratio = val_split / (val_split + cfg["data"]["test_split"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1.0 - val_ratio,
        stratify=y_temp,
        random_state=seed,
    )

    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

    print("\nSplit sizes:")
    for split, (paths, _) in splits.items():
        print(f"  {split}: {len(paths)}")

    # Copy resized images to processed dir
    for split, (paths, labels) in splits.items():
        print(f"\nProcessing {split} set...")
        for src, cls in tqdm(zip(paths, labels), total=len(paths)):
            dst = processed_dir / split / cls / src.name
            # Avoid filename collisions across datasets
            if dst.exists():
                dst = dst.with_stem(dst.stem + "_1")
            resize_and_copy(src, dst, image_size)

    print(f"\nDone. Processed data written to {processed_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    preprocess(cfg)


if __name__ == "__main__":
    main()
