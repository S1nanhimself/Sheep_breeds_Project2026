import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


CLASSES = ["Naeimi", "Najdi", "Harri", "Sawakni", "Roman", "Barbari"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def get_transforms(split: str, image_size: int, cfg: dict) -> A.Compose:
    aug = cfg.get("augmentation", {})
    cj = aug.get("color_jitter", {})

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        transforms = [
            A.Resize(image_size, image_size),
        ]
        if aug.get("horizontal_flip", True):
            transforms.append(A.HorizontalFlip(p=0.5))
        if aug.get("rotation_degrees", 0):
            transforms.append(A.Rotate(limit=aug["rotation_degrees"], p=0.5))
        if cj:
            transforms.append(A.ColorJitter(
                brightness=cj.get("brightness", 0.3),
                contrast=cj.get("contrast", 0.3),
                saturation=cj.get("saturation", 0.3),
                hue=cj.get("hue", 0.1),
                p=0.5,
            ))
        transforms += [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
        if aug.get("random_erasing_prob", 0) > 0:
            # Applied after ToTensor on the tensor side in collate
            pass
    else:
        transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]

    return A.Compose(transforms)


class SheepDataset(Dataset):
    """
    Expects processed data directory with structure:
        split/ClassName/image.jpg
    """

    def __init__(self, root: str, split: str, cfg: dict):
        self.root = Path(root) / split
        self.cfg = cfg
        image_size = cfg.get("data", {}).get("image_size", 224)
        self.transform = get_transforms(split, image_size, cfg)
        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        for cls_name in CLASSES:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            idx = CLASS_TO_IDX[cls_name]
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                for img_path in cls_dir.glob(ext):
                    self.samples.append((img_path, idx))
        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root}. "
                "Run src/preprocess.py first."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        augmented = self.transform(image=image)
        return augmented["image"], label

    def class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in CLASSES}
        for _, idx in self.samples:
            counts[CLASSES[idx]] += 1
        return counts


def mixup_collate(alpha: float):
    """Returns a collate_fn that applies mixup to a batch."""

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        if alpha <= 0 or not torch.is_grad_enabled():
            return images, labels

        lam = np.random.beta(alpha, alpha)
        batch_size = images.size(0)
        perm = torch.randperm(batch_size)

        mixed = lam * images + (1 - lam) * images[perm]
        labels_a, labels_b = labels, labels[perm]
        return mixed, (labels_a, labels_b, lam)

    return collate_fn


def build_dataloaders(cfg: dict):
    processed_dir = cfg["data"]["processed_dir"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 4)
    mixup_alpha = cfg["augmentation"].get("mixup_alpha", 0.0)

    train_ds = SheepDataset(processed_dir, "train", cfg)
    val_ds   = SheepDataset(processed_dir, "val",   cfg)
    test_ds  = SheepDataset(processed_dir, "test",  cfg)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=mixup_collate(mixup_alpha),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
