"""
Train a single model.

Usage:
    python src/train.py --model convnext --config configs/config.yaml
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset import build_dataloaders, CLASSES
from models import (
    build_model,
    freeze_backbone,
    unfreeze_all,
    get_optimizer,
    get_scheduler,
    LabelSmoothingCrossEntropy,
    SUPPORTED_MODELS,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets) -> float:
    preds = logits.argmax(dim=1)
    if isinstance(targets, tuple):
        labels_a, labels_b, lam = targets
        correct = lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()
        return correct.mean().item()
    return (preds == targets).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, scheduler, device, train: bool):
    model.train(train)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for images, targets in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            if isinstance(targets, tuple):
                targets = tuple(
                    t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                    for t in targets
                )
            else:
                targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            total_acc  += accuracy(logits, targets)
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


def train(model_key: str, cfg: dict):
    set_seed(cfg["training"]["seed"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    num_classes = len(CLASSES)

    model = build_model(model_key, cfg, num_classes=num_classes)
    model = model.to(device)

    criterion = LabelSmoothingCrossEntropy(
        smoothing=cfg["training"].get("label_smoothing", 0.1),
        num_classes=num_classes,
    )

    checkpoints_dir = Path(cfg["output"]["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    freeze_epochs = cfg["training"].get("freeze_epochs", 5)
    total_epochs  = cfg["training"]["epochs"]

    # ---- Phase 1: frozen backbone ----
    freeze_backbone(model)
    optimizer = get_optimizer(model, phase=1, cfg=cfg)
    scheduler = get_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    print(f"\n=== Phase 1: Training head ({freeze_epochs} epochs) ===")
    for epoch in range(1, freeze_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, scheduler, device, train=False)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:3d}/{freeze_epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {vl_loss:.4f} acc {vl_acc:.4f} | {elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), checkpoints_dir / f"{model_key}_best.pth")

    # ---- Phase 2: full fine-tuning ----
    unfreeze_all(model)
    optimizer = get_optimizer(model, phase=2, cfg=cfg)
    scheduler = get_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    print(f"\n=== Phase 2: Full fine-tuning ({total_epochs - freeze_epochs} epochs) ===")
    for epoch in range(freeze_epochs + 1, total_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, scheduler, device, train=False)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:3d}/{total_epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {vl_loss:.4f} acc {vl_acc:.4f} | {elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), checkpoints_dir / f"{model_key}_best.pth")

    # Save training history
    history_path = checkpoints_dir / f"{model_key}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(f"Checkpoint saved to {checkpoints_dir / f'{model_key}_best.pth'}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(args.model, cfg)


if __name__ == "__main__":
    main()
