"""
Evaluate a trained model on the test set.
Produces: confusion matrix, per-class F1, Grad-CAM samples.

Usage:
    python src/evaluate.py --model convnext --config configs/config.yaml
    python src/evaluate.py --model convnext --config configs/config.yaml --split val
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import math

from dataset import SheepDataset, CLASSES, get_transforms
from models import build_model, SUPPORTED_MODELS

from PIL import Image


def get_gradcam_target_layer(model, model_key: str):
    """Return the last conv/attention layer for Grad-CAM."""
    if model_key == "efficientnet":
        return [model.conv_head]
    elif model_key == "convnext":
        return [model.stages[-1].blocks[-1].conv_dw]
    elif model_key == "dinov2":
        return [model.blocks[-1].norm1]
    raise ValueError(f"Unknown model key: {model_key}")


def get_reshape_transform(model_key: str, image_size: int):
    """ViT outputs [B, N+1, C] — reshape patch tokens back to spatial grid."""
    if model_key != "dinov2":
        return None
    patch_size = 14
    h = w = image_size // patch_size

    def reshape_transform(tensor):
        # Remove CLS token, reshape to [B, C, H, W]
        result = tensor[:, 1:, :].reshape(tensor.size(0), h, w, tensor.size(2))
        return result.transpose(2, 3).transpose(1, 2)

    return reshape_transform


def evaluate(model_key: str, cfg: dict, split: str = "test"):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    num_classes = len(CLASSES)

    model = build_model(model_key, cfg, num_classes=num_classes)
    ckpt_path = Path(cfg["output"]["checkpoints_dir"]) / f"{model_key}_best.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    processed_dir = cfg["data"]["processed_dir"]
    # Use model-specific input size (DINOv2 requires 518, others 224)
    model_input_size = cfg["models"][model_key].get("input_size", cfg["data"]["image_size"])
    cfg = {**cfg, "data": {**cfg["data"], "image_size": model_input_size}}
    image_size = model_input_size
    dataset = SheepDataset(processed_dir, split, cfg)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=cfg["data"].get("num_workers", 4),
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating {model_key} on {split}"):
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(
        all_labels, all_preds, target_names=CLASSES, output_dict=True
    )
    print(f"\n=== {model_key} — {split} set ===")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    figures_dir = Path(cfg["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"{model_key} — Confusion Matrix ({split})")
    plt.tight_layout()
    cm_path = figures_dir / f"{model_key}_confusion_matrix_{split}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Save metrics JSON
    metrics_path = figures_dir / f"{model_key}_metrics_{split}.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    # Grad-CAM
    _save_gradcam(model, model_key, dataset, device, cfg, figures_dir, split)

    return report


def _save_gradcam(model, model_key, dataset, device, cfg, figures_dir, split, n_samples=6):
    try:
        target_layers = get_gradcam_target_layer(model, model_key)
    except Exception as e:
        print(f"Grad-CAM layer lookup failed: {e}")
        return

    reshape_transform = get_reshape_transform(model_key, cfg["data"]["image_size"])
    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape_transform)
    image_size = cfg["data"]["image_size"]

    # Pick one sample per class
    class_samples: dict[int, tuple] = {}
    for img_path, label in dataset.samples:
        if label not in class_samples:
            class_samples[label] = (img_path, label)
        if len(class_samples) == len(CLASSES):
            break

    fig, axes = plt.subplots(2, len(class_samples), figsize=(3 * len(class_samples), 6))

    for col, (label, (img_path, _)) in enumerate(sorted(class_samples.items())):
        raw_img = np.array(Image.open(img_path).convert("RGB").resize((image_size, image_size)))
        raw_float = raw_img.astype(np.float32) / 255.0

        transform = get_transforms("val", image_size, cfg)
        tensor = transform(image=raw_img)["image"].unsqueeze(0).to(device)

        targets = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
        cam_image = show_cam_on_image(raw_float, grayscale_cam, use_rgb=True)

        axes[0, col].imshow(raw_img)
        axes[0, col].set_title(CLASSES[label], fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(cam_image)
        axes[1, col].set_title("Grad-CAM", fontsize=9)
        axes[1, col].axis("off")

    plt.suptitle(f"{model_key} — Grad-CAM Visualisations ({split})", fontsize=11)
    plt.tight_layout()
    cam_path = figures_dir / f"{model_key}_gradcam_{split}.png"
    plt.savefig(cam_path, dpi=150)
    plt.close()
    print(f"Grad-CAM saved to {cam_path}")


def plot_training_curves(cfg: dict):
    """Plot loss/accuracy curves for all trained models."""
    figures_dir = Path(cfg["output"]["figures_dir"])
    checkpoints_dir = Path(cfg["output"]["checkpoints_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_key in SUPPORTED_MODELS:
        history_path = checkpoints_dir / f"{model_key}_history.json"
        if not history_path.exists():
            continue
        with open(history_path) as f:
            h = json.load(f)
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["val_loss"], label=model_key)
        axes[1].plot(epochs, h["val_acc"],  label=model_key)

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    out = figures_dir / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Training curves saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split",  default="test", choices=["val", "test"])
    parser.add_argument("--curves", action="store_true",
                        help="Also plot training curves for all models")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(args.model, cfg, split=args.split)

    if args.curves:
        plot_training_curves(cfg)


if __name__ == "__main__":
    main()
