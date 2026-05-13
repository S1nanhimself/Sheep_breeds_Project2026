# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ARTI 560 Computer Vision — Saudi local sheep breed classification (Naeimi, Najdi, Harri, Sawakni, Roman, Barbari).  
Academic deliverable: IEEE-format paper + GitHub repo + presentation.

## Stack

Python · PyTorch · timm · albumentations · scikit-learn · matplotlib

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets (Roboflow + Kaggle)
python src/download_data.py

# Preprocess: deduplicate, resize, split
python src/preprocess.py

# Train a single model (resnet / convnext / dinov2)
python src/train.py --model convnext --config configs/config.yaml

# Evaluate and generate confusion matrix + Grad-CAM
python src/evaluate.py --model convnext --checkpoint results/checkpoints/convnext_best.pth

# Run all three models sequentially
bash scripts/train_all.sh
```

## Architecture

```
ComputervisiorProj/
├── data/
│   ├── raw/              # Downloaded dataset files (gitignored)
│   └── processed/        # Deduplicated, split into train/val/test
├── src/
│   ├── dataset.py        # PyTorch Dataset class + albumentations transforms
│   ├── models.py         # Model factory using timm (EfficientNet-B3, ConvNeXt-Tiny, DINOv2-ViT-S)
│   ├── train.py          # Training loop, LR scheduling, checkpointing
│   ├── evaluate.py       # Metrics (accuracy, per-class F1), confusion matrix, Grad-CAM
│   ├── download_data.py  # Roboflow + Kaggle API download scripts
│   └── preprocess.py     # Dedup (perceptual hash), resize 224×224, stratified split
├── configs/
│   └── config.yaml       # Hyperparameters: LR, batch size, epochs, augmentation flags
├── notebooks/
│   └── exploration.ipynb # EDA, class distribution, sample visualization
├── results/
│   ├── checkpoints/      # Saved model weights (gitignored if large)
│   └── figures/          # Confusion matrices, Grad-CAM outputs, loss curves
└── scripts/
    └── train_all.sh      # Trains all 3 models sequentially
```

## Models

Three models are compared to support the paper's research question — *"Do self-supervised vision transformers outperform supervised CNNs for fine-grained livestock classification with limited data?"*

| Key | `timm` model name | Role |
|-----|-------------------|------|
| `efficientnet` | `efficientnet_b3` | Efficient CNN baseline |
| `convnext` | `convnext_tiny` | Modern CNN (proven on this dataset in Kaggle competition) |
| `dinov2` | `vit_small_patch14_dinov2` | Self-supervised ViT — main novelty angle |

All models are ImageNet-pretrained, fine-tuned with a two-phase strategy: freeze backbone → train head (5 epochs), then unfreeze full model with lower LR (25 epochs).

## Training Details

- Loss: cross-entropy with label smoothing 0.1
- Optimizer: AdamW
- Scheduler: cosine annealing
- Augmentation: horizontal flip, rotation ±15°, color jitter, random erasing, mixup/cutmix
- Input size: 224×224

## Dataset

- **Roboflow** (`sheep-breeds-4ytic`, MIT): 680 images, 6 breeds
- **Kaggle** (`sheep-classification-challenge-2025`): ~1200 images, 7 classes
- Split: 70% train / 15% val / 15% test (stratified per class)

## Novelty / Contribution

The paper's contribution is a comparative study of EfficientNet-B3 vs. ConvNeXt-Tiny vs. DINOv2-ViT-S on a small Arabian sheep breed dataset, with Grad-CAM interpretability analysis showing which visual features (face, wool texture, body shape) each architecture uses to discriminate breeds.
