# Saudi Local Sheep Breed Classification

Fine-grained image classification of local Arabian sheep breeds (Naeimi, Najdi, Harri, Sawakni, Roman, Barbari) using deep learning.

**Research question:** Do self-supervised vision transformers (DINOv2) outperform supervised CNNs (EfficientNet-B3, ConvNeXt-Tiny) for fine-grained livestock classification with limited data?

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce Results

### 1. Download datasets

```bash
# Roboflow (requires free API key from roboflow.com)
python src/download_data.py --source roboflow --api-key YOUR_KEY

# Kaggle (requires ~/.kaggle/kaggle.json)
python src/download_data.py --source kaggle

# Both at once
python src/download_data.py --source all --api-key YOUR_KEY
```

### 2. Preprocess (deduplicate, resize 224×224, stratified split 70/15/15)

```bash
python src/preprocess.py --config configs/config.yaml
```

### 3. Explore the data

```bash
jupyter notebook notebooks/exploration.ipynb
```

### 4. Train

```bash
# Single model
python src/train.py --model convnext --config configs/config.yaml
# convnext | efficientnet | dinov2

# All three models
bash scripts/train_all.sh
```

### 5. Evaluate (confusion matrix + Grad-CAM)

```bash
python src/evaluate.py --model convnext --config configs/config.yaml --split test

# Also plot combined training curves
python src/evaluate.py --model convnext --config configs/config.yaml --split test --curves
```

## Project Structure

```
├── configs/config.yaml          # All hyperparameters
├── data/
│   ├── raw/                     # Downloaded datasets (gitignored)
│   └── processed/               # Cleaned, split data (gitignored)
├── notebooks/exploration.ipynb  # EDA
├── results/
│   ├── checkpoints/             # Saved model weights
│   └── figures/                 # Plots and Grad-CAM images
├── scripts/train_all.sh         # Train + evaluate all models
└── src/
    ├── dataset.py               # PyTorch Dataset + augmentations
    ├── models.py                # Model factory (timm)
    ├── train.py                 # Training loop
    ├── evaluate.py              # Metrics + Grad-CAM
    ├── download_data.py         # Dataset download helpers
    └── preprocess.py            # Dedup + split pipeline
```

## Models

| Key | Architecture | Pretrained on |
|-----|-------------|---------------|
| `efficientnet` | EfficientNet-B3 | ImageNet-1k (supervised) |
| `convnext` | ConvNeXt-Tiny | ImageNet-1k (supervised) |
| `dinov2` | ViT-S/14 | LVD-142M (self-supervised) |

Training strategy: freeze backbone → train head (5 epochs), then full fine-tuning with cosine LR annealing (25 epochs), AdamW, label smoothing 0.1, mixup augmentation.

## Datasets

- [Roboflow sheep-breeds-4ytic](https://universe.roboflow.com/biomedical-engineering-mansoura-university/sheep-breeds-4ytic) — 680 images, MIT license
- [Kaggle sheep-classification-challenge-2025](https://www.kaggle.com/competitions/sheep-classification-challenge-2025) — ~1200 images
