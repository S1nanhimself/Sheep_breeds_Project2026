#!/usr/bin/env bash
# Train all three models sequentially, then evaluate each on the test set
# and plot combined training curves.
#
# Usage:
#   bash scripts/train_all.sh
#   bash scripts/train_all.sh --config configs/config.yaml

set -euo pipefail

CONFIG="${2:-configs/config.yaml}"
cd "$(dirname "$0")/.."   # run from project root

echo "========================================"
echo "Sheep Breed Classification — Training"
echo "Config: $CONFIG"
echo "========================================"

for MODEL in efficientnet convnext dinov2; do
    echo ""
    echo ">>> Training: $MODEL"
    python src/train.py --model "$MODEL" --config "$CONFIG"

    echo ">>> Evaluating: $MODEL"
    python src/evaluate.py --model "$MODEL" --config "$CONFIG" --split test
done

echo ""
echo ">>> Plotting combined training curves"
python src/evaluate.py --model efficientnet --config "$CONFIG" --split test --curves

echo ""
echo "All done. Results in results/figures/ and results/checkpoints/"
