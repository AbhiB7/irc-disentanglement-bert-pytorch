#!/bin/bash
# Full-scale training script for IRC Disentanglement
# Optimized for GTX 1080 Ti (11GB VRAM) - Vast.ai 15-Hour Overnight Run
# Hardware: 1x 1080 Ti, Xeon E5-2660 v4, 64GB RAM

echo "=========================================="
echo "IRC Disentanglement - Vast.ai Overnight Run"
echo "Target Hardware: GTX 1080 Ti (11GB)"
echo "Estimated Window: 15 Hours"
echo "=========================================="

# Ensure output directory exists
mkdir -p checkpoints/vast_overnight

# Training parameters:
# --batch-size 32: Safe for 11GB VRAM with BERT-base
# --epochs 10: Sufficient for 15-hour window (likely 2-4 full epochs)
# --patience 3: Early stopping to save time if model converges early
# --max-dist 30: User-specified distance window
# --learning-rate 5e-5: Stable rate for BERT fine-tuning
# --warmup-steps 1000: Increased for the larger pair count (approx 2M pairs/epoch)
# --threshold 0.3: Optimized for recall on imbalanced links

python src/train.py \
    --mode train \
    --batch-size 32 \
    --epochs 10 \
    --learning-rate 5e-5 \
    --max-length 128 \
    --max-dist 30 \
    --warmup-steps 1000 \
    --patience 3 \
    --threshold 0.3 \
    --eval-every 1 \
    --save-every 1 \
    --output-dir checkpoints/vast_overnight \
    --device cuda

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoints saved to: checkpoints/vast_overnight"
echo "=========================================="
