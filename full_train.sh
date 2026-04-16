#!/bin/bash
# Full-scale training script for IRC Disentanglement
# Optimized for RTX 4090 (24GB VRAM)
# Estimated duration: ~1-2 hours (5 epochs)

echo "=========================================="
echo "IRC Disentanglement - Full Training (4090 Optimized)"
echo "=========================================="
echo ""

# Training parameters explained:
# --mode train: Use full training dataset
# --batch-size 64: Optimized for 24GB VRAM (BERT-base @ 128 seq len)
# --epochs 5: Standard training length
# --learning-rate 2e-5: Standard for BERT fine-tuning
# --max-length 128: Standard BERT sequence length
# --max-dist 101: Consider previous 100 messages for linking
# --warmup-steps 100: Standard for linear warmup scheduler
# --eval-every 1: Evaluate on dev set every epoch
# --save-every 1: Save checkpoint every epoch
# --output-dir: Organized output directory

echo "Starting full training run..."
echo "Batch size: 64"
echo ""

# Ensure output directory exists
mkdir -p checkpoints/full_train

python src/train.py \
    --mode train \
    --batch-size 64 \
    --epochs 5 \
    --learning-rate 2e-5 \
    --max-length 128 \
    --max-dist 101 \
    --warmup-steps 100 \
    --eval-every 1 \
    --save-every 1 \
    --output-dir checkpoints/full_train \
    --device cuda

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoints saved to: checkpoints/full_train"
echo "=========================================="
