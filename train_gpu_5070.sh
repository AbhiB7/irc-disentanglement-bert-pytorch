#!/bin/bash
# Training script optimized for RTX 5070 (12GB VRAM) and 32GB RAM
# Hardware: RTX 5070, Xeon E5-2696, 32GB RAM

echo "=========================================="
echo "IRC Disentanglement - RTX 5070 Run"
echo "Target Hardware: RTX 5070 (12GB), 32GB RAM"
echo "=========================================="

# Ensure output directory exists
mkdir -p checkpoints/gpu_5070

# REASONING FOR PARAMETERS:
# 1. --batch-size 32: 
#    The previous run used 64, which likely caused a GPU Out-of-Memory (OOM) error on 12GB VRAM.
#    32 is a safer balance for BERT-base to ensure stability while maintaining throughput.
#
# 2. --num-workers 2:
#    The dataset contains ~5.8 million pairs. PyTorch workers can consume significant System RAM.
#    Reducing from 4 to 2 helps prevent the OS from killing the process due to System RAM exhaustion (32GB limit).
#
# 3. --fp16:
#    Uses Mixed Precision training. This is critical for the RTX 5070 to utilize Tensor Cores,
#    providing ~2x speedup and reducing VRAM footprint.
#
# 4. --warmup-steps 1000:
#    Helps stabilize the initial gradients given the very large number of training pairs.

python src/train.py \
    --mode train \
    --batch-size 32 \
    --num-workers 2 \
    --fp16 \
    --epochs 5 \
    --learning-rate 5e-5 \
    --max-length 128 \
    --max-dist 30 \
    --warmup-steps 1000 \
    --patience 3 \
    --threshold 0.3 \
    --eval-every 1 \
    --save-every 1 \
    --output-dir checkpoints/gpu_5070 \
    --device cuda

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoints saved to: checkpoints/gpu_5070"
echo "=========================================="
