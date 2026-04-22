#!/bin/bash
# Test 3: Large-Scale Stability & Precision Refinement
# Hardware: RTX 5070 (12GB), 32GB RAM
# Target: Half-dataset scale to improve Precision and verify long-term convergence.

echo "=========================================="
echo "IRC Disentanglement - TEST 3 (1 MILLION PAIRS)"
echo "Target: Precision Refinement & Scale"
echo "Dataset: 1 Million pairs (~15% of full dataset)"
echo "=========================================="

# Ensure output directory exists
mkdir -p checkpoints/test_3

# REASONING FOR TEST 3:
# 1. --test-end 1000000:
#    Increases data volume to 1M pairs.
#    This provides significantly more variety across the 153 training files
#    compared to Test 2, helping the model reduce False Positives (improve Precision).
#
# 2. --batch-size 64:
#    Optimized for RTX 5070 based on Test 2 telemetry (~2.7GB used at BS=16).
#    Linear scaling suggests ~7GB usage, well within 12GB VRAM.
#
# 3. --test-start 300:
#    Continues to skip the "join/quit" noise at the start of logs.
#
# 4. Hyperparameters (from Test 2):
#    Maintains 5e-5 LR and 0.3 threshold which successfully broke the "all-zero" bias.

python src/train.py \
    --mode train \
    --data-dir data \
    --batch-size 64 \
    --num-workers 4 \
    --fp16 \
    --epochs 3 \
    --test-start 300 \
    --test-end 1000000 \
    --learning-rate 5e-5 \
    --max-length 128 \
    --max-dist 30 \
    --warmup-steps 100 \
    --patience 3 \
    --threshold 0.3 \
    --eval-every 1 \
    --save-every 1 \
    --output-dir checkpoints/test_3 \
    --device cuda

echo ""
echo "=========================================="
echo "Test 3 complete!"
echo "Check checkpoints/test_3 for the best model."
echo "=========================================="
