#!/bin/bash
# Test 2: 3-Hour Mid-Range Stability Run
# Hardware: RTX 5070 (12GB), 32GB RAM
# Target: Variety across files, convergence trends, and long-term stability.

echo "=========================================="
echo "IRC Disentanglement - TEST 2 (3 HOURS)"
echo "Target: Multi-file stability & Convergence"
echo "Dataset: Conservative (~50K pairs, ~0.5%)"
echo "=========================================="

# Ensure output directory exists
mkdir -p checkpoints/test_2

# REASONING FOR TEST 2:
# 1. --test-end 50000:
#    Limits TOTAL pairs to 50K (~0.5% of full dataset).
#    Conservative RAM usage while ensuring positive examples.
#    Previous run with --test-end 500 caused data starvation (zero positive examples).
#
# 2. --epochs 3:
#    Standard BERT fine-tuning depth.
#
# 3. --patience 3:
#    Prevents over-training if the model converges early.
#
# 4. --save-every 1 / --eval-every 1:
#    Frequent enough to capture progress but not so frequent to slow down training.

python src/train.py \
    --mode train \
    --data-dir data \
    --batch-size 16 \
    --num-workers 4 \
    --fp16 \
    --epochs 3 \
    --test-start 300 \
    --test-end 50000 \
    --learning-rate 5e-5 \
    --max-length 128 \
    --max-dist 30 \
    --warmup-steps 100 \
    --patience 3 \
    --threshold 0.3 \
    --eval-every 1 \
    --save-every 1 \
    --output-dir checkpoints/test_2 \
    --device cuda

echo ""
echo "=========================================="
echo "Test 2 complete!"
echo "Check checkpoints/test_2 for the best model."
echo "=========================================="
