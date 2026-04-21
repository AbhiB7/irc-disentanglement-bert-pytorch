#!/bin/bash
# Test 1: 5-Minute Stability & Logic Check
# Hardware: RTX 5070 (12GB), 32GB RAM

echo "=========================================="
echo "IRC Disentanglement - TEST 1 (5 MIN)"
echo "Target: Stability, OOM Logging, Positive Bias"
echo "=========================================="

# Ensure output directory exists
mkdir -p checkpoints/test_1

# REASONING FOR TEST 1:
# 1. --mode train:
#    Exercises the full pipeline (forward + backward pass) to test OOM/NaN logging.
#
# 2. --test-start 0 --test-end 5000:
#    Limits the number of pairs to ensure the run completes in ~5 minutes.
#    Starting at 0 ensures we include the beginning of the dialogue where 
#    initial connections are established.
#
# 3. --batch-size 16:
#    Lower batch size to test the new OOM logging and ensure stability.
#
# 4. --epochs 1:
#    Single pass is enough for a logic check.

python src/train.py \
    --mode train \
    --data-dir data \
    --batch-size 16 \
    --num-workers 0 \
    --fp16 \
    --epochs 1 \
    --learning-rate 5e-5 \
    --max-length 128 \
    --max-dist 30 \
    --warmup-steps 10 \
    --patience 0 \
    --threshold 0.3 \
    --eval-every 1 \
    --save-every 1 \
    --output-dir checkpoints/test_1 \
    --device cuda \
    --test-start 0 \
    --test-end 5000

echo ""
echo "=========================================="
echo "Test 1 complete!"
echo "Check logs for OOM reports and positive sample stats."
echo "Note: Test 2 (1hr) and Test 3 (3hr) are planned for next stages."
echo "=========================================="
