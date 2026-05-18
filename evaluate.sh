#!/bin/bash
# Evaluation script for direct GPU node execution (no SLURM)
# Usage: bash evaluate_2.sh [checkpoint_path] [verbose_count]

set -e

# Default checkpoint (epoch 6 from most recent training run)
if [ -z "$1" ]; then
    CHECKPOINT="/scratch/user/$USER/ircbert_runs/checkpoints/checkpoint_epoch_6.pt"
else
    CHECKPOINT="$1"
fi

# Default verbose count = 3
if [ -z "$2" ]; then
    VERBOSE="--verbose 3"
else
    VERBOSE="--verbose $2"
fi

# Create logs directory
export RUN_ROOT=/scratch/user/$USER/ircbert_runs
export LOG_DIR=$RUN_ROOT/logs
mkdir -p logs $LOG_DIR

echo "=========================================="
echo "IRC Disentanglement - Checkpoint Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Verbose: $VERBOSE"
echo ""

# Run setup (handles conda env creation + activation + dependency installation)
# Use 'source' so conda activation and env vars persist in the current shell
source setup.sh

echo ""
echo "=== Evaluating on DEV set ==="
python src/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --split dev \
    --batch-size 64 \
    --max-dist 50 \
    --metrics both \
    $VERBOSE \
    --verbose-seed 42 2>&1 | tee -a logs/eval_dev_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=== Evaluating on TEST set ==="
python src/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --split test \
    --batch-size 64 \
    --max-dist 50 \
    --metrics both \
    $VERBOSE \
    --verbose-seed 42 2>&1 | tee -a logs/eval_test_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Logs saved to: logs/"
echo "=========================================="