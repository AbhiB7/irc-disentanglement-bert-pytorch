#!/bin/bash
# Script to train and evaluate on data/tiny (real IRC data)
# Designed to run directly on a live Bunya GPU node (no SLURM submission needed)
# Tests self-links exclusion by training on real data with DeBERTa-v3-base
# Follows HPC checklist: set -e, miniconda setup, /scratch output dir

set -e  # Exit immediately on any command failure (HPC checklist)

# Bunya Singularity requirements (even if not using Singularity directly)
export SINGULARITY_CACHEDIR=/scratch/user/$USER
export SINGULARITY_TMPDIR=/scratch/user/$USER

# Repository and output paths
REPO_DIR=$(pwd)  # Assumes script is run from repo root
export RUN_ROOT=/scratch/user/$USER/ircbert_runs
export LOG_DIR=$RUN_ROOT/logs
export CHECKPOINT_DIR=$RUN_ROOT/checkpoints_tiny_test
export TINY_DIR=$REPO_DIR/data/tiny

echo "=== Setting up directories ==="
mkdir -p logs $LOG_DIR $CHECKPOINT_DIR

# Setup conda environment (matches Bunya HPC setup)
echo "=== Setting up environment ==="
source setup.sh  # Loads miniconda3/23.9.0-0, creates/activates env, installs deps

# Train on data/tiny (real IRC data, 300 messages, 212 gold links)
echo ""
echo "=== Starting training on data/tiny (DeBERTa-v3-base, max_dist=20) ==="
python src/train.py \
    --mode train \
    --data-dir $TINY_DIR \
    --model-name microsoft/deberta-v3-base \
    --max-dist 20 \
    --batch-size 16 \
    --epochs 10 \
    --learning-rate 5e-5 \
    --warmup-ratio 0.1 \
    --patience 3 \
    --eval-every 1 \
    --save-every 1 \
    --test-end 300 \
    --output-dir "$CHECKPOINT_DIR" \
    --device cuda

# Evaluate latest checkpoint
echo ""
echo "=== Evaluating on data/tiny/dev ==="
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "WARNING: No checkpoint found in $CHECKPOINT_DIR - skipping evaluation"
else
    echo "Latest checkpoint: $LATEST_CHECKPOINT"
    python src/evaluate.py \
        --checkpoint "$LATEST_CHECKPOINT" \
        --data-dir $TINY_DIR \
        --split dev \
        --test-end 300 \
        --batch-size 16 \
        --metrics both \
        --verbose 3
fi

echo ""
echo "=== Training and evaluation complete ==="
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: $LOG_DIR"