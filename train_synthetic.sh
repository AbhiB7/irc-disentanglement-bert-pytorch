#!/bin/bash
# Script to train and evaluate on synthetic interleaved data
# Designed to run directly on a live Bunya GPU node (no SLURM submission needed)
# Follows HPC checklist: set -e, miniconda setup, /scratch output dir

set -e  # Exit immediately on any command failure (HPC checklist)

# Bunya Singularity requirements (even if not using Singularity directly)
export SINGULARITY_CACHEDIR=/scratch/user/$USER
export SINGULARITY_TMPDIR=/scratch/user/$USER

# Repository and output paths
REPO_DIR=$(pwd)  # Assumes script is run from repo root (c:/Users/vigyan/.../irc_dis_pytorch on local, /scratch/... on Bunya)
export RUN_ROOT=/scratch/user/$USER/ircbert_runs
export LOG_DIR=$RUN_ROOT/logs
export CHECKPOINT_DIR=$RUN_ROOT/checkpoints_synthetic
export SYNTHETIC_DIR=$REPO_DIR/data/synthetic_interleaved

echo "=== Setting up directories ==="
mkdir -p logs $LOG_DIR $CHECKPOINT_DIR
mkdir -p $SYNTHETIC_DIR/train $SYNTHETIC_DIR/dev

# Move generated synthetic files to train subdir (required by data_loader.py split logic)
echo "Moving synthetic files to train/dev subdirs..."
mv $SYNTHETIC_DIR/*.ascii.txt $SYNTHETIC_DIR/train/ 2>/dev/null || true
mv $SYNTHETIC_DIR/*.annotation.txt $SYNTHETIC_DIR/train/ 2>/dev/null || true

# Copy to dev subdir for evaluation (small synthetic dataset, same files for train/eval)
cp $SYNTHETIC_DIR/train/*.ascii.txt $SYNTHETIC_DIR/dev/
cp $SYNTHETIC_DIR/train/*.annotation.txt $SYNTHETIC_DIR/dev/

# Setup conda environment (matches Bunya HPC setup)
echo "=== Setting up environment ==="
source setup.sh  # Loads miniconda3/23.9.0-0, creates/activates env, installs deps

# Train on synthetic data (small dataset, fast training)
echo ""
echo "=== Starting training on synthetic interleaved data ==="
python src/train.py \
    --mode train \
    --data-dir $SYNTHETIC_DIR \
    --model-name bert-base-uncased \
    --max-dist 20 \
    --batch-size 16 \
    --epochs 10 \
    --learning-rate 5e-5 \
    --warmup-ratio 0.1 \
    --patience 3 \
    --eval-every 1 \
    --save-every 1 \
    --output-dir "$CHECKPOINT_DIR" \
    --device cuda

# Evaluate latest checkpoint
echo ""
echo "=== Evaluating on synthetic data ==="
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "WARNING: No checkpoint found in $CHECKPOINT_DIR - skipping evaluation"
else
    echo "Latest checkpoint: $LATEST_CHECKPOINT"
    python src/evaluate.py \
        --checkpoint "$LATEST_CHECKPOINT" \
        --data-dir $SYNTHETIC_DIR \
        --split dev \
        --batch-size 16 \
        --metrics both \
        --verbose 3
fi

echo ""
echo "=== Training and evaluation complete ==="
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: $LOG_DIR"