#!/bin/bash

# train.sh - Training execution script for UQ Bunya HPC
# Inspired by full_train.sh but optimized for A100

# Load Conda module
module load miniconda3/23.9.0-0

# Source Conda shell init using environment variable path
source "$EBROOTMINICONDA3/etc/profile.d/conda.sh"

# Activate the environment
conda activate ircbert

# Use environment variables for output directories if provided (e.g. from Slurm)
# Default to repo-local if not set
LOG_DIR=${LOG_DIR:-"logs"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"checkpoints"}

# Create output directories if they don't exist
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# Run the main training Python script
# Entry point: src/train.py
# Optimized for A100 (80GB/40GB) on Bunya
python src/train.py \
    --mode train \
    --batch-size 128 \
    --num-workers 4 \
    --fp16 \
    --epochs 3 \
    --learning-rate 5e-5 \
    --max-length 128 \
    --max-dist 30 \
    --warmup-steps 1000 \
    --patience 3 \
    --threshold 0.1 \
    --eval-every 1 \
    --save-every 1 \
    --test-end 1000000000 \
    --output-dir "$CHECKPOINT_DIR" \
    --device cuda
