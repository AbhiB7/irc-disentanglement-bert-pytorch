#!/bin/bash

# train.sh - Training execution script for UQ Bunya HPC
# Inspired by full_train.sh but optimized for A100

# Load Conda module
module load Miniforge3 || module load Miniconda3

# Source Conda shell init using environment variable path
if [ -n "$EBROOTMINIFORGE" ]; then
    source "$EBROOTMINIFORGE/etc/profile.d/conda.sh"
elif [ -n "$EBROOTMINICONDA" ]; then
    source "$EBROOTMINICONDA/etc/profile.d/conda.sh"
else
    echo "Error: Conda module root environment variable not found."
    exit 1
fi

# Activate the environment
conda activate ircbert

# Create output directories if they don't exist
mkdir -p logs checkpoints

# Run the main training Python script
# Entry point: src/train.py
# Optimized for A100 (80GB/40GB) on Bunya
python src/train.py \
    --mode train \
    --batch-size 128 \
    --num-workers 8 \
    --fp16 \
    --epochs 10 \
    --learning-rate 5e-5 \
    --max-length 128 \
    --max-dist 30 \
    --warmup-steps 1000 \
    --patience 3 \
    --threshold 0.3 \
    --eval-every 1 \
    --save-every 1 \
    --output-dir checkpoints \
    --device cuda
