#!/bin/bash

# setup.sh - Environment setup for UQ Bunya HPC
# This script is idempotent and safe to run multiple times.

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

# Check if conda environment 'ircbert' exists
if conda info --envs | grep -q "ircbert"; then
    echo "Conda environment 'ircbert' already exists. Skipping creation."
else
    echo "Creating conda environment 'ircbert'..."
    conda create -n ircbert python=3.10 -y
fi

# Activate the environment
conda activate ircbert

# Upgrade pip and core tools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies via pip
echo "Installing PyTorch with CUDA 12.1..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "Installing other dependencies..."
pip install transformers datasets sentence-transformers accelerate
pip install scikit-learn numpy pandas tqdm psutil

echo "Setup complete."
