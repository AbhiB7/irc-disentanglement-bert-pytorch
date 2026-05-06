#!/bin/bash

# setup.sh - Environment setup for UQ Bunya HPC
# This script is idempotent and safe to run multiple times.
# Designed to run inside a SLURM job (compute node) — NOT on a login node.
#
# Idempotency:
#   - conda create: skipped if env already exists
#   - pip install: skips packages already installed with matching version
#   - module load: idempotent by design

# ---- Scratch storage paths (belt-and-braces for non-interactive jobs) ----
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/user/\$USER/conda-envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/user/\$USER/conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/user/\$USER/pip-cache}"
export HF_HOME="${HF_HOME:-/scratch/user/\$USER/huggingface}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/scratch/user/\$USER}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/scratch/user/\$USER}"

# Create scratch directories if they don't exist
mkdir -p \
    "${CONDA_ENVS_PATH}" \
    "${CONDA_PKGS_DIRS}" \
    "${PIP_CACHE_DIR}" \
    "${HF_HOME}"

# ---- Conda setup ----
module load miniconda3/23.9.0-0
source "$EBROOTMINICONDA3/etc/profile.d/conda.sh"

# Create conda env if it doesn't exist (idempotent)
if conda info --envs 2>/dev/null | grep -q "ircbert"; then
    echo "Conda environment 'ircbert' already exists. Skipping creation."
else
    echo "Creating conda environment 'ircbert'..."
    conda create -n ircbert python=3.10 -y
fi

conda activate ircbert

# ---- CUDA-aware PyTorch ----
# Bunya requires CUDA module loaded on compute node before installing GPU packages
echo "Loading CUDA module..."
module load cuda 2>/dev/null || echo "  (CUDA module already loaded or unavailable — continuing)"

# Install dependencies (idempotent: pip skips already-satisfied packages)
echo "Installing PyTorch with CUDA 12.1..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "Installing other dependencies..."
pip install transformers datasets sentence-transformers accelerate
pip install scikit-learn numpy pandas tqdm psutil
pip install sentencepiece tiktoken protobuf huggingface_hub[cli]
pip install tensorboard

echo "Setup complete."
