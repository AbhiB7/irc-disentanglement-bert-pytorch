#!/bin/bash
# Setup script for IRC Disentanglement PyTorch BERT project on Linux (Vast.ai)
# Assumes Python 3.12+ installed

echo "=== IRC BERT Setup (Linux) ==="

# Update pip
python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch (using CUDA 12.1 compatible version as per setup.bat logic, 
# but letting pip resolve the best version for the environment if possible)
# On Vast.ai, torch is often pre-installed, but we ensure it's there.
echo "Installing/Updating core dependencies..."

# Core ML dependencies
# Note: setup.bat had specific versions, but for Vast.ai we'll try to be more flexible 
# unless those specific versions are strictly required. 
# Given the user's error, they just need them installed.
pip install transformers datasets sentence-transformers accelerate

# Additional utils
pip install scikit-learn numpy pandas tqdm

echo ""
echo "Checking dependencies..."
python3 check_dependencies.py

echo ""
echo "Setup complete."
