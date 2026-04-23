#!/bin/bash
#SBATCH --job-name=ircbert_smoke
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_smoke.out
#SBATCH --error=logs/%j_smoke.err
#SBATCH --qos=gpu
#SBATCH --account=a_hcc

set -e

# Bunya requirement for Singularity/Apptainer even if not used directly
export SINGULARITY_CACHEDIR=/scratch/user/$USER
export SINGULARITY_TMPDIR=/scratch/user/$USER

# Export run root and output dirs
export RUN_ROOT=/scratch/user/$USER/ircbert_runs
export LOG_DIR=$RUN_ROOT/logs
export CHECKPOINT_DIR=$RUN_ROOT/checkpoints

# Create all required directories
mkdir -p logs $LOG_DIR $CHECKPOINT_DIR

# Run setup
bash setup.sh

# Load Conda module and activate environment
module load miniconda3/23.9.0-0

# Source Conda shell init using environment variable path
source "$EBROOTMINICONDA3/etc/profile.d/conda.sh"

conda activate ircbert

# Run the training Python script with reduced smoke test parameters
python src/train.py \
    --mode train \
    --batch-size 32 \
    --num-workers 2 \
    --fp16 \
    --epochs 1 \
    --learning-rate 5e-5 \
    --max-length 128 \
    --max-dist 5 \
    --warmup-steps 10 \
    --patience 0 \
    --threshold 0.3 \
    --eval-every 1 \
    --save-every 1 \
    --test-end 500 \
    --output-dir "$CHECKPOINT_DIR" \
    --device cuda