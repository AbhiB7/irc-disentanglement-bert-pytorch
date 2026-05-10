#!/bin/bash
#SBATCH --job-name=ircbert_train
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --qos=gpu
#SBATCH --account=a_hcc

mkdir -p logs

set -e

# Bunya requirement for Singularity/Apptainer even if not used directly
export SINGULARITY_CACHEDIR=/scratch/user/$USER
export SINGULARITY_TMPDIR=/scratch/user/$USER

# Bunya storage layout: relocate heavy outputs to /scratch
export RUN_ROOT=/scratch/user/$USER/ircbert_runs
export LOG_DIR=$RUN_ROOT/logs
export CHECKPOINT_DIR=$RUN_ROOT/checkpoints

# Create directories (including repo-local logs for slurm output/error files)
mkdir -p logs $LOG_DIR $CHECKPOINT_DIR

# Run setup (handles conda env creation + activation + dependency installation)
# Use 'source' so conda activation and env vars persist in the current shell
source setup.sh

# Run training directly (not via train.sh) to avoid double conda activation
# which can cause "ModuleNotFoundError: No module named 'psutil'" errors.
#
# Hyperparameters validated by debug run on 2026-05-10:
#   - 16,754 training samples from 153 files (data/train/)
#   - 462 dev samples from 10 files (data/dev/)
#   - 4189 batches/epoch at batch_size=4
#   - ~8 min/epoch on H100, ~40 min for 5 epochs
#   - Loss: 2.35 → 1.49 over 5 epochs (still improving)
#   - Dev Accuracy: 1.0000, Test Accuracy: 1.0000
#   - Test F1: 0.7333 on 922 held-out samples
#   - 0 NaN batches, 0 OOM — fully stable
#
# Overnight config: 20 epochs, batch_size=8 (GPU has 80GB, plenty of headroom)
python src/train.py \
    --mode train \
    --batch-size 8 \
    --num-workers 4 \
    --epochs 20 \
    --learning-rate 5e-5 \
    --max-length 128 \
    --max-dist 15 \
    --warmup-ratio 0.1 \
    --patience 5 \
    --eval-every 1 \
    --save-every 1 \
    --test-end 1000000000 \
    --output-dir "$CHECKPOINT_DIR" \
    --device cuda