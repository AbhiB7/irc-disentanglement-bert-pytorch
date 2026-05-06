#!/bin/bash
#SBATCH --job-name=ircbert_smoke
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1         # Bunya requires explicit GPU type
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_smoke.out
#SBATCH --error=logs/%j_smoke.err
#SBATCH --qos=debug
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

# Run setup (handles conda env creation + activation + dependency installation)
# Use 'source' so conda activation and env vars persist in the current shell
source setup.sh

# Run the training Python script with reduced smoke test parameters
#
# Parameter rationale:
#   --max-dist 20   Small candidate window for speed but large enough for ~80% of gold links
#                   to fall within window, giving meaningful metrics (> random baseline 0.05)
#   --test-end 500  Limit to 500 samples total (shared across train+dev). If max_dist is
#                   increased, this may need to increase too so dev set isn't skipped.
#   --batch-size 32 Reduced from default 64 for faster iteration on tiny dataset
#   --epochs 1      Single epoch — just verify no OOM/NaN and pipeline runs end-to-end
#
python src/train.py \
    --mode train \
    --data-dir data/tiny \
    --batch-size 4 \
    --num-workers 2 \
    --fp16 \
    --epochs 1 \
    --learning-rate 5e-5 \
    --max-length 128 \
    --max-dist 20 \
    --warmup-ratio 0.1 \
    --patience 0 \
    --eval-every 1 \
    --save-every 1 \
    --test-end 500 \
    --output-dir "$CHECKPOINT_DIR" \
    --device cuda
