#!/bin/bash
#SBATCH --job-name=ircbert_train
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
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

REPO_DIR=$(pwd)

# Bunya storage layout: relocate heavy outputs to /scratch
export RUN_ROOT=/scratch/user/$USER/ircbert_runs
export LOG_DIR=$RUN_ROOT/logs
export CHECKPOINT_DIR=$RUN_ROOT/checkpoints_maxdist50

# Create directories (including repo-local logs for slurm output/error files)
mkdir -p logs $LOG_DIR $CHECKPOINT_DIR

# Run setup (handles conda env creation + activation + dependency installation)
# Use 'source' so conda activation and env vars persist in the current shell
source setup.sh

# Full training on Ubuntu IRC dataset with gradient accumulation
# --batch-size 4 with gradient-accumulation-steps 4 = effective batch size 16
# --max-length 96 covers 95%+ of IRC messages at reduced memory
python src/train.py \
    --mode train \
    --data-dir "$REPO_DIR/data" \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 10 \
    --learning-rate 5e-5 \
    --max-length 96 \
    --max-dist 50 \
    --warmup-ratio 0.1 \
    --patience 3 \
    --eval-every 1 \
    --save-every 1 \
    --output-dir "$CHECKPOINT_DIR" \
    --device cuda

echo ""
echo "=== Training complete. Running final evaluation on DEV and TEST sets ==="
echo ""

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "WARNING: No checkpoint found in $CHECKPOINT_DIR — skipping final evaluation."
else
    echo "Latest checkpoint: $LATEST_CHECKPOINT"
    echo ""
    echo "=== Evaluating on DEV set ==="
    python src/evaluate.py \
        --checkpoint "$LATEST_CHECKPOINT" \
        --data-dir "$REPO_DIR/data" \
        --split dev \
        --batch-size 64 \
        --max-dist 50 \
        --metrics both

    echo ""
    echo "=== Evaluating on TEST set ==="
    python src/evaluate.py \
        --checkpoint "$LATEST_CHECKPOINT" \
        --data-dir "$REPO_DIR/data" \
        --split test \
        --batch-size 64 \
        --max-dist 50 \
        --metrics both
fi