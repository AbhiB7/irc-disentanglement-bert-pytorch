#!/bin/bash
#SBATCH --job-name=ircbert_eval
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_eval.out
#SBATCH --error=logs/%j_eval.err
#SBATCH --qos=gpu
#SBATCH --account=a_hcc
#SBATCH --constraint=cuda48gb|cuda80gb

set -e

export SINGULARITY_CACHEDIR=/scratch/user/$USER
export SINGULARITY_TMPDIR=/scratch/user/$USER

export RUN_ROOT=/scratch/user/$USER/ircbert_runs
export LOG_DIR=$RUN_ROOT/logs
export CHECKPOINT_DIR=$RUN_ROOT/checkpoints_maxdist30

mkdir -p logs $LOG_DIR $CHECKPOINT_DIR

# Default checkpoint: epoch 8 from run 24836624 (highest F1, max_dist=30)
CHECKPOINT=${1:-$CHECKPOINT_DIR/checkpoint_epoch_8.pt}

echo "=========================================="
echo "IRC Disentanglement - Checkpoint Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""

# Run setup (handles conda env creation + activation + dependency installation)
# Use 'source' so conda activation and env vars persist in the current shell
source setup.sh

echo ""
echo "=== Evaluating on DEV set ==="
python src/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --split dev \
    --batch-size 64 \
    --max-length 96 \
    --max-dist 30 \
    --metrics both \
    --verbose 3 \
    --verbose-seed 42

echo ""
echo "=== Evaluating on TEST set ==="
python src/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --split test \
    --batch-size 64 \
    --max-length 96 \
    --max-dist 30 \
    --metrics both \
    --verbose 3 \
    --verbose-seed 42

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
