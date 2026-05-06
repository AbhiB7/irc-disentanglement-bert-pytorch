#!/bin/bash
#SBATCH --job-name=ircbert_eval
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_eval.out
#SBATCH --error=logs/%j_eval.err
#SBATCH --qos=gpu
#SBATCH --account=a_hcc


set -e

export SINGULARITY_CACHEDIR=/scratch/user/$USER
export SINGULARITY_TMPDIR=/scratch/user/$USER

export RUN_ROOT=/scratch/user/$USER/ircbert_runs
export LOG_DIR=$RUN_ROOT/logs
export CHECKPOINT_DIR=$RUN_ROOT/checkpoints

mkdir -p logs $LOG_DIR $CHECKPOINT_DIR

CHECKPOINT=${1:-$CHECKPOINT_DIR/best/checkpoint_epoch_3.pt}

echo "=========================================="
echo "IRC Disentanglement - Checkpoint Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""

# Run setup (handles conda env creation + activation + dependency installation)
# Use 'source' so conda activation and env vars persist in the current shell
source setup.sh

python src/evaluate.py --checkpoint "$CHECKPOINT"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="