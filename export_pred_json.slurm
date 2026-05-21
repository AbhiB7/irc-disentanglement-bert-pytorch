#!/bin/bash
#SBATCH --job-name=ircbert_export_json
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_export_json.out
#SBATCH --error=logs/%j_export_json.err
#SBATCH --qos=gpu
#SBATCH --account=a_hcc
#SBATCH --constraint=cuda48gb|cuda80gb

set -e

export SINGULARITY_CACHEDIR=/scratch/user/$USER
export SINGULARITY_TMPDIR=/scratch/user/$USER

export RUN_ROOT=/scratch/user/$USER/ircbert_runs
export LOG_DIR=$RUN_ROOT/logs
export CHECKPOINT_DIR=$RUN_ROOT/checkpoints_maxdist30

mkdir -p logs $LOG_DIR $CHECKPOINT_DIR app/predicted_data

# Default checkpoint: epoch 8 from run 24836624 (highest F1, max_dist=30)
CHECKPOINT=${1:-$CHECKPOINT_DIR/checkpoint_epoch_8.pt}

echo "=========================================="
echo "IRC Disentanglement - Export Predicted JSONs"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""

# Run setup (handles conda env creation + activation + dependency installation)
source setup.sh

echo ""
echo "=== Exporting predicted JSONs for DEV set ==="
python src/evaluate_pred.py \
    --checkpoint "$CHECKPOINT" \
    --split dev \
    --batch-size 64 \
    --max-length 96 \
    --max-dist 30 \
    --export-json \
    --output-dir app/predicted_data

echo ""
echo "=== Exporting predicted JSONs for TEST set ==="
python src/evaluate_pred.py \
    --checkpoint "$CHECKPOINT" \
    --split test \
    --batch-size 64 \
    --max-length 96 \
    --max-dist 30 \
    --export-json \
    --output-dir app/predicted_data

echo ""
echo "=========================================="
echo "Export complete! JSON files in app/predicted_data/"
echo "=========================================="