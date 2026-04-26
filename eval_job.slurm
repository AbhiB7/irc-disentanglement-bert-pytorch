#!/bin/bash
#SBATCH --job-name=ircbert_eval
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_eval.out
#SBATCH --error=logs/%j_eval.err
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

# Checkpoint path - can be overridden via command line
CHECKPOINT=${1:-$CHECKPOINT_DIR/best/checkpoint_epoch_3.pt}

echo "=========================================="
echo "IRC Disentanglement - Checkpoint Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""

# Run setup
bash setup.sh

# Load Conda module and activate environment
module load miniconda3/23.9.0-0
source "$EBROOTMINICONDA3/etc/profile.d/conda.sh"
conda activate ircbert

# Run evaluation
python src/evaluate.py --checkpoint "$CHECKPOINT"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
