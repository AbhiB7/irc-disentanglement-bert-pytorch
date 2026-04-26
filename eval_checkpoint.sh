#!/bin/bash
# Evaluate checkpoint on IRC Disentanglement

# Checkpoint path - can be overridden
CHECKPOINT=${1:-/scratch/user/s4901673/ircbert_runs/checkpoints/best/checkpoint_epoch_3.pt}

echo "=========================================="
echo "IRC Disentanglement - Checkpoint Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""

# Run setup first to ensure dependencies are installed
bash setup.sh

# Load Conda module and activate environment (needed because setup.sh runs in subshell)
module load miniconda3/23.9.0-0
source "$EBROOTMINICONDA3/etc/profile.d/conda.sh"
conda activate ircbert

# Run evaluation
python src/evaluate.py --checkpoint "$CHECKPOINT"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
