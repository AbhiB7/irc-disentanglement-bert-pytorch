#!/bin/bash
# Evaluate checkpoint on IRC Disentanglement

# Checkpoint path - can be overridden
CHECKPOINT=${1:-/scratch/user/s4901673/ircbert_runs/checkpoints/best/checkpoint_epoch_3.pt}

echo "=========================================="
echo "IRC Disentanglement - Checkpoint Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""

# Run setup (idempotent - loads conda environment)
bash setup.sh

# Run evaluation
python src/evaluate.py --checkpoint "$CHECKPOINT"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
