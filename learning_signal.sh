#!/bin/bash
# learning_signal.sh — Validate training produces a real learning signal
# Full dataset, max_dist=15, DeBERTa-v3-base, 10 epochs, 1/8 data
# Run on Bunya interactive GPU node: bash learning_signal.sh
# All output tee'd to logs/learning_signal_DATETIME.log

set -e

export SINGULARITY_CACHEDIR=/scratch/user/$USER
export SINGULARITY_TMPDIR=/scratch/user/$USER

REPO_DIR=$(pwd)
export RUN_ROOT=/scratch/user/$USER/ircbert_runs
export CHECKPOINT_DIR=$RUN_ROOT/checkpoints_maxdist15
export DATA_DIR=$REPO_DIR/data

mkdir -p $RUN_ROOT/logs $CHECKPOINT_DIR logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/learning_signal_$TIMESTAMP.log"

echo "================================================" | tee -a "$LOG_FILE"
echo " Learning Signal Validation" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo " Log file:    $LOG_FILE" | tee -a "$LOG_FILE"
echo " Checkpoints: $CHECKPOINT_DIR" | tee -a "$LOG_FILE"
echo " Data:        $DATA_DIR (full train/dev/test)" | tee -a "$LOG_FILE"
echo " max_dist:    15" | tee -a "$LOG_FILE"
echo " test_end:    156 (1/8 of ~1250 msgs per file)" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo ""

echo "=== Setting up environment ===" | tee -a "$LOG_FILE"
source setup.sh 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Training on full dataset (max_dist=15, test_end=156) ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
python src/train.py \
    --mode train \
    --data-dir "$DATA_DIR" \
    --model-name microsoft/deberta-v3-base \
    --max-dist 15 \
    --batch-size 16 \
    --epochs 10 \
    --learning-rate 5e-5 \
    --warmup-ratio 0.1 \
    --patience 3 \
    --eval-every 1 \
    --save-every 1 \
    --test-end 156 \
    --output-dir "$CHECKPOINT_DIR" \
    --device cuda 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Evaluating on DEV set ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
LATEST=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
    echo "ERROR: No checkpoint found in $CHECKPOINT_DIR" | tee -a "$LOG_FILE"
    exit 1
fi
echo "Checkpoint: $LATEST" | tee -a "$LOG_FILE"
python src/evaluate.py \
    --checkpoint "$LATEST" \
    --data-dir "$DATA_DIR" \
    --split dev \
    --batch-size 16 \
    --max-dist 15 \
    --metrics both \
    --verbose 3 \
    --verbose-seed 42 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Evaluating on TEST set ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
python src/evaluate.py \
    --checkpoint "$LATEST" \
    --data-dir "$DATA_DIR" \
    --split test \
    --batch-size 16 \
    --max-dist 15 \
    --metrics both \
    --verbose 3 \
    --verbose-seed 42 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo " Done!" | tee -a "$LOG_FILE"
echo " Log:        $LOG_FILE" | tee -a "$LOG_FILE"
echo " Checkpoints: $CHECKPOINT_DIR" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"