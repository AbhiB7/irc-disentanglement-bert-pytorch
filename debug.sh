#!/bin/bash
# ==============================================================================
# debug.sh — Fast iterative debugging for Bunya interactive node sessions
#
# Usage (on Bunya interactive node):
#   salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=16G \
#     --job-name=GPUInteractive --time=01:00:00 \
#     --partition=gpu_cuda --qos=debug \
#     --gres=gpu:1 \
#     --account=a_hcc \
#     srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l
#   ./debug.sh                          # Baseline (no fp16, DeBERTa, tiny)
#   ./debug.sh --fp16                   # Test fp16
#   ./debug.sh --model bert-base-uncased --batch-size 2
#   ./debug.sh --max-dist 30 --fp16
#
# Purpose: Replace SLURM job submission during debugging. Each run takes
#          < 1 minute on tiny dataset. Edit → run → check → repeat.
# ==============================================================================

set -e

# ── Config (overridable via CLI args or env vars) ──────────────────────────
FP16=${FP16:-false}
MODEL=${MODEL:-microsoft/deberta-v3-base}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_DIST=${MAX_DIST:-15}
TEST_END=${TEST_END:-500}
DATA_DIR=${DATA_DIR:-data/tiny}
# --medium shortcut: uses train_small (~10 files, ~10K samples)
MEDIUM=${MEDIUM:-false}
# --epochs shortcut: override default 1 epoch (e.g., --epochs 3 for more training)
EPOCHS=${EPOCHS:-1}

# ── Parse CLI args ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --fp16) FP16=true ;;
    --model) MODEL="$2"; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift ;;
    --max-dist) MAX_DIST="$2"; shift ;;
    --data-dir) DATA_DIR="$2"; shift ;;
    --medium) MEDIUM=true ;;
    --epochs) EPOCHS="$2"; shift ;;
    --test-end) TEST_END="$2"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

# ── Print config ───────────────────────────────────────────────────────────
echo "================================================"
echo "debug.sh — Fast Iterative Debug Run"
echo "================================================"
echo "  FP16:      $FP16"
echo "  Model:     $MODEL"
echo "  Batch:     $BATCH_SIZE"
echo "  Max dist:  $MAX_DIST"
echo "  Test end:  $TEST_END"
echo "  Data dir:  $DATA_DIR"
echo "  Medium:    $MEDIUM"
echo "================================================"

# ── Activate environment ───────────────────────────────────────────────────
# IMPORTANT: Activate your conda environment FIRST before running this script.
#   source setup.sh    (one-time: installs all dependencies)
#   conda activate irc-bert
# Do NOT put source setup.sh inside this script — it re-downloads PyTorch.

# ── Apply --medium shortcut ───────────────────────────────────────────────
if [ "$MEDIUM" = "true" ]; then
    DATA_DIR="data/train_small"
    TEST_END=1000000
    echo "  >> --medium: using data/train_small with test_end=$TEST_END"
fi

# ── Run training (stdout+stderr tee'd to logs/) ────────────────────────────
LOG_FILE="logs/debug_$(date +%Y%m%d_%H%M%S).log"
python src/train.py \
    --mode train \
    --data-dir "$DATA_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 2 \
    --epochs "$EPOCHS" \
    --learning-rate 5e-5 \
    --max-length 128 \
    --max-dist "$MAX_DIST" \
    --warmup-ratio 0.1 \
    --patience 0 \
    --eval-every 1 \
    --save-every 1 \
    --test-end "$TEST_END" \
    --output-dir /scratch/user/$USER/ircbert_runs/debug_checkpoints \
    --device cuda \
    $( [ "$FP16" = "true" ] && echo "--fp16" ) \
    2>&1 | tee "$LOG_FILE"
echo ""
echo "Log saved to: $LOG_FILE"
