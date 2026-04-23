#!/bin/bash
#SBATCH --job-name=ircbert_train
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --qos=gpu
#SBATCH --account=a_hcc

mkdir -p logs

set -e

# Bunya requirement for Singularity/Apptainer even if not used directly
export SINGULARITY_CACHEDIR=/scratch/user/$USER
export SINGULARITY_TMPDIR=/scratch/user/$USER

# Bunya storage layout: relocate heavy outputs to /scratch
export RUN_ROOT=/scratch/user/$USER/ircbert_runs
export LOG_DIR=$RUN_ROOT/logs
export CHECKPOINT_DIR=$RUN_ROOT/checkpoints

# Create directories
mkdir -p $LOG_DIR $CHECKPOINT_DIR

# Run setup (idempotent)
bash setup.sh

# Run training
bash train.sh
