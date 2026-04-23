# Project Progress & Status

> [!IMPORTANT]
> **<u>ANTI-DRIFT RULE</u>**: This file is the **ONLY** place for **DYNAMIC WORKING STATE**.
> - **ALL** progress, status updates, next steps, and temporary decisions belong here.
> - **DO NOT** move this information to `context.md` or `instructions.md`.

This file tracks the dynamic working state, recent completions, and immediate next steps.

## Current Status
- ✅ **Test 1 (Iteration 3)**: Successfully ran on `data/tiny`. Resolved "all-zero" prediction issue. Achieved **0.48 F1** and **92.3% Recall** on tiny dev set.
- ✅ **Robustness**: OOM recovery and NaN detection implemented and verified.
- ✅ **Model Fix**: Resolved "all-zero" prediction issue via hyperparameter tuning.
- ✅ **Early Stopping**: Implemented and verified in `train.py`.
- ✅ **Documentation**: Refactored into three distinct files (`instructions.md`, `context.md`, `progress.md`).
- ✅ **Environment**: Python 3.13 compatibility confirmed on Windows.
- ✅ **Optimization**: Default `max_dist` reduced to 30 for local GPU feasibility.
- ✅ **Test 2 Success**: 3-hour stability run completed. Achieved **0.1454 F1** and **57.36% Recall** on dev set. Pipeline is stable on RTX 5070.
- ✅ **Bunya Smoke Test**: Created [`smoke_test.slurm`](../smoke_test.slurm) for UQ Bunya HPC (A100) verification.
- ✅ **Bunya Compliance Fix**: Added `--qos=gpu`, `--account=a_hcc`, and `mkdir -p logs` to both SLURM files.
- ✅ **Conda Module Fix**: Updated all HPC files to use `miniconda3/23.9.0-0` and `$EBROOTMINICONDA3` (removed Miniforge3/Miniconda3 fallbacks).
- ✅ **Error Handling**: Added `set -e` to `run_job.slurm` and `smoke_test.slurm`.

## Recent Completions (2026-04-23)
- **Class Imbalance Fix (pos_weight cap)**: Raised `pos_weight` cap from 300 to 1500 in [`src/model.py:154`](src/model.py:154). With ~746:1 negative-to-positive ratio, the old cap of 300 was insufficient (negatives still dominated loss 746 > 300). New cap of 1500 allows proper loss weighting for the imbalance.

## Recent Completions (2026-04-23) - Previous
- **All-Zero Prediction Fix (Iteration 4)**: Implemented three targeted fixes for all-zero prediction collapse:
    - **Fix 1**: Dynamic `pos_weight` in [`src/model.py`](src/model.py:148) — computes `(num_neg / (num_pos + 1e-8)).clamp(min=10.0, max=300.0)` per batch instead of hardcoded 5.0.
    - **Fix 2**: Reduced epochs in [`train.sh`](train.sh:31) from 10 to 3 — ensures LR decay completes within actual training window.
    - **Fix 3**: Lowered threshold in [`train.sh`](train.sh:37) from 0.3 to 0.1 — handles 748:1 class imbalance where sigmoid outputs are calibrated low.

## Next Steps
- **Step 2**: Threshold optimization after re-training with new pos_weight cap

## Recent Completions (2026-04-22)
- **Test 2 Success**: Completed stability run on RTX 5070.
    - **Metrics**: F1: 0.1454, Recall: 57.36%, Precision: 8.33%.
    - **Stability**: No OOMs or NaNs. GPU memory usage was low (~1.7GB).
    - **Finding**: Model is successfully identifying links but over-predicting (high FP count), likely due to the small training slice (50k pairs).

## Recent Completions (2026-04-21)
- **Test 1 Success**: Verified model logic on `data/tiny`. The model now predicts positive links correctly (Recall: 92.3%, F1: 0.48) instead of all zeros.
- **OOM Logging**: Implemented comprehensive CUDA OOM catching and system diagnostic logging in `train.py`.
- **Numerical Stability**: Added NaN/Inf loss detection and batch skipping.
- **Test 1 Setup**: Rewrote `train_gpu_5070.sh` for a 5-minute stability and logic check.
- **Label Fix**: Decoupled `skip_labels` from data limiting to allow real metrics on small subsets.
- **Test 2 Diagnosis**: Identified that `--test-end 500` limits TOTAL pairs to 500 (not per file), causing data starvation and all-zero predictions. Fixed by increasing to 500K pairs.

## Recent Completions (2026-04-19)
- **Model Fix**: Resolved "all-zero" prediction issue by reducing `pos_weight` (14.0 -> 5.0), increasing learning rate (2e-5 -> 5e-5), and lowering threshold (0.5 -> 0.3).
- **Diagnostic Logging**: Added "Smart Logging" to `train.py` to track probability distributions and positive batch statistics.
- **Dataset Fix**: Recreated `tiny` dataset with guaranteed gold links to enable valid local verification.
- **Optimization**: Reduced default `max_dist` from 101 to 30 to support local training on RTX 4070.
- **Feature Refactoring**: Refactored `compute_features` to dynamically accept `max_dist` for correct normalization.
- **Early Stopping**: Added `--patience` argument and logic to `train.py`. Verified via `args.json`.
- **Context Audit**: Consolidated redundant documentation and separated behavioral rules from project knowledge.
- **Code Explanation**: Clarified `data_loader.py` entry points and `args` object usage for the user.

## Active Task: Documentation Refactoring
- [x] Audit `instructions.md`, `context.md`, and `progress.md`.
- [x] Define clear responsibilities for each file.
- [x] Move stable knowledge to `context.md`.
- [x] Move behavioral rules to `instructions.md`.
- [x] Update `progress.md` with current state.

## Next Steps
1. **Test 3 (Immediate)**: Large-scale stability run using `train_test_3.sh`. Uses **1 Million pairs** and `batch-size=64` to improve Precision and verify long-term convergence on RTX 5070.
4. **GPU Training**: Execute `full_train.sh` on Vast.ai GTX 1080 Ti (15-hour overnight run).
5. **Inference**: Run the trained model on all 10 dev files.
3. **Evaluation**: Use `graph-eval.py` to generate final link-level F1 metrics.
4. **Comparison**: Compare BERT results against the Study 1 DyNet baseline.

---

## Project History (Brief)

### 2026-04-16: Smoke Tests & Bug Fixes
- **Verification**: Ran smoke test on tiny dataset (`--test-end 10`).
- **Fixes**: Resolved test range limiting in `data_loader.py`, fixed logging conflicts, and implemented a robust `save_checkpoint` to handle Windows file locking (Error 1224).

### 2026-04-15: Environment & Unit Testing
- **Compatibility**: Updated dependencies for Python 3.13 (Transformers 5.5.4, Sentence-Transformers 5.4.1).
- **Refactoring**: Renamed `data.py` to `data_loader.py`.
- **Testing**: Passed 15 data loader tests and 22 model tests.

### Pre-2026: Study 1 Completion
- **Baseline**: Established DyNet (GloVe + FFNN) baseline at ~62.6% F1.
- **Constraint**: Identified computational bottleneck for full-scale training on consumer GPUs.
