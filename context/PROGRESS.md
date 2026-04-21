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

## Recent Completions (2026-04-21)
- **Test 1 Success**: Verified model logic on `data/tiny`. The model now predicts positive links correctly (Recall: 92.3%, F1: 0.48) instead of all zeros.
- **OOM Logging**: Implemented comprehensive CUDA OOM catching and system diagnostic logging in `train.py`.
- **Numerical Stability**: Added NaN/Inf loss detection and batch skipping.
- **Test 1 Setup**: Rewrote `train_gpu_5070.sh` for a 5-minute stability and logic check.
- **Label Fix**: Decoupled `skip_labels` from data limiting to allow real metrics on small subsets.

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
1. **Test 2 (Immediate)**: 3-hour stability run using `train_test_2.sh`. Uses first 500 messages of every training file to verify multi-file stability and convergence trends on Vast.ai RTX 5070.
2. **Test 3 (Prospective)**: Stress test on full dev set or significant portion of train set.
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
