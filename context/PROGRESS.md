# Project Progress & Status

> [!IMPORTANT]
> **<u>ANTI-DRIFT RULE</u>**: This file is the **ONLY** place for **DYNAMIC WORKING STATE**.
> - **ALL** progress, status updates, next steps, and temporary decisions belong here.
> - **DO NOT** move this information to `context.md` or `instructions.md`.

This file tracks the dynamic working state, recent completions, and immediate next steps.

## Current Status
- ✅ **Early Stopping**: Implemented and verified in `train.py`.
- ✅ **Documentation**: Refactored into three distinct files (`instructions.md`, `context.md`, `progress.md`).
- ✅ **Environment**: Python 3.13 compatibility confirmed on Windows.
- 🔄 **Next Major Milestone**: Execute full training run on GPU (Vast.ai).

## Recent Completions (2026-04-19)
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
1. **GPU Training**: Prepare `full_train.sh` for execution on a Vast.ai RTX 4090 instance.
2. **Inference**: Run the trained model on all 10 dev files.
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
