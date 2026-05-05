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
- ✅ **Lazy Tokenization Fix**: Implemented on-the-fly tokenization in `data_loader.py` to eliminate OOM. Now stores raw text pairs instead of pre-tokenized tensors. Tokenization happens in `__getitem__` instead of `__init__`. Reduces per-worker memory from ~1.5GB to ~200MB (~85% reduction).
- 📋 **Supervisor Meeting**: Created `research/supervisor_meeting_20260424.md` with discussion points for 2026-04-24 meeting.
- ✅ **Threshold Fix**: Changed default threshold from 0.3 to 0.5 in `src/train.py`. Removed `--threshold` flags from all training scripts. This fixes the "predict everything as positive" failure mode (Recall=100%, Precision=0.69%, F1=0.0137).
- ✅ **Evaluation Script**: Created [`src/evaluate.py`](src/evaluate.py) for checkpoint evaluation with threshold sweep support. Usage: `python src/evaluate.py --checkpoint <path> [--threshold 0.5] [--sweep-thresholds]`
- 🔄 **Full Dataset Training**: Currently running on Bunya A100 with threshold=0.5. A100 provides 80GB VRAM enabling full dataset training without OOM issues.
- ✅ **DeBERTa-v3-base**: Updated default model to `microsoft/deberta-v3-base` in `model.py` and `train.py` (ROCLING 2025 SOTA).
- ✅ **max_dist=50**: Increased max_dist from 30 to 50 in data loader and training (StructBERT uses kh=50).
- ✅ **Multiclass Architecture**: Properly implemented multiclass reframing (Priority 3):
  - Data loader creates C separate samples (one per candidate)
  - Model processes child message independently, outputs C probabilities
  - Uses CrossEntropyLoss (no pos_weight needed)
  - Ready for training

## Recent Completions (2026-05-04)
- **Proper Multiclass Implementation**: Complete refactor of multiclass architecture:
  - **Data Loader**: [`src/data_loader.py`](src/data_loader.py) - Creates C separate samples (one per candidate) instead of concatenated input. Each sample: `(parent_text, child_text, features, label)` where label = gold parent index (0 to C-1).
  - **Model**: [`src/model.py`](src/model.py) - Processes child message independently, outputs `[batch_size, C]` probabilities. Uses softmax instead of sigmoid.
  - **Training**: [`src/train.py`](src/train.py) - Updated log messages to say "samples" instead of "pairs".
  - **Evaluation**: [`src/evaluate.py`](src/evaluate.py) - Removed `--threshold` argument, updated defaults to DeBERTa-v3-base and max_dist=50.
  - **Status**: ✅ Ready for training

## Bug Fixes (2026-05-05)

### 1. `num_features=4` bug in `train.py:main()`
Model was created with `num_features=4` while data loader outputs 5 features. This caused a shape mismatch (772 vs 773) at the `torch.cat` in `forward()`. Fixed to `num_features=5`.

### 2. Hardcoded `warmup-steps=100` replaced with `--warmup-ratio 0.1`
Warmup should scale with dataset size (standard practice: 10% of total steps). Old fixed value of 100 steps was negligible for full training (~0.04% of 270K steps). Now computes `int(total_steps * 0.1)` automatically.

## Recent Completions (2026-05-05)
- **Test Coverage of `src/data_loader.py`**: Comprehensive test suite covering every function and class:
  - **`tests/test_create_samples.py`** (5 tests): `_create_samples_for_conversation`, `compute_features`
  - **`tests/test_data_loader.py`** (8 tests): `__getitem__`, `parse_irc_line`
  - **`tests/test_load_conversation.py`** (6 tests): `load_conversation`, `load_dataset_files` — includes real file test from `data/tiny/` (300 msgs, 212 links, 32 users)
  - **Nothing left untested**:

  | Component | Coverage | Test file(s) |
  |-----------|----------|-------------|
  | `IRCMessage` dataclass | ✅ Indirectly | All 3 test files |
  | `IRCConversation` dataclass | ✅ Indirectly | All 3 test files |
  | `parse_irc_line()` | ✅ Directly | `test_data_loader.py` |
  | `extract_targets()` | ✅ Indirectly | `test_create_samples.py` |
  | `load_conversation()` | ✅ Directly | `test_load_conversation.py` |
  | `compute_features()` | ✅ Directly | `test_create_samples.py` |
  | `IRCDisentanglementDataset.__init__` | ✅ Indirectly | All dataset tests |
  | `_create_samples_for_conversation` | ✅ Directly | `test_create_samples.py` |
  | `__len__` | ✅ Indirectly | All dataset tests |
  | `__getitem__` | ✅ Directly | `test_data_loader.py` |
  | `load_dataset_files()` | ✅ Directly | `test_load_conversation.py` |

- **Test Coverage of `src/model.py`**: Complete multiclass test suite with 23 tests:
  - **`tests/test_model.py`** (23 tests across 6 classes):
    - Init (7): Default, DeBERTa-v3-base, custom params, device, freeze_bert, params count, combined_size
    - Forward (8): With/without labels, without features, probs sum to 1, single sample, candidate masking, different C, no token_type_ids
    - Predict (3): Argmax, probs sum to 1, single sample
    - Architecture (2): Classifier output shape, dropout behavior
    - Loss (2): Non-negative, finite
    - Smoke test (1): End-to-end BERT-base verification
  - Removed embedded `test_model()` from `src/model.py` — replaced with redirect to pytest
  - Updated module docstring with architecture description and test annotation
  - All 23 tests run in ~30s on CPU with bert-base-uncased and tiny inputs (seq=32, batch=2, C=5)

## Recent Completions (2026-04-23)
- **Class Imbalance Fix (pos_weight cap)**: Raised `pos_weight` cap from 300 to 1500 in [`src/model.py:154`](src/model.py:154). With ~746:1 negative-to-positive ratio, the old cap of 300 was insufficient (negatives still dominated loss 746 > 300). New cap of 1500 allows proper loss weighting for the imbalance.

## Recent Completions (2026-04-23) - Previous
- **All-Zero Prediction Fix (Iteration 4)**: Implemented three targeted fixes for all-zero prediction collapse:
    - **Fix 1**: Dynamic `pos_weight` in [`src/model.py`](src/model.py:148) — computes `(num_neg / (num_pos + 1e-8)).clamp(min=10.0, max=300.0)` per batch instead of hardcoded 5.0.
    - **Fix 2**: Reduced epochs in [`train.sh`](train.sh:31) from 10 to 3 — ensures LR decay completes within actual training window.
    - **Fix 3**: Lowered threshold in [`train.sh`](train.sh:37) from 0.3 to 0.1 — handles 748:1 class imbalance where sigmoid outputs are calibrated low.

## Recent Completions (2026-04-26)
- **Full Data Training (Run 24009010)**: Completed 3-epoch training on full dataset (~5.8M pairs).
    - **Metrics**: F1: 0.0137, Recall: 100%, Precision: 0.69%, Accuracy: 80.77%
    - **Problem**: Model predicted ALL pairs as positive (threshold=0.3 too low for probability calibration)
    - **Root Cause**: Probability range [0.0007, 0.7632] meant ~45% of samples exceeded threshold=0.3
    - **Fix**: Changed default threshold to 0.5 in `src/train.py:173`, removed explicit --threshold from all scripts

## Next Steps
- **Current Priority**: Run training with proper multiclass architecture. Expected F1 improvement from 0.0137 to ~0.10-0.15.
- **Post-Training**: Evaluate on full dev set using `src/evaluate.py`.
- **Future: Improve Convergence Detection**: Current early stopping uses patience-based heuristic. Consider implementing more robust convergence detection:
  - **Gradient-based convergence**: Monitor gradient norms approaching zero
  - **Loss plateau detection**: Track when loss stops decreasing significantly
- **Remaining Priorities**:
  - Priority 4: Union-Find clustering + thread metrics (VI, ARI, MCF)
  - Priority 5: Speaker-masked MHSA (structural attention masking)
  - Priority 6: @mention r-GCN (uses existing `msg.targets` data)
  - **Multiple metric convergence**: Require F1, loss, AND precision to plateau
  - **Learning rate decay**: Reduce LR when validation loss plateaus, then apply early stopping
  - **Priority**: Low - current patience-based approach is sufficient for now, focus on full pipeline first

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

## Active Task: Full Dataset Training on Bunya
- [ ] Submit full training job on Bunya A100 with threshold=0.5
- [ ] Monitor training progress and logs
- [ ] Evaluate best checkpoint on dev set
- [ ] Compare results against Study 1 DyNet baseline (~62.6% F1)

## Next Steps (Archived/Completed)
- ~~**Test 3 (Immediate)**: Large-scale stability run using `train_test_3.sh`.~~ (Archived - now using Bunya A100 for full training)
- ~~**GPU Training on Vast.ai**: Execute `full_train.sh` on Vast.ai GTX 1080 Ti.~~ (Completed - now using Bunya A100)
- ~~**Inference**: Run the trained model on all 10 dev files.~~ (Pending post-training)
- ~~**Evaluation**: Use `graph-eval.py` to generate final link-level F1 metrics.~~ (Pending post-training)
- ~~**Comparison**: Compare BERT results against the Study 1 DyNet baseline.~~ (Pending post-training)

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
