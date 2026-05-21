# Project Progress & Status

> [!IMPORTANT]
> **<u>ANTI-DRIFT RULE</u>**: This file is the **ONLY** place for **DYNAMIC WORKING STATE**.
> - **ALL** progress, status updates, next steps, and temporary decisions belong here.
> - **DO NOT** move this information to `context.md` or `instructions.md`.

This file tracks the dynamic working state, recent completions, and immediate next steps.

## Current Status
- ✅ **Run 24836624 completed successfully**: 10 epochs, no NaN events. Best dev F1=0.3696 (epoch 8), test F1=0.4097 (epoch 8). Dev ARI=0.6053, Test ARI=0.5491.
- ✅ **Best checkpoint selection bug fixed**: `src/train.py` was comparing by `accuracy` instead of `f1` — epochs 6–8 (F1 0.3637–0.3698) were not saved as `best/` despite outperforming epoch 5 (F1 0.3633). Fixed.
- ✅ **max_length mismatch fixed**: `run_job.slurm` and `eval_job.slurm` evaluation calls omitted `--max-length`, defaulting to 128 while training used 96. Both now pass `--max-length 96`. `eval_job.slurm` also updated to use `checkpoints_maxdist30` dir and `--max-dist 30`.
- ✅ **Logit collapse diagnosed**: All 49 logits drift negative over training (mean -4.6 at epoch 1 → -37 at epoch 9). Label smoothing + pointwise scoring is the root cause. Only positional features survive as discriminators. This is an architectural limitation — each candidate is scored independently with no cross-candidate interaction.
- ✅ **NaN Fix C confirmed**: logit clamping + label smoothing + AdamW eps=1e-4 ran clean for 10 epochs. No NaN events. The fix is proven.

## Next Steps
- [ ] **Architecture fix**: Replace Linear(773,1) pointwise scoring with bilinear interaction mechanism. Child CLS embedding should interact with each candidate CLS embedding in a shared space before softmax.
- [ ] **Remove label smoothing**: Set to 0.0 or 0.05 after bilinear fix. Current 0.1 dilutes gradient signal across 48 wrong classes.
- [ ] **SELF-as-candidate**: Add SELF token to candidate list so new-thread detection is a learnable class rather than a threshold heuristic.
- [ ] **Poster presentation**: Finalize poster layout with revised text and results figure.
- [ ] **Re-train with bilinear architecture**: Submit `run_job.slurm` with updated model for cross-candidate interaction.

---

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

## Bug Fixes (2026-05-06)

### 3. `label=-1` crash in `data_loader.py`
Messages whose gold parent is outside `max_dist` got `gold_parent_idx=-1`, which crashes `CrossEntropyLoss` (target -1 is out of bounds). Fixed by skipping those samples during training with `if not self.skip_labels and gold_parent_idx < 0: continue.

### 4. Hardcoded model name in `evaluate.py`
`load_checkpoint_for_eval()` hardcoded `microsoft/deberta-v3-base` instead of reading `model_name` from the checkpoint's saved args. This caused a shape mismatch when loading a `bert-base-uncased` checkpoint. Fixed to read from `checkpoint["args"]`.

### 5. Variable-C probs concatenation in `evaluate()`
`evaluate()` tried to `torch.cat(all_probs)` but different batches have different numbers of candidates (C varies), so probs tensors have different shapes. Fixed by keeping `all_probs` as a list instead of concatenating.

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

- **Test Coverage of `src/train.py` pipeline functions**: collate_fn + create_dataloaders verified:
  - **`tests/test_train_pipeline.py`** (9 tests):
    - `TestCollateFn` (6): Padding, zero-fill, feature/label preservation, dtype — catches variable-C batch mismatch
  - `TestCreateDataloaders` (3): real data/ files → train/dev loaders created, batch shapes match model.forward()
  - **Bug caught**: collate_fn expected features [batch, 5] but data loader returns per-candidate [batch, C, 5]. Fixed collate_fn and model.forward() to use [batch, C, 5] consistently.

- **Test Coverage of `src/model.py`**: Complete multiclass test suite with 23 tests:
  - **`tests/test_model.py`** (23 tests across 6 - Init (7): Default, DeBERTa-v3-base, custom params, device, freeze_bert, params count, combined_size
  - Forward (8): With/without features, probs sum to 1, single sample, candidate masking, different C, no token_type_ids
  - Predict (3): Argmax, probs sum to 1, single sample
  - Architecture (2): Classifier output shape, dropout behavior
  - Loss (2): Non-negative, finite
  - Smoke test (1): End-to-end BERT-base verification
  - Removed embedded `test_model()` from `src/model.py` — replaced with redirect to pytest
  - Updated module docstring with architecture description and test annotation
  - All 23 tests run in ~30s on CPU with bert-base-uncased and tiny inputs (seq=32, batch=2, C=5)

## Recent Completions (2026-05-06)
- **Test Coverage of `src/train.py` remaining functions**: evaluate(), save_checkpoint(), load_checkpoint(), parse_args():
  - **`tests/test_evaluate.py`** (12 tests): Return keys, metric values (accuracy/precision/recall/F1), loss behavior, edge cases (empty predictions, all same class)
  - **`tests/test_checkpoint.py`** (14 tests): Save creates file with expected keys, epoch/metrics values, best_model.pt logic (saves when F1 present, skips when absent, overwrites on subsequent saves), load returns correct epoch/metrics, weights match after save/load, load without optimizer/scheduler, nonexistent file raises error, save without scheduler/optimizer, multiple epochs saved separately
  - **`tests/test_parse_args.py`** (20 tests): All 20 CLI args verified — defaults match expected values, custom values parsed with correct types (int/float/str), boolean flags (--freeze-bert, --fp16), mode choices enforced (invalid mode raises SystemExit), device parsing
  - **Bug caught and fixed**: `save_checkpoint()` crashed with `AttributeError: 'NoneType' object has no attribute 'state_dict'` when `optimizer=None` (test mode). Fixed with guard: `optimizer.state_dict() if optimizer else None`.
  - **Total test count**: 99 tests across 5 test files, all passing in ~40s on CPU.

- **Local Smoke Test (train.py + evaluate.py)**: Full pipeline verified end-to-end on `data/tiny` with `bert-base-uncased`:
  - **Training**: 31 batches, avg_loss=0.8456, 34.91s on CPU
  - **Evaluation**: 33 batches, Loss=0.4081, **Accuracy=0.8615**, Precision=0.7813, Recall=0.8911, F1=0.8007
  - **Checkpoints saved**: `checkpoints_tiny/checkpoint_epoch_1.pt` + `checkpoints_tiny/best/checkpoint_epoch_1.pt`
  - **evaluate.py** loaded checkpoint correctly and produced identical metrics
  - **Bugs caught and fixed**:
    1. `data_loader.py`: Messages with gold parent outside `max_dist` got `label=-1`, crashing `CrossEntropyLoss`. Fixed by skipping those samples during training.
    2. `evaluate.py`: `load_checkpoint_for_eval()` hardcoded `microsoft/deberta-v3-base` instead of reading `model_name` from checkpoint's saved args. Fixed to read from `checkpoint["args"]`.
  - **Pipeline is ready for Bunya smoke test**.

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

## Recent Completions (2026-05-17)
- **100% Accuracy Root Cause Analysis**: Traced through evaluate.py, train.py`, `data_loader.py`, and `model.py`. Found that 100% accuracy likely stems from gold labels being dominated by the immediately previous message (recency bias), making the pairwise metric trivial.
- **Human-Readable Validation Output**: Added `--verbose N` flag to `src/evaluate.py`:
  - Randomly samples N conversations (reproducible via `--verbose-seeded)
  - Formats gold and predicted threads side-by-side for manual inspection
  - Added `format_conversation_threads()` function
- **Synthetic Data Generation**: Created `scripts/generate_synthetic_data.py`:
  - Generates conversations with clear thread structure (Python help, Weather, Food, Gaming)
  - Creates `.ascii.txt` and `.annotation.txt` files
  - Topics are clearly separated to prevent recency shortcut
- **Updated `eval_job.slurm`**: Added `--verbose 3` to dev/test evaluation calls. Added synthetic data evaluation step.
- **Fixed Invalid Log Filenames on Bunya Linux**: Removed stray double quotes from `evaluate.sh` (lines 47, 59) and `evaluate_2.sh` (lines 57, 69). Stray quotes caused shell to interpret subsequent `echo` statements and newlines as part of the filename, creating invalid filenames like `'eval_test_20260517_190243.log'$'\n\n''echo '$'\n''echo ==='`. Verified both files use LF line endings for Linux compatibility.
- **Added Predicted Pair Debugging to `src/evaluate.py`**:
  - New function `debug_predicted_pairs()` prints (child, predicted parent, gold parent) with message indices, speakers, and text snippets
  - Integrated into `--verbose` block to show actual predicted pairs for samples
  - Helps diagnose if model is predicting self-links, SYSTEM→SYSTEM links, or meaningful thread links
  - Fixed `KeyError: 3` by using `sample["labels"]` dict access instead of tuple indexing (since `__getitem__` returns dict)
  - **Added softmax probability debugging**: Now prints P(self), P(pred), P(gold) for each sampled pair to understand if model is confidently predicting self-links
  - Modified `debug_predicted_pairs()` to accept `all_probs` from `evaluate()` function
  - Updated call site in `main()` to pass `metrics["probs"]` to `debug_predicted_pairs()`
  - All 120 tests pass after changes (`pytest tests/ -x -q`)

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