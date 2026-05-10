# Project Context: IRC BERT Conversation Disentanglement

> [!IMPORTANT]
> **<u>ANTI-DRIFT RULE</u>**: This file is for **STABLE PROJECT KNOWLEDGE** only.
> - **DO NOT** add active task progress, current status, or "Next Steps" here.
> - **DO NOT** add temporary planning notes or recent completions here.
> - **ONLY** edit this to add long-lived research data, architectural changes, or new technical invariants.

This file serves as the stable knowledge base for the project, containing research background, architectural decisions, and technical invariants.

---

## 1. Research Background & Thesis Narrative

### The Problem
Conversation disentanglement is the task of separating interleaved chat messages into distinct threads. This project uses the **Ubuntu IRC dataset** (Kummerfeld et al., 2019), which contains 77,563 messages with gold-standard reply annotations.

### Study 1: The Baseline
- **Model**: DyNet feedforward model using GloVe word vectors.
- **Results**: ~62.6% link-level F1 on dev file `2004-11-15_03`.
- **Key Finding**: Full-scale training was computationally infeasible on consumer hardware (RTX 3060 Laptop) due to the O(N × max_dist) pair-scoring complexity (~1M pairs per epoch).

### Study 2: The Current Focus
- **Research Question**: *Which models best balance accuracy and computational feasibility, and how do modern Transformer-based approaches compare to the original DyNet baseline?*
- **Approach**: Replace the GloVe encoder with a fine-tuned **BERT cross-encoder** while maintaining the identical pairwise scoring architecture. This enables a controlled comparison where only the representation layer changes.

### Field Progression
- **Pre-2019**: Handcrafted features (~35% F1).
- **2019**: GloVe + FFNN (Kummerfeld et al. — Study 1 baseline).
- **2021-2022**: Fine-tuned BERT/ALBERT/DeBERTa + handcrafted features (~72% F1).
- **2022**: StructBERT (ACL 2022) — Speaker-masked MHSA + r-GCN for reference dependency → **52.6% F1** (vs 33.5% for plain BERT).
- **2022**: Bi-Level Contrastive Learning (SOTA ~80%+ F1).
- **2025**: ROCLING 2025 benchmark on Ubuntu IRC dataset:
  - DeBERTa-v3-base: **0.7230** Link F1 (vs 0.7144 for BERT-base)
  - ModernBERT-base: 0.7087 Link F1

---

## 2. Data Specification

### Data Format
| Field         | Example                                     | Notes                                                                            |
| ------------- | ------------------------------------------- | -------------------------------------------------------------------------------- |
| `id`          | `1050`                                      | Integer message index                                                            |
| `ascii`       | `"[03:57] <Xophe> (also, I'm guessing...)"` | Contains **timestamp** and **speaker name** (critical for features)              |
| `connections` | `[1048, 1054, 1055]`                        | **Gold links**: indices of messages this one replies to                          |

### Data Splits
| Split | Messages | Files      |
| ----- | -------- | ---------- |
| Train | 67,463   | ~158 files |
| Dev   | 2,500    | ~10 files  |
| Test  | 5,000    | ~20 files  |

---

## 3. Architecture & Implementation

### Multiclass Model (Current Architecture)
The problem is framed as "which of C candidates is the parent of message i?":
- **Input**: Child message encoded once, C candidate parents processed independently
- **Shape**: `input_ids [batch, C, seq_len]` — C candidates per sample, padded to max C in batch
- **Processing**: Flatten to `[batch*C, seq_len]` for BERT, reshape back to `[batch, C, hidden]`
- **Output**: Softmax over C candidates — probability distribution over candidate parents
- **Loss**: `CrossEntropyLoss` — no pos_weight needed, no threshold needed
- **Prediction**: Argmax over C candidates

### Handcrafted Feature Augmentation
Zhu et al. (2021) found a 25-point F1 gap between raw BERT and BERT + features.
- **Time difference**: Minutes between messages (most critical feature).
- **Speaker match**: 1 if same speaker, 0 otherwise.
- **Position distance**: `i - j`.
- **Word overlap**: Jaccard similarity of word sets.
- **Directedness**: 1 if child message mentions parent's speaker (explicit @mention).

**Integration**: Features are concatenated to the 768-dim [CLS] vector, resulting in a 773-dim input to the classification head.

### Structural Features (from StructBERT ACL 2022)
- **Speaker-masked MHSA**: Attention is masked so tokens only attend within same-speaker utterances (M[i,j]=0 if same speaker, -∞ otherwise). Boosts F1 from 33.5 → 45.0.
- **r-GCN (Reference Dependency)**: Graph where edges connect utterances containing @username mentions to all prior utterances by that user. Boosts F1 to 47.4.
- **Combined**: Both together → 52.6 F1 (vs 33.5 for plain BERT).
- **Current Implementation**: `msg.targets` set in `IRCMessage` already contains the reference dependency graph needed for r-GCN.

### Proper Multiclass Architecture (Implemented 2026-05-04)
- **Data Loader**: Creates C separate samples (one per candidate) instead of concatenated input.
  - Each sample: `(parent_text, child_text, features, label)` where label = gold parent index (0 to C-1)
  - Input format: `[seq_len]` for child message only
  - Each candidate has its own features (previously shared)
- **Model**: Processes child message independently, outputs C probabilities.
  - Input: `[batch_size, seq_len]` for child message
  - Output: `[batch_size, C]` probabilities (one per candidate)
  - Uses softmax instead of sigmoid
  - Loss function: `CrossEntropyLoss` (no pos_weight needed)
- **Training**: Batch size represents C samples per message
- **Evaluation**: Uses argmax for multiclass predictions
- **Status**: ✅ Ready for training

---

## 4. Hardware & Training Strategy

### Available GPUs (UQ Bunya HPC)
| GPU         | VRAM    | Notes                                                       |
|-------------|---------|-------------------------------------------------------------|
| **A100**    | 40 GB   | Reliable, widely available. Default for smoke tests.        |
| **L40S**    | 48 GB   | Newer architecture, faster. Available on `gpu_cuda` queue.  |
| **H100**    | 80 GB   | Highest throughput. Use for full training runs.             |

### Training Hyperparameters
- **Learning Rate**: 5e-5 (Standard for BERT fine-tuning).
- **Epochs**: 3 (BERT typically converges in 2-4 epochs).
- **Batch Size**: 64 (Feasible on all Bunya GPUs; adjust up for H100 if needed).
- **Early Stopping**: Implemented via `--patience` (default 3) to monitor Dev F1.

## 5. Priority Improvement Roadmap (vs. SOTA)

| Priority | Improvement | Effort | Expected Gain | Status |
|----------|-------------|--------|---------------|--------|
| 1 | **DeBERTa-v3-base backbone** | Change 1 string in `model.py` | ~+1% F1 | ✅ Complete |
| 2 | **Increase max_dist to 50** | Change 1 default argument | Improves recall | ✅ Complete |
| 3 | **Multiclass reframing** | Restructure loss in `model.py` | Eliminates pos_weight heuristic | ✅ Complete |
| 4 | **Union-Find clustering + thread metrics** | New module needed | Required for VI/ARI in thesis | ⏳ Pending |

## 6. Robustness & Diagnostics
- **OOM Recovery**: Training and evaluation loops catch CUDA Out-of-Memory errors, log memory state, clear cache, and skip the problematic batch.
- **Numerical Safety**: NaN/Inf loss detection triggers batch skipping to prevent weight corruption.
- **NaN Loss Prevention — Fix A (collate_fn, 2026-05-10)**: Two fixes prevent NaN from out-of-range labels:
  1. `max_candidates` cap follows `--max-dist` instead of hardcoded 15 (Option B).
  2. Labels are clamped to `min(label, max_candidates - 1)` to prevent `CrossEntropyLoss` from receiving an out-of-range target.
  See [`tests/test_train_pipeline.py`](tests/test_train_pipeline.py) for `test_label_clamp_out_of_bounds`.
- **NaN Loss Prevention — Fix B (model, 2026-05-10)**: **THE REAL ROOT CAUSE.**
  - **What**: Padded candidates were masked with `torch.finfo(dtype).min` (= `-3.4e38` for fp32) before softmax. CrossEntropyLoss backward through `-3.4e38` produces mathematically undefined gradients (effectively INF). One optimizer step with INF gradient corrupts ALL model weights to NaN, cascading to every subsequent batch.
  - **Fix attempt 1**: `-1e4`. STILL NaN. Reason: `exp(-10000)` underflows to 0 in fp32 → `log(0) = -inf` → `0 * -inf = NaN` in backward.
  - **Fix attempt 2 (✅ FINAL)**: `-100.0`. `exp(-100) ≈ 3.7e-44` (well above fp32 minimum of ~1.4e-45), so `log(exp(-100)) = -100` is finite. Gradients stay well-behaved. Verified on DeBERTa-v3-base + L40S.
  - Also added `torch.nan_to_num` safety net for BERT LayerNorm NaN (all-zero attention mask → zero output → 0/0 normalization).
  - **Debugging arc**: 2 weeks, 4 HPC runs, 3 fix attempts. Diagnostics finally pin-pointed the `-inf` logit through logit min/max/has_nan logging.
  - **Invariant**: NEVER use `torch.finfo(dtype).min` in a `masked_fill` that participates in softmax + CrossEntropyLoss backward. Even `-1e4` is too extreme because `exp(-10000)` underflows to 0 in fp32. Use `-100.0` — large enough for softmax to assign ~0 probability, but small enough that `exp(-100)` is representable.
- **Smart Logging**: Logs probability distribution stats every 50 batches (avg/min/max prob) to monitor model calibration.
- **Data Starvation Prevention**: Test runs must use message offsets (e.g., 300+ or 1000+) or the `tiny` dataset to avoid the link-less "join/quit" noise at the start of IRC logs.
- **Atomic Checkpointing**: Checkpoints are saved to `.tmp` files and renamed to avoid Windows file-locking conflicts (Error 1224).

---

## 7. Clustering & Thread-Level Metrics
- **Current**: Only link-level F1 is computed.
- **Gap**: Actual task is conversation disentanglement (grouping messages into threads). Needs a **clustering step** after link prediction.
- **Methods**: Union-Find (simple) or bipartite graph matching.
- **Metrics Needed**: VI (Variational Inference), ARI (Adjusted Rand Index), MCF (Message Clustering F1) for thesis visualization component.

## 8. Technical Reference

### Project Structure
- `src/data_loader.py`: Handles file discovery, message parsing, and multiclass sample generation.
- `src/model.py`: Defines `CrossEncoderWithFeatures` and model initialization (multiclass output).
- `src/train.py`: Main entry point for training, evaluation, and checkpointing. Includes **Smart Logging** for imbalanced data diagnostics.
- `src/evaluate.py`: Evaluation script for multiclass predictions.
- `tests/`: Comprehensive unit tests — 99 tests across 5 files, all passing in ~40s on CPU:
  - `tests/test_data_loader.py` (8): `__getitem__`, `parse_irc_line`
  - `tests/test_create_samples.py` (5): `_create_samples_for_conversation`, `compute_features`
  - `tests/test_load_conversation.py` (6): `load_conversation`, `load_dataset_files`
  - `tests/test_model.py` (44): Model init, forward, prediction, architecture, loss
  - `tests/test_train_pipeline.py` (9): `collate_fn`, `create_dataloaders`
  - `tests/test_evaluate.py` (12): `evaluate()` metrics, loss, edge cases
  - `tests/test_checkpoint.py` (14): `save_checkpoint`, `load_checkpoint`
  - `tests/test_parse_args.py` (20): All 20 CLI argument defaults and parsing

### Setup Instructions
- **Windows**: Run [`setup.bat`](../setup.bat). Requires Python 3.13.11+.
- **Linux/Remote**: Run [`setup.sh`](../setup.sh).
- **Verification**: Run `python src/train.py --help` (Note: slow load time is normal due to PyTorch/Transformers initialization).

### HPC Setup (UQ Bunya)
- **Cluster**: UQ Bunya HPC (SLURM scheduler)
- **Available GPUs**: A100 40GB, L40S 48GB, H100 80GB (all on `gpu_cuda` partition)
- **Required SLURM directives**: `--qos=gpu` and `--account=a_hcc` for all GPU jobs on `gpu_cuda` partition
- **Conda Module**: `miniconda3/23.9.0-0` (via `$EBROOTMINICONDA3`)
- **Smoke Test**: [`smoke_test.slurm`](../smoke_test.slurm) — minimal end-to-end test (30 min, 500 pairs)
- **Full Training**: [`run_job.slurm`](../run_job.slurm) — 8-hour training job
- **Error handling**: `set -e` in SLURM scripts aborts on any command failure
- **Output directories**: All heavy outputs relocated to `/scratch/user/$USER/ircbert_runs`

### Known Issues & Fixes
- **OOM with num_workers>0**: Resolved via **lazy tokenization** in `data_loader.py`. Dataset now stores raw text pairs instead of pre-tokenized tensors. Tokenization happens in `__getitem__` on-the-fly, reducing per-worker memory from ~1.5GB to ~200MB (~85% reduction).

---

## 9. Key References
1. Kummerfeld et al. (2019). "A Large-Scale Corpus for Conversation Disentanglement." ACL 2019.
2. Zhu et al. (2021). "BERT for Conversation Disentanglement." (Key feature comparison paper).
3. Huang et al. (2022). "Bi-Level Contrastive Learning for Conversation Disentanglement."
4. **StructBERT** (ACL 2022). "Structural Encoding for Conversation Disentanglement." [aclanthology.org/2022.acl-long.23.pdf](https://aclanthology.org/2022.acl-long.23.pdf)
5. **ROCLING 2025**. "Benchmarking Transformer Models on Ubuntu IRC Dataset." [aclanthology.org/2025.rocling-main.31.pdf](https://aclanthology.org/2025.rocling-main.31.pdf)
