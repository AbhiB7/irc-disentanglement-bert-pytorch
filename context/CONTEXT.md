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

### Known Limitation: Pointwise Scoring (Discovered 2026-05-20)
The current architecture scores each candidate independently through a single `Linear(773, 1)` layer. There is no cross-candidate interaction until the softmax step. This means:
- The model cannot learn relative rankings (A > B > C). It only learns absolute scores.
- All 49 logits drift negative during training (mean -4.6 at epoch 1 → -37 at epoch 9) because there is no mechanism to elevate the correct candidate above others.
- Only positional features survive as discriminators — recency bias dominates predictions.
- **Fix**: Replace pointwise scoring with bilinear interaction where the child message CLS embedding is compared against each candidate embedding in a shared learned space. This is the standard cross-encoder formulation used in Kummerfeld 2019 and Zhu 2021.
- **Invariant**: Any future classifier architecture must include candidate interaction before softmax. Pointwise scoring without cross-candidate comparison will reproduce logit collapse.

---

## 4. Hardware & Training Strategy

### Available GPUs (UQ Bunya HPC)
| GPU         | VRAM    | Notes                                                       |
|-------------|---------|-------------------------------------------------------------|
| **A100 (full)** | 40/80 GB | Full GPU. Request with `--gres=gpu:1 --constraint=cuda80gb`. |
| **A100 MIG** | 10/20/40 GB | **AVOID**. MIG partitions share one A100. `--constraint=cuda48gb|cuda80gb` excludes them. |
| **L40S**    | 48 GB   | Newer architecture, fast. Available on `gpu_cuda` queue.  |
| **H100**    | 80 GB   | Highest throughput. Use for full training runs.             |

### Training Hyperparameters
- **Learning Rate**: 5e-5 (Standard for BERT fine-tuning).
- **Epochs**: 10 (with gradient accumulation and patience=3 for early stopping).
- **Batch Size**: 4 (with `--gradient-accumulation-steps 4` → effective batch size 16).
- **Max Length**: 96 tokens (covers 95%+ of IRC messages; saves ~25% memory vs 128).
- **Max Dist**: 50 (covers 98.1% of gold cross-links).
- **Early Stopping**: Implemented via `--patience` (default 3) to monitor Dev F1.
- **Always pass `--constraint=cuda48gb|cuda80gb` to SLURM** to avoid MIG partitions.

## 5. Priority Improvement Roadmap (vs. SOTA)

| Priority | Improvement | Effort | Expected Gain | Status |
|----------|-------------|--------|---------------|--------|
| 1 | **DeBERTa-v3-base backbone** | Change 1 string in `model.py` | ~+1% F1 | ✅ Complete |
| 2 | **Increase max_dist to 50** | Change 1 default argument | Improves recall | ✅ Complete |
| 3 | **Multiclass reframing** | Restructure loss in `model.py` | Eliminates pos_weight heuristic | ✅ Complete |
| 4 | **Union-Find clustering + thread metrics** | New module needed | Required for VI/ARI in thesis | ⏳ Pending |

## 6. Robustness & Diagnostics
- **OOM Recovery**: Training and evaluation loops catch CUDA Out-of-Memory errors, log memory state, clear cache, and skip the problematic batch.
- **Gradient Accumulation (2026-05-18)**: DeBERTa-v3-base with batch=16, candidate count up to 50, and seq_len=128 generates ~102K tokens/forward pass per batch. With gradient+AdamW states, this exceeds 48GB VRAM. Fix: `--batch-size 4` + `--gradient-accumulation-steps 4` + `--max-length 96`. Loss divided by `accumulation_steps` before backward (averages gradients over N batches). `optimizer.step()` and `scheduler.step()` only trigger every `accumulation_steps` batches. `optimizer.zero_grad()` runs after optimizer step (not before backward). Scheduler `total_steps` counts optimizer steps (`(len(train_loader) * epochs) // accumulation_steps`). Invariant: gradient accumulation is ON for any training run using DeBERTa-v3-base at batch_size > 1.
- **SLURM Constraint (2026-05-18)**: First `learning_signal.sh` run landed on a MIG `1g.10gb` partition (9.5 GiB) instead of a full GPU. Even batch=4 OOM'd. Fix: always include `--constraint=cuda48gb|cuda80gb` in SLURM directives, even for interactive sessions: `srun --partition=gpu_cuda --gres=gpu:1 --constraint=cuda48gb|cuda80gb ...`.
- **Annotation Format Bug (Found 2026-05-18)**: The annotation files use format `PARENT CHILD -` (first column = parent, second = child). The code was parsing this as `child = int(parts[0]), parent = int(parts[1])`, which assigned every cross-link to the wrong message index. Only self-links survived because they are symmetric. Fixed by swapping the variable assignment. Invariant: annotation column 0 is always the parent.
- **Numerical Safety**: NaN/Inf loss detection triggers batch skipping to prevent weight corruption.
- **NaN Loss Prevention — Fix A (collate_fn, 2026-05-10)**: Two fixes prevent NaN from out-of-range labels:
  1. `max_candidates` cap follows `--max-dist` instead of hardcoded 15 (Option B).
  2. Labels are clamped to `min(label, max_candidates - 1)` to prevent `CrossEntropyLoss` from receiving an out-of-range target.
  See [`tests/test_train_pipeline.py`](tests/test_train_pipeline.py) for `test_label_clamp_out_of_bounds`.
- **NaN Loss Prevention — Fix B (model + optimizer, 2026-05-10)**: **ORIGINAL ROOT CAUSE — Partially superseded by Fix C.**
  - **What**: AdamW's default `eps=1e-8` was too small for DeBERTa-v3-base fine-tuning. The update formula `grad / sqrt(v + eps)` with `eps=1e-8` caused numerical instability when `v` was very small.
  - **Fix**: Changed AdamW `eps` from `1e-8` to `1e-6` (DeBERTa's HuggingFace training guide recommendation). Also moved `nan_to_num` safety net to **before** the classifier (was after — useless placement). Added pre-clip gradient norm logging, weight NaN check after `optimizer.step()`, and LR logging.
  - **Debugging arc**: 2 weeks, 5 HPC runs, HPC runs, 4 fix attempts.
  - **Status**: Superseded by Fix C on 2026-05-19. The primary NaN driver was logit overflow from 49-class CrossEntropyLoss, not AdamW epsilon alone.
- **NaN Loss Prevention — Fix C (model + train.py + config, 2026-05-19)**: **DEFINITIVE FIX — 49-class logit overflow prevention.**
  - **What**: The true root cause is logit overflow in the 49-class CrossEntropyLoss softmax. Unclamped logits drifted to ~80–100 during training, causing `exp(100)` to overflow fp32 to `inf`, making the softmax denominator `inf/inf = NaN`. This cascaded to corrupt weights, and the `nan_to_num` fix on `cls_embedding` silently masked the symptom while letting contamination continue.
  - **Three changes in `src/model.py`**:
    1. **Removed `nan_to_num` on `cls_embedding`** (was actively dangerous — masked NaN and let weight contamination continue). Now NaN propagates naturally to produce NaN loss, which `train.py` catches and skips.
    2. **Added `torch.clamp(logits, -50, 50)` before loss computation**: Hard numerical ceiling guarantees `exp(z)` stays within fp32 range. `exp(50) ≈ 5.18e21` (well below fp32 max 3.4e38), making softmax denominator finite by construction.
    3. **Added `label_smoothing=0.1` to `CrossEntropyLoss`**: Prevents the model from pushing the correct-class logit to +inf by capping the target probability at 0.9. The remaining 0.1 is distributed across all classes, creating a soft target instead of a one-hot spike.
  - **Two changes in `src/train.py`**:
    1. NaN loss skip path now calls `optimizer.zero_grad()` + `torch.cuda.empty_cache()` (was just `continue` with no cleanup). Without this, accumulated gradients from previous batches in the accumulation window survived and got applied on the next valid optimizer step, reintroducing instability.
    2. AdamW `eps` increased from `1e-6` to `1e-4` — stronger `grad / sqrt(v + eps)` stabilization across the full training run (not just the first few steps).
  - **Config changes in `run_job.slurm`**: batch=8, accumulation=2, lr=3e-5, warmup=15% (was batch=4, acc=4, lr=5e-5, warmup=10%). Reduces gradient noise accumulation window.
  - **Invariant**: Any training on 49-class softmax with DeBERTa-v3-base must use (1) logit clamping to [-50, 50], (2) label smoothing ≥ 0.1, (3) AdamW eps ≥ 1e-4. The `nan_to_num` safety net on `cls_embedding` is explicitly forbidden — it masks the symptom and leaves weights corrupted.
  - See `research/handover.md` for the full analysis chain.
- **Smart Logging**: Logs probability distribution stats every 50 batches (avg/min/max prob) to monitor model calibration.
- **Data Starvation Prevention**: Test runs must use message offsets (e.g., 300+ or 1000+) or the `tiny` dataset to avoid the link-less "join/quit" noise at the start of IRC logs.
- **Atomic Checkpointing**: Checkpoints are saved to `.tmp` files and renamed to avoid Windows file-locking conflicts (Error 1224).

---

## 7. Evaluation Results — INVALIDATED (2026-05-18)

**The 100% accuracy results reported below are invalid.** The parent/child column swap bug in annotation parsing (see Section 6) caused all cross-links to be silently assigned to the wrong message index. The model was only ever trained and evaluated on self-links (24.1% of the dataset). The 100% accuracy was a self-link memorization artifact of the bug, not genuine disentanglement.

The bug was fixed on 2026-05-18. All evaluation results prior to this date should be disregarded.

### Pre-Fix Results (Kept for reference — no longer valid)
- Pairwise accuracy: **100%** on both dev (462 samples) and test (922 samples)
- "Predict last candidate" baseline: 16.0% dev, 15.4% test
- "Predict most common position" baseline: 16.2% dev, 16.6% test
- Recency check (positions 46-49): 59.5% dev, 53.6% test
- Clusters: Dev 494 (55% singleton), Test 961 (63% singleton)
- ARI ≈ 0, VI ≈ 0 (not meaningful due to singleton dominance)

### Post-Fix Status (Pre-Run)
- Cross-links: 52,641 (75.9%) of total gold links
- Training dataset: 49,676 valid samples
- **First training attempt OOM'd**: Landed on MIG 1g.10gb (9.5 GiB) instead of a full GPU. Batch=4 with gradient accumulation OOM'd on first forward pass. Evaluation on the untrained random init produced Loss=3.8236, F1=0.008 (near-uniform distribution over 49 candidates; ln(49) ≈ 3.89).
- **Fix**: `--constraint=cuda48gb|cuda80gb` added to SLURM headers. `run_job.slurm` ready to submit.

### Run 24796535 — DeBERTa-v3-base (2026-05-19)
| Metric | Dev | Test |
|--------|-----|------|
| Best Epoch | 6 | 6 |
| Link-level F1 | **0.3386** | — |
| Top-1 Accuracy | **51.55%** | **49.54%** |
| Precision | 0.2500 | — |
| Recall | 0.6132 | — |
| Loss (best epoch) | ~3.82 | — |

**Training config**: H100 GPU, 5.2h, batch=8, accumulation=2 (effective 16), lr=3e-5, warmup=15%, AdamW eps=1e-4, label_smoothing=0.1, logit_clamp=[-50,50], max_length=96, max_dist=50.

**Key observations**:
- No NaN events across all 9 epochs — Fix C confirmed stable.
- Logit mean drifted from -4.6 (epoch 1) to -37 (epoch 9). Max logit was below 0 from epoch 2 onward. Predicted positions concentrated in positions 40-48 (recency).
- Best epoch 6 was chosen by dev F1 early stopping (patience=3).
- Dev accuracy (51.55%) is 5x the majority-class baseline (10.6%) and 3.2x the recency heuristic (16%), confirming the model extracts genuine signal.
- The GloVe baseline (62.6% F1) is NOT directly comparable — it used a binary pairwise formulation on a different metric.

## 8. Clustering & Thread-Level Metrics
- **Current**: Only link-level F1 is computed.
- **Gap**: Actual task is conversation disentanglement (grouping messages into threads). Needs a **clustering step** after link prediction.
- **Methods**: Union-Find (simple) or bipartite graph matching.
- **Metrics Needed**: VI (Variational Inference), ARI (Adjusted Rand Index), MCF (Message Clustering F1) for thesis visualization component.
- **Status**: ARI/VI implemented but not meaningful on this dataset due to annotation sparsity (55-63% singleton clusters). Pairwise accuracy is the primary metric.

## 9. Self-Links Architecture Note

### Status (2026-05-18)
The "self-links are dominant" theory is **incorrect**. Diagnostic confirmed:
- Self-links: 16,754 (24.1%) — minority of gold links
- Cross-links: 52,641 (75.9%) — majority
- Cross-link median distance: 3 messages
- Cross-links within max_dist=50: 51,632 (98.1%)

Self-links are still meaningful as "new thread" labels, but they are NOT the primary training signal. The real problem was the annotation format bug (parent/child columns swapped), which masked all cross-links.

### The Original Argument (Kept for reference — partially valid)
Self-links in the Kummerfeld et al. annotation schema are **not bugs** — they are the dataset's way of encoding "this message starts a new conversation thread." 
A message whose gold parent is itself is a valid training label.

### Proposed Architecture: SELF-as-Candidate
Reformulate the candidate list to include a special SELF token:

```
candidates = [msg_0, msg_1, ..., msg_{i-1}, SELF]
```

Where `SELF` is a special embedding appended at the end of the candidate list. The gold label is either:
- The index of the gold parent (if cross-message link exists within window)
- The index of `SELF` (if the message starts a new thread)

**Why this works:**
- No samples are dropped (every message has a valid candidate)
- Self-link prediction becomes a learnable signal — the model learns "does this message start a new thread?"
- Cross-message links train the content-based ranking
- Recency bias now competes with a real "new thread" class, breaking the trivial shortcut

**Literature support**: Kummerfeld's own feedforward model uses a threshold below which a message links to itself. This formulation makes that explicit as a candidate class.

**Thesis contribution**: *"We identified that prior work conflates two sub-tasks (new thread detection vs. reply linking) and propose a unified candidate formulation."*

### Current Priority (2026-05-18)
- Self-links are **excluded** via `range(..., i)` in `data_loader.py` line 376 (correct behavior — self-links are a separate task)
- **Priority**: Train the model with corrected annotations first. If cross-link accuracy is reasonable, SELF-as-candidate is unnecessary.
- The SELF-as-candidate refactor remains a valid longer-term option but should not be prioritized until cross-link-only training is evaluated.

## 10. Technical Reference

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

## 11. Key References
1. Kummerfeld et al. (2019). "A Large-Scale Corpus for Conversation Disentanglement." ACL 2019.
2. Zhu et al. (2021). "BERT for Conversation Disentanglement." (Key feature comparison paper).
3. Huang et al. (2022). "Bi-Level Contrastive Learning for Conversation Disentanglement."
4. **StructBERT** (ACL 2022). "Structural Encoding for Conversation Disentanglement." [aclanthology.org/2022.acl-long.23.pdf](https://aclanthology.org/2022.acl-long.23.pdf)
5. **ROCLING 2025**. "Benchmarking Transformer Models on Ubuntu IRC Dataset." [aclanthology.org/2025.rocling-main.31.pdf](https://aclanthology.org/2025.rocling-main.31.pdf)
