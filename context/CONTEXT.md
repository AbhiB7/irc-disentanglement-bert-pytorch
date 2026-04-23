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
- **2022**: Bi-Level Contrastive Learning (SOTA ~80%+ F1).

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

### Cross-Encoder Model
Input: `[CLS] message_i [SEP] message_j [SEP]` → BERT → [CLS] embedding → Linear Head → Sigmoid.
BERT can attend across both messages, which is the standard approach for high-accuracy link prediction.

### Handcrafted Feature Augmentation
Zhu et al. (2021) found a 25-point F1 gap between raw BERT and BERT + features.
- **Time difference**: Minutes between messages (most critical feature).
- **Speaker match**: 1 if same speaker, 0 otherwise.
- **Position distance**: `i - j`.
- **Word overlap**: Jaccard similarity of word sets.

**Integration**: Features are concatenated to the 768-dim [CLS] vector, resulting in a 772-dim input to the classification head.

### Pair Generation & Class Imbalance
- **Window**: `MAX_DIST` (default 30). Reduced from 101 to optimize for local 4070 GPU memory/speed.
- **Imbalance**: Handled via `pos_weight` in `BCEWithLogitsLoss`.
- **Solution**: Use `pos_weight=5.0` in `BCEWithLogitsLoss` (reduced from 14.0 to prevent gradient instability).

---

## 4. Hardware & Training Strategy

### GPU Selection & Local Optimization
| GPU                        | VRAM      | Price (USD/hr)  | Notes                                     |
| -------------------------- | --------- | --------------- | ----------------------------------------- |
| **RTX 4090 (Recommended)** | **24 GB** | **~$0.29–0.39** | Fits BERT-base with batch_size=64.        |
| RTX 4070 (Local)           | 12 GB     | -               | Requires `max_dist=30` for feasibility.   |
| A100 40GB                  | 40 GB     | ~$0.63          | Overkill for BERT-base but very stable.   |

### Training Hyperparameters
- **Learning Rate**: 5e-5 (Increased from 2e-5 to overcome majority-class bias).
- **Epochs**: 3 (BERT typically converges in 2-4 epochs).
- **Batch Size**: 64 (Optimized for RTX 5070 12GB; uses ~6-7GB VRAM).
- **Threshold**: 0.3 (Lowered from 0.5 to improve recall on rare positive samples).
- **Early Stopping**: Implemented via `--patience` (default 3) to monitor Dev F1.

### Multi-Stage Testing Plan
- **Test 1 (5 min)**: Stability and logic check. Uses `train` mode on the **Tiny Dataset** (`data/tiny`). Verifies OOM logging, NaN detection, and positive sample handling (guaranteed links).
- **Test 2 (1 hour)**: Mid-range stability run. Verified pipeline on RTX 5070 with ~50K pairs.
- **Test 3 (3-6 hours)**: Large-scale stability run. Uses **1 Million pairs** and **Batch Size 64** to refine Precision and verify long-term convergence.

## 5. Robustness & Diagnostics
- **OOM Recovery**: Training and evaluation loops catch CUDA Out-of-Memory errors, log memory state, clear cache, and skip the problematic batch.
- **Numerical Safety**: NaN/Inf loss detection triggers batch skipping to prevent weight corruption.
- **Smart Logging**: Automatic logging of any batch containing a positive sample (`label=1`) to monitor minority class behavior.
- **Data Starvation Prevention**: Test runs must use message offsets (e.g., 300+ or 1000+) or the `tiny` dataset to avoid the link-less "join/quit" noise at the start of IRC logs.
- **Atomic Checkpointing**: Checkpoints are saved to `.tmp` files and renamed to avoid Windows file-locking conflicts (Error 1224).

---

## 5. Technical Reference

### Project Structure
- `src/data_loader.py`: Handles file discovery, message parsing, and pair generation.
- `src/model.py`: Defines `CrossEncoderWithFeatures` and model initialization.
- `src/train.py`: Main entry point for training, evaluation, and checkpointing. Includes **Smart Logging** for imbalanced data diagnostics.
- `tests/`: Comprehensive unit tests for data and model logic.

### Setup Instructions
- **Windows**: Run [`setup.bat`](../setup.bat). Requires Python 3.13.11+.
- **Linux/Remote**: Run [`setup.sh`](../setup.sh).
- **Verification**: Run `python src/train.py --help` (Note: slow load time is normal due to PyTorch/Transformers initialization).

### HPC Setup (UQ Bunya)
- **Cluster**: UQ Bunya HPC (SLURM scheduler)
- **GPU**: NVIDIA A100 40GB (full job uses H100)
- **Required SLURM directives**: `--qos=gpu` and `--account=a_hcc` for all GPU jobs on `gpu_cuda` partition
- **Smoke Test**: [`smoke_test.slurm`](../smoke_test.slurm) — minimal end-to-end test (30 min, 500 pairs)
- **Full Training**: [`run_job.slurm`](../run_job.slurm) — 8-hour training job
- **Conda Environment**: `ircbert` (loaded via `Miniforge3` module with environment variable paths)
- **Output directories**: All heavy outputs relocated to `/scratch/user/$USER/ircbert_runs`

---

## 6. Key References
1. Kummerfeld et al. (2019). "A Large-Scale Corpus for Conversation Disentanglement." ACL 2019.
2. Zhu et al. (2021). "BERT for Conversation Disentanglement." (Key feature comparison paper).
3. Huang et al. (2022). "Bi-Level Contrastive Learning for Conversation Disentanglement."
