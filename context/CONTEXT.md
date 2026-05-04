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
  - **Gap**: StructBERT and ROCLING 2025 use `kh=50`. At max_dist=30, the model cannot predict replies more than 30 messages back — this sets a hard ceiling on recall.
- **Imbalance**: Handled via dynamic `pos_weight` in `BCEWithLogitsLoss`.
- **Solution**: Compute `pos_weight = (num_neg / (num_pos + 1e-8)).clamp(min=10.0, max=1500.0)` per batch to adapt to actual label distribution.
- **Gap**: SOTA methods reframe as **multiclass** (which candidate is the parent?) instead of binary, eliminating class imbalance entirely.

### Structural Features (from StructBERT ACL 2022)
- **Speaker-masked MHSA**: Attention is masked so tokens only attend within same-speaker utterances (M[i,j]=0 if same speaker, -∞ otherwise). Boosts F1 from 33.5 → 45.0.
- **r-GCN (Reference Dependency)**: Graph where edges connect utterances containing @username mentions to all prior utterances by that user. Boosts F1 to 47.4.
- **Combined**: Both together → 52.6 F1 (vs 33.5 for plain BERT).
- **Current Implementation**: `msg.targets` set in `IRCMessage` already contains the reference dependency graph needed for r-GCN.

### Multiclass vs. Binary Framing
- **Current**: Binary classification "is pair (i, j) linked?" with BCE loss and pos_weight.
- **SOTA**: Multiclass over window "which of C candidates is the parent of message i?" with cross-entropy loss.
- **Benefits**: Eliminates class imbalance, produces probability distribution over candidates.

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
- **Threshold**: 0.1 (Lowered from 0.3 to handle 748:1 class imbalance where sigmoid outputs are calibrated low).
- **Early Stopping**: Implemented via `--patience` (default 3) to monitor Dev F1.

### Multi-Stage Testing Plan
- **Test 1 (5 min)**: Stability and logic check. Uses `train` mode on the **Tiny Dataset** (`data/tiny`). Verifies OOM logging, NaN detection, and positive sample handling (guaranteed links).
- **Test 2 (1 hour)**: Mid-range stability run. Verified pipeline on RTX 5070 with ~50K pairs.
- **Test 3 (3-6 hours)**: Large-scale stability run. Uses **1 Million pairs** and **Batch Size 64** to refine Precision and verify long-term convergence.

## 5. Priority Improvement Roadmap (vs. SOTA)

| Priority | Improvement | Effort | Expected Gain |
|----------|-------------|--------|---------------|
| 1 | **DeBERTa-v3-base backbone** | Change 1 string in `model.py` | ~+1% F1 |
| 2 | **Increase max_dist to 50** | Change 1 default argument | Improves recall |
| 3 | **Multiclass reframing** | Restructure loss in `model.py` | Eliminates pos_weight heuristic |
| 4 | **Union-Find clustering + thread metrics** | New module needed | Required for VI/ARI in thesis |
| 5 | **Speaker-masked MHSA** | Add structural module to `model.py` | ~+11 F1 points over BERT baseline |
| 6 | **@mention r-GCN** | Use existing `msg.targets` | ~+5 F1 points |

**Items 1–4** are straightforward given existing codebase. **Items 5–6** match what's already computed in `data_loader.py` — the `msg.targets` set in `IRCMessage` is already the reference dependency graph for r-GCN.

**Summary**: Current architecture leaves ~14+ F1 points on the table vs. StructBERT (ACL 2022 SOTA). Biggest gains come from structural modules (speaker MHSA + reference r-GCN).

## 6. Robustness & Diagnostics
- **OOM Recovery**: Training and evaluation loops catch CUDA Out-of-Memory errors, log memory state, clear cache, and skip the problematic batch.
- **Numerical Safety**: NaN/Inf loss detection triggers batch skipping to prevent weight corruption.
- **Smart Logging**: Automatic logging of any batch containing a positive sample (`label=1`) to monitor minority class behavior.
- **Data Starvation Prevention**: Test runs must use message offsets (e.g., 300+ or 1000+) or the `tiny` dataset to avoid the link-less "join/quit" noise at the start of IRC logs.
- **Atomic Checkpointing**: Checkpoints are saved to `.tmp` files and renamed to avoid Windows file-locking conflicts (Error 1224).

---

## 5. Clustering & Thread-Level Metrics
- **Current**: Only link-level F1 is computed.
- **Gap**: Actual task is conversation disentanglement (grouping messages into threads). Needs a **clustering step** after link prediction.
- **Methods**: Union-Find (simple) or bipartite graph matching.
- **Metrics Needed**: VI (Variational Inference), ARI (Adjusted Rand Index), MCF (Message Clustering F1) for thesis visualization component.

## 6. Technical Reference

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
- **Conda Module**: `miniconda3/23.9.0-0` (via `$EBROOTMINICONDA3`)
- **Smoke Test**: [`smoke_test.slurm`](../smoke_test.slurm) — minimal end-to-end test (30 min, 500 pairs)
- **Full Training**: [`run_job.slurm`](../run_job.slurm) — 8-hour training job
- **Error handling**: `set -e` in SLURM scripts aborts on any command failure
- **Output directories**: All heavy outputs relocated to `/scratch/user/$USER/ircbert_runs`

### Known Issues & Fixes
- **OOM with num_workers>0**: Resolved via **lazy tokenization** in `data_loader.py`. Dataset now stores raw text pairs instead of pre-tokenized tensors. Tokenization happens in `__getitem__` on-the-fly, reducing per-worker memory from ~1.5GB to ~200MB (~85% reduction).

---

## 8. Key References
1. Kummerfeld et al. (2019). "A Large-Scale Corpus for Conversation Disentanglement." ACL 2019.
2. Zhu et al. (2021). "BERT for Conversation Disentanglement." (Key feature comparison paper).
3. Huang et al. (2022). "Bi-Level Contrastive Learning for Conversation Disentanglement."
4. **StructBERT** (ACL 2022). "Structural Encoding for Conversation Disentanglement." [aclanthology.org/2022.acl-long.23.pdf](https://aclanthology.org/2022.acl-long.23.pdf)
5. **ROCLING 2025**. "Benchmarking Transformer Models on Ubuntu IRC Dataset." [aclanthology.org/2025.rocling-main.31.pdf](https://aclanthology.org/2025.rocling-main.31.pdf)
