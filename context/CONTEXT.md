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

| Priority | Improvement | Effort | Expected Gain | Status |
|----------|-------------|--------|---------------|--------|
| 1 | **DeBERTa-v3-base backbone** | Change 1 string in `model.py` | ~+1% F1 | ✅ Complete |
| 2 | **Increase max_dist to 50** | Change 1 default argument | Improves recall | ✅ Complete |
| 3 | **Multiclass reframing** | Restructure loss in `model.py` | Eliminates pos_weight heuristic | ✅ Complete |
| 4 | **Union-Find clustering + thread metrics** | New module needed | Required for VI/ARI in thesis | ⏳ Pending |

---

## 9. 🚀 Successful Training Run Guide (2026-05-04)

This section provides the verified instructions for running the multiclass architecture.

### ✅ Verification Status
- **Architecture**: Multiclass Cross-Entropy over search window.
- **Data Model**: Child message encoded once, paired with multiple candidate parents in batch via custom `collate_fn`.
- **Validation**: Pipeline verified locally with `data/tiny`.

### 📊 1. Local Verification (Tiny Test)
Run this to ensure the logic and environment are stable (takes ~15 mins on CPU).
```bash
python src/train.py --data-dir data/tiny --epochs 1 --batch-size 16 --test-end 100
```

### ⚡ 2. HPC Full Run (UQ Bunya)
This is the recommended path for a "successful today" result using A100 GPUs.

#### 🔍 HPC Resource Monitoring (Are GPUs free?)
Use these commands on Bunya to check the queue and GPU availability:

#### 🛠️ Interactive Debug Session (General Partition)
If you need to troubleshoot `ls` issues or run small Python tests interactively:
1.  **Request allocation**: 
    `salloc --partition=general --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=4G --time=01:00:00 --account=a_hcc --qos=debug`
2.  **Drop into the node**: 
    `srun --pty bash`
3.  **Verify**: 
    `hostname` (should show `bunXXX` instead of `bunya3`)

*   **Check GPU availability**: 
    `sinfo -p gpu_cuda -o "%P %G %D %t"` 
    (Look for `idle` under the `STATE` column—that means there are free GPUs ready to take your job!)
*   **Check your specific queue status**: 
    `squeue -u $USER` 
    (If `ST` is `PD`, your job is Pending. If `R`, it is Running.)
*   **See why a job is pending**: 
    `squeue -j [JOB_ID] -o %r` 
    (Common reasons: `Resources` = waiting for GPU, `Priority` = waiting in line.)
*   **Check GPU stats while running**: 
    `srun --jobid [JOB_ID] nvidia-smi` 
    (Check if your model is actually using the GPU VRAM).

**Step A: Submit Smoke Test (30 mins)**
Ensure the Bunya environment handles the new multiclass logic.
```bash
sbatch smoke_test.slurm
```
*Check logs with `tail -f logs/[job_id]_smoke.out`*

**Step B: Submit Full Training (3-8 hours)**
Execute the primary thesis baseline training.
```bash
sbatch train.sh
```

### 📈 Metrics for Success
- **Baseline Accuracy**: Should exceed 0.10 immediately (random). 
- **Link F1**: Target is >0.70 (per ROCLING 2025 benchmarks).
- **Behavior**: Monitor `[POSITIVE BATCH]` tags in the log to verify parent-child link detection.

| Priority | Improvement | Effort | Expected Gain | Status |
|----------|-------------|--------|---------------|--------|
| 1 | **DeBERTa-v3-base backbone** | Change 1 string in `model.py` | ~+1% F1 | ✅ Complete |
| 2 | **Increase max_dist to 50** | Change 1 default argument | Improves recall | ✅ Complete |
| 3 | **Multiclass reframing** | Restructure loss in `model.py` | Eliminates pos_weight heuristic | ✅ Complete |
| 4 | **Union-Find clustering + thread metrics** | New module needed | Required for VI/ARI in thesis | ⏳ Pending |
| 5 | **Speaker-masked MHSA** | Add structural module to `model.py` | ~+11 F1 points over BERT baseline | ⏳ Pending |
| 6 | **@mention r-GCN** | Use existing `msg.targets` | ~+5 F1 points | ⏳ Pending |

**Items 1–3** are complete. **Items 4–6** remain for future implementation.

**Summary**: Current architecture leaves ~14+ F1 points on the table vs. StructBERT (ACL 2022 SOTA). Biggest gains come from structural modules (speaker MHSA + reference r-GCN).

## 6. Robustness & Diagnostics
- **OOM Recovery**: Training and evaluation loops catch CUDA Out-of-Memory errors, log memory state, clear cache, and skip the problematic batch.
- **Numerical Safety**: NaN/Inf loss detection triggers batch skipping to prevent weight corruption.
- **Smart Logging**: Automatic logging of any batch containing a positive sample (`label=1`) to monitor minority class behavior.
- **Data Starvation Prevention**: Test runs must use message offsets (e.g., 300+ or 1000+) or the `tiny` dataset to avoid the link-less "join/quit" noise at the start of IRC logs.
- **Atomic Checkpointing**: Checkpoints are saved to `.tmp` files and renamed to avoid Windows file-locking conflicts (Error 1224).

---

## 6. Clustering & Thread-Level Metrics
- **Current**: Only link-level F1 is computed.
- **Gap**: Actual task is conversation disentanglement (grouping messages into threads). Needs a **clustering step** after link prediction.
- **Methods**: Union-Find (simple) or bipartite graph matching.
- **Metrics Needed**: VI (Variational Inference), ARI (Adjusted Rand Index), MCF (Message Clustering F1) for thesis visualization component.

## 7. Technical Reference

### Project Structure
- `src/data_loader.py`: Handles file discovery, message parsing, and multiclass sample generation.
- `src/model.py`: Defines `CrossEncoderWithFeatures` and model initialization (multiclass output).
- `src/train.py`: Main entry point for training, evaluation, and checkpointing. Includes **Smart Logging** for imbalanced data diagnostics.
- `src/evaluate.py`: Evaluation script for multiclass predictions.
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
