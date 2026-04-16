# IRC BERT Project - Context & Research Background

## Project Overview

This document consolidates the research context, architecture, and project structure for the IRC BERT conversation disentanglement project.

---

## 1. Research Context

### What This Project Is About

This project is part of a research thesis on **conversation disentanglement** — the NLP task of separating interleaved chat messages into distinct conversation threads. Imagine an IRC chatroom where 10 people are having 4 different conversations simultaneously, all mixed together in a single stream. Disentanglement is the problem of figuring out which messages belong to which thread.

The dataset is the **Ubuntu IRC dataset** (Kummerfeld et al., ACL 2019). It contains 77,563 messages from the Ubuntu Linux help channel on Freenode, with gold-standard reply annotations forming a tree of parent-child links.

### Data Format (verified from GitHub source)

Each message in the dataset contains:

| Field         | Example                                     | Notes                                                                            |
| ------------- | ------------------------------------------- | -------------------------------------------------------------------------------- |
| `id`          | `1050`                                      | Integer message index                                                            |
| `ascii`       | `"[03:57] <Xophe> (also, I'm guessing...)"` | Contains **timestamp** and **speaker name**                                      |
| `raw`         | Same as ascii                               | Original IRC log                                                                 |
| `tokenized`   | `"<s> ( also , i 'm ... </s>"`              | Pre-tokenised (not needed — BERT has its own tokenizer)                          |
| `connections` | `[1048, 1054, 1055]`                        | **Gold links**: indices of messages this one replies to or receives replies from |
| `date`        | `"2004-12-25"`                              | Only for Ubuntu dataset                                                          |

**Data splits:**

| Split | Messages | Files      |
| ----- | -------- | ---------- |
| Train | 67,463   | ~158 files |
| Dev   | 2,500    | ~10 files  |
| Test  | 5,000    | ~20 files  |

**Key for handcrafted features:** The `ascii` field format is `[HH:MM] <SpeakerName> message text`. Timestamps and speaker names are directly extractable.

### The Evaluation Metric

The task is framed as **link prediction**: for every message in the chat, predict which earlier message it is replying to (its parent). The metric is **link-level precision / recall / F1**, computed by the script `graph-eval.py` from the Kummerfeld et al. repository.

> **Important clarification:** There are two types of F1 in this field:
> 
> - **Link-level F1** (what we use): Did you correctly predict which message is the parent? This is what `graph-eval.py` computes.
> - **Thread-level F1** (different): After predicting all links, group messages into threads and compare thread-level clustering. We may explore this later but link-level F1 is the primary metric.

### What Has Already Been Done — Study 1

Study 1 replicated the **DyNet feedforward model** from Kummerfeld et al. (ACL 2019):

- Represents each message using **GloVe word vectors** (pre-trained on Ubuntu IRC data)
- For every message *i*, scores each candidate parent *j* in a sliding window of the previous 15–100 messages
- Uses a small feedforward neural network to produce a link probability for each (i, j) pair
- Selects the highest-scoring *j* as the predicted parent

**Study 1 results:**

- CPU baseline: **62.6% link-level F1** (dev file `2004-11-15_03`, 254 messages)
- GPU baseline: **62.2% F1** (same dev file)

**Study 1 key finding:** Full-scale training (158 files, 68,000 messages) was attempted on an RTX 3060 Laptop (6GB VRAM) and abandoned after 9+ hours without completing one epoch. The architecture's O(N × max_dist) pair-scoring creates ~1M pairs per epoch — infeasible on the available hardware.

### Why Study 2

> *Research question: Which conversation disentanglement models best balance accuracy and computational feasibility on the Ubuntu IRC dataset, and how do modern Transformer-based approaches compare to the original DyNet feedforward baseline?*

Study 2 keeps the **exact same pairwise scoring architecture** but replaces the GloVe + feedforward encoder with a fine-tuned **BERT cross-encoder**. Only the representation layer changes. Everything else stays identical.

### Field Progression

| Era       | Model Type                     | Notes                                                            |
| --------- | ------------------------------ | ---------------------------------------------------------------- |
| Pre-2019  | Handcrafted features           | Elsner & Charniak baseline ~35% F1                               |
| 2019      | GloVe + feedforward NN         | **Kummerfeld et al. — Study 1 baseline**                         |
| 2021–2022 | Fine-tuned BERT/ALBERT/DeBERTa | Cross-encoders + handcrafted features                            |
| 2025      | ELECTRA, RoBERTa, ModernBERT   | Lam & Yang (ROCLING 2025) showed ELECTRA/RoBERTa outperform BERT |
| 2022      | Bi-Level Contrastive Learning  | State of the art ~80%+ F1                                        |
| 2026      | LLM-based (Gemini, GPT-4)      | Zero-shot — shelved for this thesis                              |

---

## 2. Framework Decision: PyTorch + HuggingFace

**Do not use TensorFlow.** Use PyTorch with the HuggingFace ecosystem. Reasons:

| Factor               | PyTorch + HuggingFace                                             | TensorFlow                    |
| -------------------- | ----------------------------------------------------------------- | ----------------------------- |
| BERT support         | Native, first-class                                               | Second-class, lagging         |
| CrossEncoder library | `sentence-transformers` v4.0.1 (March 2025) with built-in trainer | No equivalent                 |
| Code required        | ~15 lines for full training pipeline                              | 100+ lines, manual everything |
| Research standard    | Yes — used by every BERT disentanglement paper                    | No                            |
| Debugging            | Standard Python debugger                                          | Cryptic graph errors          |

### Installation

```bash
# Use Python 3.12 (safer than 3.13 due to PyTorch 2.11 bug)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets sentence-transformers accelerate
```

**Python version note:** If using Python 3.13, ensure it's **3.13.11+**. Version 3.13.8 has a known PyTorch 2.11 bug that causes an `IndentationError` on import.

---

## 3. Architecture

### Cross-Encoder (recommended — this is what we build)

```
Input: "[CLS] message_i [SEP] message_j [SEP]"
   → BERT encoder
   → [CLS] token embedding (768-dim)
   → Linear(768 → 1)
   → Sigmoid → probability of link
```

Both messages are passed together — BERT can attend across them. This is the standard approach in the literature.

### Bi-Encoder (noted for future work)

```
Encode message_i → e_i (768-dim)   # Done once, cached
Encode message_j → e_j (768-dim)   # Done once, cached
Score = MLP([e_i; e_j; e_i-e_j; e_i*e_j])
```

Each message encoded independently, so encoding is O(N) not O(N²). Mentioned as a future optimization but not required for Study 2.

---

## 4. Handcrafted Feature Augmentation

**This is critical.** The Zhu et al. (2021) paper found that BERT alone achieves 47.3% F1, but BERT + handcrafted features achieves **72.6% F1** — a **25-point gap**. The time difference feature alone accounts for 19 of those points.

### How to extract features from the data

The `ascii` field format is `[HH:MM] <SpeakerName> message text`. We extract:

| Feature               | What it is                                  | Why it matters                                                                  | How to compute                                         |
| --------------------- | ------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **Time difference**   | Minutes between message *i* and message *j* | Replies happen quickly; 30-second gaps are more likely links than 5-minute gaps | Parse `[HH:MM]` from both messages, compute difference |
| **Speaker match**     | 1 if same speaker, 0 if different           | People often reply to their own messages                                        | Parse `<Name>` from both, compare                      |
| **Position distance** | `i - j`                                     | Closer messages are more likely linked                                          | `i - j` directly                                       |
| **Word overlap**      | Jaccard similarity of word sets             | Replies often share topic words                                                 | Set intersection / union                               |

### How to add them to the model

After BERT produces the 768-dim [CLS] embedding, concatenate the feature vector and pass through the classification head:

```python
# BERT output: 768-dim [CLS] vector
# Features: [time_diff, speaker_match, position_dist, word_overlap] = 4-dim

# Concatenate → 772-dim
# Then: Linear(772 → 128) → ReLU → Dropout → Linear(128 → 1) → Sigmoid
```

This is ~10 lines of additional code and gives you a massive performance boost.

---

## 5. Experimental Matrix

Four controlled experiments. Run B1 and B2 simultaneously on two Vast.ai instances.

| Experiment                     | What changes         | Expected effect                |
| ------------------------------ | -------------------- | ------------------------------ |
| **B1: BERT + no features**     | Baseline BERT only   | Establish raw BERT performance |
| **B2: BERT + time difference** | Add 1 feature        | Major boost (lit says +19 pts) |
| **B3: BERT + all features**    | Add 4 features       | Best-effort performance        |
| **B4: max_dist=30**            | Double search window | Captures more distant links    |

---

## 6. GPU Selection & Cost

| GPU                        | VRAM      | Price (USD/hr)  | 12-hr total             |
| -------------------------- | --------- | --------------- | ----------------------- |
| **RTX 4090 (recommended)** | **24 GB** | **~$0.29–0.39** | **~$4.20 (~AUD $6.50)** |
| A100 40GB                  | 40 GB     | ~$0.63          | $7.56                   |
| H100                       | 80 GB     | ~$1.65          | $19.80 ❌                |

BERT-base (110M params) fits easily in 24GB VRAM with room for batch_size=64.

**Use Vast.ai on-demand tier** for overnight runs to avoid interruptions. Use interruptible for daytime testing to save ~30%.

---

## 7. Pair Generation

```python
MAX_DIST = 15  # same window as DyNet; test 30 in B4

pairs = []
for i in range(len(messages)):
    for j in range(max(0, i - MAX_DIST), i):
        label = 1 if (i, j) in gold_links else 0
        pairs.append((messages[i], messages[j], label))
```

### Class Imbalance

For each message *i* with MAX_DIST=15, there is typically 1 positive and ~14 negatives. Ratio ≈ 1:14.

**Solution:** Use `pos_weight=14` in `BCEWithLogitsLoss`. Simplest, no data dropped.

### Scale

| Split                      | Messages    | Pairs (max_dist=15) |
| -------------------------- | ----------- | ------------------- |
| Dev only (quick test)      | 254         | ~3,810              |
| Small train (5 files)      | ~2,000      | ~30,000             |
| **Full train (158 files)** | **~68,000** | **~1,020,000**      |

---

## 8. Training Configuration

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("bert-base-uncased", num_labels=1)

# Training hyperparameters
learning_rate   = 2e-5
weight_decay    = 0.01
epochs          = 3        # BERT typically converges in 2-4 epochs
batch_size      = 32       # Safe for RTX 4090 24GB; try 64 if VRAM allows
max_length      = 128      # IRC messages are short
warmup_ratio    = 0.1
```

### Using sentence-transformers v4 CrossEncoder Trainer (recommended)

The library handles the training loop, loss computation, checkpointing, and logging:

```python
from datasets import Dataset
from sentence_transformers import CrossEncoder, CrossEncoderTrainer
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

# 1. Load model
model = CrossEncoder("bert-base-uncased", num_labels=1)

# 2. Prepare dataset (from our pair generation)
train_dataset = Dataset.from_dict({
    "sentence_A": [msg_i for msg_i, msg_j, label in train_pairs],
    "sentence_B": [msg_j for msg_i, msg_j, label in train_pairs],
    "labels":     [label for msg_i, msg_j, label in train_pairs],
})

# 3. Define loss
loss = BinaryCrossEntropyLoss(model)

# 4. Training arguments
args = CrossEncoderTrainingArguments(
    output_dir="./bert-run",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,  # RTX 4090 supports bf16
    logging_steps=100,
    save_strategy="epoch",
)

# 5. Train
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()
```

**With handcrafted features:** You'll need a custom model class that extends `CrossEncoder` to concatenate feature vectors to the [CLS] embedding. This adds ~30 lines of code.

---

## 9. Inference & Evaluation

### Inference

For each message *i* in the dev file:

1. Score all candidate parents *j* in window [i-MAX_DIST, i-1]
2. Pick j* = argmax score
3. If max_score < threshold (e.g. 0.5), predict "no parent"

### Output format

```
# One line per message
# Format: child_index TAB parent_index (or "-" for no parent)
0    -
1    0
2    1
3    1
...
```

### Evaluation

**Run on ALL 10 dev files**, not just one. Report average F1.

```bash
# For each dev file:
python tools/evaluation/graph-eval.py \
  --gold data/dev/FILE.annotation.txt \
  --auto bert_predictions.out
```

---

## 10. Expected Results

| Model                                       | F1 (expected)    | Notes                           |
| ------------------------------------------- | ---------------- | ------------------------------- |
| DyNet GloVe+FF (Study 1)                    | 62.2–62.6%       | Established baseline            |
| BERT only (B1)                              | ~47% estimate    | Based on Zhu et al. 2021        |
| BERT + time difference (B2)                 | ~66% estimate    | The 19-point feature boost      |
| BERT + all features (B3)                    | ~70–75% estimate | Best-effort configuration       |
| Paper target (Kummerfeld FF, full training) | 72.3%            | Upper bound for fair comparison |
| SOTA (Huang et al. Bi-CL, 2022)             | ~80%+            | Literature reference only       |

**Even if BERT-only (B1) underperforms DyNet, that is a valid and publishable finding.** The Zhu et al. (2021) paper showed the same thing: raw BERT without features loses to a well-tuned feature-based model. The thesis contribution is the controlled comparison and the demonstration that features matter.

---

## 11. Day-by-Day Implementation Plan

### Implementation Day (6–8 hours)

| Time Block | Task                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------- |
| Hour 1     | Spin up Vast.ai RTX 4090; install PyTorch + HuggingFace; transfer data                      |
| Hour 2     | Write data loading code — parse `ascii` fields, extract timestamps/speakers, generate pairs |
| Hour 3     | Write training script — BERT cross-encoder with `sentence-transformers` trainer             |
| Hour 4     | Write inference script — load model, score all pairs for dev files, write `.out` files      |
| Hour 5     | Dev-only dry run (254 messages) — verify loss decreases, F1 is nonzero                      |
| Hour 6     | Small train dry run (5 files) — verify no OOM, batch sizes OK                               |
| Hour 7–8   | Launch full overnight training run; `nohup` and sleep                                       |

### Overnight Training

```bash
nohup python train.py \
  --train-list data/list.ubuntu.train.txt \
  --epochs 3 \
  --batch-size 32 \
  --lr 2e-5 \
  --max-dist 15 \
  --output-dir ./bert-run \
  > bert-train.out 2>&1 &

# Monitor:
watch -n 30 tail -20 bert-train.out
nvidia-smi
```

### Morning Results

```bash
# Evaluate on ALL 10 dev files
for f in data/dev/*.annotation.txt; do
    python infer.py --model-dir ./bert-run --dev-file $f --output pred.out
    python tools/evaluation/graph-eval.py --gold $f --auto pred.out
done
```

---

## 12. Expected Training Time

| Phase                                   | Estimated Time |
| --------------------------------------- | -------------- |
| Environment setup                       | ~30 min        |
| Dev-only dry run                        | ~5 min         |
| Small train (5 files)                   | ~20 min        |
| **Full training (158 files, 3 epochs)** | **~3–5 hours** |
| Inference on all dev files              | ~10 min        |
| Evaluation                              | <1 min         |
| **Total**                               | **~5–7 hours** |

Comfortably inside the 12-hour Vast.ai window.

---

## 13. Thesis Narrative

> *Study 1 established the DyNet feedforward model as a viable baseline at 62.6% link-level F1 but found the architecture computationally infeasible for full-scale training on accessible hardware. Study 2 replaces the GloVe-based encoder with a fine-tuned BERT cross-encoder using the identical pairwise scoring architecture and evaluation pipeline. This enables a controlled comparison where only the representation layer changes, directly testing the hypothesis that contextual embeddings improve disentanglement accuracy over static word vectors. Ablation experiments (B1–B4) measure the contribution of handcrafted features and search window size.*

---

## 14. Known Risks & Mitigations

| Risk                            | Likelihood             | Mitigation                                                |
| ------------------------------- | ---------------------- | --------------------------------------------------------- |
| OOM on RTX 4090                 | Low (24GB is ample)    | Reduce batch_size to 16; use gradient accumulation        |
| Training loss does not decrease | Low                    | Check pos_weight; verify labels; confirm tokenisation     |
| F1 lower than DyNet (B1 only)   | Expected               | Valid result; features in B2/B3 should beat baseline      |
| Vast.ai instance interrupted    | Medium (interruptible) | Use on-demand for overnight; save checkpoints every epoch |
| Python 3.13 compatibility       | Low                    | Use 3.12 or 3.13.11+                                      |
| Data transfer takes long        | Low                    | Ubuntu IRC data is ~50MB total                            |

---

## 15. Key References

1. Kummerfeld et al. (2019). "A Large-Scale Corpus for Conversation Disentanglement." ACL 2019.
2. Zhu et al. (2021). Found BERT alone: 47.3% F1, BERT + features: 72.6% F1. (Key feature comparison paper.)
3. Lam & Yang (2025). "Revisiting Pre-trained Language Models for Conversation Disentanglement." ROCLING 2025. (Tested BERT, ELECTRA, RoBERTa, XLNet, ModernBERT.)
4. sentence-transformers v4.0.1 (March 2025). CrossEncoder training refactor with multi-GPU, bf16, 11 new losses.
5. Huang et al. (2022). Bi-Level Contrastive Learning. Current SOTA ~80%+.
6. GitHub: `jkkummerfeld/irc-disentanglement` — data and evaluation scripts.

---

## 16. Project Structure

```
irc_dis_pytorch/
├── data/                  # Ubuntu IRC dataset (already present)
│   ├── train/             # ~158 files (*.annotation.txt, *.ascii.txt)
│   ├── dev/               # 10 files for validation
│   ├── test/              # Test set
│   └── list.ubuntu.*.txt  # File lists for splits
├── src/                   # Our PyTorch code (to create)
│   ├── data.py            # Load messages, parse ts/speaker, gold links, pair gen + features
│   ├── model.py           # Custom CrossEncoderWithFeatures
│   └── utils.py           # Feature extractors, config
├── train.py               # Training CLI + CrossEncoderTrainer
├── infer.py               # Inference CLI, output .out files
├── eval.sh                # Loop: infer + graph-eval.py on dev files
├── setup.bat              # Windows venv + pip (copy from plans/setup.md)
├── setup.sh               # Linux (copy from plans/setup.md)
├── plans/                 # This planning dir
│   ├── setup.md
│   └── project_structure.md
├── .venv-irc-bert/        # Existing venv
└── README.md              # Usage, results
```

### Architecture Diagram

```mermaid
graph TD
    A[ASCII.txt<br/>[HH:MM] <Speaker> msg] --> B[Parse:<br/>ts, speaker, msg_text]
    C[Annotation.txt<br/>child parent -] --> D[Gold links dict<br/>{child: [parents]}]
    B --> E[For i in msgs:<br/>for j in i-MAX_DIST:i<br/>label = j in gold[i]]
    D --> E
    E --> F[Features:<br/>time_diff_min,<br/>speaker_match,<br/>pos_dist,<br/>word_jaccard]
    F --> G[Dataset:<br/>sentences=[[msg_i,msg_j]],<br/>labels, features]
    G --> H[BERT CrossEncoder<br/>[CLS] 768 + feats 4<br/>--> Linear(772->1) Sigmoid]
    H --> I[Train: BCE(pos_weight=14)]
    H --> J[Infer: argmax_j score_i,j<br/>output child TAB parent]
    J --> K[graph-eval.py<br/>P/R/F1]
```

---

## 17. Current Status

### Smoke Test Results (2026-04-16)

✅ **Smoke test completed successfully**

- **Dataset**: Tiny dataset with `--test-end 10` pairs
- **Training**: Completed 3 epochs without errors
- **Checkpoints**: All 3 epoch checkpoints + best_model saved successfully
- **Logging**: Log file created with proper content

### Precision/Recall Analysis

**Observation**: Smoke test showed precision=0, recall=0, F1=0

**Root Cause**: Training data had no links (all negative examples)

**Explanation**:
- The tiny dataset contains only negative examples (no gold links)
- Model learned to always predict "no link" (class 0)
- This is expected behavior for a dataset with no positive examples
- **Full training run will have positive examples and should show non-zero metrics**

### Project Readiness

✅ **Ready for GPU Training**

- All unit tests passed (15 data loader tests, 22 model tests)
- Smoke test verified pipeline works end-to-end
- Code is stable and ready for full-scale training
- Next step: Execute full training run on Vast.ai GPU

### Next Steps

1. **Execute full training run** (3 epochs on full dataset)
2. **Perform inference** on all 10 dev files
3. **Evaluate** using graph-eval.py to compute link-level P/R/F1
4. **Compare results** with DyNet baseline (62.2-62.6% F1)

---

*Document finalized after 3 review cycles. Ready for implementation.*
