# IRC Conversation Disentanglement — BERT/DeBERTa Cross-Encoder

A PyTorch implementation of conversation disentanglement for multi-party IRC chat logs, using a BERT/DeBERTa cross-encoder with handcrafted feature augmentation.

**Task**: Given an interleaved multi-party chat log (no explicit reply-to mechanism), determine which previous message each new message replies to — i.e., reconstruct coherent conversational threads from a flat message stream.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Key References](#key-references)

---

## Quick Start

```bash
# Windows
setup.bat

# Linux
chmod +x setup.sh && ./setup.sh

# Verify setup
pytest tests/ -x -q
```

Requires Python 3.13.11+. The first `pytest` run will be slow due to PyTorch/Transformers initialization.

---

## Dataset

**Ubuntu IRC Dataset** (Kummerfeld et al., ACL 2019) — the largest public benchmark for conversation disentanglement.

| Split   | Messages | Files |
|---------|----------|-------|
| Train   | 67,463   | ~158  |
| Dev     | 2,500    | ~10   |
| Test    | 5,000    | ~20   |

Each message has a gold annotation: the index of the message it replies to. 75.9% are cross-links (replying to someone else), 24.1% are self-links (starting a new thread).

**Annotation format** (critical — a bug was fixed here):
```
PARENT CHILD -
```
Column 0 = parent index, column 1 = child index. The original code had these swapped, which silently invalidated all cross-links during early training runs.

**Data format** per message:
| Field         | Example                                     |
|---------------|---------------------------------------------|
| `id`          | `1050`                                      |
| `ascii`       | `"[03:57] <Xophe> (also, I'm guessing...)"` |
| `connections` | `[1048, 1054, 1055]`                        |

The `ascii` field contains timestamp and speaker name, which are parsed for handcrafted features.

---

## Architecture

### Multiclass Formulation

The problem is framed as "which of C candidates is the parent of message i?":

- **Input**: Child message encoded once, C candidate parents processed independently through BERT
- **Shape**: `input_ids [batch, C, seq_len]` — C candidates per sample, padded to max C in batch
- **Processing**: Flatten to `[batch*C, seq_len]` for BERT, reshape back to `[batch, C, hidden]`
- **Output**: Softmax over C candidates — probability distribution over candidate parents
- **Loss**: `CrossEntropyLoss` with label smoothing — no pos_weight needed, no threshold needed
- **Prediction**: Argmax over C candidates

### Handcrafted Features

Five features concatenated to the 768-dim BERT [CLS] vector (→ 773-dim input to classifier):

1. **Time difference**: Minutes between messages (strongest single signal)
2. **Speaker match**: 1 if same speaker, 0 otherwise
3. **Position distance**: `i - j`
4. **Word overlap**: Jaccard similarity of word sets
5. **Directedness**: 1 if child message @mentions parent's speaker

Zhu et al. (2021) found a 25-point F1 gap between raw BERT and BERT + features.

---

## Training

```bash
python src/train.py \
  --model-name microsoft/deberta-v3-base \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --learning-rate 3e-5 \
  --max-length 96 \
  --max-dist 30 \
  --epochs 10 \
  --patience 3 \
  --warmup-ratio 0.15
```

Key hyperparameters:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--model-name` | `microsoft/deberta-v3-base` | Also supports `bert-base-uncased` |
| `--max-length` | 96 | Covers 95%+ of IRC messages |
| `--max-dist` | 50 | Candidate window. 98.1% of gold cross-links within 50 |
| `--batch-size` | 8 | Effective batch = batch × accumulation |
| `--gradient-accumulation-steps` | 2 | Required for DeBERTa-v3 at batch > 1 |
| `--learning-rate` | 3e-5 | Standard BERT fine-tuning range |
| `--warmup-ratio` | 0.15 | Linear warmup over 15% of total steps |
| `--label-smoothing` | 0.1 | Prevents logit overflow in 49-class softmax |
| `--freeze-bert` | (flag) | Freeze BERT, train only classifier head |

**Numerical stability** (hard-won): Any training on 49-class softmax with DeBERTa-v3-base must use (1) logit clamping to [-50, 50], (2) label smoothing ≥ 0.1, (3) AdamW eps ≥ 1e-4. Without all three, logits overflow fp32 and produce NaN loss.

### HPC (UQ Bunya)

Training was conducted on UQ Bunya HPC (SLURM scheduler) using H100 and A100 GPUs. Submit via:

```bash
sbatch run_job.slurm
```

Always pass `--constraint=cuda48gb|cuda80gb` to avoid MIG partitions. See `run_job.slurm` for the full SLURM configuration.

---

## Evaluation

```bash
python src/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --split dev
```

Metrics: accuracy, precision, recall, link-level F1, ARI (Adjusted Rand Index), VI (Variation of Information).

For HPC evaluation:
```bash
sbatch eval_job.slurm
```

### Exporting Predictions for Visualization

```bash
python src/evaluate_pred.py \
  --checkpoint checkpoints/best_model.pt \
  --split test \
  --export-json
```

Outputs JSON files matching the visualizer schema in `app/predicted_data/`.

---

## Visualization

Two web apps for side-by-side qualitative comparison of gold vs. predicted thread structure:

```bash
# Gold annotations (port 8080)
cd app && python -m http.server 8080

# Model predictions (port 8081)
cd app2 && python -m http.server 8081
```

**Three-panel layout**:
- **Left** — Thread legend with coloured dots and message counts
- **Center** — Chronological message list with thread-coloured left borders
- **Right** — SVG node-link diagram with Bezier curves (static vertical alignment, not force-directed)

Hover any message or node to highlight its entire thread and dim unrelated messages.

**Data pipeline**:
- Gold JSON: `scripts/export_chat_json.py` → parses `.ascii.txt` + `.annotation.txt` files
- Predicted JSON: `src/evaluate_pred.py --export-json` → runs model checkpoint on evaluation data

---

## Testing

99 tests across 8 test files, all passing in ~40s on CPU:

```bash
pytest tests/ -x -q
```

| Test file | Tests | What it covers |
|-----------|-------|----------------|
| `test_data_loader.py` | 8 | `__getitem__`, `parse_irc_line` |
| `test_create_samples.py` | 5 | `_create_samples_for_conversation`, `compute_features` |
| `test_load_conversation.py` | 6 | `load_conversation`, `load_dataset_files` |
| `test_model.py` | 44 | Init, forward, prediction, architecture, loss |
| `test_train_pipeline.py` | 9 | `collate_fn`, `create_dataloaders` |
| `test_evaluate.py` | 12 | `evaluate()` metrics, loss, edge cases |
| `test_checkpoint.py` | 14 | `save_checkpoint`, `load_checkpoint` |
| `test_parse_args.py` | 20 | All 20 CLI argument defaults and parsing |

---

## Project Structure

```
├── src/
│   ├── data_loader.py      # File discovery, message parsing, multiclass sample generation
│   ├── model.py            # CrossEncoderWithFeatures — BERT/DeBERTa + classifier head
│   ├── train.py            # Training loop, evaluation, checkpointing
│   ├── evaluate.py         # Evaluation script (metrics, verbose output)
│   └── evaluate_pred.py    # Prediction export to JSON (for visualizer)
├── tests/                  # 99 pytest tests
├── app/                    # Gold annotation visualizer (port 8080)
├── app2/                   # Predicted thread visualizer (port 8081)
├── scripts/
│   ├── export_chat_json.py # Export gold annotations to JSON
│   ├── generate_synthetic_data.py  # Synthetic conversation generator
│   ├── analyze_gold_clusters.py    # Cluster analysis
│   └── diagnose_links.py           # Link diagnostics
├── data/                   # Ubuntu IRC dataset
│   ├── tiny/               # Small subset for local testing
│   ├── train/              # Training files
│   ├── dev/                # Dev files
│   └── test/               # Test files
├── context/                # Project knowledge base
│   ├── CONTEXT.md          # Stable project knowledge (architecture, invariants)
│   ├── PROGRESS.md         # Dynamic working state and history
│   └── INSTRUCTIONS.md     # Agent behavioral rules
├── research/               # Research notes and analysis
├── run_job.slurm           # SLURM training job
├── eval_job.slurm          # SLURM evaluation job
├── setup.bat               # Windows setup
└── setup.sh                # Linux setup
```

---

## Key References

1. Kummerfeld et al. (2019). "A Large-Scale Corpus for Conversation Disentanglement." ACL 2019. — *Dataset and GloVe baseline.*
2. Zhu et al. (2021). "BERT for Conversation Disentanglement." — *Feature comparison: BERT + handcrafted features.*
3. Huang et al. (2022). "Bi-Level Contrastive Learning for Conversation Disentanglement." — *SOTA approach.*
4. StructBERT (ACL 2022). "Structural Encoding for Conversation Disentanglement." — *Speaker-masked MHSA + r-GCN.*
5. ROCLING 2025. "Benchmarking Transformer Models on Ubuntu IRC Dataset." — *DeBERTa-v3-base achieves 72.30% link F1.*

