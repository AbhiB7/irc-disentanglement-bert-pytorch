# IRC Disentanglement Project - Technical Handover

## 1. Current Architecture

### Model: Multiclass CrossEncoder with Features
- **File**: `src/model.py`
- **Architecture**: BERT-based CrossEncoder (DeBERTa-v3-base default) with 5 handcrafted features
- **Input**: For each child message, we create C candidate samples (one per potential parent within `max_dist=50`)
- **Each sample**: `(parent_text, child_text, features, label)` where label = gold parent index (0 to C-1)
- **Forward pass**:
  1. BERT processes each candidate independently: `[batch, C, seq]` → flatten → `[batch*C, seq]`
  2. Extract `[CLS]` token embedding (768-dim for BERT-base, 768 for DeBERTa-v3)
  3. Concatenate with 5 features → 773-dim vector per candidate
  4. Linear layer (773 → 1) per candidate → unflatten to `[batch, C]`
  5. Softmax over C candidates → `CrossEntropyLoss`

### Data Loader: `src/data_loader.py`
- **Key function**: `_create_samples_for_conversation()` (lines 350-434)
- **Candidate selection**: For child message at index `i`, collect all `j < i` where `i - j <= max_dist`
- **Self-links**: Currently excluded via `range(max(0, i - max_dist + 1), i)` — j < i, no self-link possible
- **Gold label**: Index of the gold parent within the candidate list (or -1 if outside window)
- **Output**: List of samples, each with `(parent_text, child_text, features_tensor, gold_parent_idx)`

### Training Pipeline: `src/train.py`
- **Loss**: `CrossEntropyLoss` (multiclass, no `pos_weight` needed)
- **Optimizer**: AdamW with linear warmup (10% of total steps)
- **Evaluation**: Pairwise accuracy (fraction of samples where predicted parent = gold parent)

---

## 2. The 100% Accuracy Problem (Recency Bias)

### What Happened
We achieved **100% pairwise accuracy** on dev (462 samples) and test (922 samples) using DeBERTa-v3-base + features. This seemed great, but analysis revealed it was a **mirage**.

### Root Cause: Recency Bias
The gold annotations in the IRC dataset (Kummerfeld et al., ACL 2019) are dominated by the **immediately previous message**. Our diagnostic baselines proved this:

| Baseline | Dev | Test |
|----------|-----|------|
| Our model | 100.0% | 100.0% |
| "Predict the last candidate" (position 49) | 16.0% | 15.4% |
| "Predict most common position" | 16.2% (pos 47) | 16.6% (pos 48) |
| Recency check (% in positions 46-49) | 59.5% | 53.6% |

**Key insight**: Gold parents are spread across 21-24 positions in the window, but the model likely learned a **recency shortcut**—always predicting the most recent candidate—rather than learning content-based thread structure.

---

## 3. Self-Links Exclusion — The Real Problem

### What We Did
On 2026-05-17, we modified `src/data_loader.py` to exclude self-links. The current code (line 376) uses:
```python
for j in range(max(0, i - self.max_dist + 1), i):
```
This correctly excludes `j = i` from the range. No self-link check needed.

### The Real Problem: Self-Links ARE the Training Signal
The Ubuntu IRC dataset's gold annotations (Kummerfeld et al., ACL 2019) have **two types of gold links**:

1. **Self-links** — message links to itself = "this message starts a new conversation thread"
2. **Cross-message links** — message replies to a prior message

**Self-links are NOT bugs.** They are the dataset's way of encoding "new thread starts here." In the Kummerfeld annotation schema, every message has exactly one gold parent. If a message starts a new thread, its gold parent is itself.

### The Consequence of Excluding Self-Links
- Most gold labels in the Ubuntu IRC dataset are self-links (new conversation starters)
- The remaining cross-message links often span **hundreds of messages apart** in IRC conversations
- With self-links excluded and `max_dist=50`, near-zero cross-message links fall within the 50-message window
- **Result: 0 samples created for almost every file**

### Evidence from Logs
```
File 149/153: 2017-03-23.train-c - 1500 messages, 0 total samples so far
File 150/153: 2017-05-09.train-c - 1500 messages, 0 total samples so far
File 151/153: 2017-07-15.train-c - 1500 messages, 0 total samples so far
```

Training completes in **2.30s for 10 epochs** — because there are 0 samples, each epoch is just "save checkpoint" with no actual training.

Evaluation shows:
```
COVERAGE: 0/12500 messages have predictions (0.0%)
Loss: 0.0000, Accuracy: 0.0000, Precision: 0.0000, Recall: 0.0000, F1: 0.0000
```

---

## 4. The Fix: SELF-as-Candidate Architecture

### The Correct Approach
Don't exclude self-links. Instead, make "self" an explicit candidate:

```
candidates = [msg_0, msg_1, ..., msg_{i-1}, SELF]
```

Where `SELF` is a special token/embedding appended at the end of the candidate list. The gold label is either:
- The index of the gold parent (if cross-message link exists within window)
- The index of `SELF` (if the message starts a new thread)

### Why This Works
- **No samples are dropped** — every message has a valid candidate
- **Self-link prediction becomes a learnable signal** — the model learns "does this message start a new thread?"
- **Cross-message links train the content-based ranking** you actually want
- **Recency bias now competes with a real "new thread" class**, breaking the trivial shortcut

### Literature Support
Kummerfeld's own feedforward model uses a threshold below which a message links to itself. This formulation makes that explicit as a candidate class.

### Thesis Contribution
*"We identified that prior work conflates two sub-tasks (new thread detection vs. reply linking) and propose a unified candidate formulation."*

### Implementation Requirements
1. Add a SELF token/embedding to the model
2. Modify `_create_samples_for_conversation` to include SELF as the last candidate
3. Update `collate_fn` to handle variable-C with SELF
4. Update evaluation to handle SELF predictions

### Current Status (2026-05-18)
- Self-links are currently **excluded** via `range(..., i)` in `data_loader.py` line 376
- `max_dist=50` is used as a workaround but produces 0 samples because cross-message links are rare
- The SELF-as-candidate refactor is documented in `context/CONTEXT.md` Section 9

---

## 5. What's Working

- ✅ Multiclass architecture refactor complete (data_loader, model, train, evaluate)
- ✅ Comprehensive test suite (99+ tests, all passing)
- ✅ Synthetic data with interleaved threads (`data/synthetic_interleaved/`)
- ✅ Human-readable evaluation output (`--verbose` flag in `evaluate.py`)
- ✅ All scripts use `--max-dist 50` and `$CONDA_PREFIX/bin/python` (no psutil errors)
- ✅ `context/CONTEXT.md` Section 9 documents the SELF-as-candidate architecture

## 6. What's Broken

- ❌ **Self-links excluded → 0 samples**: The self-link exclusion was conceptually wrong. Self-links ARE the training signal in this dataset.
- ❌ **Training runs on empty data**: 10 epochs in 2.30s, all metrics = 0.0000
- ❌ **Evaluation has 0 predictions**: 0/12500 messages have predictions

## 7. Key Files & Line Numbers

| File | Key Section | Lines |
|------|-------------|-------|
| `src/data_loader.py` | Candidate selection loop (self-link exclusion) | 376 |
| `src/data_loader.py` | `_create_samples_for_conversation()` | 350-434 |
| `src/model.py` | `CrossEncoderWithFeatures.forward()` | 95-221 |
| `src/train.py` | Training loop | (entire file) |
| `src/evaluate.py` | Evaluation with `--verbose` | (entire file) |
| `context/CONTEXT.md` | SELF-as-candidate architecture documentation | Section 9 |
| `learning_signal.sh` | Training script for Bunya GPU | (entire file) |

## 8. Dataset Notes

- **Coverage is only 4-6%**: Dataset only annotates messages 1000+ in each file
- **Gold clusters**: 55-63% singletons (size=1), making clustering metrics (ARI) meaningless
- **Recency bias**: Gold parents are concentrated in the last few positions of the candidate window
- **Self-links dominate**: Most gold labels are self-links (new thread starts)
- **Literature comparison**: Our 100% accuracy beats ALT 2021 (~85%) and ROCLING 2025 (~88%), but this was due to the recency shortcut + self-links

## 9. Relevant Code Snippets

### Bug Location: `src/data_loader.py` (lines 372-387)
```python
# Collect candidates within max_dist (j < i, no self-link possible)
candidates = []
candidate_indices = []  # (conv_idx, msg_i_idx, candidate_idx)

for j in range(max(0, i - self.max_dist + 1), i):
    msg_j = messages[j]

    # Skip system messages as parents
    if msg_j.is_system:
        continue

    candidates.append(msg_j.text)
    candidate_indices.append((conv_idx, i, j))

if not candidates:
    continue  # Skip messages with no valid candidates
```

**Problem**: The loop correctly excludes self-links, but most gold labels ARE self-links. After excluding them, near-zero messages have a gold parent within the candidate window.

**Fix**: Implement SELF-as-candidate architecture (see Section 4).

### Model Architecture: `src/model.py` (forward pass)
```python
def forward(self, input_ids, attention_mask, token_type_ids=None, features=None, labels=None):
    batch_size, num_candidates, seq_len = input_ids.shape
    flat_input_ids = input_ids.view(-1, seq_len)
    flat_attention_mask = attention_mask.view(-1, seq_len)
    bert_outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask, ...)
    cls_embedding = bert_outputs.last_hidden_state[:, 0, :]
    cls_embedding = self.dropout(cls_embedding)
    if features is not None:
        expanded_features = features.reshape(-1, self.num_features)
        combined = torch.cat([cls_embedding, expanded_features], dim=-1)
    logits = self.classifier(combined)
    logits = logits.view(batch_size, num_candidates)
    candidate_probs = torch.softmax(logits, dim=-1)
    ...
```

### Training Script: `learning_signal.sh`
```bash
# Full dataset, max_dist=50, DeBERTa-v3-base, 10 epochs, ALL messages
# Run on Bunya interactive GPU node: bash learning_signal.sh
# All output tee'd to logs/learning_signal_DATETIME.log
$PYTHON src/train.py \
    --mode train \
    --data-dir "$DATA_DIR" \
    --model-name microsoft/deberta-v3-base \
    --max-dist 50 \
    --batch-size 16 \
    --epochs 10 \
    --learning-rate 5e-5 \
    --warmup-ratio 0.1 \
    --patience 3 \
    --eval-every 1 \
    --save-every 1 \
    --output-dir "$CHECKPOINT_DIR" \
    --device cuda 2>&1 | tee -a "$LOG_FILE"
```

## 10. Log Files for Analysis

| Log | Description |
|-----|-------------|
| `logs/learning_signal_20260518_192715.log` | Latest run: max_dist=50, 0 samples, 2.30s training |
| `logs/learning_signal_20260518_190346.log` | Previous run: max_dist=15, test_end=156, 0 samples |
| `logs/train_20260518_192732.log` | Training log from latest run |
| `logs/eval_20260518_192836.log` | Dev evaluation from latest run |
| `logs/eval_20260518_192852.log` | Test evaluation from latest run |