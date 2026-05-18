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
- **Self-links**: Previously included `(i, i)` as a candidate, now excluded (see Section 3)
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

### Why Self-Links Were Suspicious
In the original implementation, we included `(i, i)` as a candidate (self-link). This is semantically invalid (a message cannot be its own parent), but it was included as a "negative candidate" during training. The model could have been "gaming" the system by learning to avoid self-links and default to the nearest valid candidate (the immediately previous message).

---

## 3. Self-Links Exclusion Attempt (Current Bug)

### What We Did
On 2026-05-17, we modified `src/data_loader.py` (lines 376-378) to exclude self-links:
```python
for j in range(max(0, i - self.max_dist + 1), i + 1):
    if j == i:
        continue  # Exclude self-link
    # ... rest of candidate collection
```

### Why It Made Sense
- Self-links `(i, i)` are semantically invalid
- Removing them forces the model to learn content-based relationships
- Synthetic data (`data/synthetic_interleaved/`) was created with interleaved threads to test this

### The Bug Introduced
**The code creates 0 samples for early messages**, causing training to fail silently:

1. **Loop range issue**: `range(max(0, i - max_dist + 1), i + 1)` includes `j = i`
2. **Self-link exclusion**: Skips `j == i`, but for early messages (small `i`), the range may ONLY contain `j = i`
3. **Example**:
   - Message `i=0`: range is `range(0, 1)` → only `[0]`. After skipping `j==i`, candidates = `[]`
   - Message `i=1`: range is `range(0, 2)` → `[0, 1]`. If message 0 is a system message (skipped), only `j=1` remains → skipped as self-link

4. **Result**: For `data/tiny` (250 messages, 188 gold links), **0 samples are created** → training runs 10 epochs in 3.44s with no data, all metrics = 0.0000

### The Fix Needed
Change line 376 to exclude `i` from the range entirely:
```python
# OLD (buggy):
for j in range(max(0, i - self.max_dist + 1), i + 1):

# FIX:
for j in range(max(0, i - self.max_dist + 1), i):  # j < i, no self-link possible
```
Then **remove lines 377-378** (self-link check is no longer needed).

---

## 4. Current State & Problems

### What's Working
- ✅ Multiclass architecture refactor complete (data_loader, model, train, evaluate)
- ✅ Comprehensive test suite (99+ tests, all passing before self-links change)
- ✅ Synthetic data with interleaved threads (`data/synthetic_interleaved/`)
- ✅ Human-readable evaluation output (`--verbose` flag in `evaluate.py`)

### What's Broken
- ❌ **Self-links exclusion bug**: 0 samples created → training produces no learning signal
- ❌ **Log files not recording**: `train_synthetic.sh` doesn't redirect output to dated log files in `/scratch/user/$USER/ircbert_runs/logs/`

### What We're Trying to Do
1. **Fix the 0 samples bug** by correcting the candidate selection loop
2. **Verify the fix** produces samples and allows real training
3. **Train without self-links** to see if the model learns content-based relationships (not just recency)
4. **Evaluate on synthetic data** to confirm the model can handle interleaved threads

### Open Questions
- Will removing self-links actually force the model to learn content, or will it just learn to predict the second-to-last message?
- Is the recency bias a dataset artifact (gold labels ARE mostly recent) or a model failure?
- Should we use the synthetic interleaved data to prove the model can handle non-recent threads?

---

## 5. Key Files & Line Numbers

| File | Key Section | Lines |
|------|-------------|-------|
| `src/data_loader.py` | Candidate selection loop (BUG HERE) | 376-378 |
| `src/data_loader.py` | `_create_samples_for_conversation()` | 350-434 |
| `src/model.py` | `CrossEncoderWithFeatures.forward()` | 95-221 |
| `src/train.py` | Training loop | (entire file) |
| `src/evaluate.py` | Evaluation with `--verbose` | (entire file) |
| `train_synthetic.sh` | Training script for Bunya GPU | (entire file) |

---

## 6. Next Steps for Fix

1. **Fix `src/data_loader.py` line 376**: Change `i + 1` to `i` in the range, remove self-link check
2. **Run tests**: `pytest tests/ -x -q` to verify fix doesn't break anything
3. **Test on `data/tiny`**: Confirm samples are created (should be ~250 samples, not 0)
4. **Train on Bunya**: Use `train_synthetic.sh` (after fixing log redirection)
5. **Evaluate**: Check if accuracy drops from 100% (indicating real learning vs recency shortcut)

---

## 7. Dataset Notes

- **Coverage is only 4-6%**: Dataset only annotates messages 1000+ in each file
- **Gold clusters**: 55-63% singletons (size=1), making clustering metrics (ARI) meaningless
- **Recency bias**: Gold parents are concentrated in the last few positions of the candidate window
- **Literature comparison**: Our 100% accuracy beats ALT 2021 (~85%) and ROCLING 2025 (~88%), but this may be due to the recency shortcut

---

## 8. Relevant Code Snippets

### Bug Location: `src/data_loader.py` (lines 370-390)
```python
# Collect candidates within max_dist (excluding self-link)
candidates = []
candidate_indices = []  # (conv_idx, msg_i_idx, candidate_idx)

for j in range(max(0, i - self.max_dist + 1), i + 1):
    if j == i:
        continue  # Exclude self-link
    msg_j = messages[j]

    # Skip system messages as parents
    if msg_j.is_system:
        continue

    candidates.append(msg_j.text)
    candidate_indices.append((conv_idx, i, j))

if not candidates:
    continue  # Skip messages with no valid candidates
```

**Problem**: The loop range `range(max(0, i - self.max_dist + 1), i + 1)` includes `j = i`. For early messages (small `i`), the range may ONLY contain `j = i`, which gets skipped → empty candidates list → 0 samples.

**Fix**: Change line 376 to `for j in range(max(0, i - self.max_dist + 1), i):` and remove lines 377-378.

### Model Architecture: `src/model.py` (forward pass, lines 95-221)
```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None,
    features: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    batch_size, num_candidates, seq_len = input_ids.shape

    # Reshape for BERT: [batch_size * C, seq_len]
    flat_input_ids = input_ids.view(-1, seq_len)
    flat_attention_mask = attention_mask.view(-1, seq_len)
    
    # Get BERT embeddings
    bert_outputs = self.bert(
        input_ids=flat_input_ids,
        attention_mask=flat_attention_mask,
        token_type_ids=flat_token_type_ids,
        return_dict=True,
    )
    
    # Use [CLS] token embedding
    cls_embedding = bert_outputs.last_hidden_state[:, 0, :]
    cls_embedding = self.dropout(cls_embedding)
    
    # Concatenate with features
    if features is not None:
        expanded_features = features.reshape(-1, self.num_features)
        combined = torch.cat([cls_embedding, expanded_features], dim=-1)
    
    # Classification head: [batch_size * C, 1]
    logits = self.classifier(combined)
    logits = logits.view(batch_size, num_candidates)
    
    # Mask out padded candidates
    candidate_mask = attention_mask.sum(dim=-1) > 0
    logits = logits.masked_fill(~candidate_mask, -100.0)
    
    candidate_probs = torch.softmax(logits, dim=-1)
    
    outputs = {"logits": logits, "probs": candidate_probs}
    
    if labels is not None:
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        outputs["loss"] = loss
    
    return outputs
```

### Training Script: `train_synthetic.sh`
```bash
#!/bin/bash
set -e

REPO_DIR=$(pwd)
export RUN_ROOT=/scratch/user/$USER/ircbert_runs
export LOG_DIR=$RUN_ROOT/logs
export CHECKPOINT_DIR=$RUN_ROOT/checkpoints_tiny_test
export TINY_DIR=$REPO_DIR/data/tiny

echo "=== Setting up directories ==="
mkdir -p logs $LOG_DIR $CHECKPOINT_DIR

source setup.sh

echo "=== Starting training on data/tiny (DeBERTa-v3-base, max_dist=50) ==="
python src/train.py \
    --mode train \
    --data-dir $TINY_DIR \
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
    --device cuda

# Evaluate latest checkpoint
echo "=== Evaluating on data/tiny/dev ==="
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "WARNING: No checkpoint found in $CHECKPOINT_DIR - skipping evaluation"
else
    echo "Latest checkpoint: $LATEST_CHECKPOINT"
    python src/evaluate.py \
        --checkpoint "$LATEST_CHECKPOINT" \
        --data-dir $TINY_DIR \
        --split dev \
        --batch-size 16 \
        --metrics both \
        --verbose 3
fi
```

**Note**: This script doesn't redirect output to a log file. Add `2>&1 | tee $LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log` to capture output.
