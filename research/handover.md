# IRC Conversation Disentanglement - NaN Loss Cascade (2026-05-10)

## 1. The Problem

Training on an **NVIDIA L40 (45.5 GB VRAM)** with `--batch-size 4 --max-dist 15 --max-length 128` (no fp16) produces a **NaN loss cascade** that starts at Batch 1 and corrupts every subsequent batch.

### Log Evidence (latest run: `logs/debug_20260510_170934.log`)

```
[Epoch 1 Batch 0 DIAGNOSTIC] C=15 | logits: min=-100.0000 max=0.6819 mean=-21.4971 has_nan=False | labels=[2, 13, 14, 14] max_label=14
Epoch 1 Batch 0 gradient norm: 5.0690

[Epoch 1 Batch 1 DIAGNOSTIC] C=14 | logits: min=nan max=nan mean=nan has_nan=True | labels=[12, 13, 10, 12] max_label=13
Batch 2: NaN or Inf loss detected! Skipping batch.

[Epoch 1 Batch 2 DIAGNOSTIC] C=15 | logits: min=nan max=nan mean=nan has_nan=True | labels=[12, 14, 14, 14] max_label=14
Batch 3: NaN or Inf loss detected! Skipping batch.
... (every batch from 2 onwards is NaN)
```

### Key Observations

1. **Batch 0 works perfectly**: logits are finite (min=-100.0, max=0.68), gradient norm=5.07 (healthy), loss is finite (2.41).
2. **Batch 1 logits are ALL NaN**: every single logit is NaN, not just the masked ones.
3. **The NaN appears between Batch 0's backward pass and Batch 1's forward pass** — i.e., the optimizer step corrupts the model weights.
4. **This happens with ALL masking values tried**: `-3.4e38` (finfo.min), `-1e4`, and `-100.0`. The masking value is NOT the root cause.
5. **The label clamp is working**: labels are all within bounds (max_label=14, C=15).

## 2. What We've Tried (and Why It Didn't Work)

### Fix A: Label Clamp in `collate_fn` (2026-05-10)
```python
# src/train.py, line 293
batch_labels[i] = min(int(labels), max_candidates - 1)
```
**Result**: Still NaN. The labels were already in bounds — the problem is elsewhere.

### Fix B: Replace `-inf` masking with `-100.0` (2026-05-10)
```python
# src/model.py, line 184
fill_value = -100.0
logits = logits.masked_fill(~candidate_mask, fill_value)
```
**Result**: Still NaN. Batch 0 logits show `min=-100.0` (correct), but Batch 1+ are all NaN.

## 3. The Code Path (What Happens Between Batch 0 and Batch 1)

### Forward Pass (model.py lines 116-198)
```python
batch_size, num_candidates, seq_len = input_ids.shape  # [4, 15, 128]

# Flatten for BERT: [batch*C, seq] = [60, 128]
flat_input_ids = input_ids.view(-1, seq_len)
flat_attention_mask = attention_mask.view(-1, seq_len)

# BERT forward
bert_outputs = self.bert(
    input_ids=flat_input_ids,
    attention_mask=flat_attention_mask,
    token_type_ids=flat_token_type_ids,
    return_dict=True,
)

# [CLS] embedding: [60, 768]
cls_embedding = bert_outputs.last_hidden_state[:, 0, :]
cls_embedding = self.dropout(cls_embedding)

# Concatenate features: [60, 768] + [60, 5] = [60, 773]
combined = torch.cat([cls_embedding, expanded_features], dim=-1)

# Classifier: [60, 773] -> [60, 1] -> reshape to [4, 15]
logits = self.classifier(combined)
logits = logits.view(batch_size, num_candidates)

# Mask padded candidates
candidate_mask = attention_mask.sum(dim=-1) > 0  # [4, 15]
fill_value = -100.0
logits = logits.masked_fill(~candidate_mask, fill_value)

# Softmax + loss
candidate_probs = torch.softmax(logits, dim=-1)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

### Backward Pass (train.py lines 741-765)
```python
optimizer.zero_grad()
loss.backward()

# NaN gradient check
grad_has_nan = False
for p in model.parameters():
    if p.grad is not None:
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            grad_has_nan = True
            break

if grad_has_nan:
    # Skip batch — but this is NOT triggered for Batch 0
    ...

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
optimizer.step()
```

### The Critical Gap
The NaN gradient check passes for Batch 0 (gradient norm=5.07, no NaN). But **after `optimizer.step()`**, the model weights become NaN. We don't check for NaN weights after the optimizer step.

## 4. Hypotheses for the Root Cause

### Hypothesis 1: AdamW + Weight Decay + `-100.0` Logits
The `-100.0` logits produce gradients that, when combined with AdamW's weight decay, cause numerical instability. Specifically:
- `softmax(-100.0) ≈ 0` for masked candidates
- CrossEntropyLoss gradient for masked candidates: `softmax(masked) - 0 = ~0` (if not the correct class)
- But if a masked candidate IS the correct class: `softmax(masked) - 1 ≈ -1`
- The gradient for the classifier weight connected to that candidate is `-1 * cls_embedding`
- If `cls_embedding` has large values (e.g., from BERT's LayerNorm on all-zero input), the gradient could be large
- AdamW's weight decay (`weight_decay=0.01`) multiplies weights by `(1 - lr * weight_decay)` each step
- Combined with a large gradient update, this could push weights to NaN

**How to test**: Log `cls_embedding` values for masked candidates. If they're large (e.g., >1e4), that's the problem.

### Hypothesis 2: BERT LayerNorm on All-Zero Input
When a candidate is fully padded (all zeros), BERT's LayerNorm computes:
```
LayerNorm(0) = (0 - mean(0)) / std(0) = 0 / 0 = NaN
```
The `nan_to_num` safety net in `model.py` (lines 169-170) is supposed to catch this, but it's placed **after** the classifier has already used `cls_embedding`:
```python
# Line 145: combined = torch.cat([cls_embedding, expanded_features], dim=-1)  ← USED HERE
# Line 156: logits = self.classifier(combined)                                 ← USED HERE
# Line 159: logits = logits.view(batch_size, num_candidates)
# Line 164: candidate_mask = attention_mask.sum(dim=-1) > 0
# Line 169: if torch.isnan(cls_embedding).any():                                ← CHECKED HERE (TOO LATE!)
# Line 170:     cls_embedding = torch.nan_to_num(cls_embedding, ...)
```

The safety net is **useless** where it is — it checks `cls_embedding` after the classifier already used it. But this shouldn't cause NaN in Batch 1 because Batch 0's logits are clean.

**How to test**: Move the `nan_to_num` check to **before** the classifier (before line 145). If Batch 0's logits were NaN before the fix, this would explain everything. But they're not — so this is probably not the root cause.

### Hypothesis 3: Scheduler Step Causes NaN LR
The scheduler has 1 warmup step. After `scheduler.step()` is called (line 802), the learning rate changes. If the scheduler produces a NaN or Inf LR, the optimizer step would corrupt the weights.

```python
# train.py line 802
if scheduler is not None:
    scheduler.step()
```

With `warmup_ratio=0.1` and 16 total steps, there's 1 warmup step. After step 1, the LR transitions from warmup to the main schedule. If there's a bug in the scheduler's LR calculation (e.g., division by zero), the LR could become NaN.

**How to test**: Log `scheduler.get_last_lr()` after each step.

### Hypothesis 4: `torch.amp.autocast` + DeBERTa Interaction
The forward pass uses `torch.amp.autocast("cuda", enabled=fp16)`. Even though `--fp16` is not set (so `fp16=False`), the autocast context manager is still active. On PyTorch 2.5.1 (Bunya's version), there might be a bug where autocast interacts badly with DeBERTa's disentangled attention, producing NaN in the backward pass for certain input configurations.

**How to test**: Remove the autocast context manager entirely and see if the NaN persists.

### Hypothesis 5: `torch.nn.utils.clip_grad_norm_` Causes NaN
Gradient clipping with `max_norm=10.0` can produce NaN if the total norm is 0 (division by zero). If all gradients are exactly 0 (e.g., because all logits are -100 and the correct class is masked), then:
```
total_norm = sqrt(sum(grad_i^2)) = 0
clip_coef = 10.0 / 0 = inf
gradients = gradients * inf = NaN
```

This would happen if:
1. All candidates in a batch are masked (C=0 real candidates)
2. The labels are clamped to 0 (the only "available" candidate)
3. The model predicts 0 with high confidence (logit for candidate 0 is much higher than -100)
4. The gradient for candidate 0 is `softmax(0) - 1 ≈ 0` (if logit=0 is much higher than -100)
5. All other gradients are 0 (because softmax assigns ~0 probability to -100 candidates)
6. Total gradient norm = 0
7. `clip_grad_norm_` divides by 0 → NaN

**How to test**: Log the gradient norm **before** clipping, and check if it's 0.

## 5. Diagnostic Gaps (What We Need to Log)

| Question | How to Check | Priority |
| :------- | :----------- | :------- |
| Are model weights NaN after `optimizer.step()` on Batch 0? | Log `torch.isnan(p).any()` on all params after step | **HIGH** |
| Is the gradient norm 0 before clipping? | Log `total_norm` before `clip_grad_norm_` | **HIGH** |
| Is the learning rate NaN after `scheduler.step()`? | Log `scheduler.get_last_lr()` | **HIGH** |
| Are BERT's hidden states NaN for padded candidates? | Log `cls_embedding` min/max/has_nan before classifier | **MEDIUM** |
| Does removing autocast fix it? | Remove `with torch.amp.autocast(...)` entirely | **MEDIUM** |
| Does `--model bert-base-uncased` also NaN? | Run `./debug.sh --model bert-base-uncased` | **MEDIUM** |
| Does reducing LR to 2e-5 fix it? | Run with `--learning-rate 2e-5` | **LOW** |

## 6. Files Involved

### `src/model.py` — `CrossEncoderWithFeatures.forward()`
```python
# Lines 116-198
batch_size, num_candidates, seq_len = input_ids.shape

# Flatten for BERT
flat_input_ids = input_ids.view(-1, seq_len)
flat_attention_mask = attention_mask.view(-1, seq_len)

# BERT forward
bert_outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask, ...)
cls_embedding = bert_outputs.last_hidden_state[:, 0, :]  # [batch*C, hidden]
cls_embedding = self.dropout(cls_embedding)

# Concatenate features
combined = torch.cat([cls_embedding, expanded_features], dim=-1)

# Classifier
logits = self.classifier(combined)  # [batch*C, 1]
logits = logits.view(batch_size, num_candidates)  # [batch, C]

# Mask padded candidates
candidate_mask = attention_mask.sum(dim=-1) > 0
fill_value = -100.0  # TRIED: -3.4e38, -1e4, -100.0 — ALL STILL NAN
logits = logits.masked_fill(~candidate_mask, fill_value)

# Loss
candidate_probs = torch.softmax(logits, dim=-1)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

### `src/train.py` — `collate_fn()`
```python
# Lines 235-300
def collate_fn(batch, max_dist=50):
    max_candidates = max(item["input_ids"].shape[0] for item in batch)
    max_candidates = min(max_candidates, max_dist)  # Cap at max_dist
    ...
    for i, item in enumerate(batch):
        ...
        batch_labels[i] = min(int(labels), max_candidates - 1)  # Label clamp
    return {"input_ids": ..., "attention_mask": ..., "features": ..., "labels": batch_labels}
```

### `src/train.py` — `train_epoch()` (backward pass)
```python
# Lines 741-802
optimizer.zero_grad()
loss.backward()

# NaN gradient check
grad_has_nan = False
for p in model.parameters():
    if p.grad is not None:
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            grad_has_nan = True
            break

if grad_has_nan:
    optimizer.zero_grad()
    continue

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
optimizer.step()

# Gradient norm logging (only for batch 0)
if batch_idx == 0:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    logger.info(f"Epoch {epoch} Batch 0 gradient norm: {total_norm:.4f}")

if scheduler is not None:
    scheduler.step()
```

## 7. Recommended Next Steps

### Immediate (on Bunya interactive node):
1. **Run with `--model bert-base-uncased`**: `./debug.sh --model bert-base-uncased` — if it works, the problem is DeBERTa-specific.
2. **Remove autocast**: Temporarily remove `with torch.amp.autocast("cuda", enabled=fp16):` from `train_epoch()` to rule out autocast interaction.
3. **Add weight NaN check after optimizer step**: Log `torch.isnan(p).any()` on all params after `optimizer.step()`.

### If still NaN:
4. **Add gradient norm logging before clipping**: Log `total_norm` before `clip_grad_norm_` to check for zero-norm gradients.
5. **Add LR logging**: Log `scheduler.get_last_lr()` after each step.
6. **Move `nan_to_num` before classifier**: Move the safety net to before `combined = torch.cat(...)`.

### If fixed:
7. **Run on medium dataset**: `./debug.sh --medium` to verify stability with more data.
8. **Update `smoke_test.slurm`** with confirmed-working config.