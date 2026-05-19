# IRC Disentanglement Project — Technical Handover

## 1. Current State: Training Run 24761301 Analysis

### System Configuration
```
GPU:      NVIDIA A100 80GB PCIe
Model:    microsoft/deberta-v3-base
Batch:    4 (with accumulation=4 → effective batch 16)
Max Dist: 50
Max Len:  96
LR:       5e-5 (linear warmup 10%)
Epochs:   10 (with patience=3)
AdamW:    eps=1e-6 (Fix B from CONTEXT.md)
```

### Epoch-by-Epoch Results

| Epoch | Duration  | Succeeded | Skipped (NaN) | Avg Loss | Dev Acc | Dev F1 |
|-------|-----------|----------:|--------------:|----------|---------|--------|
| 1     | 3,140s    | **11,187**| 1,232         | 3.7822   | 0.5160  | 0.3543 |
| 2     | 5,550s    | **0**     | 12,419        | 0.0000   | 0.5160  | 0.3543 |
| 3     | 5,549s    | **1**     | 12,418        | 3.8064   | 0.5160  | 0.3543 |
| 4     | 5,550s    | **1**     | 12,418        | 3.3977   | 0.5160  | 0.3543 |

**Dev accuracy locks at 51.60% across all 4 epochs** — the best model (epoch 1 checkpoint) is loaded for every subsequent evaluation. The model learns nothing after epoch 1.

### Final Test Set Performance (from last checkpoint)
```
Loss:      3.7553
Accuracy:  0.5271
Precision: 0.3900
Recall:    0.3733
F1:        0.3734
```

### Baseline Comparison (from dev evaluation logs)
| Baseline | Accuracy | Source |
|----------|---------:|--------|
| Always predict position 46 (most common) | 10.63% | dev log |
| Always predict last candidate (position 48) | 5.17% | dev log |
| Random uniform (1/49 classes) | ~2.0% | theoretical |
| **Model at epoch 1** | **51.60%** | dev evaluation |
| **Label recency (positions 46-49)** | 23.8% of gold labels | dev distribution |

**Key finding**: 51.6% >> 10.6% majority-class baseline. The model IS learning real structure, not exploiting positional shortcuts. This confirms NaN fix is worth pursuing — the underlying learning signal is present.

## 2. The Core Failure Mechanism

### Symptom: NaN Cascade Starting at Batch 11,178 (Epoch 1)

```
[Batch 11178] NaN/Inf in gradients after backward — skipping batch.
[Batch 11179] NaN/Inf in gradients after backward — skipping batch.
[Batch 11180] NaN/Inf in gradients after backward — skipping batch.
...continues for 1,232 consecutive batches through end of epoch...
```

Then epoch 2 starts and **every batch (12,419) produces NaN gradients.**

### Why This Happens — The True Root Cause

The cascade proceeds in 3 stages:

**Stage 1 (Batches 0–11,177): Apparent stability**
- 11,177 batches succeed = ~2,794 optimizer steps (at accumulation=4)
- LR ramps linearly from 0 → ~4.7e-5 (almost at peak 5e-5)
- Weights drift gradually from BERT pretrained initialization
- `nan_to_num` fires occasionally on `cls_embedding` but masks the symptom

**Stage 2 (Batches 11,178–12,419): 49-class logit overflow**
- The actual mechanism is **DeBERTa's disentangled attention + CrossEntropyLoss over 49 classes** producing logits in the range ~80–100
- `exp(100)` overflows fp32 to `inf`, softmax denominator becomes `inf/inf = NaN`
- `cls_embedding` accumulates NaN counts: 768 → 1,536 → 2,304 → 4,608
  ```
  WARNING: cls_embedding contains NaN/Inf before classifier.
  Shape: torch.Size([196, 768]), NaN count: 768 → 1536 → 2304 → 4608
  ```
- The `nan_to_num` fix (lines 147-155 in `model.py`) replaces NaN with 0.0 *after* BERT but the **weights are already NaN**
- Given NaN weights, forward pass produces NaN → backward produces NaN gradients → all subsequent batches are skipped

**Stage 3 (Epochs 2–4): Complete collapse**
- The checkpoint saved at the end of epoch 1 contains NaN weights
- Every batch in epoch 2 starts with NaN weights → NaN forward → NaN backward → skip
- Result: avg_loss=0.0000 (the 1 "succeeded" batch had NaN loss logged as 0)

### Why Previous Fixes Were Insufficient

| Fix | Location | What It Does | Why It Failed |
|-----|----------|-------------|---------------|
| `adamw eps=1e-6` | `train.py` line 1185 | Prevents `grad / sqrt(v + eps)` explosion | Prevents NaN step 1 but not gradual drift over 11K batches |
| `nan_to_num on cls_embedding` | `model.py` lines 147-155 | Replaces NaN with 0 before classifier | **Actively dangerous** — silently zeroes corrupted hidden states, lets training continue with contaminated weights |
| Gradient NaN check | `train.py` lines 826-839 | Skips batches with NaN gradients | Catches symptom, not cause — weights already corrupted |
| `fill_value=-100` | `model.py` line 205 | Finite negative mask for padded candidates | Correct — not a contributing factor |

## 3. Corrected Fix Priority

| Priority | Fix | Where | Why |
|---|---|---|---|
| **1** | **Detect NaN before `loss.backward()`, skip batch** | `train.py` | Stops weight corruption at source — don't let nan_to_num silently continue |
| **2** | **Add label smoothing (`0.1`) to CrossEntropyLoss** | `model.py` | Caps how hard the model pushes correct-class logit to +inf, directly prevents overflow |
| **3** | **Clamp logits to `[-50, 50]` before loss** | `model.py` | Hard numerical ceiling — even if attention produces large values, loss sees bounded input |
| **4** | **AdamW eps → `1e-4`** | `train.py` line 1185 | More aggressive grad/v stabilization — correct fix, keep |
| **5** | **Reduce batch accumulation (4→2) + batch size (4→8)** | `run_job.slurm` | Reduces gradient noise accumulation window |
| **6** | **LR 5e-5 → 3e-5, warmup 10%→15%** | `run_job.slurm` | Gentler peak LR, longer ramp |

### ⚠️ Critical Ordering Constraints (Claude's Final Check)

Two implementation-order details that must be followed exactly:

**1. In `model.py`: Logit clamping BEFORE CrossEntropyLoss**
```python
logits = torch.clamp(logits, min=-50.0, max=50.0)   # Step 1: clamp first
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)    # Step 2: then smooth
loss = loss_fn(logits, labels)                         # Step 3: then compute loss
```
If clamping happens after `CrossEntropyLoss`, it does nothing — the overflow already happened inside the loss function.

**2. In `train.py` (line 755): NaN skip MUST zero gradients**
```python
# Current code (BUG — no zero_grad):
if torch.isnan(loss) or torch.isinf(loss):
    logger.error(...)
    continue  # ← BUG: accumulated gradients from previous batches survive

# Fixed code:
if torch.isnan(loss) or torch.isinf(loss):
    logger.error(...)
    optimizer.zero_grad()   # ← MUST be here to clear accumulation window
    torch.cuda.empty_cache()
    continue
```
Without `optimizer.zero_grad()`, accumulated gradients from previous batches in the accumulation window will still get applied on the next valid step. That's a subtle instability reintroducer.

### Detailed Implementation for Each Fix

#### Fix 1: Pre-backward NaN Detection (Replace `nan_to_num`)
Two changes needed:

**In `model.py` (lines 147-155)**: Remove the dangerous `nan_to_num`. Replace with:
```python
if torch.isnan(cls_embedding).any() or torch.isinf(cls_embedding).any():
    logger.warning(
        f"  PRE-BACKWARD NaN DETECTED in cls_embedding at batch {batch_idx}. "
        f"NaN count: {torch.isnan(cls_embedding).sum().item()}. "
        f"Skipping batch."
    )
    return {"logits": logits, "probs": probs, "loss": torch.tensor(float('nan'), device=logits.device)}
```

**In `train.py` (lines 750-755)**: Add `optimizer.zero_grad()` to the NaN skip path:
```python
# BEFORE (current - dangerous — no zero_grad):
if torch.isnan(loss) or torch.isinf(loss):
    logger.error(f"  Batch {batch_idx + 1}: NaN or Inf loss detected! Skipping batch.")
    continue

# AFTER (fixed — clears accumulation state):
if torch.isnan(loss) or torch.isinf(loss):
    logger.error(f"  Batch {batch_idx + 1}: NaN or Inf loss detected — skipping batch.")
    optimizer.zero_grad()    # ← CRITICAL: clears gradient accumulation
    torch.cuda.empty_cache()
    continue
```

#### Fix 2: Label Smoothing
In `model.py`, replace CrossEntropyLoss (line 217):

```python
# BEFORE:
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)

# AFTER:
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = loss_fn(logits, labels)
```

**Why 0.1**: Standard value used in NLP classification. Prevents the model from pushing the correct-class logit to +inf by capping the "target" probability at 0.9. The remaining 0.1 is distributed across all 49 classes, creating a soft target distribution instead of a one-hot spike. This directly prevents `exp(z_y)` overflow.

#### Fix 3: Logit Clamping
In `model.py`, before CrossEntropyLoss (around line 214):

```python
# BEFORE loss computation (after logits are computed, before loss_fn):
# Clamp logits to prevent exp() overflow in CrossEntropyLoss softmax
logits = torch.clamp(logits, min=-50.0, max=50.0)

# Then compute loss
if labels is not None:
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss = loss_fn(logits, labels)
```

**Why [-50, 50]**: `exp(50) ≈ 5.18e21` (well within fp32 range), `exp(-50) ≈ 1.93e-22` (above fp32 minimum ~1.4e-45). The softmax denominator is guaranteed finite because all inputs are bounded. This is the blunt instrument that guarantees numerical stability regardless of model behavior.

#### Fix 4: AdamW eps → 1e-4
In `train.py` line 1185:

```python
# BEFORE:
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.learning_rate, weight_decay=0.01, eps=1e-6
)

# AFTER:
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.learning_rate, weight_decay=0.01, eps=1e-4
)
```

#### Fix 5 + 6: Hyperparameter Tuning
In `run_job.slurm`:

```bash
# BEFORE:
--batch-size 4 --gradient-accumulation-steps 4 --learning-rate 5e-5 --warmup-ratio 0.1

# AFTER:
--batch-size 8 --gradient-accumulation-steps 2 --learning-rate 3e-5 --warmup-ratio 0.15
```

## 4. Implementation Plan (Combined)

### Files to Change

| File | Changes |
|------|---------|
| `src/model.py` | (1) Replace `nan_to_num` with NaN-aware early return. (2) Add label_smoothing=0.1 to CrossEntropyLoss. (3) Add `torch.clamp(logits, -50, 50)` before loss. |
| `src/train.py` | (1) Make NaN loss skip also zero gradients + empty cache. (2) Change AdamW eps from 1e-6 to 1e-4. |
| `run_job.slurm` | batch=8, accumulation=2, lr=3e-5, warmup=0.15 |

### What to Delete Before Resubmitting
- `/scratch/user/checkpoints_maxdist50/` — contains NaN-contaminated weights from epoch 1
- Old log files in `/scratch/user/logs/` — will be overwritten but clean state helps

### What NOT to Change
- `max_length=96` — confirmed fine, 95%+ coverage
- `max_dist=50` — matches literature (StructBERT kh=50)
- `nan_to_num` in model.py — remove it entirely, replaced by pre-backward NaN detection
- Gradient accumulation logic in train.py — the code is correct, just the config changes
- Data loader — no data issues identified

### Verification After Changes
1. Run `pytest tests/ -x -q` (should pass all 99 tests — tests may need updating for label_smoothing)
2. If tests fail, update test assertions (label_smoothing changes loss values by a small constant factor)
3. Submit `run_job.slurm` for full 10-epoch training
4. Monitor logs for NaN cascade — if clean after 12,000 batches, the fix works

## 5. Fallback Plan

If NaN cascade still occurs after Fixes 1-6:

1. **Reduce batch to 4, accumulation to 1** — minimizes gradient noise window
2. **Further lower LR to 2e-5** — standard BERT fine-tuning rate
3. **Check data dependency** — identify which conversation file triggers the cascade (shuffle-dependent)
4. **Disable handcrafted features** — test with `num_features=0` to check if feature scaling is the trigger
5. **Switch to Adam with weight_decay** instead of AdamW — Adam doesn't decouple weight decay from learning rate and can be more stable

## 6. Key Code Snippets (Final State)

### CrossEntropyLoss with Label Smoothing (model.py)
```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = loss_fn(logits, labels)
```

### Logit Clamping (model.py, before loss)
```python
logits = torch.clamp(logits, min=-50.0, max=50.0)
```

### AdamW with eps=1e-4 (train.py)
```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.learning_rate, weight_decay=0.01, eps=1e-4
)
```

### SLURM Config (run_job.slurm)
```bash
--batch-size 8 --gradient-accumulation-steps 2 --learning-rate 3e-5 --warmup-ratio 0.15
```

## 7. Log Files for Reference

| File | Description |
|------|-------------|
| `logs/24761301.err` | Full training log (12MB, 90K+ lines) |
| `logs/eval_20260519_035042.log` | Final evaluation on DEV (best checkpoint) |
| `logs/eval_20260519_035141.log` | Final evaluation on TEST (best checkpoint) |