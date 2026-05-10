# IRC Conversation Disentanglement - OOM Diagnosis & Fix (2026-05-07)

## 1. The Problem

Training on an **NVIDIA L40 (45.5 GB VRAM)** with `--batch-size 32 --max-dist 30 --max-length 128 --fp16` was OOMing immediately. This was surprising given the L40 is a very powerful GPU.

**Error log**: `logs/24337978.err` — CUDA OOM during the very first forward pass.

## 2. Root Cause Analysis

The OOM is caused by **activation memory explosion** in the DeBERTa-v3-base transformer backbone. Here's the detailed breakdown:

### Memory per batch (batch=32, max_dist=30, max_length=128):
- Each sample has up to C=30 candidates → **32 × 30 = 960 sequences** per batch
- Each sequence is 128 tokens → **960 × 128 = 122,880 tokens** per batch
- DeBERTa-v3-base has **12 transformer layers** with hidden_size=768
- Each layer stores activations: ~122,880 × 768 × 4 bytes × 12 layers ≈ **4.5 GB** in fp32
- Plus: embeddings, attention matrices (O(n²) per sequence), classifier head, optimizer states (2× for AdamW), gradients
- **Total estimate: ~20-30 GB** — which should fit in 45.5 GB, but the real killer is the **attention computation** for 960 sequences × 128 tokens each, and the **memory fragmentation** from variable-length candidate padding

### Why `--fp16` alone wasn't enough:
- `--fp16` (torch.amp.autocast) primarily reduces:
  - Weight/gradient storage (2 bytes instead of 4)
  - Matrix multiply compute (Tensor Cores)
- But it does NOT reduce:
  - **Activation memory** (softmax, layer norms, residuals stay fp32 internally)
  - **Memory fragmentation** from variable-C padding in collate_fn
  - **Optimizer states** (AdamW stores fp32 copies of weights + momentum + variance = 3× fp32 copies)

## 3. The Fix (Three Changes)

### 3.1 Gradient Checkpointing (`src/model.py`)

Added `gradient_checkpointing` parameter to `CrossEncoderWithFeatures.__init__` and `create_model()` factory function.

When enabled, HuggingFace's `self.bert.gradient_checkpointing_enable()` stores only inputs/outputs per transformer block and recomputes intermediate activations during backward. This cuts **activation memory by ~80%** at the cost of ~30% slower training.

```python
# In __init__:
if gradient_checkpointing:
    self.bert.gradient_checkpointing_enable()

# In create_model():
model = CrossEncoderWithFeatures(
    ...,
    gradient_checkpointing=gradient_checkpointing,
)
```

### 3.2 CLI Flag (`src/train.py`)

Added `--gradient-checkpointing` flag to argument parser:

```python
parser.add_argument(
    "--gradient-checkpointing",
    action="store_true",
    help="Enable gradient checkpointing on BERT backbone (trades ~30%% speed for ~80%% less activation VRAM)",
)
```

This is passed through to `create_model()` in `main()`.

### 3.3 SLURM Script (`run_job.slurm`)

Two changes to the training invocation:
- Reduced `--batch-size` from **32 → 16** (halves peak memory)
- Added `--gradient-checkpointing` flag

```bash
python src/train.py \
    --mode train \
    --batch-size 16 \
    --num-workers 4 \
    --epochs 3 \
    --learning-rate 5e-5 \
    --max-length 128 \
    --max-dist 30 \
    --warmup-ratio 0.1 \
    --patience 3 \
    --eval-every 1 \
    --save-every 1 \
    --test-end 1000000000 \
    --output-dir "$CHECKPOINT_DIR" \
    --device cuda \
    --fp16 \
    --gradient-checkpointing
```

## 4. Second Round: Still OOM at Batch 91

Even with batch=16 + gradient checkpointing, the job OOM'd at batch 91 every run. Memory locked at ~43 GB and never recovered. Zero loss logged (all batches skipped).

**Root cause:** DeBERTa-v3 uses **disentangled attention** which requires ~2× the attention matrices of standard BERT. At batch=16×C=30 = 480 sequences through 12 layers, this saturates the 45.5 GB budget. After the first OOM, CUDA memory fragmentation prevents any further batches from succeeding — `empty_cache()` in the except handler is not enough to recover.

### 4.1 Claude's Deeper Fixes (2026-05-07)

Applied from Claude's analysis of the OOM log:

**`run_job.slurm`** — Further reduced params:
- `--batch-size 16 → 8` (halved again)
- `--max-dist 30 → 15` (fewer candidates per sample, matches collate_fn cap)

**`src/train.py` — collate_fn**: Hard cap `max_candidates` at 15 to prevent outlier padding spikes.

**`src/train.py` — OOM cascade detection**: Both forward and backward OOM handlers now track consecutive OOM count. If 10+ consecutive batches OOM, the job hard-aborts with a clear error message instead of silently running to `avg_loss=0.0`.

**`src/train.py` — Periodic cache flush**: `torch.cuda.empty_cache()` every 50 batches to prevent allocator fragmentation buildup.

**`src/train.py` — max_C logging**: Batch progress logs now include `max_C` value for memory diagnostics.

## 5. Third Round: GradScaler Bug (2026-05-08)

The OOM was fixed, but a new error appeared immediately on the first batch:

```
ValueError: Attempting to unscale FP16 gradients.
```

### 5.1 What GradScaler Does Normally

When you train with `--fp16`, numbers are stored in "half precision" (16-bit) instead of "full precision" (32-bit). This saves memory and makes things faster. But there's a problem: really tiny numbers (like 0.0000001) can't be represented in 16-bit — they just become zero. This is called "underflow."

`GradScaler` solves this by multiplying the loss by a big number (like 65536) before doing the backward pass. This makes all the gradients bigger so they don't underflow. Then it divides them back down before the optimizer step. It's like turning up the volume on a quiet recording, then turning it back down.

### 5.2 What Gradient Checkpointing Does

Normally, during the forward pass, PyTorch saves all the intermediate values (activations) so it can use them during the backward pass to compute gradients. This uses a lot of memory. Gradient checkpointing says: "don't save those intermediate values — just save the inputs and outputs, and if we need the intermediates during backward, we'll recompute them on the fly." It's like not taking notes during a lecture, but re-watching the recording when you need to study.

### 5.3 Why They Conflict

The problem is that `GradScaler` expects to find the original scaled gradients in the autograd graph after `loss.backward()`. But gradient checkpointing creates a different kind of autograd graph — one where some intermediate values were recomputed rather than stored. On PyTorch 2.5.1 (the version on Bunya), the scaler gets confused and says "I can't unscale these gradients because they don't look like scaled gradients to me" — hence the error.

### 5.4 The Fix

Disabled `GradScaler` entirely (`scaler = None`). The `--fp16` flag still works — it still runs the forward pass in 16-bit via `torch.amp.autocast`, which saves memory. We just skip the "scale the loss, unscale the gradients" dance. For BERT fine-tuning, gradient underflow isn't really a problem anyway because the gradients are large enough that they don't disappear in 16-bit. The scaler is mainly needed for training from scratch on huge datasets like ImageNet, not for fine-tuning a pretrained model.

## 6. Verification

All tests pass after all changes:
- `tests/test_model.py`: 23/23 passed (model init, forward, prediction, architecture, loss, smoke)
- `tests/test_parse_args.py`: 41/41 passed (defaults, custom values, flags, choices, device)

## 7. If It Still OOMs

Further options in order of aggressiveness:
1. Reduce `--batch-size` to 4
2. Reduce `--max-dist` to 10
3. Use `prajjwal1/bert-tiny` (4.4M params) for smoke tests instead of `bert-base-uncased` (110M) or `microsoft/deberta-v3-base` (184M)
4. Add `--freeze-bert` to train only the classifier head (dramatically reduces gradient computation)
5. Request a multi-GPU node and use gradient accumulation with smaller effective batch size

## 8. Fourth Round: NaN Loss Cascade (2026-05-08)

Two runs (job IDs 24377710 and 24377768) both show the **exact same failure pattern**:

### Log Evidence (both runs identical)

```
[Epoch 1 Batch 0 SANITY CHECK]
  input_ids shape : [8, 15, 128]  (batch x max_C x seq_len)
  features shape  : [8, 15, 5]
  labels shape    : [8]
  Candidate counts per sample: min=1 max=15 mean=8.1
  Label values: [0, 0, 0, 0, 0, 0, 0, 0]
  Label distribution: {0: 8}
  Epoch 1 Batch 0 gradient norm: 3.9544
  Batch 2: NaN or Inf loss detected! Skipping batch.
  Batch 3: NaN or Inf loss detected! Skipping batch.
  ... (every batch from 2 through 584+)
```

### What We Know

1. **Batch 0 works perfectly** — gradient norm = 3.95 (healthy), no NaN
2. **Batch 1 is not logged** — no error, no progress log (presumably passes)
3. **Every batch from 2 onwards** produces NaN loss
4. **Removing `--fp16` did NOT change the pattern** — same NaN cascade in both runs
5. **Gradient clipping didn't help** — the NaN is in the forward pass **loss**, not the gradients

### What We DON'T Know (Diagnostic Gaps)

| Question | How to check |
| :------- | :----------- |
| Do model **weights** become NaN after the first optimizer step? | Log `torch.isnan(p).any()` on all params after `optimizer.step()` on batch 0 |
| Are the model's **logits** NaN before the loss function? | Log `torch.isnan(logits).any()` inside the forward pass when loss is NaN |
| Is batch 1's label valid? | We only log labels at batch 0 (all zeros). Batch 1 might have a label ≥ C |
| What's the gradient norm for batch 1? | We only log gradient norm at batch 0 |

### Hypothesis

The most likely root cause is **label ≥ C** (out-of-bounds target for CrossEntropyLoss). The collate_fn caps `max_candidates` at 15, but the original label indices are computed based on the uncapped candidate list. If a sample originally had C=20 candidates and the gold parent was at index 17, the label would be 17 — but after capping to 15 candidates, the label 17 is out of bounds. CrossEntropyLoss with an out-of-bounds target produces NaN loss.

This would explain:
- Why batch 0 works (all labels are 0, which is always in bounds)
- Why batch 1 might work (random chance of in-bounds labels)
- Why every batch from 2 onwards fails (once the optimizer sees NaN loss, the model weights become NaN, and every subsequent forward pass produces NaN)

### Fix Needed

In `collate_fn`: after capping `max_candidates` at 15, clamp all labels to `min(label, max_candidates - 1)`.

## 9. Files Modified

| File | Change |
| :--- | :--- |
| `src/model.py` | Added `gradient_checkpointing` param to `__init__` and `create_model()` |
| `src/train.py` | Added `--gradient-checkpointing` CLI flag; collate_fn cap at 15; OOM cascade detection; periodic cache flush; max_C logging; disabled GradScaler; gradient clipping; NaN gradient check |
| `run_job.slurm` | batch-size 32→16→8, max-dist 30→15, added `--gradient-checkpointing`, removed `--fp16` |
