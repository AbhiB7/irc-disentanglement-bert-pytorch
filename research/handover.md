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

## 4. Verification

All tests pass after the changes:
- `tests/test_model.py`: 23/23 passed (model init, forward, prediction, architecture, loss, smoke)
- `tests/test_parse_args.py`: 41/41 passed (defaults, custom values, flags, choices, device)

## 5. If It Still OOMs

Further options in order of aggressiveness:
1. Reduce `--batch-size` to 8
2. Reduce `--max-dist` to 20 (fewer candidates per sample)
3. Use `prajjwal1/bert-tiny` (4.4M params) for smoke tests instead of `bert-base-uncased` (110M) or `microsoft/deberta-v3-base` (184M)
4. Add `--freeze-bert` to train only the classifier head (dramatically reduces gradient computation)
5. Request a multi-GPU node and use gradient accumulation with smaller effective batch size

## 6. Files Modified

| File | Change |
| :--- | :--- |
| `src/model.py` | Added `gradient_checkpointing` param to `__init__` and `create_model()` |
| `src/train.py` | Added `--gradient-checkpointing` CLI flag |
| `run_job.slurm` | Reduced batch-size 32→16, added `--gradient-checkpointing` |