# IRC Disentanglement Project — Technical Handover

## 1. Model: Multiclass CrossEncoder with Features

`src/model.py` — BERT-based CrossEncoder (DeBERTa-v3-base default) with 5 handcrafted features.

For each child message, C candidate samples are created. Each sample: `(parent_text, child_text, features, label)` where label = gold parent index (0 to C-1).

Forward pass: `[batch, C, seq]` → flatten → `[batch*C, seq]` through BERT → `[CLS]` embedding (768-dim) → concat with 5 features (773-dim) → linear(773→1) → unflatten to `[batch, C]` → softmax → CrossEntropyLoss.

## 2. Data Loader

`src/data_loader.py` — key function: `_create_samples_for_conversation()` (lines 350-434).

Candidate selection: For child message `i`, collect all `j < i` where `i - j <= max_dist`. Self-links are excluded via `range(max(0, i - max_dist + 1), i)`.

## 3. The Real Bug (Found 2026-05-18)

The annotation file format is `PARENT CHILD -`, but the code was reading it as `CHILD PARENT -`.

Evidence from annotation files:
```
1000 1000 -    → self-link (symmetric, unaffected by bug)
1009 -   → message 1009 replies to message 1000
1020 1021 -    → message 1021 replies to message 1020
```

If read as `child = 1000, parent 1009= `parent > child`, which is impossible in IRC.

**Fix** (lines 171-172 of `data_loader.py`):
```python
# BEFORE (wrong):
child = int(parts[0])   # first column = parent
parent = int(parts[1])  # second column = child

# AFTER (correct):
parent = int(parts[0])
child = int(parts[1])
```

This single change fixes the 0 samples problem.

## 4. Diagnostic Results

**Cross-link distance analysis** (`scripts/diagnose_links.py` on all train files):
- Total gold links: 69,395
- Self-links: 16,754 (24.1%)
- Cross-links: 52,641 (75.9%)
- Cross-link distance median: 3 messages
- Cross-links within max_dist=50: 51,632 (98.1%)

Self-links are NOT dominant. The earlier theory was wrong.

**Training dataset after fix** (from `learning_signal_20260518_203340.log`):
- Training: 49,676 samples from 153 files, 3,105 batches/epoch
- Validation: 1,994 samples from 10 dev files
- Coverage: ~26.7% of messages (only annotated suffix has gold links)

## 5. Training Log (learning_signal_20260518_203340.log)

**What happened**: The `learning_signal.sh` script ran on Bunya L40 (48GB GPU). Training **failed with OOM cascade** — every single training batch OOM'd on the forward pass.

### OOM Details
```
Batch 1:   CUDA Out of Memory — 44478/44532MB
Batch 2:   CUDA Out of Memory — 44477/44532MB
...
Batch 10:  CUDA Out of Memory — 44477/44532MB
```
Root exception:
```
torch.OutOfMemoryError: Tried to allocate 1.15 GiB. GPU 0 has 44.39 GiB total,
409.25 MiB free. 43.99 GiB in use, 43.43 GiB allocated by PyTorch.
```
Followed by:
```
RuntimeError: OOM cascade: 10 consecutive OOM batches. Reduce --batch-size or --max-dist.
```

### Why
The model was loaded from the old checkpoint (trained on 0 samples). The training batch processes `batch_size * C * seq_len` tokens per forward pass. At batch_size=16, C up to 50, seq_len=128 → 102,400 tokens/batch through DeBERTa-v3-base (184M params). With gradients + optimizer states (AdamW: 2x model size for momentum + variance), total memory exceeds the L40's 48GB.

The old checkpoint weights being near-random don't cause OOM. The OOM is architecture-driven: DeBERTa-v3-base's memory footprint at batch_size=16 with variable C (up to 50) is too large.

### What Didn't OOM
The evaluation phase (same checkpoint, batch_size=64) ran successfully because no gradients are computed:
- **Accuracy: 0.0098** (random — 1/49 ≈ 2%)
- **F1: 0.0078**
- **Loss: 3.8236**
- Evaluation at batch_size=64 uses only ~20GB due to no backprop

### Solution
Reduce batch_size or use gradient accumulation. Options:
1. `--batch-size 8` (halves memory)
2. `--batch-size 4` with `--gradient-accumulation-steps 4` (effective batch=16, peak memory = batch=4)
3. `--max-length 96` instead of 128

**Bottom line**: The fix works (49,676 valid samples produced). Training needs `--batch-size` reduced to 4-8.

## 6. Clustering Metrics (From eval log)

- ARI: 0.1198, VI: 3.6586
- Per-conversation ARI ranges from -0.0027 to 0.3306
- 10 conversations evaluated, avg coverage 81.3%
- Predicted threads: 5-10 per conversation vs gold: 54-183

These metrics are from an untrained checkpoint — meaningless for evaluation but useful as a sanity check.

## 7. Human-Readable Validation

`--verbose 3` flag in `evaluate.py` produces side-by-side gold vs predicted threads for random conversations. The log shows predicted threads are clearly wrong (5 threads instead of 183 for `2007-01-11_12`), confirming random weights.

Debug predicted pairs show `P(self)=0.0000` for all samples — the model never predicts self-links, which is correct given they're excluded from candidates.

## 8. Key Files

| File | Key Section | Lines |
|------|-------------|-------|
| `src/data_loader.py` | Candidate selection + gold link parsing | 166-172, 350-434 |
| `src/data_loader.py` | Gold link parsing (THE FIX) | 171-172 |
| `src/model.py` | CrossEncoderWithFeatures.forward() | 95-221 |
| `scripts/diagnose_links.py` | Cross-link distance diagnostic | entire file |
| `learning_signal.sh` | Training script for Bunya GPU | entire file |

## 9. Log Files for Analysis

| Log | Description |
|-----|-------------|
| `logs/learning_signal_20260518_203340.log` | Full training + eval with parent/child fix (~49K samples, random weights) |
| `logs/train_20260518_203632.log` | Eval-only on test set with same checkpoint |
| `logs/learning_signal_20260518_192715.log` | Previous run: 0 samples, 2.30s training |

## 10. Next Step

Delete old checkpoints at `/scratch/user/checkpoints_maxdist50/` and run `bash learning_signal.sh` again for a fresh training run from random initialization.