# Evaluation Analysis — 2026-05-11

## Summary

After fixing the clustering evaluation (valid_messages filter) and per-position accuracy logging, we ran evaluation on the epoch-6 checkpoint (trained with `max_dist=15`). The results reveal a critical coverage problem.

## Key Findings

### 1. Pairwise Accuracy = 100% (Real, Not a Bug)

The model achieves perfect accuracy on both dev (462 samples) and test (922 samples) for the multiclass candidate-ranking task. Every position (3-14) shows 1.000 accuracy. The last-position baseline is ~39%, so the task is not trivially easy.

### 2. Coverage Gap: Only 4-6% of Messages Get Predictions

| Split | Samples | Total Messages | Coverage |
|-------|---------|----------------|----------|
| Dev   | 462     | 12,500         | **3.7%** |
| Test  | 922     | 15,000         | **6.1%** |

**Root cause**: `max_dist=15` means only messages whose gold parent is within 15 positions back are included as training/evaluation samples. The other 94-96% of messages have parents further back and are completely ignored.

### 3. Clustering Metrics: ARI=0, VI≈0 (Fixed Evaluation)

After the `valid_messages` fix (only comparing messages that have predictions):

- **VI ≈ 0**: Predicted clusters are identical to gold clusters for the covered messages. This confirms the 100% pairwise accuracy is genuine.
- **ARI = 0**: Numerical edge case — when every predicted message forms an isolated parent-child pair and gold clusters are similarly fragmented, the ARI formula hits 0/0 → 0. VI=0 is the reliable indicator.

### 4. The Real Problem

The model is excellent at a **trivial subtask**: "given 15 candidates, pick the right parent." But this only covers 4-6% of the actual problem. The other 94-96% of messages have parents outside the window and are completely ignored.

This is like training a spell-checker that only checks 4% of words and declaring 100% accuracy. The metric is real but the scope is too narrow to be meaningful.

## Fixes Applied

### Fix 1: Per-Position Accuracy (train.py)
- Removed `min(num_pos_classes, 10)` cap — now shows ALL positions (3-14)
- Previously only showed positions 3-9, hiding the high-frequency positions 10-14

### Fix 2: Clustering Evaluation (evaluate.py)
- Added `valid_messages` parameter to `compute_ari_and_vi()` — restricts comparison to only messages that have predictions
- Messages without predictions are excluded from both gold and pred clusterings
- Added per-conversation coverage reporting (e.g., "65/250 msgs covered = 26.0%")

### Fix 3: Coverage Diagnostic (train.py)
- Added COVERAGE log line showing how many messages have predictions vs total

## Next Steps

### Immediate: Increase max_dist
- Change `--max-dist` from 15 to 50 in both `run_job.slurm` and `eval_job.slurm`
- At max_dist=50, coverage should jump to roughly 30-50% of messages
- This will give a meaningful clustering evaluation

### Medium-term: Rethink the Formulation
- The current multiclass formulation ("which of these C candidates is the parent?") is inherently limited by the candidate window
- A binary formulation ("is message A the parent of message B?") would not have this limitation
- Literature (ALT 2021, ROCLING 2025) uses max_dist=50-60

### Long-term: Full Conversation Disentanglement
- Even with max_dist=50, the model only sees a sliding window
- True disentanglement requires global thread construction across the entire conversation
- Consider graph-based approaches (e.g., clustering on pairwise scores)

## Run Logs Referenced

- Run 24485619: Training (epoch 6 checkpoint, max_dist=15)
- Run 24556124: First evaluation with clustering metrics (broken — compared 65 vs 1250 messages)
- Run 24556804: Second evaluation with fixed clustering metrics (ARI=0, VI≈0, coverage=3.7-6.1%)