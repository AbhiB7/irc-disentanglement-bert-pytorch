# Run 23918880: pos_weight Cap Fix

## Issue
Training run 23918880 produced F1: 0.0867 with severe class imbalance (~746:1 negative-to-positive ratio). Model predicted positive for nearly everything (Recall: 98.7%, Precision: 4.53%).

## Root Cause
`pos_weight` was capped at 300 in `src/model.py:154`, but the actual imbalance ratio was ~746:1. Loss contributions still favored negatives (746 > 300).

## Fix
Raised `pos_weight` cap from 300 → 1500 in `src/model.py:154`.

```python
# Before
pos_weight = (num_neg / (num_pos + 1e-8)).clamp(min=10.0, max=300.0)

# After  
pos_weight = (num_neg / (num_pos + 1e-8)).clamp(min=10.0, max=1500.0)
```

## Impact
With cap=1500, pos_weight can reach ~746 (matching the imbalance ratio), allowing proper loss balancing where positive samples dominate gradient updates.

## Next Step
Re-train with new cap, then optimize decision threshold.
