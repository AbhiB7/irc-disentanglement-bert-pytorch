# IRC Conversation Disentanglement - Research Handover (2026-05-04)

## ⚠️ CRITICAL UPDATE: Multiclass Architecture Shift
The project has recently undergone a major architectural shift from **Binary Classification** to **Multiclass Classification** (Priority 3 in the roadmap). 

**STATUS**: The implementation is partially complete in `src/`, but **Broken in `tests/`** and contains several logic bugs in the training/evaluation scripts.

---

## 1. Architectural Changes

### Data Loader (`src/data_loader.py`)
- **Shift**: Instead of generating negative/positive pairs, it now frames the problem as "Which of these $C$ candidates is the parent of message $i$?".
- **Mechanism**: 
    - `IRCDisentanglementDataset` generates samples where each sample contains a child message and ALL its potential parents within `max_dist`.
    - `__getitem__` returns a batch of candidate encodings of shape `[C, seq_len]`.
- **Known Bug**: The logic for reconstructing candidate texts in `__getitem__` via string matching is fragile and inefficient. It should use the `conversation_map` indices directly.

### Model (`src/model.py`)
- **Shift**: Replaced Sigmoid/BCE loss with **Softmax/CrossEntropy loss**.
- **Input**: Expects `input_ids` of shape `[batch_size, C, seq_len]`.
- **Head**: The classifier now outputs a probability distribution over the $C$ candidates for each sample in the batch.
- **Known Bug**: `test_model()` at the bottom of the file is still using the old binary architecture and will fail.

### Training & Evaluation (`src/train.py` & `src/evaluate.py`)
- **Complexity**: Variable candidate counts ($C$) are handled via a custom `collate_fn` that pads the candidate dimension to the maximum $C$ in the batch.
- **Metrics**: Accuracy is now the primary metric for the "parent picking" task.
- **Known Bugs**:
    - `src/evaluate.py`: Passes invalid arguments to `create_model()`.
    - `src/train.py`: The `evaluate()` function is still being called with a `threshold` argument (remnant of binary mode).
    - `src/train.py`: TP/FP/FN calculations in `evaluate()` are mathematically incorrect for multiclass sets.
    - `src/train.py`: "Smart Logging" still looks for `labels == 1` to identify "positive batches", which is incorrect for multiclass indices.

---

## 2. File Status Audit

| File | Status | Action Needed |
| :--- | :--- | :--- |
| `src/data_loader.py` | 🛠️ Working but fragile | Refactor `__getitem__` to use indices instead of string matching. |
| `src/model.py` | ✅ Updated | Update the `test_model()` internal test function. |
| `src/train.py` | 🛠️ Buggy | Fix metric calculations, logging, and argument passing. |
| `src/evaluate.py` | ❌ Broken | Fix `create_model` calls. |
| `tests/test_model.py` | ❌ Broken | Completely rewrite to test `[batch, C, seq]` logic. |
| `tests/test_data_loader.py`| ❌ Broken | Update to verify multiclass sample structure. |
| `context/CONTEXT.md` | ✅ Up-to-date | Stable knowledge base. |
| `context/PROGRESS.md` | ✅ Up-to-date | Tracks current run status. |

---

## 3. Immediate Next Steps for the AI Successor

1. **Fix the Tests First**: Rewrite `tests/test_model.py` and `tests/test_data_loader.py` to match the `[batch, C, seq]` architecture. This will reveal the API mismatches.
2. **Refactor Data Loader**: Fix the string-matching bottleneck in `data_loader.py:__getitem__`.
3. **Repair Evaluation Metrics**: Correct the F1/Precision/Recall logic in `train.py` to handle the multiclass output properly (e.g., macro-averaging over the window candidates).
4. **Clean up `evaluate.py`**: Fix the initialization parameters to match the model's new signature.

---

## 4. Technical Implementation Snippets

To help the next model hit the ground running, here are the key data shapes and logic blocks currently in the `src/`.

### 4.1 Data Shape Mismatch (The `[batch, C, seq]` logic)
The model now expects a 3D tensor for input IDs because it encodes $C$ candidates for every batch item.

**Current `src/data_loader.py` logic:**
```python
# Returns:
item = {
    "input_ids": all_candidate_input_ids,        # [C, seq_len]
    "attention_mask": all_candidate_attention_mask, # [C, seq_len]
    "features": torch.tensor(features),          # [num_features]
    "labels": torch.tensor(gold_parent_idx)      # Single integer index
}
```

**Current `src/model.py` forward pass:**
```python
def forward(self, input_ids, attention_mask, ...):
    batch_size, num_candidates, seq_len = input_ids.shape
    # Flattening for BERT processing
    flat_input_ids = input_ids.view(-1, seq_len) # [batch_size * C, seq_len]
    bert_outputs = self.bert(input_ids=flat_input_ids, ...)
    # Reshaping back for multiclass logits
    logits = self.classifier(combined).view(batch_size, num_candidates)
    return {"logits": logits, "probs": torch.softmax(logits, dim=-1)}
```

### 4.2 The String-Matching Fragility
The `data_loader.py:__getitem__` contains this block which must be replaced with index-based lookup:
```python
candidate_texts = []
# ... logic to find candidates ...
for i, c_text in enumerate(candidate_texts):
    if c_text == parent_text: # <--- FRAGILE: Fails on duplicate messages
        current_candidate_idx_in_full_list = i
```

---

## 5. Source File Requirements
*Note for the successor model:* To allow for precise, line-by-line code generation and unambiguous instructions, the following files must be examined in their entirety:

1. `src/data_loader.py`
2. `src/model.py`
3. `src/train.py`
4. `src/evaluate.py`
5. `tests/test_model.py`
6. `tests/test_data_loader.py`

Once these are provided, the goal is to produce a **complete, unambiguous instruction set** for the Cline agent—specific line-by-line changes with exact replacements for all identified bugs.

## 6. Current Training Strategy
- **Backbone**: `microsoft/deberta-v3-base`
- **Max Distance**: 50 (to match StructBERT/ROCLING 2025)
- **Loss**: Cross-Entropy (Multiclass)
- **Hardware**: Targeting A100/H100 on UQ Bunya.

---
*End of Handover*
