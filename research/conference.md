## Hyperparameter Defaults (with Literature Sources)

### `--model-name`: `microsoft/deberta-v3-base`
ROCLING 2025 (Lam & Yang): Evaluated 6 transformer models for IRC disentanglement. DeBERTa-v3 (He et al., 2021) achieved SOTA performance. Alternatives: BERT-base (Devlin et al., 2019), RoBERTa (Liu et al., 2019).

### `--max-length`: 128
Devlin et al. (2019): BERT uses 128 tokens for 90% of pretraining, 512 for remaining 10%. IRC messages are short (<50 tokens typically). 128 captures >95% without wasted padding.

### `--max-dist`: 50
ROCLING 2025 (Lam & Yang): StructBERT uses kh=50. ALT 2021 (Zhu et al.): kc=60 past utterances as candidates. Trade-off: larger values increase recall but also memory/noise.

### `--batch-size`: 64
ALT 2021 (Zhu et al.): BERT+MF uses batch-size=64 for IRC disentanglement. Devlin et al. (2019): 32 for GLUE tasks. 64 is feasible on 12GB GPUs (RTX 5070 verified ~7GB VRAM usage).

### `--learning-rate`: 5e-5
Devlin et al. (2019): recommends 2e-5 to 5e-5 for fine-tuning. ALT 2021: uses Adamax with 5e-5. Bi-CL (Huang et al., 2024): uses Adam with 5e-5.

### `--epochs`: 3
Devlin et al. (2019): 3 epochs for all GLUE tasks. General practice: 3-5 for classification fine-tuning. More epochs risk overfitting.

### `--warmup-ratio`: 0.1 (10%)
Standard practice from BERT paper and HuggingFace defaults. Linear warmup over 10% of total steps prevents early training instability.

### `--dropout`: 0.1
Devlin et al. (2019): dropout=0.1 on classification head. Confirmed by ACL 2025 SemEval and Stanford CS224n 2024 projects. Increase for small datasets.

### `--patience`: 3
Standard early stopping hyperparameter. If validation F1 doesn't improve for 3 consecutive epochs, stop training to prevent overfitting.

## Bugs Fixed (2026-05-05)

### 1. `num_features=4` in train.py (was `main()`)
The model was being created with `num_features=4` in `main()` (line 753) while the data loader output 5 features. This caused a shape mismatch at `combined = torch.cat([cls_embedding, expanded_features], dim=-1)` — the model expected a concatenated vector of size 772 (768+4) but received size 773 (768+5) from the data loader. Fixed to `num_features=5`.

### 2. `warmup-steps=100` (hardcoded) replaced with `--warmup-ratio 0.1`
The original `--warmup-steps 100` was a fixed integer that should scale with dataset size. Standard practice is 10% of total training steps (BERT paper; HuggingFace default). Changed to `--warmup-ratio 0.1` so it scales automatically: e.g., for ~270K total steps (full data, batch=64, 3 epochs) → 27K warmup steps instead of 100.

## Dropout = 0.1

Standard for BERT classification heads. Devlin et al. (2019) original BERT paper uses dropout=0.1 on the classification head. Confirmed in practice: ACL 2025 SemEval paper (dropout=0.1 for BERT multi-label classification), Stanford CS224n 2024 projects (hidden dropout probability = 0.1), and common BERT fine-tuning guides (mbrenndoerfer.com: "dropout (0.1): Applied in the classifier head. Increase for small datasets to reduce overfitting.").

Applied via `nn.Dropout(0.1)` to the [CLS] embedding before the linear classifier layer.

## max_length = 128

Pragmatic choice. BERT supports up to 512 (Devlin et al., 2019). IRC messages are short, 128 is enough for most. Remaining positions padded with `[PAD]` token.

## Self-Links

Standard in the literature. Kummerfeld et al. (2019): *"If a message started a new conversation it was linked to itself."* Our code follows this — self-link (j=i) is always included as a candidate parent.


## Example of data
Concrete example. Say a conversation has 5 messages:

```
0: [10:00] <Alice> hello
1: [10:01] <Bob> hi             
2: [10:02] <Alice> weather is nice
3: [10:03] <Charlie> anyone here?
4: [10:04] <Bob> yes charlie   ← replies to message 3
```

Gold link: message 4's parent is message 3.

With `max_dist=50`, when processing child message **i=4**, candidates are messages **j=0..4** (all within window).

After `_create_samples_for_conversation`, the three lists look like:

---

**`self.conversations`** — holds the parsed `IRCConversation` objects, one per file.

**`self.samples`** — one entry per child message. For message 4:

```python
(
    "anyone here?",           # parent_text (the gold parent at j=3)
    "yes charlie",            # child_text (message at i=4)
    torch.Tensor([5, 4]),     # features: 5 candidates × 4 feature values
    3                         # gold_parent_idx: candidate #3 is correct
)
```

**`self.conversation_map`** — maps sample index back to source. For this same sample:

```python
(
    0,                                 # conv_idx: first conversation (index 0)
    4,                                 # msg_i_idx: child is message at index 4
    [                                  # candidate_indices: (conv, child, candidate)
        (0, 4, 0),  # candidate 0 = message 0
        (0, 4, 1),  # candidate 1 = message 1
        (0, 4, 2),  # candidate 2 = message 2
        (0, 4, 3),  # candidate 3 = message 3  ← this is the gold parent
        (0, 4, 4),  # candidate 4 = message 4 (self-link)
    ]
)
```

---

Then in `__getitem__`, it uses `conversation_map` to fetch the actual candidate texts from the `conversations` object, tokenises all 5, and spits out `input_ids: [5, 128]`.

So the three lists serve different purposes:
- `conversations`: permanent store of loaded data
- `samples`: lightweight metadata for fast iteration (text + features + label)
- `conversation_map`: back-reference so `__getitem__` can reconstruct the full tensor


When `skip_labels=True`, the dataset pretends there's no answer key. It sets every label to -1 (meaning "unknown").

You'd use it when running the model on brand new data that has no manual annotations — you still want the model to make predictions, but there's nothing to compare against for accuracy. You're just running inference, not training or evaluating.
