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
