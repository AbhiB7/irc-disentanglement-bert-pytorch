"""
Test script for IRCDisentanglementDataset.__getitem__.
Validates that __getitem__ returns correctly shaped tensors with correct labels.

Run with: python tests/test_data_loader.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import logging

# Suppress logging from the data_loader module during test
logging.basicConfig(level=logging.WARNING)
logging.getLogger("data_loader").setLevel(logging.WARNING)

from data_loader import (
    IRCMessage,
    IRCConversation,
    parse_irc_line,
    compute_features,
    IRCDisentanglementDataset,
)


# =============================================================
# Step 1: Dummy tokenizer — returns fixed-shape tensors
# =============================================================
class DummyTokenizer:
    """Mimics a HuggingFace tokenizer without downloading anything."""

    def __init__(self, vocab_size=30522, max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102

    def __call__(
        self,
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    ):
        """Return dummy tensors with correct structure."""
        seq_len = min(max_length, self.max_length)
        input_ids = torch.zeros((1, seq_len), dtype=torch.long)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


# =============================================================
# Step 2: Build a synthetic conversation (same as test_create_samples.py)
# =============================================================
def build_test_conversation():
    """
    Create a conversation with 10 messages.

    Index | Speaker | Text              | Gold parent | Notes
    ------|---------|-------------------|-------------|----------------------
     0     | alice   | "hello"           | — (self)    | first message
     1     | bob     | "hi alice"        | 0           | replies to alice
     2     | alice   | "how are you"     | 1           | replies to bob
     3     | SYSTEM  | "=== topic ==="   | —           | system message
     4     | bob     | "where is alice"  | 4 (self)    | starts new thread
     5     | alice   | "i am here"       | 4           | replies to bob's new thread
     6     | bob     | "good"            | 5           | replies to alice
     7     | bob     | "alice?"          | 5           | also replies to alice
     8     | alice   | "yes?"            | 7           | replies to bob's "alice?"
     9     | bob     | "never mind"      | 8           | replies to alice

    Gold links dict: {1:[0], 2:[1], 4:[4], 5:[4], 6:[5], 7:[5], 8:[7], 9:[8]}
    Messages 0 and 3 have no gold entries.
    """
    messages = []

    message_data = [
        (0, (10, 0), "alice", "hello", False),
        (1, (10, 1), "bob", "hi alice", False),
        (2, (10, 2), "alice", "how are you", False),
        (3, None, "SYSTEM", "=== topic ===", True),  # system msg
        (4, (10, 5), "bob", "where is alice", False),
        (5, (10, 6), "alice", "i am here", False),
        (6, (10, 7), "bob", "good", False),
        (7, (10, 8), "bob", "alice?", False),
        (8, (10, 9), "alice", "yes?", False),
        (9, (10, 10), "bob", "never mind", False),
    ]

    all_users = {"alice", "bob"}

    for idx, ts, speaker, text, is_system in message_data:
        msg = IRCMessage(
            index=idx,
            timestamp=ts,
            speaker=speaker,
            text=text,
            is_system=is_system,
            is_bot=False,
            targets=set(),
            last_from_same_user=None,
            next_from_same_user=None,
        )
        messages.append(msg)

    # Fill in targets (who each message mentions)
    targets_map = {
        1: {"alice"},  # bob says "hi alice" → mentions alice
        4: {"alice"},  # bob says "where is alice" → mentions alice
        7: {"alice"},  # bob says "alice?" → mentions alice
        8: {"bob"},  # alice says "yes?" → potentially replies to bob
    }
    for msg in messages:
        if msg.index in targets_map:
            msg.targets = targets_map[msg.index]

    # Build user_message_indices
    user_message_indices = {}
    for msg in messages:
        if not msg.is_system:
            user_message_indices.setdefault(msg.speaker, []).append(msg.index)

    # Set last_from_same_user and next_from_same_user
    for user, indices in user_message_indices.items():
        for i, idx in enumerate(indices):
            messages[idx].last_from_same_user = indices[i - 1] if i > 0 else None
            messages[idx].next_from_same_user = (
                indices[i + 1] if i < len(indices) - 1 else None
            )

    # Gold links
    gold_links = {
        1: [0],
        2: [1],
        4: [4],  # self-link — starts new thread
        5: [4],
        6: [5],
        7: [5],
        8: [7],
        9: [8],
    }

    conv = IRCConversation(
        name="test_conv",
        messages=messages,
        gold_links=gold_links,
        user_message_indices=user_message_indices,
    )

    return conv


# =============================================================
# Helper: create a dataset pre-populated with our synthetic conv
# =============================================================
def create_test_dataset(max_dist=50):
    """Create an IRCDisentanglementDataset with synthetic conversation."""
    tokenizer = DummyTokenizer()

    # Create temporary dummy files so __init__ doesn't crash on file loading
    with tempfile.NamedTemporaryFile(
        suffix=".ascii.txt", delete=False, mode="w"
    ) as f_ascii:
        f_ascii.write("[10:00] <dummy> test\n")
        dummy_ascii = f_ascii.name
    with tempfile.NamedTemporaryFile(
        suffix=".annotation.txt", delete=False, mode="w"
    ) as f_ann:
        f_ann.write("- -1\n")
        dummy_ann = f_ann.name

    dataset = IRCDisentanglementDataset(
        ascii_files=[dummy_ascii],
        annotation_files=[dummy_ann],
        tokenizer=tokenizer,
        max_dist=max_dist,
        max_length=128,
        test_start=0,
        test_end=10,
    )

    # Replace loaded conversation with our synthetic one
    dataset.conversations = [build_test_conversation()]
    dataset.samples = []
    dataset.conversation_map = []

    # Now manually call _create_samples_for_conversation
    dataset._create_samples_for_conversation(dataset.conversations[0], 0)

    # Clean up temp files
    os.unlink(dummy_ascii)
    os.unlink(dummy_ann)

    return dataset


# =============================================================
# Step 3: Tests
# =============================================================


def test_getitem_output_structure():
    """
    TEST 1: Verify __getitem__ returns a dict with the correct keys and tensor shapes.
    """
    print("\n" + "=" * 60)
    print("TEST 1: __getitem__ output structure")
    print("=" * 60)

    dataset = create_test_dataset(max_dist=50)
    all_pass = True

    for idx in range(len(dataset)):
        item = dataset[idx]

        # Check that output is a dict
        is_dict = isinstance(item, dict)
        if not is_dict:
            print(f"  FAIL sample {idx}: output is not a dict, got {type(item)}")
            all_pass = False
            continue

        # Check expected keys
        expected_keys = {"input_ids", "attention_mask", "features", "labels"}
        actual_keys = set(item.keys())
        missing_keys = expected_keys - actual_keys
        extra_keys = actual_keys - expected_keys

        if missing_keys:
            print(f"  FAIL sample {idx}: missing keys: {missing_keys}")
            all_pass = False
        if extra_keys:
            print(f"  FAIL sample {idx}: unexpected keys (may be ok): {extra_keys}")

        # Check input_ids: [C, seq_len]
        input_ids = item["input_ids"]
        is_long = input_ids.dtype == torch.long
        is_2d = input_ids.dim() == 2
        if not (is_long and is_2d):
            print(
                f"  FAIL sample {idx}: input_ids shape={input_ids.shape}, dtype={input_ids.dtype}"
            )
            all_pass = False

        # Check attention_mask: [C, seq_len]
        attention_mask = item["attention_mask"]
        is_long_mask = attention_mask.dtype == torch.long
        is_2d_mask = attention_mask.dim() == 2
        if not (is_long_mask and is_2d_mask):
            print(
                f"  FAIL sample {idx}: attention_mask shape={attention_mask.shape}, dtype={attention_mask.dtype}"
            )
            all_pass = False

        # Check input_ids and attention_mask have same shape
        if input_ids.shape != attention_mask.shape:
            print(
                f"  FAIL sample {idx}: input_ids shape {input_ids.shape} != attention_mask shape {attention_mask.shape}"
            )
            all_pass = False

        # Check features: [C, 5]
        features = item["features"]
        is_float = features.dtype == torch.float32 or features.dtype == torch.float64
        is_2d_feat = features.dim() == 2
        has_5_features = features.shape[1] == 5 if is_2d_feat else False
        if not (is_float and is_2d_feat and has_5_features):
            print(
                f"  FAIL sample {idx}: features shape={features.shape}, dtype={features.dtype}"
            )
            all_pass = False

        # Check that C dimension matches between input_ids and features
        if input_ids.shape[0] != features.shape[0]:
            print(
                f"  FAIL sample {idx}: C mismatch: input_ids={input_ids.shape[0]}, features={features.shape[0]}"
            )
            all_pass = False

        # Check labels: scalar LongTensor
        labels = item["labels"]
        is_scalar = labels.dim() == 0 or labels.numel() == 1
        is_long_label = labels.dtype == torch.long
        if not (is_scalar and is_long_label):
            print(
                f"  FAIL sample {idx}: labels shape={labels.shape}, dtype={labels.dtype}"
            )
            all_pass = False

    if all_pass:
        print(f"  All {len(dataset)} samples have correct structure — PASS")

    print(f"\n>>> TEST 1 {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


def test_getitem_label_correctness():
    """
    TEST 2: Verify gold parent labels are correct for known cases.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Label correctness")
    print("=" * 60)

    dataset = create_test_dataset(max_dist=50)
    all_pass = True

    # Samples are only created for messages with gold labels (msg0 and msg3 are skipped).
    # Messages with gold labels: 1,2,4,5,6,7,8,9 → 8 samples.
    # Sample ordering matches message order after filtering.
    # Label = index of gold parent within candidate list (SYSTEM msg3 filtered out).
    #   msg1: candidates [0,1] → gold parent 0 at index 0
    #   msg2: candidates [0,1,2] → gold parent 1 at index 1
    #   msg4: candidates [0,1,2,4] → gold parent 4 at index 3
    #   msg5: candidates [0,1,2,4,5] → gold parent 4 at index 3
    #   msg6: candidates [0,1,2,4,5,6] → gold parent 5 at index 4
    #   msg7: candidates [0,1,2,4,5,6,7] → gold parent 5 at index 4
    #   msg8: candidates [0,1,2,4,5,6,7,8] → gold parent 7 at index 6
    #   msg9: candidates [0,1,2,4,5,6,7,8,9] → gold parent 8 at index 7
    expected_labels = [0, 1, 3, 3, 4, 4, 6, 7]

    # Verify we have the expected number of samples
    if len(dataset) != len(expected_labels):
        print(f"  FAIL: expected {len(expected_labels)} samples, got {len(dataset)}")
        all_pass = False
        return all_pass

    for idx, expected in enumerate(expected_labels):
        item = dataset[idx]
        actual = item["labels"].item()
        passed = actual == expected
        if not passed:
            print(f"  FAIL sample {idx}: label={actual}, expected={expected}")
            all_pass = False

    if all_pass:
        print(f"  All {len(dataset)} samples have correct labels — PASS")

    print(f"\n>>> TEST 2 {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


def test_getitem_candidate_count():
    """
    TEST 3: Verify the number of candidates (C dimension) is correct.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Candidate count correctness")
    print("=" * 60)

    dataset = create_test_dataset(max_dist=50)
    all_pass = True

    # Build expected candidate counts dynamically.
    # Dataset skips msg0 and msg3 (no gold labels).
    # Messages with gold labels: 1,2,4,5,6,7,8,9 (8 samples).
    # For each sample index i in [0..7], the corresponding message index is:
    #   sample0 → msg1, sample1 → msg2, sample2 → msg4, etc.
    # Candidate count = number of preceding non-filtered messages + self-link.
    # System msg3 is filtered out as parent candidate.
    msg_indices = [1, 2, 4, 5, 6, 7, 8, 9]
    # Preceding non-system messages for each:
    #   msg1: [0,1] → 2 (msg0 is not system)
    #   msg2: [0,1,2] → 3
    #   msg4: [0,1,2,4] → 4 (msg3 filtered)
    #   msg5: [0,1,2,4,5] → 5
    #   msg6: [0,1,2,4,5,6] → 6
    #   msg7: [0,1,2,4,5,6,7] → 7
    #   msg8: [0,1,2,4,5,6,7,8] → 8
    #   msg9: [0,1,2,4,5,6,7,8,9] → 9
    expected_counts = [2, 3, 4, 5, 6, 7, 8, 9]

    for idx in range(len(dataset)):
        expected_c = expected_counts[idx]
        item = dataset[idx]
        actual_c = item["input_ids"].shape[0]
        passed = actual_c == expected_c
        if not passed:
            print(
                f"  FAIL sample {idx}: input_ids has {actual_c} candidates, expected {expected_c}"
            )
            all_pass = False

    if all_pass:
        print(
            f"  All {len(dataset.samples)} samples have correct candidate counts — PASS"
        )

    print(f"\n>>> TEST 3 {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


def test_getitem_features_preserved():
    """
    TEST 4: Verify features tensor is preserved through __getitem__.

    The features stored in self.samples[idx][2] should be the same tensor
    returned in item["features"], not a different tensor with different values.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Features tensor preservation")
    print("=" * 60)

    dataset = create_test_dataset(max_dist=50)
    all_pass = True

    for idx in range(len(dataset)):
        item = dataset[idx]

        # Get original features from self.samples
        original_features = dataset.samples[idx][2]  # [C, 5]
        returned_features = item["features"]  # [C, 5]

        # Check shape matches
        if original_features.shape != returned_features.shape:
            print(
                f"  FAIL sample {idx}: feature shape mismatch: original={original_features.shape}, returned={returned_features.shape}"
            )
            all_pass = False
            continue

        # Check values are exactly equal
        if not torch.allclose(original_features, returned_features):
            print(f"  FAIL sample {idx}: feature values differ!")
            diff_mask = ~torch.isclose(original_features, returned_features)
            num_diffs = diff_mask.sum().item()
            print(f"    {num_diffs} out of {original_features.numel()} elements differ")
            all_pass = False

    if all_pass:
        print(f"  All {len(dataset.samples)} samples preserve features exactly — PASS")

    print(f"\n>>> TEST 4 {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


def test_getitem_with_max_dist():
    """
    TEST 5: Verify __getitem__ works correctly with max_dist limiting.
    """
    print("\n" + "=" * 60)
    print("TEST 5: __getitem__ with max_dist=3")
    print("=" * 60)

    dataset = create_test_dataset(max_dist=3)
    all_pass = True

    # Dataset skips msg0 and msg3 (no gold labels).
    # Messages with gold labels: 1,2,4,5,6,7,8,9 (8 samples).
    # For max_dist=3, range(max(0, i-2), i+1):
    #   msg1: range(0,2) → [0,1] → 2 candidates
    #   msg2: range(0,3) → [0,1,2] → 3 candidates
    #   msg4: range(2,5) → [2,3,4] → msg3 system filtered → [2,4] → 2
    #   msg5: range(3,6) → [3,4,5] → msg3 filtered → [4,5] → 2
    #   msg6: range(4,7) → [4,5,6] → 3 candidates
    #   msg7: range(5,8) → [5,6,7] → 3 candidates
    #   msg8: range(6,9) → [6,7,8] → 3 candidates
    #   msg9: range(7,10) → [7,8,9] → 3 candidates
    expected_counts = [2, 3, 2, 2, 3, 3, 3, 3]

    for idx in range(len(dataset)):
        expected_c = expected_counts[idx]
        item = dataset[idx]
        actual_c = item["input_ids"].shape[0]
        passed = actual_c == expected_c
        if not passed:
            print(
                f"  FAIL sample {idx}: input_ids has {actual_c} candidates, expected {expected_c}"
            )
            all_pass = False

    if all_pass:
        print(
            f"  All {len(dataset.samples)} samples have correct candidate counts with max_dist=3 — PASS"
        )

    print(f"\n>>> TEST 5 {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


def test_getitem_seq_length():
    """
    TEST 6: Verify seq_len dimension matches max_length.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Sequence length correctness")
    print("=" * 60)

    max_length = 128
    tokenizer = DummyTokenizer(max_length=max_length)

    with tempfile.NamedTemporaryFile(
        suffix=".ascii.txt", delete=False, mode="w"
    ) as f_ascii:
        f_ascii.write("[10:00] <dummy> test\n")
        dummy_ascii = f_ascii.name
    with tempfile.NamedTemporaryFile(
        suffix=".annotation.txt", delete=False, mode="w"
    ) as f_ann:
        f_ann.write("- -1\n")
        dummy_ann = f_ann.name

    dataset = IRCDisentanglementDataset(
        ascii_files=[dummy_ascii],
        annotation_files=[dummy_ann],
        tokenizer=tokenizer,
        max_dist=50,
        max_length=max_length,
        test_start=0,
        test_end=10,
    )

    dataset.conversations = [build_test_conversation()]
    dataset.samples = []
    dataset.conversation_map = []
    dataset._create_samples_for_conversation(dataset.conversations[0], 0)

    os.unlink(dummy_ascii)
    os.unlink(dummy_ann)

    all_pass = True
    for idx in range(len(dataset)):
        item = dataset[idx]
        actual_seq_len = item["input_ids"].shape[1]
        passed = actual_seq_len == max_length
        if not passed:
            print(
                f"  FAIL sample {idx}: seq_len={actual_seq_len}, expected {max_length}"
            )
            all_pass = False

    if all_pass:
        print(f"  All {len(dataset.samples)} samples have seq_len={max_length} — PASS")

    print(f"\n>>> TEST 6 {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


def test_getitem_consistency():
    """
    TEST 7: Verify calling __getitem__ twice returns the same values (determinism).
    """
    print("\n" + "=" * 60)
    print("TEST 7: __getitem__ determinism")
    print("=" * 60)

    dataset = create_test_dataset(max_dist=50)
    all_pass = True

    # Dataset has 8 samples (indices 0-7). Use valid indices only.
    valid_indices = [i for i in [0, 2, 5, 9] if i < len(dataset)]
    if not valid_indices:
        print("  No valid indices to check determinism")
        return True()

    for idx in valid_indices:
        item1 = dataset[idx]
        item2 = dataset[idx]

        # Compare all tensors
        for key in ["input_ids", "attention_mask", "features", "labels"]:
            t1 = item1[key]
            t2 = item2[key]
            if not torch.equal(t1, t2):
                print(f"  FAIL sample {idx}, key '{key}': values differ between calls")
                all_pass = False

    if all_pass:
        print(f"  All checked samples are deterministic — PASS")

    print(f"\n>>> TEST 7 {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


def test_parse_irc_line():
    """
    TEST 8: Verify parse_irc_line handles all message formats correctly.
    """
    print("\n" + "=" * 60)
    print("TEST 8: parse_irc_line")
    print("=" * 60)

    all_pass = True

    # 1. Normal message
    ts, speaker, text, is_system = parse_irc_line("[10:30] <bob> hi alice")
    passed = (
        ts == (10, 30)
        and speaker == "bob"
        and text == "hi alice"
        and is_system == False
    )
    print(f"  Normal message: PASS={passed}")
    if not passed:
        print(f"    Got: ts={ts}, speaker='{speaker}', text='{text}', system={is_system}")
    all_pass = all_pass and passed

    # 2. System message
    ts, speaker, text, is_system = parse_irc_line("=== topic change ===")
    passed = (
        ts is None
        and speaker == "SYSTEM"
        and text == "=== topic change ==="
        and is_system == True
    )
    print(f"  System message: PASS={passed}")
    if not passed:
        print(f"    Got: ts={ts}, speaker='{speaker}', text='{text}', system={is_system}")
    all_pass = all_pass and passed

    # 3. Speaker with spaces in name
    ts, speaker, text, is_system = parse_irc_line("[00:05] <john_doe> hello world")
    passed = (
        ts == (0, 5)
        and speaker == "john_doe"
        and text == "hello world"
        and is_system == False
    )
    print(f"  Underscore name: PASS={passed}")
    if not passed:
        print(f"    Got: ts={ts}, speaker='{speaker}', text='{text}', system={is_system}")
    all_pass = all_pass and passed

    # 4. Garbage / unparseable line
    ts, speaker, text, is_system = parse_irc_line("this is not a valid irc line")
    passed = (
        ts is None
        and speaker == "UNKNOWN"
        and text == "this is not a valid irc line"
        and is_system == True
    )
    print(f"  Garbage text:  PASS={passed}")
    if not passed:
        print(f"    Got: ts={ts}, speaker='{speaker}', text='{text}', system={is_system}")
    all_pass = all_pass and passed

    print(f"\n>>> TEST 8 {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


# =============================================================
# Main
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING IRCDisentanglementDataset.__getitem__")
    print("=" * 60)

    t1 = test_getitem_output_structure()
    t2 = test_getitem_label_correctness()
    t3 = test_getitem_candidate_count()
    t4 = test_getitem_features_preserved()
    t5 = test_getitem_with_max_dist()
    t6 = test_getitem_seq_length()
    t7 = test_getitem_consistency()
    t8 = test_parse_irc_line()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Output Structure):      {'PASS' if t1 else 'FAIL'}")
    print(f"Test 2 (Label Correctness):     {'PASS' if t2 else 'FAIL'}")
    print(f"Test 3 (Candidate Count):       {'PASS' if t3 else 'FAIL'}")
    print(f"Test 4 (Features Preserved):    {'PASS' if t4 else 'FAIL'}")
    print(f"Test 5 (max_dist):              {'PASS' if t5 else 'FAIL'}")
    print(f"Test 6 (Sequence Length):       {'PASS' if t6 else 'FAIL'}")
    print(f"Test 7 (Determinism):           {'PASS' if t7 else 'FAIL'}")
    print(f"Test 8 (parse_irc_line):        {'PASS' if t8 else 'FAIL'}")
    print(
        f"\nOverall: {'ALL PASSED' if all([t1, t2, t3, t4, t5, t6, t7, t8]) else 'SOME FAILED'}"
    )
