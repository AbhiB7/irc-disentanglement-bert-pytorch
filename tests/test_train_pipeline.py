"""
Tests for src/train.py pipeline functions (collate_fn, create_dataloaders)

Tests the data batching and dataloader creation logic that bridges
data_loader.py output and model.py input.

Run with: python -m pytest tests/test_train_pipeline.py -v
"""

import sys
import torch
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from train import collate_fn, create_dataloaders
from data_loader import IRCDisentanglementDataset, load_dataset_files
from transformers import AutoTokenizer


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def tokenizer():
    """BERT tokenizer, loaded once per module"""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


# ─────────────────────────────────────────────
# collate_fn tests
# ─────────────────────────────────────────────

class TestCollateFn:
    """Test the custom collate function that pads variable-sized candidate lists"""

    def test_basic_padding(self):
        """Test padding when items have different candidate counts"""
        # Create 3 fake batch items with different numbers of candidates
        seq_len = 8
        num_features = 5
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (3, seq_len)),   # 3 candidates
                "attention_mask": torch.ones(3, seq_len, dtype=torch.long),
                "features": torch.randn(3, num_features),              # [C, 5]
                "labels": torch.tensor(1, dtype=torch.long),
            },
            {
                "input_ids": torch.randint(0, 1000, (5, seq_len)),   # 5 candidates
                "attention_mask": torch.ones(5, seq_len, dtype=torch.long),
                "features": torch.randn(5, num_features),              # [C, 5]
                "labels": torch.tensor(3, dtype=torch.long),
            },
            {
                "input_ids": torch.randint(0, 1000, (2, seq_len)),   # 2 candidates
                "attention_mask": torch.ones(2, seq_len, dtype=torch.long),
                "features": torch.randn(2, num_features),              # [C, 5]
                "labels": torch.tensor(0, dtype=torch.long),
            },
        ]

        result = collate_fn(batch)

        # All should be padded to 5 candidates (the max)
        assert result["input_ids"].shape == (3, 5, 8), \
            f"Expected [3, 5, 8], got {result['input_ids'].shape}"
        assert result["attention_mask"].shape == (3, 5, 8)
        assert result["features"].shape == (3, 5, 5), \
            f"Expected features [3, 5, 5], got {result['features'].shape}"
        assert result["labels"].shape == (3,)

    def test_padded_entries_are_zero(self):
        """Test that padded candidate slots are all zeros"""
        seq_len = 4
        num_features = 5
        batch = [
            {
                "input_ids": torch.full((1, seq_len), 42, dtype=torch.long),  # 1 candidate, all 42
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                "features": torch.zeros(1, num_features),                      # [1, 5]
                "labels": torch.tensor(0, dtype=torch.long),
            },
            {
                "input_ids": torch.full((3, seq_len), 99, dtype=torch.long),  # 3 candidates, all 99
                "attention_mask": torch.ones(3, seq_len, dtype=torch.long),
                "features": torch.zeros(3, num_features),                      # [3, 5]
                "labels": torch.tensor(1, dtype=torch.long),
            },
        ]

        result = collate_fn(batch)

        # First item has only 1 real candidate; slots 1 and 2 should be zero-padded
        assert torch.all(result["input_ids"][0, 0, :] == 42), "First candidate should be 42"
        assert torch.all(result["input_ids"][0, 1:, :] == 0), "Padded slots should be 0"
        assert torch.all(result["attention_mask"][0, 0, :] == 1), "First candidate mask should be 1"
        assert torch.all(result["attention_mask"][0, 1:, :] == 0), "Padded slots mask should be 0"
        assert torch.all(result["features"][0, 1:, :] == 0), "Padded feature slots should be 0"

    def test_features_and_labels_preserved(self):
        """Test that features and labels are preserved through collation"""
        seq_len = 4
        num_features = 5
        feat_a = torch.randn(2, num_features)
        feat_b = torch.randn(2, num_features)
        batch = [
            {
                "input_ids": torch.randn(2, seq_len).long(),
                "attention_mask": torch.ones(2, seq_len, dtype=torch.long),
                "features": feat_a,                                               # [2, 5]
                "labels": torch.tensor(0, dtype=torch.long),
            },
            {
                "input_ids": torch.randn(2, seq_len).long(),
                "attention_mask": torch.ones(2, seq_len, dtype=torch.long),
                "features": feat_b,                                               # [2, 5]
                "labels": torch.tensor(1, dtype=torch.long),
            },
        ]

        result = collate_fn(batch)

        assert torch.allclose(result["features"][0], feat_a)
        assert torch.allclose(result["features"][1], feat_b)
        assert result["labels"][0].item() == 0
        assert result["labels"][1].item() == 1

    def test_single_item_batch(self):
        """Test with a batch of size 1 (no padding needed)"""
        seq_len = 8
        num_features = 5
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (4, seq_len)),
                "attention_mask": torch.ones(4, seq_len, dtype=torch.long),
                "features": torch.randn(4, num_features),                          # [4, 5]
                "labels": torch.tensor(2, dtype=torch.long),
            },
        ]

        result = collate_fn(batch)

        assert result["input_ids"].shape == (1, 4, 8)
        assert result["features"].shape == (1, 4, 5)
        assert result["labels"][0].item() == 2

    def test_labels_stay_as_long(self):
        """Test labels are dtype long (required by CrossEntropyLoss)"""
        seq_len = 4
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (1, seq_len)),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                "features": torch.zeros(1, 5),                                     # [1, 5]
                "labels": torch.tensor(0, dtype=torch.long),
            },
        ]
        result = collate_fn(batch)
        assert result["labels"].dtype == torch.long, \
            f"Expected long, got {result['labels'].dtype}"
        assert result["input_ids"].dtype == torch.long

    def test_label_clamp_within_bounds(self):
        """Test labels are preserved when within candidate range"""
        seq_len = 4
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (5, seq_len)),
                "attention_mask": torch.ones(5, seq_len, dtype=torch.long),
                "features": torch.zeros(5, 5),
                "labels": torch.tensor(2, dtype=torch.long),  # label=2, max_candidates=5 → stays as 2
            },
        ]
        result = collate_fn(batch)
        assert result["labels"][0].item() == 2, \
            f"Expected label 2, got {result['labels'][0].item()}"

    def test_label_clamp_out_of_bounds(self):
        """Test labels are clamped when label >= max_candidates (prevents NaN cascade)"""
        seq_len = 4
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (3, seq_len)),  # 3 candidates w/ max_dist=5 → capped at 3
                "attention_mask": torch.ones(3, seq_len, dtype=torch.long),
                "features": torch.zeros(3, 5),
                "labels": torch.tensor(4, dtype=torch.long),  # label=4 > max_candidates-1=2 → clamped to 2
            },
        ]
        # Call with max_dist=5 (same as the test default), but label=4 > 3-1=2
        result = collate_fn(batch)
        assert result["labels"][0].item() == 2, \
            f"Expected clamped label 2 (max_candidates-1), got {result['labels'][0].item()}"

    def test_all_same_candidates(self):
        """Test when all items have the same number of candidates (no padding)"""
        seq_len = 8
        num_features = 5
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (3, seq_len)),
                "attention_mask": torch.ones(3, seq_len, dtype=torch.long),
                "features": torch.randn(3, num_features),                          # [3, 5]
                "labels": torch.tensor(0, dtype=torch.long),
            },
            {
                "input_ids": torch.randint(0, 1000, (3, seq_len)),
                "attention_mask": torch.ones(3, seq_len, dtype=torch.long),
                "features": torch.randn(3, num_features),                          # [3, 5]
                "labels": torch.tensor(1, dtype=torch.long),
            },
        ]

        result = collate_fn(batch)

        assert result["input_ids"].shape == (2, 3, seq_len)
        assert result["features"].shape == (2, 3, 5)
        # No padding needed, so non-zero values should match
        assert not torch.all(result["input_ids"][0, 2, :] == 0)  # Last candidate should have real data


# ─────────────────────────────────────────────
# create_dataloaders integration tests
# ─────────────────────────────────────────────

class TestCreateDataloaders:
    """Test create_dataloaders with real data/tiny/ files"""

    def test_creates_train_and_dev_loaders(self, tokenizer):
        """Test that create_dataloaders returns both train and dev loaders"""
        class MockArgs:
            mode = "train"
            data_dir = "data"
            model_name = "bert-base-uncased"
            max_length = 128
            max_dist = 50
            batch_size = 4
            learning_rate = 5e-5
            epochs = 1
            warmup_ratio = 0.1
            dropout = 0.1
            freeze_bert = False
            patience = 3
            output_dir = "checkpoints"
            save_every = 1
            eval_every = 1
            device = "cpu"
            num_workers = 0
            fp16 = False
            test_start = 1000
            test_end = 1300
            resume_from = None

        train_loader, dev_loader = create_dataloaders(MockArgs(), tokenizer)

        assert train_loader is not None, "train_loader should not be None"
        assert dev_loader is not None, "dev_loader should not be None"
        assert len(train_loader.dataset) > 0, "Train dataset should have samples"
        assert len(dev_loader.dataset) > 0, "Dev dataset should have samples"

    def test_batch_shapes_match_model_input(self, tokenizer):
        """Test that a batch from the dataloader has the right shapes for the model"""
        class MockArgs:
            mode = "train"
            data_dir = "data"
            model_name = "bert-base-uncased"
            max_length = 128
            max_dist = 50
            batch_size = 2
            learning_rate = 5e-5
            epochs = 1
            warmup_ratio = 0.1
            dropout = 0.1
            freeze_bert = False
            patience = 3
            output_dir = "checkpoints"
            save_every = 1
            eval_every = 1
            device = "cpu"
            num_workers = 0
            fp16 = False
            test_start = 1000
            test_end = 1300
            resume_from = None

        train_loader, _ = create_dataloaders(MockArgs(), tokenizer)

        # Get first batch
        batch = next(iter(train_loader))

        # Check shapes match what CrossEncoderWithFeatures.forward() expects
        assert "input_ids" in batch, "Missing input_ids"
        assert "attention_mask" in batch, "Missing attention_mask"
        assert "features" in batch, "Missing features"
        assert "labels" in batch, "Missing labels"

        batch_size = batch["input_ids"].shape[0]
        assert batch_size == 2, f"Expected batch_size=2, got {batch_size}"

        # input_ids: [batch, C, seq_len]
        assert batch["input_ids"].dim() == 3, \
            f"Expected 3D input_ids, got {batch['input_ids'].dim()}D"
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].shape[2] == 128  # max_length

        # attention_mask: [batch, C, seq_len], same shape as input_ids
        assert batch["attention_mask"].shape == batch["input_ids"].shape

        # features: [batch, C, 5] (per-candidate features)
        assert batch["features"].dim() == 3, \
            f"Expected 3D features, got {batch['features'].dim()}D"
        assert batch["features"].shape[0] == 2
        assert batch["features"].shape[2] == 5

        # labels: [batch] long
        assert batch["labels"].shape == (2,), \
            f"Expected labels [2], got {batch['labels'].shape}"
        assert batch["labels"].dtype == torch.long, \
            f"Expected labels dtype long, got {batch['labels'].dtype}"

        # All candidates should have at least some real tokens (not all zeros)
        assert torch.any(batch["attention_mask"] > 0), \
            "At least some candidates should have non-zero attention masks"

    def test_dev_only_mode(self, tokenizer):
        """Test dev-only mode returns None for train_loader"""
        class MockArgs:
            mode = "dev-only"
            data_dir = "data"
            model_name = "bert-base-uncased"
            max_length = 128
            max_dist = 50
            batch_size = 2
            learning_rate = 5e-5
            epochs = 1
            warmup_ratio = 0.1
            dropout = 0.1
            freeze_bert = False
            patience = 3
            output_dir = "checkpoints"
            save_every = 1
            eval_every = 1
            device = "cpu"
            num_workers = 0
            fp16 = False
            test_start = 1000
            test_end = 1300
            resume_from = None

        train_loader, dev_loader = create_dataloaders(MockArgs(), tokenizer)

        assert train_loader is None, "train_loader should be None in dev-only mode"
        assert dev_loader is not None, "dev_loader should not be None"
        assert len(dev_loader.dataset) > 0, "Dev dataset should have samples"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])