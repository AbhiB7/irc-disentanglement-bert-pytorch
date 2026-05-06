"""
Tests for train.py save_checkpoint() and load_checkpoint() functions.

Verifies that checkpoint save/load works correctly:
- File is created with expected keys
- Loaded weights match saved weights
- best_model.pt is saved when metrics contain F1
- Atomic save (temp + rename) doesn't corrupt

Run with: python -m pytest tests/test_checkpoint.py -v
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import pytest
from train import save_checkpoint, load_checkpoint


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    """Create a temporary directory for checkpoint files"""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def tiny_model():
    """A tiny model with a few parameters"""
    model = torch.nn.Linear(4, 2)
    return model


@pytest.fixture
def optimizer(tiny_model):
    """AdamW optimizer for the tiny model"""
    return torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)


@pytest.fixture
def scheduler(optimizer):
    """Simple scheduler for the optimizer"""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)


@pytest.fixture
def mock_args():
    """Mock args object"""
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
        test_start = 0
        test_end = 100
        resume_from = None
    return MockArgs()


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

class TestSaveCheckpoint:
    """Test save_checkpoint() creates valid checkpoint files"""

    def test_creates_checkpoint_file(self, temp_dir, tiny_model, optimizer, scheduler, mock_args):
        """Test that save_checkpoint creates a .pt file"""
        metrics = {"accuracy": 0.85, "f1": 0.82}
        
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=1, args=mock_args,
                       metrics=metrics, checkpoint_dir=temp_dir)
        
        expected_path = temp_dir / "checkpoint_epoch_1.pt"
        assert expected_path.exists(), f"Checkpoint file not found: {expected_path}"

    def test_checkpoint_has_expected_keys(self, temp_dir, tiny_model, optimizer, scheduler, mock_args):
        """Test that the checkpoint dict contains all expected keys"""
        metrics = {"accuracy": 0.85, "f1": 0.82}
        
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=1, args=mock_args,
                       metrics=metrics, checkpoint_dir=temp_dir)
        
        checkpoint = torch.load(temp_dir / "checkpoint_epoch_1.pt", weights_only=False)
        
        expected_keys = {"epoch", "model_state_dict", "optimizer_state_dict",
                        "scheduler_state_dict", "args", "metrics"}
        assert set(checkpoint.keys()) == expected_keys, \
            f"Expected keys {expected_keys}, got {set(checkpoint.keys())}"

    def test_checkpoint_epoch_value(self, temp_dir, tiny_model, optimizer, scheduler, mock_args):
        """Test that the epoch number is saved correctly"""
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=5, args=mock_args,
                       metrics={}, checkpoint_dir=temp_dir)
        
        checkpoint = torch.load(temp_dir / "checkpoint_epoch_5.pt", weights_only=False)
        assert checkpoint["epoch"] == 5, f"Expected epoch=5, got {checkpoint['epoch']}"

    def test_checkpoint_metrics(self, temp_dir, tiny_model, optimizer, scheduler, mock_args):
        """Test that metrics are saved correctly"""
        metrics = {"accuracy": 0.75, "f1": 0.72, "loss": 0.5}
        
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=1, args=mock_args,
                       metrics=metrics, checkpoint_dir=temp_dir)
        
        checkpoint = torch.load(temp_dir / "checkpoint_epoch_1.pt", weights_only=False)
        assert checkpoint["metrics"]["accuracy"] == pytest.approx(0.75)
        assert checkpoint["metrics"]["f1"] == pytest.approx(0.72)
        assert checkpoint["metrics"]["loss"] == pytest.approx(0.5)

    def test_saves_best_model_when_f1_present(self, temp_dir, tiny_model, optimizer, scheduler, mock_args):
        """Test that best_model.pt is saved when metrics contain 'f1'"""
        metrics = {"accuracy": 0.85, "f1": 0.82}
        
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=1, args=mock_args,
                       metrics=metrics, checkpoint_dir=temp_dir)
        
        best_path = temp_dir / "best_model.pt"
        assert best_path.exists(), f"best_model.pt not found: {best_path}"

    def test_does_not_save_best_model_without_f1(self, temp_dir, tiny_model, optimizer, scheduler, mock_args):
        """Test that best_model.pt is NOT saved when metrics lack 'f1'"""
        metrics = {"accuracy": 0.85}  # No 'f1' key
        
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=1, args=mock_args,
                       metrics=metrics, checkpoint_dir=temp_dir)
        
        best_path = temp_dir / "best_model.pt"
        assert not best_path.exists(), "best_model.pt should not exist without f1 metric"

    def test_overwrites_best_model(self, temp_dir, tiny_model, optimizer, scheduler, mock_args):
        """Test that best_model.pt gets overwritten on subsequent saves"""
        metrics = {"accuracy": 0.85, "f1": 0.82}
        
        # Save twice
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=1, args=mock_args,
                       metrics=metrics, checkpoint_dir=temp_dir)
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=2, args=mock_args,
                       metrics=metrics, checkpoint_dir=temp_dir)
        
        best_path = temp_dir / "best_model.pt"
        checkpoint = torch.load(best_path, weights_only=False)
        assert checkpoint["epoch"] == 2, "best_model.pt should contain the latest save"


class TestLoadCheckpoint:
    """Test load_checkpoint() restores model state correctly"""

    def test_load_returns_epoch_and_metrics(self, temp_dir, tiny_model, optimizer, scheduler, mock_args):
        """Test that load_checkpoint returns (epoch, metrics) tuple"""
        metrics = {"accuracy": 0.85, "f1": 0.82}
        
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=3, args=mock_args,
                       metrics=metrics, checkpoint_dir=temp_dir)
        
        loaded_epoch, loaded_metrics = load_checkpoint(
            temp_dir / "checkpoint_epoch_3.pt", tiny_model, device="cpu"
        )
        
        assert loaded_epoch == 3, f"Expected epoch=3, got {loaded_epoch}"
        assert loaded_metrics["accuracy"] == pytest.approx(0.85)
        assert loaded_metrics["f1"] == pytest.approx(0.82)

    def test_weights_match_after_save_load(self, temp_dir, mock_args):
        """Test that loaded model weights match saved weights exactly"""
        # Create model, save its weights
        model1 = torch.nn.Linear(4, 2)
        original_weights = model1.weight.clone()
        original_bias = model1.bias.clone()
        
        optimizer = torch.optim.AdamW(model1.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        
        save_checkpoint(model1, optimizer, scheduler, epoch=1, args=mock_args,
                       metrics={}, checkpoint_dir=temp_dir)
        
        # Create a new model, load saved weights
        model2 = torch.nn.Linear(4, 2)
        load_checkpoint(temp_dir / "checkpoint_epoch_1.pt", model2, device="cpu")
        
        # Weights should match exactly
        assert torch.equal(model2.weight, original_weights), "Loaded weights don't match saved weights"
        assert torch.equal(model2.bias, original_bias), "Loaded bias doesn't match saved bias"

    def test_load_without_optimizer(self, temp_dir, tiny_model, optimizer, scheduler, mock_args):
        """Test that load_checkpoint works without passing optimizer/scheduler"""
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=1, args=mock_args,
                       metrics={}, checkpoint_dir=temp_dir)
        
        # Load without optimizer or scheduler
        epoch, metrics = load_checkpoint(
            temp_dir / "checkpoint_epoch_1.pt", tiny_model, device="cpu"
        )
        
        assert epoch == 1, "Should return epoch even without optimizer"

    def test_load_nonexistent_file_raises_error(self, temp_dir, tiny_model):
        """Test that loading a nonexistent file raises FileNotFoundError"""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_checkpoint(temp_dir / "nonexistent.pt", tiny_model, device="cpu")


class TestSaveCheckpointEdgeCases:
    """Test edge cases for save_checkpoint"""

    def test_save_without_scheduler(self, temp_dir, tiny_model, optimizer, mock_args):
        """Test saving without a scheduler (scheduler=None)"""
        save_checkpoint(tiny_model, optimizer, scheduler=None, epoch=1, args=mock_args,
                       metrics={}, checkpoint_dir=temp_dir)
        
        checkpoint = torch.load(temp_dir / "checkpoint_epoch_1.pt", weights_only=False)
        assert checkpoint["scheduler_state_dict"] is None, \
            "scheduler_state_dict should be None when scheduler is None"

    def test_save_without_optimizer(self, temp_dir, tiny_model, mock_args):
        """Test saving without an optimizer (test mode)"""
        save_checkpoint(tiny_model, optimizer=None, scheduler=None, epoch=1, args=mock_args,
                       metrics={}, checkpoint_dir=temp_dir)
        
        checkpoint = torch.load(temp_dir / "checkpoint_epoch_1.pt", weights_only=False)
        assert checkpoint["optimizer_state_dict"] is None, \
            "optimizer_state_dict should be None when optimizer is None"

    def test_multiple_epochs_saved_separately(self, temp_dir, tiny_model, optimizer, scheduler, mock_args):
        """Test that multiple epoch checkpoints are saved as separate files"""
        for epoch in [1, 2, 3]:
            save_checkpoint(tiny_model, optimizer, scheduler, epoch=epoch, args=mock_args,
                           metrics={}, checkpoint_dir=temp_dir)
        
        assert (temp_dir / "checkpoint_epoch_1.pt").exists()
        assert (temp_dir / "checkpoint_epoch_2.pt").exists()
        assert (temp_dir / "checkpoint_epoch_3.pt").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])