"""
Tests for train.py evaluate() function.
Verifies that multiclass metrics (accuracy, precision, recall, F1) are computed correctly.

Strategy: Create a mock model that returns fixed logits/probs, feed it fake batch data
with known labels, and verify the returned metrics match expected values.

Run with: python -m pytest tests/test_evaluate.py -v
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import pytest
from train import evaluate, collate_fn
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────
# Mock Model — returns fixed predictions
# ─────────────────────────────────────────────

class MockModel(torch.nn.Module):
    """
    A fake model that returns known logits instead of running BERT.
    Used to test evaluate() metric calculations deterministically.
    
    Args:
        logits: Fixed logits to return each forward call [batch_size, C]
    """
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("_fixed_logits", logits)
        self._call_count = 0
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, features=None, labels=None):
        """Return fixed logits/probs regardless of input"""
        self._call_count += 1
        
        # Use the fixed logits (trim to match batch size if needed)
        batch_size = input_ids.shape[0]
        logits = self._fixed_logits[:batch_size]
        
        probs = torch.softmax(logits, dim=-1)
        
        outputs = {
            'logits': logits,
            'probs': probs,
        }
        
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs['loss'] = loss
        
        return outputs
    
    def to(self, device):
        self._fixed_logits = self._fixed_logits.to(device)
        return super().to(device)


# ─────────────────────────────────────────────
# Fake Dataset — yields known batch items
# ─────────────────────────────────────────────

class FakeEvaluationDataset(Dataset):
    """
    A dataset that produces fake batch items with known labels.
    Each item has input_ids/attention_mask/features at fixed sizes.
    """
    def __init__(self, num_samples: int, num_candidates: int, seq_len: int = 4, num_features: int = 5):
        self.num_samples = num_samples
        self.num_candidates = num_candidates
        self.seq_len = seq_len
        self.num_features = num_features
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.zeros(self.num_candidates, self.seq_len, dtype=torch.long),
            "attention_mask": torch.ones(self.num_candidates, self.seq_len, dtype=torch.long),
            "features": torch.randn(self.num_candidates, self.num_features),
            "labels": torch.tensor(0, dtype=torch.long),  # label will be overridden per test
        }


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

class TestEvaluateReturnKeys:
    """Test that evaluate() returns the expected dictionary keys"""

    def test_returns_all_keys(self):
        """Test evaluate() returns loss, accuracy, precision, recall, f1"""
        # 2 samples, 3 candidates, fixed logits where class 0 wins
        logits = torch.tensor([
            [2.0, 0.0, -1.0],  # sample 0: predicts class 0
            [1.0, 0.5, 0.0],   # sample 1: predicts class 0
        ])
        model = MockModel(logits)
        
        # Create dataset with 2 samples, all labeled 0
        dataset = FakeEvaluationDataset(num_samples=2, num_candidates=3)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        # Override labels after collation to known values
        # We need to patch the dataloader's output
        metrics = evaluate(model, dataloader, device="cpu")
        
        expected_keys = {"loss", "accuracy", "precision", "recall", "f1", "predictions", "labels", "probs"}
        assert set(metrics.keys()) == expected_keys, \
            f"Expected keys {expected_keys}, got {set(metrics.keys())}"
    
    def test_all_correct_perfect_score(self):
        """Test all predictions correct => accuracy=1.0, precision=1.0, recall=1.0, f1=1.0"""
        # 4 samples, 2 candidates. Labels = [0, 1, 0, 1]
        # Logits set so predictions match labels perfectly
        logits = torch.tensor([
            [5.0, -5.0],  # predicts 0, label=0 ✓
            [-5.0, 5.0],  # predicts 1, label=1 ✓
            [5.0, -5.0],  # predicts 0, label=0 ✓
            [-5.0, 5.0],  # predicts 1, label=1 ✓
        ])
        model = MockModel(logits)
        
        # Create 4 samples with 2 candidates each, labeled [0,1,0,1]
        dataset = FakeEvaluationDataset(num_samples=4, num_candidates=2)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        # Override: create custom dataset with known labels
        # Use a custom collate-like approach
        metrics = _run_evaluate_with_labels(model, logits, labels=torch.tensor([0, 1, 0, 1]))
        
        assert metrics["accuracy"] == pytest.approx(1.0), f"Expected accuracy=1.0, got {metrics['accuracy']}"
        assert metrics["precision"] == pytest.approx(1.0), f"Expected precision=1.0, got {metrics['precision']}"
        assert metrics["recall"] == pytest.approx(1.0), f"Expected recall=1.0, got {metrics['recall']}"
        assert metrics["f1"] == pytest.approx(1.0), f"Expected f1=1.0, got {metrics['f1']}"
        
    def test_all_wrong_zero_score(self):
        """Test all predictions wrong => accuracy=0.0, precision=0.0, recall=0.0"""
        # 2 samples, 2 candidates. Labels = [0, 0], predictions = [1, 1]
        logits = torch.tensor([
            [-5.0, 5.0],  # predicts 1, label=0 ✗
            [-5.0, 5.0],  # predicts 1, label=0 ✗
        ])
        model = MockModel(logits)
        
        metrics = _run_evaluate_with_labels(model, logits, labels=torch.tensor([0, 0]))
        
        assert metrics["accuracy"] == pytest.approx(0.0), f"Expected accuracy=0.0, got {metrics['accuracy']}"
        assert metrics["precision"] == pytest.approx(0.0), f"Expected precision=0.0, got {metrics['precision']}"
        assert metrics["recall"] == pytest.approx(0.0), f"Expected recall=0.0, got {metrics['recall']}"
        assert metrics["f1"] == pytest.approx(0.0), f"Expected f1=0.0, got {metrics['f1']}"


class TestEvaluateMetricValues:
    """Test specific metric values for known prediction/label combinations"""

    def test_accuracy_half_correct(self):
        """Test accuracy=0.5 when half the predictions are correct"""
        # 4 samples, 2 candidates. Labels = [0, 1, 0, 1]
        # Predictions = [0, 0, 1, 1] → 2 correct, 2 wrong
        logits = torch.tensor([
            [5.0, -5.0],  # predicts 0, label=0 ✓
            [5.0, -5.0],  # predicts 0, label=1 ✗
            [-5.0, 5.0],  # predicts 1, label=0 ✗
            [-5.0, 5.0],  # predicts 1, label=1 ✓
        ])
        model = MockModel(logits)
        
        metrics = _run_evaluate_with_labels(model, logits, labels=torch.tensor([0, 1, 0, 1]))
        
        assert metrics["accuracy"] == pytest.approx(0.5), f"Expected accuracy=0.5, got {metrics['accuracy']}"
    
    def test_precision_recall_f1_mixed(self):
        """
        Test macro P/R/F1 for a specific known case.
        
        4 samples, 3 candidates. Labels = [0, 1, 2, 0], Predictions = [0, 0, 0, 0]
        
        class 0: TP=2(allpred0&label0), FP=2(pred0butlabel1or2), FN=0 → P=2/4=0.5, R=2/2=1.0, F1=0.667
        class 1: TP=0, FP=0, FN=1 → P=0.0, R=0.0, F1=0.0
        class 2: TP=0, FP=0, FN=1 → P=0.0, R=0.0, F1=0.0
        
        Macro: P=(0.5+0+0)/3=0.1667, R=(1.0+0+0)/3=0.3333, F1=(0.667+0+0)/3=0.2222
        """
        # 4 samples, 3 candidates
        logits = torch.tensor([
            [5.0, -5.0, -5.0],  # predicts 0
            [5.0, -5.0, -5.0],  # predicts 0
            [5.0, -5.0, -5.0],  # predicts 0
            [5.0, -5.0, -5.0],  # predicts 0
        ])
        model = MockModel(logits)
        
        metrics = _run_evaluate_with_labels(model, logits, labels=torch.tensor([0, 1, 2, 0]))
        
        assert metrics["accuracy"] == pytest.approx(0.5), f"Expected accuracy=0.5, got {metrics['accuracy']}"
        assert metrics["precision"] == pytest.approx(0.1667, abs=1e-4), f"Expected precision≈0.1667, got {metrics['precision']}"
        assert metrics["recall"] == pytest.approx(0.3333, abs=1e-4), f"Expected recall≈0.3333, got {metrics['recall']}"
        assert metrics["f1"] == pytest.approx(0.2222, abs=1e-4), f"Expected f1≈0.2222, got {metrics['f1']}"
    
    def test_single_sample(self):
        """Test evaluate works with a single sample (batch_size=1)"""
        logits = torch.tensor([
            [5.0, -5.0],  # predicts 0, label=0 ✓
        ])
        model = MockModel(logits)
        
        metrics = _run_evaluate_with_labels(model, logits, labels=torch.tensor([0]))
        
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
    
    def test_single_candidate(self):
        """Test evaluate with only 1 candidate (C=1) — trivial case"""
        logits = torch.tensor([
            [0.0],  # only 1 candidate → always predicts class 0
            [0.0],
            [0.0],
        ])
        model = MockModel(logits)
        
        metrics = _run_evaluate_with_labels(model, logits, labels=torch.tensor([0, 0, 0]))
        
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
    
    def test_loss_is_finite_and_non_negative(self):
        """Test that the returned loss is a finite, non-negative number"""
        logits = torch.tensor([
            [2.0, 0.0, -1.0],
            [1.0, 0.5, 0.0],
        ])
        model = MockModel(logits)
        
        metrics = _run_evaluate_with_labels(model, logits, labels=torch.tensor([0, 1]))
        
        assert torch.isfinite(torch.tensor(metrics["loss"])), f"Loss is not finite: {metrics['loss']}"
        assert metrics["loss"] >= 0.0, f"Loss is negative: {metrics['loss']}"
    
    def test_many_candidates(self):
        """Test evaluate with many candidates (C=10)"""
        # 5 samples, 10 candidates, all predict correctly (label=0 for all)
        logits = torch.zeros(5, 10)
        logits[:, 0] = 5.0  # class 0 wins for all
        
        model = MockModel(logits)
        
        metrics = _run_evaluate_with_labels(model, logits, labels=torch.tensor([0, 0, 0, 0, 0]))
        
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["recall"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["f1"] == pytest.approx(1.0, abs=1e-4)


class TestEvaluateLoss:
    """Test loss computation via evaluate()"""

    def test_better_predictions_have_lower_loss(self):
        """
        CrossEntropyLoss should be lower when predictions match labels.
        Good case: logits favor correct label
        Bad case:  logits disfavor correct label
        """
        # Good predictions: correct class gets high logit
        good_logits = torch.tensor([
            [5.0, 0.0, -5.0],  # predicts 0, label=0
        ])
        good_model = MockModel(good_logits)
        good_metrics = _run_evaluate_with_labels(good_model, good_logits, labels=torch.tensor([0]))
        
        # Bad predictions: wrong class gets high logit
        bad_logits = torch.tensor([
            [-5.0, 0.0, 5.0],  # predicts 2, label=0
        ])
        bad_model = MockModel(bad_logits)
        bad_metrics = _run_evaluate_with_labels(bad_model, bad_logits, labels=torch.tensor([0]))
        
        assert good_metrics["loss"] < bad_metrics["loss"], \
            f"Good loss ({good_metrics['loss']}) should be < bad loss ({bad_metrics['loss']})"


class TestEvaluateEdgeCases:
    """Test edge cases and robustness"""

    def test_empty_predictions(self):
        """Test evaluate handles empty dataloader gracefully"""
        # Empty dataset, should not crash
        dataset = FakeEvaluationDataset(num_samples=0, num_candidates=3)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        # Dummy model (won't be called)
        logits = torch.tensor([[5.0, 0.0, -5.0]])
        model = MockModel(logits)
        
        metrics = evaluate(model, dataloader, device="cpu")
        
        # With no data, loss should be 0, accuracy 0
        assert metrics["loss"] == pytest.approx(0.0)
        assert metrics["accuracy"] == pytest.approx(0.0)
        assert metrics["precision"] == pytest.approx(0.0)
        assert metrics["recall"] == pytest.approx(0.0)
        assert metrics["f1"] == pytest.approx(0.0)
    
    def test_all_samples_same_class(self):
        """Test when all samples have the same label and same prediction
        
        num_classes = max(0, 0) + 1 = 1 (only 1 class detected)
        Class 0: TP=3, FP=0, FN=0 → P=1.0, R=1.0, F1=1.0
        Macro: P=R=F1=1.0/1=1.0
        """
        # 3 samples, 4 candidates, all labeled 0, all predict 0
        logits = torch.tensor([
            [5.0, -5.0, -5.0, -5.0],
            [5.0, -5.0, -5.0, -5.0],
            [5.0, -5.0, -5.0, -5.0],
        ])
        model = MockModel(logits)
        
        metrics = _run_evaluate_with_labels(model, logits, labels=torch.tensor([0, 0, 0]))
        
        # Only 1 class detected (max label/pred = 0), so macro = per-class
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)


# ─────────────────────────────────────────────
# Helper: run evaluate with known labels
# ─────────────────────────────────────────────

def _run_evaluate_with_labels(model, logits, labels):
    """
    Run evaluate() by bypassing the dataset and feeding batch items directly.
    
    This creates fake batch items that match the model's expectation,
    then manually patches the dataloader to return labelled data.
    """
    batch_size = logits.shape[0]
    num_candidates = logits.shape[1]
    seq_len = 4
    num_features = 5
    
    # Create a single batch with known labels
    batch = []
    for i in range(batch_size):
        batch.append({
            "input_ids": torch.zeros(num_candidates, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(num_candidates, seq_len, dtype=torch.long),
            "features": torch.zeros(num_candidates, num_features, dtype=torch.float32),
            "labels": labels[i].clone().detach(),
        })
    
    collated = collate_fn(batch)
    
    # Create a fake dataset that returns our collated batch
    class FakeLoader:
        def __init__(self, collated_batch, dataset_len):
            self.dataset = _FakeDataset(dataset_len)
            self._batch = collated_batch
            self._iterated = False
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self._iterated:
                raise StopIteration
            self._iterated = True
            return self._batch
    
    return evaluate(model, FakeLoader(collated, len(labels)), device="cpu")


class _FakeDataset:
    def __init__(self, length):
        self._length = length
    def __len__(self):
        return self._length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])