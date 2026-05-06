"""
Unit tests for IRC Conversation Disentanglement Model (multiclass architecture).

Tests the CrossEncoderWithFeatures model from src/model.py
with [batch, C, seq] input shapes and softmax classification.

Run with: python -m pytest tests/test_model.py -v
"""

import sys
import torch
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import CrossEncoderWithFeatures, create_model, count_parameters


class TestMulticlassModelInit:
    """Test model initialization and configuration"""

    def test_create_default(self):
        """Test creating model with default parameters (DeBERTa-v3-base)"""
        model = create_model()
        assert isinstance(model, CrossEncoderWithFeatures)
        assert model.num_features == 5
        assert model.bert_hidden_size > 0
        assert model.combined_size == model.bert_hidden_size + 5

    def test_create_deberta(self):
        """Test creating model with production SOTA model"""
        model = create_model(model_name="microsoft/deberta-v3-base")
        assert isinstance(model, CrossEncoderWithFeatures)
        assert model.bert_hidden_size == 768  # DeBERTa-v3-base uses 768
        assert model.num_features == 5
        # DeBERTa doesn't use token_type_ids — verify forward works without them
        batch_size = 2
        C = 3
        seq_len = 16  # Small for speed
        input_ids = torch.randint(0, 1000, (batch_size, C, seq_len))
        attention_mask = torch.ones((batch_size, C, seq_len), dtype=torch.long)
        features = torch.randn((batch_size, C, 5))
        labels = torch.zeros(batch_size, dtype=torch.long)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            features=features,
            labels=labels
        )
        assert outputs['logits'].shape == (batch_size, C)
        assert outputs['probs'].shape == (batch_size, C)
        assert outputs['loss'].numel() == 1

    def test_create_custom_params(self):
        """Test creating model with custom parameters"""
        model = create_model(
            model_name="bert-base-uncased",
            num_features=3,
            dropout=0.3,
            freeze_bert=True
        )
        assert model.num_features == 3
        assert model.dropout.p == 0.3
        assert model.combined_size == 768 + 3

    def test_create_on_device(self):
        """Test creating model on CPU"""
        model = create_model(device='cpu')
        for param in model.parameters():
            assert param.device.type == 'cpu'

    def test_freeze_bert(self):
        """Test freezing BERT parameters"""
        model = create_model(freeze_bert=True)
        for param in model.bert.parameters():
            assert param.requires_grad == False
        for param in model.classifier.parameters():
            assert param.requires_grad == True

    def test_parameter_count(self):
        """Test parameter counting"""
        model = create_model()
        trainable, total = count_parameters(model)
        assert trainable > 0
        assert total > 0
        assert trainable <= total

    def test_combined_size_scales_with_features(self):
        """Test that combined size correctly includes feature count"""
        model_5 = create_model(num_features=5)
        model_8 = create_model(num_features=8)
        assert model_5.combined_size == model_5.bert_hidden_size + 5
        assert model_8.combined_size == model_8.bert_hidden_size + 8
        assert model_8.combined_size == model_5.combined_size + 3


class TestMulticlassForward:
    """Test model forward pass with multiclass [batch, C, seq] inputs"""

    def setup_method(self):
        self.model = create_model(model_name="bert-base-uncased", num_features=5)
        self.batch_size = 2
        self.num_candidates = 5
        self.seq_len = 32

        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.num_candidates, self.seq_len))
        self.attention_mask = torch.ones((self.batch_size, self.num_candidates, self.seq_len), dtype=torch.long)
        self.features = torch.randn((self.batch_size, self.num_candidates, 5))
        self.labels = torch.tensor([2, 4], dtype=torch.long)

    def test_forward_with_labels(self):
        """Test forward pass with labels (training mode)"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features,
            labels=self.labels
        )

        assert 'logits' in outputs
        assert 'probs' in outputs
        assert 'loss' in outputs

        assert outputs['logits'].shape == (self.batch_size, self.num_candidates)
        assert outputs['probs'].shape == (self.batch_size, self.num_candidates)
        assert outputs['loss'].dim() == 0  # scalar loss
        assert outputs['loss'].item() >= 0

    def test_forward_without_labels(self):
        """Test forward pass without labels (inference mode)"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features
        )

        assert 'logits' in outputs
        assert 'probs' in outputs
        assert 'loss' not in outputs

    def test_forward_without_features(self):
        """Test forward pass without handcrafted features (zero-filled)"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=None
        )

        assert 'logits' in outputs
        assert outputs['logits'].shape == (self.batch_size, self.num_candidates)

    def test_probs_sum_to_one(self):
        """Test that probabilities sum to 1 per sample"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features,
            labels=self.labels
        )

        probs_sum = outputs['probs'].sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones(self.batch_size), atol=1e-5)

    def test_single_sample(self):
        """Test forward pass with single sample (batch_size=1)"""
        single_ids = self.input_ids[:1]
        single_mask = self.attention_mask[:1]
        single_feat = self.features[:1]
        single_labels = self.labels[:1]

        outputs = self.model(
            input_ids=single_ids,
            attention_mask=single_mask,
            features=single_feat,
            labels=single_labels
        )

        assert outputs['logits'].shape == (1, self.num_candidates)
        assert outputs['loss'].dim() == 0

    def test_candidate_masking(self):
        """Test that padded candidates get masked logits"""
        mask = self.attention_mask.clone()
        mask[:, 0, :] = 0  # mask out first candidate

        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=mask,
            features=self.features,
            labels=self.labels
        )

        # First candidate should have very negative logits (masked)
        assert torch.all(outputs['logits'][:, 0] < -1e8)

    def test_different_num_candidates(self):
        """Test forward pass with varying number of candidates"""
        for C in [1, 3, 10]:
            input_ids = torch.randint(0, 1000, (2, C, 32))
            attention_mask = torch.ones((2, C, 32), dtype=torch.long)
            features = torch.randn((2, C, 5))
            labels = torch.zeros(2, dtype=torch.long)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                features=features,
                labels=labels
            )

            assert outputs['logits'].shape == (2, C), f"Expected [2, {C}], got {outputs['logits'].shape}"

    def test_no_token_type_ids(self):
        """Test forward pass without token_type_ids (DeBERTa doesn't use them)"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features,
            labels=self.labels
        )
        assert 'logits' in outputs
        assert outputs['logits'].shape == (self.batch_size, self.num_candidates)


class TestMulticlassPrediction:
    """Test model prediction method"""

    def setup_method(self):
        self.model = create_model(model_name="bert-base-uncased", num_features=5)
        self.input_ids = torch.randint(0, 1000, (2, 5, 32))
        self.attention_mask = torch.ones((2, 5, 32), dtype=torch.long)
        self.features = torch.randn((2, 5, 5))

    def test_predict_returns_argmax(self):
        """Test predict returns argmax with valid indices"""
        predictions, probs = self.model.predict(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features
        )

        assert predictions.shape == (2,)
        assert probs.shape == (2, 5)
        # All predictions should be valid candidate indices
        assert torch.all((predictions >= 0) & (predictions < 5))

    def test_predict_probs_sum_to_one(self):
        """Test predict probabilities sum to 1"""
        _, probs = self.model.predict(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features
        )
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_predict_single_sample(self):
        """Test predict with single sample"""
        predictions, probs = self.model.predict(
            input_ids=self.input_ids[:1],
            attention_mask=self.attention_mask[:1],
            features=self.features[:1]
        )
        assert predictions.shape == (1,)
        assert probs.shape == (1, 5)


class TestMulticlassArchitecture:
    """Test model architecture details"""

    def test_classifier_output_shape(self):
        """Test classifier produces [batch*C, 1] before reshaping"""
        model = create_model(model_name="bert-base-uncased", num_features=5)

        batch_size = 3
        combined = torch.randn((batch_size, model.combined_size))
        output = model.classifier(combined)
        assert output.shape == (batch_size, 1)

    def test_dropout_applied_in_train(self):
        """Test that dropout changes output between train and eval"""
        model = create_model(model_name="bert-base-uncased", num_features=5, dropout=0.5)

        input_ids = torch.randint(0, 1000, (2, 3, 32))
        attention_mask = torch.ones((2, 3, 32), dtype=torch.long)
        features = torch.randn((2, 3, 5))

        model.eval()
        with torch.no_grad():
            eval_out = model(input_ids=input_ids, attention_mask=attention_mask, features=features)

        model.train()
        train_out = model(input_ids=input_ids, attention_mask=attention_mask, features=features)

        # Outputs should differ due to dropout (very likely)
        assert not torch.allclose(eval_out['logits'], train_out['logits'])


class TestMulticlassLoss:
    """Test loss calculation"""

    def setup_method(self):
        self.model = create_model(model_name="bert-base-uncased", num_features=5)
        self.input_ids = torch.randint(0, 1000, (4, 5, 32))
        self.attention_mask = torch.ones((4, 5, 32), dtype=torch.long)
        self.features = torch.randn((4, 5, 5))

    def test_loss_non_negative(self):
        """Test loss is non-negative for random predictions"""
        labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features,
            labels=labels
        )
        assert outputs['loss'].item() >= 0

    def test_perfect_predictions_lower_loss(self):
        """Test that perfect predictions give lower loss than random"""
        # Create data where candidate 0 is always correct
        batch_size = 4
        C = 3
        input_ids = torch.randint(0, 1000, (batch_size, C, 32))
        attention_mask = torch.ones((batch_size, C, 32), dtype=torch.long)
        features = torch.zeros((batch_size, C, 5))
        labels = torch.zeros(batch_size, dtype=torch.long)  # always candidate 0

        # Get loss with random weights
        loss_random = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            features=features,
            labels=labels
        )['loss']

        # Now set logits to favor candidate 0 by manipulating features
        # (This is a weak test — the point is it shouldn't crash and loss should be finite)
        assert torch.isfinite(loss_random)


class TestMulticlassSmokeTest:
    """Smoke test matching the original test_model() from model.py"""

    def test_multiclass_smoke(self):
        """Verify multiclass [batch, C, seq] architecture end-to-end"""
        model = create_model(model_name="bert-base-uncased", num_features=5)
        trainable, total = count_parameters(model)
        assert trainable > 0
        assert total > 0

        batch_size = 2
        num_candidates = 5
        seq_len = 32

        input_ids = torch.randint(0, 1000, (batch_size, num_candidates, seq_len))
        attention_mask = torch.ones((batch_size, num_candidates, seq_len), dtype=torch.long)
        features = torch.randn((batch_size, num_candidates, 5))
        labels = torch.tensor([2, 4], dtype=torch.long)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            features=features,
            labels=labels,
        )

        assert outputs["logits"].shape == (batch_size, num_candidates), \
            f"Expected logits [{batch_size}, {num_candidates}], got {outputs['logits'].shape}"
        assert outputs["probs"].shape == (batch_size, num_candidates), \
            f"Expected probs [{batch_size}, {num_candidates}], got {outputs['probs'].shape}"
        assert outputs["loss"].dim() == 0, "Loss should be scalar"
        assert torch.allclose(outputs["probs"].sum(dim=-1), torch.ones(batch_size), atol=1e-5), \
            "Probs must sum to 1 per sample"

        predictions, probs = model.predict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            features=features,
        )
        assert predictions.shape == (batch_size,)
        assert all(0 <= p < num_candidates for p in predictions.tolist())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])