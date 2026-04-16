"""
Unit tests for IRC Conversation Disentanglement Model

Tests the CrossEncoderWithFeatures model from src/model.py
"""

import sys
import torch
import pytest
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import CrossEncoderWithFeatures, create_model, count_parameters


class TestModelInitialization:
    """Test model initialization and configuration"""
    
    def test_model_creation_default(self):
        """Test creating model with default parameters"""
        model = create_model()
        
        assert model is not None
        assert isinstance(model, CrossEncoderWithFeatures)
        assert model.bert_hidden_size == 768  # BERT base hidden size
        assert model.num_features == 4
        assert model.combined_size == 772  # 768 + 4
    
    def test_model_creation_custom_params(self):
        """Test creating model with custom parameters"""
        model = create_model(
            model_name="bert-base-uncased",
            num_features=4,
            dropout=0.2,
            freeze_bert=False
        )
        
        assert model.dropout.p == 0.2
        assert model.num_features == 4
    
    def test_model_parameter_count(self):
        """Test parameter counting"""
        model = create_model()
        trainable, total = count_parameters(model)
        
        assert trainable > 0
        assert total > 0
        assert trainable <= total
        
        # BERT base has ~110M parameters
        assert total > 100_000_000
    
    def test_model_freeze_bert(self):
        """Test freezing BERT parameters"""
        model = create_model(freeze_bert=True)
        
        # Check that BERT parameters are frozen
        for param in model.bert.parameters():
            assert param.requires_grad == False
        
        # Check that classifier parameters are not frozen
        for param in model.classifier.parameters():
            assert param.requires_grad == True


class TestModelForward:
    """Test model forward pass"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = create_model()
        self.batch_size = 2
        self.seq_len = 128
        
        # Create dummy inputs
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones((self.batch_size, self.seq_len))
        self.token_type_ids = torch.zeros((self.batch_size, self.seq_len), dtype=torch.long)
        self.features = torch.randn((self.batch_size, 4))
        self.labels = torch.tensor([1.0, 0.0])
    
    def test_forward_pass_with_labels(self):
        """Test forward pass with labels (training mode)"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids,
            features=self.features,
            labels=self.labels
        )
        
        # Check output structure
        assert 'logits' in outputs
        assert 'probs' in outputs
        assert 'loss' in outputs
        
        # Check output shapes
        assert outputs['logits'].shape == (self.batch_size,)
        assert outputs['probs'].shape == (self.batch_size,)
        
        # Check loss is a scalar
        assert outputs['loss'].dim() == 0
        
        # Check probabilities are in [0, 1]
        assert torch.all(outputs['probs'] >= 0)
        assert torch.all(outputs['probs'] <= 1)
    
    def test_forward_pass_without_labels(self):
        """Test forward pass without labels (inference mode)"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids,
            features=self.features
        )
        
        # Check output structure
        assert 'logits' in outputs
        assert 'probs' in outputs
        assert 'loss' not in outputs
        
        # Check output shapes
        assert outputs['logits'].shape == (self.batch_size,)
        assert outputs['probs'].shape == (self.batch_size,)
    
    def test_forward_pass_without_features(self):
        """Test forward pass without handcrafted features"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids,
            features=None
        )
        
        # Should still work with zero-padded features
        assert 'logits' in outputs
        assert 'probs' in outputs
        assert outputs['logits'].shape == (self.batch_size,)
    
    def test_forward_pass_without_token_type_ids(self):
        """Test forward pass without token_type_ids (some tokenizers don't use them)"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features
        )
        
        assert 'logits' in outputs
        assert 'probs' in outputs
        assert outputs['logits'].shape == (self.batch_size,)
    
    def test_feature_dimension_mismatch(self):
        """Test that wrong feature dimension raises error"""
        wrong_features = torch.randn((self.batch_size, 3))  # Should be 4
        
        with pytest.raises(ValueError, match="Expected 4 features"):
            self.model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                features=wrong_features
            )
    
    def test_single_sample(self):
        """Test forward pass with single sample (batch_size=1)"""
        single_input = self.input_ids[:1]
        single_mask = self.attention_mask[:1]
        single_features = self.features[:1]
        
        outputs = self.model(
            input_ids=single_input,
            attention_mask=single_mask,
            features=single_features
        )
        
        assert outputs['logits'].shape == (1,)
        assert outputs['probs'].shape == (1,)


class TestModelPrediction:
    """Test model prediction method"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = create_model()
        self.batch_size = 2
        self.seq_len = 128
        
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones((self.batch_size, self.seq_len))
        self.token_type_ids = torch.zeros((self.batch_size, self.seq_len), dtype=torch.long)
        self.features = torch.randn((self.batch_size, 4))
    
    def test_predict_default_threshold(self):
        """Test prediction with default threshold (0.5)"""
        predictions, probs = self.model.predict(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features
        )
        
        assert predictions.shape == (self.batch_size,)
        assert probs.shape == (self.batch_size,)
        
        # Check predictions are binary (0 or 1)
        assert torch.all((predictions == 0) | (predictions == 1))
        
        # Check probabilities are in [0, 1]
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)
    
    def test_predict_custom_threshold(self):
        """Test prediction with custom threshold"""
        threshold = 0.3
        predictions, probs = self.model.predict(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features,
            threshold=threshold
        )
        
        # Check that predictions match threshold
        expected_predictions = (probs >= threshold).long()
        assert torch.all(predictions == expected_predictions)
    
    def test_predict_high_threshold(self):
        """Test prediction with high threshold (should get more 0s)"""
        threshold = 0.9
        predictions, probs = self.model.predict(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features,
            threshold=threshold
        )
        
        # With high threshold, most predictions should be 0
        assert torch.all(predictions == 0) or torch.mean(predictions.float()) < 0.5
    
    def test_predict_low_threshold(self):
        """Test prediction with low threshold (should get more 1s)"""
        threshold = 0.1
        predictions, probs = self.model.predict(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features,
            threshold=threshold
        )
        
        # With low threshold, most predictions should be 1
        assert torch.all(predictions == 1) or torch.mean(predictions.float()) > 0.5


class TestModelArchitecture:
    """Test model architecture details"""
    
    def test_combined_size_calculation(self):
        """Test that combined size is correctly calculated"""
        model = create_model(num_features=4)
        assert model.combined_size == 768 + 4
        
        model = create_model(num_features=8)
        assert model.combined_size == 768 + 8
    
    def test_classifier_output_shape(self):
        """Test classifier output is single value per sample"""
        model = create_model()
        
        # Create dummy combined features
        batch_size = 3
        combined = torch.randn((batch_size, model.combined_size))
        
        # Pass through classifier
        output = model.classifier(combined)
        
        assert output.shape == (batch_size, 1)
    
    def test_dropout_applied(self):
        """Test that dropout is applied during forward pass"""
        model = create_model(dropout=0.5)
        
        # Create same input twice
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones((2, 128))
        features = torch.randn((2, 4))
        
        # Set model to eval mode (dropout disabled)
        model.eval()
        with torch.no_grad():
            outputs_eval = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                features=features
            )
        
        # Set model to train mode (dropout enabled)
        model.train()
        outputs_train = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            features=features
        )
        
        # Outputs should be different due to dropout
        # (though this is probabilistic, it's very likely)
        assert not torch.allclose(outputs_eval['logits'], outputs_train['logits'])


class TestModelDevice:
    """Test model device handling"""
    
    def test_model_on_cpu(self):
        """Test model on CPU"""
        model = create_model(device='cpu')
        
        # Check model parameters are on CPU
        for param in model.parameters():
            assert param.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_cuda(self):
        """Test model on CUDA (if available)"""
        model = create_model(device='cuda')
        
        # Check model parameters are on CUDA
        for param in model.parameters():
            assert param.device.type == 'cuda'
    
    def test_forward_pass_device_consistency(self):
        """Test that forward pass works with different devices"""
        model = create_model(device='cpu')
        
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones((2, 128))
        features = torch.randn((2, 4))
        
        # Forward pass should work
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            features=features
        )
        
        assert outputs['logits'].device.type == 'cpu'


class TestModelLossCalculation:
    """Test loss calculation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = create_model()
        self.batch_size = 4
        self.seq_len = 128
        
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones((self.batch_size, self.seq_len))
        self.token_type_ids = torch.zeros((self.batch_size, self.seq_len), dtype=torch.long)
        self.features = torch.randn((self.batch_size, 4))
    
    def test_loss_with_balanced_labels(self):
        """Test loss calculation with balanced labels"""
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features,
            labels=labels
        )
        
        assert 'loss' in outputs
        assert outputs['loss'].dim() == 0  # Scalar loss
        assert outputs['loss'].item() >= 0  # Non-negative loss
    
    def test_loss_with_imbalanced_labels(self):
        """Test loss calculation with imbalanced labels"""
        # All positive labels (imbalanced)
        labels = torch.tensor([1.0, 1.0, 1.0, 1.0])
        
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features,
            labels=labels
        )
        
        assert 'loss' in outputs
        assert outputs['loss'].item() >= 0
    
    def test_loss_with_all_negative_labels(self):
        """Test loss calculation with all negative labels"""
        labels = torch.tensor([0.0, 0.0, 0.0, 0.0])
        
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            features=self.features,
            labels=labels
        )
        
        assert 'loss' in outputs
        assert outputs['loss'].item() >= 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
