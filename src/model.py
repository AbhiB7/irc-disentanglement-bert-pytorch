"""
IRC Conversation Disentanglement Model - BERT CrossEncoder with Handcrafted Features

Architecture:
1. BERT CrossEncoder processes message pairs
2. Extract [CLS] token embedding (768-dim)
3. Concatenate with 4 handcrafted features → 772-dim vector
4. Linear layer (772 → 1) + Sigmoid for binary classification

Matches the architecture described in context/CONTEXT.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple, Dict


class CrossEncoderWithFeatures(nn.Module):
    """
    BERT-based CrossEncoder with additional handcrafted features.
    
    Input: 
    - Tokenized message pairs (input_ids, attention_mask, token_type_ids)
    - 4 handcrafted features: [time_diff, speaker_match, pos_dist, word_jaccard]
    
    Output:
    - Probability that message_j is a reply to message_i (0-1)
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_features: int = 4,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        super().__init__()
        
        # Load BERT model for CrossEncoder
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Freeze BERT layers if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # BERT hidden size (typically 768 for bert-base-uncased)
        bert_hidden_size = self.config.hidden_size
        
        # Combined feature size
        combined_size = bert_hidden_size + num_features
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(combined_size, 1)
        
        # Initialize classifier weights
        self._init_weights(self.classifier)
        
        # Store dimensions for reference
        self.bert_hidden_size = bert_hidden_size
        self.num_features = num_features
        self.combined_size = combined_size
    
    def _init_weights(self, module):
        """Initialize weights for linear layers"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (segment IDs) [batch_size, seq_len]
            features: Handcrafted features [batch_size, num_features]
            labels: Ground truth labels [batch_size]
            
        Returns:
            Dictionary with:
            - logits: Raw model outputs [batch_size]
            - probs: Sigmoid probabilities [batch_size]
            - loss: BCE loss (if labels provided)
        """
        batch_size = input_ids.shape[0]
        
        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Use [CLS] token embedding (first token)
        cls_embedding = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply dropout
        cls_embedding = self.dropout(cls_embedding)
        
        # Concatenate with handcrafted features
        if features is not None:
            # Ensure features have correct shape
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            # Verify feature dimension
            if features.shape[-1] != self.num_features:
                raise ValueError(
                    f"Expected {self.num_features} features, got {features.shape[-1]}"
                )
            
            # Concatenate BERT embedding with features
            combined = torch.cat([cls_embedding, features], dim=-1)  # [batch_size, hidden_size + num_features]
        else:
            # If no features provided, use zero-padded features
            zero_features = torch.zeros(
                batch_size, self.num_features,
                device=cls_embedding.device,
                dtype=cls_embedding.dtype
            )
            combined = torch.cat([cls_embedding, zero_features], dim=-1)
        
        # Classification head
        logits = self.classifier(combined).squeeze(-1)  # [batch_size]
        probs = torch.sigmoid(logits)
        
        # Prepare output
        outputs = {
            'logits': logits,
            'probs': probs
        }
        
        # Compute loss if labels provided
        if labels is not None:
            # Dynamic pos_weight based on actual batch label distribution
            # Clamp prevents explosion on batches with zero positives and caps at 300
            num_neg = (labels == 0).sum().float()
            num_pos = (labels == 1).sum().float()
            pos_weight = (num_neg / (num_pos + 1e-8)).clamp(min=10.0, max=300.0)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
            loss = loss_fn(logits, labels)
            outputs['loss'] = loss
        
        return outputs
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with thresholding.
        
        Returns:
            - predictions: Binary predictions (0 or 1) [batch_size]
            - probabilities: Sigmoid probabilities [batch_size]
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                features=features
            )
            
            probs = outputs['probs']
            predictions = (probs >= threshold).long()
            
            return predictions, probs


def create_model(
    model_name: str = "bert-base-uncased",
    num_features: int = 4,
    dropout: float = 0.1,
    freeze_bert: bool = False,
    device: str = None
) -> CrossEncoderWithFeatures:
    """
    Factory function to create and initialize model.
    
    Args:
        model_name: Pretrained BERT model name
        num_features: Number of handcrafted features
        dropout: Dropout probability
        freeze_bert: Whether to freeze BERT parameters
        device: Device to load model on (cuda/cpu)
        
    Returns:
        Initialized CrossEncoderWithFeatures model
    """
    model = CrossEncoderWithFeatures(
        model_name=model_name,
        num_features=num_features,
        dropout=dropout,
        freeze_bert=freeze_bert
    )
    
    # Move to device if specified
    if device:
        model = model.to(device)
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters.
    
    Returns:
        (trainable_params, total_params)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


# Test function to verify model works
def test_model():
    """Test the model with dummy data"""
    print("Testing CrossEncoderWithFeatures model...")
    
    # Create model
    model = create_model(
        model_name="bert-base-uncased",
        num_features=4,
        dropout=0.1,
        freeze_bert=False
    )
    
    # Count parameters
    trainable, total = count_parameters(model)
    print(f"  Parameters: {trainable:,} trainable, {total:,} total")
    print(f"  BERT hidden size: {model.bert_hidden_size}")
    print(f"  Combined size: {model.combined_size}")
    
    # Create dummy batch
    batch_size = 2
    seq_len = 128
    
    # Random inputs (simulating tokenized message pairs)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    features = torch.randn((batch_size, 4))
    labels = torch.tensor([1.0, 0.0])
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        features=features,
        labels=labels
    )
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Probs shape: {outputs['probs'].shape}")
    print(f"  Loss: {outputs.get('loss', 'N/A')}")
    
    # Test prediction
    predictions, probs = model.predict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        features=features,
        threshold=0.5
    )
    
    print(f"  Predictions: {predictions}")
    print(f"  Probabilities: {probs}")
    
    # Verify architecture
    assert outputs['logits'].shape == (batch_size,), "Logits should have shape [batch_size]"
    assert outputs['probs'].shape == (batch_size,), "Probs should have shape [batch_size]"
    assert model.combined_size == 768 + 4, f"Combined size should be 772, got {model.combined_size}"
    
    print("\n✓ Model test passed!")
    return model


if __name__ == "__main__":
    # Run test
    model = test_model()