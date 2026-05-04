"""
IRC Conversation Disentanglement Model - BERT CrossEncoder with Handcrafted Features

Architecture:
1. BERT CrossEncoder processes message pairs
2. Extract [CLS] token embedding (768-dim)
3. Concatenate with 5 handcrafted features → 773-dim vector
4. Linear layer (773 → 1) + Sigmoid for binary classification

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
    - 5 handcrafted features: [time_diff, speaker_match, pos_dist, word_jaccard, directedness]
    
    Output:
    - Probability that message_j is a reply to message_i (0-1)
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_features: int = 5,
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
            input_ids: Token IDs [batch_size, C, seq_len]
            attention_mask: Attention mask [batch_size, C, seq_len]
            token_type_ids: Token type IDs [batch_size, C, seq_len]
            features: Handcrafted features [batch_size, num_features]
            labels: Ground truth labels [batch_size] (gold parent index)
             
        Returns:
            Dictionary with:
            - logits: Raw model outputs [batch_size, C]
            - probs: Softmax probabilities [batch_size, C]
            - loss: CrossEntropy loss (if labels provided)
        """
        batch_size, num_candidates, seq_len = input_ids.shape
        
        # Reshape for BERT: [batch_size * C, seq_len]
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)
        flat_token_type_ids = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None

        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            return_dict=True
        )
        
        # Use [CLS] token embedding
        cls_embedding = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size * C, hidden_size]
        
        # Apply dropout
        cls_embedding = self.dropout(cls_embedding)
        
        # Reshape features to match cls_embedding: [batch_size, num_features] -> [batch_size, C, num_features] -> [batch_size * C, num_features]
        if features is not None:
            expanded_features = features.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, self.num_features)
            combined = torch.cat([cls_embedding, expanded_features], dim=-1)
        else:
            zero_features = torch.zeros(
                cls_embedding.shape[0], self.num_features,
                device=cls_embedding.device,
                dtype=cls_embedding.dtype
            )
            combined = torch.cat([cls_embedding, zero_features], dim=-1)
        
        # Classification head: [batch_size * C, 1]
        logits = self.classifier(combined) 
        
        # Reshape back to [batch_size, C]
        logits = logits.view(batch_size, num_candidates)
        
        # Mask out padded candidates (where attention mask is all zeros or similar)
        # In our case, the collate_fn uses 0 for padding.
        # Check if the whole candidate was padding:
        candidate_mask = attention_mask.sum(dim=-1) > 0 # [batch_size, C]
        logits = logits.masked_fill(~candidate_mask, -1e9)
        
        candidate_probs = torch.softmax(logits, dim=-1)  # [batch_size, C]
        
        # Prepare output
        outputs = {
            'logits': logits,
            'probs': candidate_probs
        }
        
        # Compute loss if labels provided
        if labels is not None:
            # Use CrossEntropyLoss for multiclass classification
            # No pos_weight needed - each sample has exactly 1 positive class
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs['loss'] = loss
        
        return outputs
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make multiclass predictions.
        
        Returns:
            - predictions: Candidate indices (0 to C-1) [batch_size]
            - probabilities: Softmax probabilities [batch_size, C]
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                features=features
            )
            
            probs = outputs['probs']
            predictions = torch.argmax(probs, dim=-1)  # [batch_size]
            
            return predictions, probs


def create_model(
    model_name: str = "microsoft/deberta-v3-base",
    num_features: int = 5,
    dropout: float = 0.1,
    freeze_bert: bool = False,
    device: str = None
) -> CrossEncoderWithFeatures:
    """
    Factory function to create and initialize model.
    
    Args:
        model_name: Pretrained BERT model name (default: DeBERTa-v3-base for SOTA performance)
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
    """Smoke test for multiclass [batch, C, seq] architecture"""
    print("Testing CrossEncoderWithFeatures model (multiclass)...")

    model = create_model(model_name="bert-base-uncased", num_features=5)
    trainable, total = count_parameters(model)
    print(f"  Parameters: {trainable:,} trainable, {total:,} total")

    batch_size = 2
    num_candidates = 5
    seq_len = 32

    input_ids = torch.randint(0, 1000, (batch_size, num_candidates, seq_len))
    attention_mask = torch.ones((batch_size, num_candidates, seq_len), dtype=torch.long)
    features = torch.randn((batch_size, 5))
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

    print("  test_model() passed.")
    return model


if __name__ == "__main__":
    # Run test
    model = test_model()